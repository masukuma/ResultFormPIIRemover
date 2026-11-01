//! `/v1/files`: uploads (multipart, or the raw body with its Content-Type),
//! checked by their first bytes and size, zips checked before they are
//! accepted. Stored by id under the account. Originals are never served
//! back: a file can be described and deleted, not downloaded.

use axum::Extension;
use axum::body::Body;
use axum::extract::{FromRequest, Multipart, Path, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use http_body_util::BodyExt;
use rusqlite::{OptionalExtension, Row, params};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiQuery, PageQuery, db, page};
use crate::auth::{Caller, Scope};
use crate::problem::{ApiError, ApiResult};
use crate::server::AppState;
use crate::storage::{self, Kind, Received, Sink};

/// A stored file, as described.
pub struct FileRow {
    pub id: String,
    pub account_id: String,
    pub name: Option<String>,
    pub kind: String,
    pub bytes: i64,
    pub sha256: String,
    pub created_at: i64,
    pub deleted_at: Option<i64>,
    pub deleted_why: Option<String>,
}

pub const COLUMNS: &str = "id, account_id, name, kind, bytes, sha256, created_at, deleted_at, deleted_why";

pub fn from_row(r: &Row) -> rusqlite::Result<FileRow> {
    Ok(FileRow {
        id: r.get(0)?,
        account_id: r.get(1)?,
        name: r.get(2)?,
        kind: r.get(3)?,
        bytes: r.get(4)?,
        sha256: r.get(5)?,
        created_at: r.get(6)?,
        deleted_at: r.get(7)?,
        deleted_why: r.get(8)?,
    })
}

impl FileRow {
    pub fn resource(&self, unused_hours: i64) -> Value {
        json!({
            "id": self.id, "object": "file", "name": self.name, "kind": self.kind, "bytes": self.bytes, "sha256": self.sha256,
            "status": if self.deleted_at.is_some() { "deleted" } else { "available" },
            "created_at": crate::clock::rfc3339(self.created_at),
            "delete_unused_after": crate::clock::rfc3339(self.created_at + unused_hours * 3600),
            "deleted_at": self.deleted_at.map(crate::clock::rfc3339),
            "deleted_why": self.deleted_why,
        })
    }
}

pub fn load(conn: &rusqlite::Connection, account: &str, id: &str) -> anyhow::Result<Option<FileRow>> {
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM files WHERE id = ?1 AND account_id = ?2"), params![id, account], from_row).optional()?)
}

#[derive(Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct UploadQuery {
    /// the file's name, for a raw upload
    pub name: Option<String>,
}

/// `POST /v1/files`.
pub async fn create(State(s): State<AppState>, Extension(caller): Extension<Caller>, ApiQuery(q): ApiQuery<UploadQuery>, req: Request) -> ApiResult<Response> {
    caller.require(Scope::JobsWrite)?;
    let multipart = req.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("multipart/form-data"));
    let (received, name) = if multipart {
        let mp = Multipart::from_request(req, &s).await.map_err(|e| ApiError::bad_request(e.body_text()))?;
        receive_multipart(&s, mp).await?
    } else {
        (receive_body(&s, req.into_body()).await?, q.name)
    };
    let kind = match accept(&s, &received).await {
        Ok(k) => k,
        Err(e) => {
            let _ = tokio::fs::remove_file(&received.path).await;
            return Err(e);
        }
    };
    let row = store(&s, &caller, received, kind, name).await?;
    let mut res = (StatusCode::CREATED, axum::Json(row.resource(s.cfg.retention_unused_file_hours))).into_response();
    if let Ok(v) = format!("/v1/files/{}", row.id).parse() {
        res.headers_mut().insert(header::LOCATION, v);
    }
    Ok(res)
}

fn too_large(s: &AppState) -> ApiError {
    ApiError::too_large(format!("files are accepted up to {} bytes", s.cfg.max_upload_bytes), s.cfg.max_upload_bytes)
}

/// Write a raw body to disk, counting it against the limit.
async fn receive_body(s: &AppState, body: Body) -> ApiResult<Received> {
    let mut sink = Sink::create(&s.cfg.data_dir, s.cfg.max_upload_bytes).await.map_err(|e| ApiError::internal(&e.into()))?;
    let mut body = body;
    while let Some(frame) = body.frame().await {
        let frame = match frame {
            Ok(f) => f,
            Err(e) => {
                sink.discard().await;
                return Err(ApiError::bad_request(format!("the upload was cut off: {e}")));
            }
        };
        if let Ok(chunk) = frame.into_data()
            && let Err(e) = write(&mut sink, &chunk).await
        {
            sink.discard().await;
            return Err(e.unwrap_or_else(|| too_large(s)));
        }
    }
    finish(sink).await
}

/// Write a chunk; Err(None) when it passes the limit.
async fn write(sink: &mut Sink, chunk: &[u8]) -> Result<(), Option<ApiError>> {
    match sink.write(chunk).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(Some(ApiError::internal(&e.into()))),
        Err(storage::TooLarge) => Err(None),
    }
}

async fn finish(sink: Sink) -> ApiResult<Received> {
    if sink.bytes == 0 {
        sink.discard().await;
        return Err(ApiError::bad_request("the upload is empty"));
    }
    sink.finish().await.map_err(|e| ApiError::internal(&e.into()))
}

/// Write the multipart field `file` (or the first field with a file name)
/// to disk; with the name the client gave it.
async fn receive_multipart(s: &AppState, mut mp: Multipart) -> ApiResult<(Received, Option<String>)> {
    while let Some(mut field) = mp.next_field().await.map_err(|e| ApiError::bad_request(e.body_text()))? {
        if field.name() != Some("file") && field.file_name().is_none() {
            continue;
        }
        let name = field.file_name().map(String::from);
        let mut sink = Sink::create(&s.cfg.data_dir, s.cfg.max_upload_bytes).await.map_err(|e| ApiError::internal(&e.into()))?;
        loop {
            let chunk = match field.chunk().await {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(e) => {
                    sink.discard().await;
                    let status = e.status();
                    return Err(if status == StatusCode::PAYLOAD_TOO_LARGE { too_large(s) } else { ApiError::bad_request(e.body_text()) });
                }
            };
            if let Err(e) = write(&mut sink, &chunk).await {
                sink.discard().await;
                return Err(e.unwrap_or_else(|| too_large(s)));
            }
        }
        return Ok((finish(sink).await?, name));
    }
    Err(ApiError::bad_request("no file in the form: send it as the field \"file\""))
}

/// The file's kind, from its first bytes; a zip must pass its checks.
async fn accept(s: &AppState, r: &Received) -> ApiResult<Kind> {
    let kind = storage::sniff(&r.head).ok_or_else(|| {
        ApiError::unsupported_media("the file is not a PDF, JPEG, PNG, TIFF or zip (told by its first bytes, not its name)")
            .with("accepted", json!(["pdf", "jpeg", "png", "tiff", "zip"]))
    })?;
    if kind == Kind::Zip {
        let (path, entries, bytes) = (r.path.clone(), s.cfg.max_zip_entries, s.cfg.max_zip_uncompressed_bytes);
        let checked = tokio::task::spawn_blocking(move || storage::check_zip(&path, entries, bytes)).await.map_err(|e| ApiError::internal(&e.into()))?;
        if let Err(why) = checked {
            return Err(ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, Some("unsafe-zip"), "Zip refused", format!("the zip was refused: {why}"))
                .with("reason", why)
                .with("max_entries", entries)
                .with("max_uncompressed_bytes", bytes));
        }
    }
    Ok(kind)
}

/// Record the file and move it to its place.
async fn store(s: &AppState, caller: &Caller, r: Received, kind: Kind, name: Option<String>) -> ApiResult<FileRow> {
    let id = crate::ids::new_id("file");
    let dest = storage::file_path(&s.cfg.data_dir, &caller.account_id, &id);
    if let Some(dir) = dest.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| ApiError::internal(&e.into()))?;
    }
    tokio::fs::rename(&r.path, &dest).await.map_err(|e| ApiError::internal(&e.into()))?;
    let safe = name.as_deref().and_then(|n| storage::safe_name(n, kind));
    let (now, account) = (s.clock.now(), caller.account_id.clone());
    let row = db(s, move |c| {
        c.execute(
            "INSERT INTO files (id, account_id, name, kind, bytes, sha256, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![id, account, safe, kind.name(), r.bytes as i64, r.sha256, now],
        )?;
        Ok(c.query_row(&format!("SELECT {COLUMNS} FROM files WHERE id = ?1"), [&id], from_row)?)
    })
    .await;
    if row.is_err() {
        let _ = tokio::fs::remove_file(&dest).await;
    }
    row
}

/// `GET /v1/files`.
pub async fn list(State(s): State<AppState>, Extension(caller): Extension<Caller>, ApiQuery(q): ApiQuery<PageQuery>) -> ApiResult<Response> {
    caller.require(Scope::JobsRead)?;
    let limit = q.limit()?;
    let (account, after) = (caller.account_id.clone(), q.after.clone());
    let (rows, more) = db(&s, move |c| {
        let mut st = c.prepare(&format!("SELECT {COLUMNS} FROM files WHERE account_id = ?1 AND (?2 IS NULL OR id < ?2) ORDER BY id DESC LIMIT ?3"))?;
        let mut rows = st.query_map(params![account, after, limit + 1], from_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
        let more = rows.len() as i64 > limit;
        rows.truncate(limit as usize);
        Ok((rows, more))
    })
    .await?;
    let h = s.cfg.retention_unused_file_hours;
    Ok(page("/v1/files", rows.iter().map(|r| r.resource(h)).collect(), more, limit))
}

/// `GET /v1/files/{id}`: the file described; its content is never served.
pub async fn get(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::JobsRead)?;
    let account = caller.account_id.clone();
    let row = db(&s, move |c| load(c, &account, &id)).await?.ok_or_else(|| ApiError::not_found("file"))?;
    Ok(axum::Json(row.resource(s.cfg.retention_unused_file_hours)))
}

/// `DELETE /v1/files/{id}`: refused while a job waiting or running uses it.
pub async fn remove(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::JobsWrite)?;
    let (account, now, data_dir) = (caller.account_id.clone(), s.clock.now(), s.cfg.data_dir.clone());
    let row = db(&s, move |c| {
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let Some(f) = load(&tx, &account, &id)? else { return Ok(Err(ApiError::not_found("file"))) };
        let busy: i64 =
            tx.query_row("SELECT COUNT(*) FROM job_files jf JOIN jobs j ON j.id = jf.job_id WHERE jf.file_id = ?1 AND j.status IN ('queued', 'running')", [&id], |r| r.get(0))?;
        if busy > 0 {
            return Ok(Err(ApiError::conflict("a job that is queued or running uses this file; cancel it first")));
        }
        if f.deleted_at.is_none() {
            storage::remove(&storage::file_path(&data_dir, &account, &id))?;
            let why = "deleted at the client's request";
            tx.execute("UPDATE files SET deleted_at = ?2, deleted_why = ?3 WHERE id = ?1", params![id, now, why])?;
            crate::audit::deletion(&tx, now, &account, None, Some(&id), &format!("files/{account}/{id}"), why)?;
        }
        let f = load(&tx, &account, &id)?.ok_or_else(|| anyhow::anyhow!("file vanished"))?;
        tx.commit()?;
        Ok(Ok(f))
    })
    .await??;
    Ok(axum::Json(row.resource(s.cfg.retention_unused_file_hours)))
}
