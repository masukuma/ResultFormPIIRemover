//! `POST /v1/verify`: anyone can check a published form against its
//! manifest (and, if they have them, the original, the publisher's key and
//! the ledger) with `rfpii verify`. Public, free, limited per IP address;
//! the files are deleted when the check ends.

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

use crate::problem::{ApiError, ApiResult};
use crate::rfpii::VerifyFiles;
use crate::server::AppState;
use crate::storage::{self, Kind};

/// The form fields, and the file each is saved as.
const FIELDS: [(&str, &str); 5] =
    [("pdf", "redacted.pdf"), ("manifest", "redacted.manifest.json"), ("original", "original"), ("pubkey", "signing_key.pub.pem"), ("ledger", "ledger.jsonl")];

/// Removes the check's folder when dropped.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub async fn verify(State(s): State<AppState>, mut mp: Multipart) -> ApiResult<axum::Json<Value>> {
    let dir = TempDir(storage::tmp_dir(&s.cfg.data_dir).join(crate::ids::new_id("verify")));
    tokio::fs::create_dir_all(&dir.0).await.map_err(|e| ApiError::internal(&e.into()))?;
    let mut got: Vec<(&str, PathBuf)> = Vec::new();
    while let Some(mut field) = mp.next_field().await.map_err(|e| multipart_error(&s, e))? {
        let Some((name, file)) = FIELDS.iter().find(|(n, _)| Some(*n) == field.name()).copied() else { continue };
        if got.iter().any(|(n, _)| *n == name) {
            return Err(ApiError::bad_request(format!("the field {name:?} is given twice")));
        }
        let path = dir.0.join(file);
        let mut f = tokio::fs::File::create(&path).await.map_err(|e| ApiError::internal(&e.into()))?;
        while let Some(chunk) = field.chunk().await.map_err(|e| multipart_error(&s, e))? {
            f.write_all(&chunk).await.map_err(|e| ApiError::internal(&e.into()))?;
        }
        f.flush().await.map_err(|e| ApiError::internal(&e.into()))?;
        got.push((name, path));
    }
    let files = arrange(&dir.0, &got)?;
    let _slot = s.verify_slots.acquire().await.map_err(|e| ApiError::internal(&e.into()))?;
    let v = crate::rfpii::verify(&s.cfg.rfpii_bin, &files).await.map_err(|e| ApiError::internal(&e))?;
    Ok(axum::Json(json!({"object": "verification", "verified": v.verified, "checks": v.checks, "notes": v.notes, "result": v.result})))
}

fn multipart_error(s: &AppState, e: axum::extract::multipart::MultipartError) -> ApiError {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::too_large(format!("a verification is accepted up to {} bytes", s.cfg.max_verify_bytes), s.cfg.max_verify_bytes)
    } else {
        ApiError::bad_request(e.body_text())
    }
}

/// The files for `rfpii verify`: the PDF (which must be one) and the
/// manifest are required; the original keeps the extension of its kind,
/// which `rfpii` reads it by.
fn arrange(dir: &Path, got: &[(&str, PathBuf)]) -> ApiResult<VerifyFiles> {
    let get = |n: &str| got.iter().find(|(k, _)| *k == n).map(|(_, p)| p.clone());
    let (Some(pdf), Some(manifest)) = (get("pdf"), get("manifest")) else {
        return Err(ApiError::validation("the fields \"pdf\" and \"manifest\" are required (\"original\", \"pubkey\" and \"ledger\" are optional)"));
    };
    if storage::sniff(&head(&pdf)) != Some(Kind::Pdf) {
        return Err(ApiError::unsupported_media("the field \"pdf\" is not a PDF"));
    }
    let original = match get("original") {
        Some(p) => {
            let kind = storage::sniff(&head(&p)).filter(|k| *k != Kind::Zip).ok_or_else(|| ApiError::unsupported_media("the original must be a PDF, JPEG, PNG or TIFF"))?;
            let named = dir.join(format!("original.{}", kind.ext()));
            std::fs::rename(&p, &named).map_err(|e| ApiError::internal(&e.into()))?;
            Some(named)
        }
        None => None,
    };
    Ok(VerifyFiles { pdf, manifest, original, pubkey: get("pubkey"), ledger: get("ledger") })
}

fn head(p: &Path) -> Vec<u8> {
    use std::io::Read;
    let mut buf = vec![0u8; storage::HEAD];
    let n = std::fs::File::open(p).and_then(|mut f| f.read(&mut buf)).unwrap_or(0);
    buf.truncate(n);
    buf
}
