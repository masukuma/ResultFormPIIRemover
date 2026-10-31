//! `/v1/jobs`: create a job (a long-running operation: `202 Accepted` and
//! a `Location`), list, read, cancel and delete jobs, and read what they
//! released.

use axum::Extension;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use rusqlite::params;
use serde_json::{Value, json};
use std::path::PathBuf;

use super::{ApiJson, ApiQuery, PageQuery, db, page};
use crate::accounts::{self, Account};
use crate::auth::{Caller, Scope};
use crate::http::files::{self, FileRow};
use crate::jobs::{self, JobRequest, JobRow};
use crate::problem::{ApiError, ApiResult};
use crate::rfpii::{self, Counted, JobFile};
use crate::server::AppState;
use crate::storage::{self, Kind};
use crate::worker::JobPaths;

/// Longest instructions accepted, in characters.
const MAX_INSTRUCTIONS: usize = 20_000;
/// Largest metadata object, as JSON.
const MAX_METADATA_BYTES: usize = 8 * 1024;

/// `POST /v1/jobs`.
pub async fn create(State(s): State<AppState>, Extension(caller): Extension<Caller>, headers: HeaderMap, ApiJson(body): ApiJson<Value>) -> ApiResult<Response> {
    caller.require(Scope::JobsWrite)?;
    let req: JobRequest = serde_json::from_value(body.clone()).map_err(|e| ApiError::validation(format!("{e}")))?;
    let key = idempotency_key(&headers)?;
    let hash = jobs::request_hash(&body);
    if let Some(k) = &key
        && let Some(found) = replay(&s, &caller, k, &hash).await?
    {
        return Ok(found);
    }
    validate(&s, &req)?;
    let account = load_account(&s, &caller).await?;
    ready(&account, &req)?;
    let files = load_files(&s, &caller, &req.file_ids).await?;
    let job_id = crate::ids::new_id("job");
    let paths = JobPaths::new(&s.cfg.data_dir, &caller.account_id, &job_id);
    match prepare_and_insert(&s, &caller, &account, &req, &files, &job_id, &paths, key.as_deref(), &hash).await {
        Ok(Inserted::New(job, warnings)) => {
            tracing::info!(job = %job.id, forms = job.forms_requested, "job queued");
            let mut v = job.resource();
            v["warnings"] = json!(warnings);
            Ok(accepted(&job.id, v, false))
        }
        Ok(Inserted::Replay(job)) => {
            let _ = storage::remove(&paths.dir);
            Ok(accepted(&job.id, job.resource(), true))
        }
        Err(e) => {
            let _ = storage::remove(&paths.dir);
            Err(e)
        }
    }
}

/// `202 Accepted`, with the job's `Location` (RFC 9110 §15.3.3).
fn accepted(id: &str, body: Value, replayed: bool) -> Response {
    let mut res = (StatusCode::ACCEPTED, axum::Json(body)).into_response();
    if let Ok(v) = HeaderValue::from_str(&format!("/v1/jobs/{id}")) {
        res.headers_mut().insert(header::LOCATION, v);
    }
    if replayed {
        res.headers_mut().insert("idempotent-replayed", HeaderValue::from_static("true"));
    }
    res
}

/// The `Idempotency-Key` header, if given: 1 to 255 visible characters.
fn idempotency_key(h: &HeaderMap) -> ApiResult<Option<String>> {
    let Some(v) = h.get("idempotency-key") else { return Ok(None) };
    let k = v.to_str().map_err(|_| ApiError::bad_request("Idempotency-Key must be visible ASCII"))?.trim().to_string();
    if k.is_empty() || k.len() > 255 || !k.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(ApiError::bad_request("Idempotency-Key must be 1 to 255 visible ASCII characters"));
    }
    Ok(Some(k))
}

fn key_reused() -> ApiError {
    ApiError::new(
        StatusCode::UNPROCESSABLE_ENTITY,
        Some("idempotency-key-reused"),
        "Idempotency key reused",
        "this Idempotency-Key was used in the last 24 hours with a different request",
    )
}

/// The job an earlier request with the same key created, if there is one;
/// 422 if that request was a different one.
async fn replay(s: &AppState, caller: &Caller, key: &str, hash: &str) -> ApiResult<Option<Response>> {
    let (account, key, hash, now) = (caller.account_id.clone(), key.to_string(), hash.to_string(), s.clock.now());
    let found = db(s, move |c| {
        let Some((h, id)) = jobs::idempotent(c, &account, &key, now)? else { return Ok(None) };
        Ok(Some((h == hash, jobs::load(c, &account, &id)?)))
    })
    .await?;
    match found {
        None => Ok(None),
        Some((false, _)) => Err(key_reused()),
        Some((true, Some(job))) => Ok(Some(accepted(&job.id, job.resource(), true))),
        Some((true, None)) => Ok(None),
    }
}

fn validate(s: &AppState, req: &JobRequest) -> ApiResult<()> {
    let n = req.file_ids.len();
    if n == 0 || n > s.cfg.max_files_per_job {
        return Err(ApiError::validation(format!("file_ids must name 1 to {} files", s.cfg.max_files_per_job)));
    }
    let mut seen = std::collections::BTreeSet::new();
    if !req.file_ids.iter().all(|f| f.len() <= 64 && seen.insert(f)) {
        return Err(ApiError::validation("file_ids must be distinct file ids"));
    }
    if req.instructions.as_ref().is_some_and(|i| i.chars().count() > MAX_INSTRUCTIONS) {
        return Err(ApiError::validation(format!("instructions are limited to {MAX_INSTRUCTIONS} characters")));
    }
    if let Some(m) = &req.metadata
        && (!m.is_object() || m.to_string().len() > MAX_METADATA_BYTES)
    {
        return Err(ApiError::validation(format!("metadata must be a JSON object of at most {MAX_METADATA_BYTES} bytes")));
    }
    if let Some(u) = &req.webhook_url {
        crate::webhooks::check_url(u, s.cfg.webhook_allow_private).map_err(ApiError::validation)?;
    }
    Ok(())
}

async fn load_account(s: &AppState, caller: &Caller) -> ApiResult<Account> {
    let id = caller.account_id.clone();
    db(s, move |c| accounts::load(c, &id)).await?.ok_or_else(|| ApiError::not_found("account"))
}

/// 422 until the account has what `rfpii job` needs: a controller record
/// and a DPIA (HARNESS.md §9, conditions 8 and 9), and a webhook secret if
/// a webhook is asked for.
fn ready(account: &Account, req: &JobRequest) -> ApiResult<()> {
    let missing = accounts::missing_for_jobs(account);
    if !missing.is_empty() {
        let how: Vec<&str> = missing.iter().map(|m| if *m == "controller" { "PUT /v1/account/controller" } else { "PUT /v1/account/dpia" }).collect();
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            Some("account-not-ready"),
            "Account not ready for jobs",
            format!(
                "a job needs the account's {} first ({}): rfpii job refuses to run without a controller record, and its release gate needs a DPIA",
                missing.join(" and "),
                how.join(", ")
            ),
        )
        .with("missing", json!(missing)));
    }
    if req.webhook_url.is_some() && account.webhook_secret.is_none() {
        return Err(ApiError::validation("a webhook needs the account's signing secret: create it with POST /v1/account/webhook-secret"));
    }
    Ok(())
}

/// The files named, which must be the account's and not deleted.
async fn load_files(s: &AppState, caller: &Caller, ids: &[String]) -> ApiResult<Vec<FileRow>> {
    let (account, ids) = (caller.account_id.clone(), ids.to_vec());
    let found: Vec<(String, Option<FileRow>)> = db(s, move |c| ids.into_iter().map(|id| Ok((id.clone(), files::load(c, &account, &id)?))).collect()).await?;
    let bad: Vec<String> = found.iter().filter(|(_, f)| f.as_ref().is_none_or(|f| f.deleted_at.is_some())).map(|(id, _)| id.clone()).collect();
    if !bad.is_empty() {
        return Err(ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, Some("unknown-files"), "Unknown files", "some file ids are not files of this account, or were deleted")
            .with("file_ids", json!(bad)));
    }
    Ok(found.into_iter().filter_map(|(_, f)| f).collect())
}

enum Inserted {
    New(JobRow, Vec<String>),
    Replay(JobRow),
}

/// Make the job's folder (its inputs, controller record and job file),
/// count its forms with `rfpii inventory`, then, in one transaction, check
/// the plan's allowance, reserve the forms and queue the job.
#[allow(clippy::too_many_arguments)]
async fn prepare_and_insert(
    s: &AppState,
    caller: &Caller,
    account: &Account,
    req: &JobRequest,
    files: &[FileRow],
    job_id: &str,
    paths: &JobPaths,
    key: Option<&str>,
    hash: &str,
) -> ApiResult<Inserted> {
    let names = link_inputs(s, &caller.account_id, files, paths)?;
    let inv = rfpii::inventory(&s.cfg.rfpii_bin, &paths.inputs()).await.map_err(|e| ApiError::internal(&e.context("counting the job's forms")))?;
    let per_page = req.per_page.map(|p| p.value());
    let counted = rfpii::count_forms(&inv, per_page.unwrap_or_default());
    if counted.forms == 0 {
        return Err(
            ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, Some("no-forms"), "No forms", "none of the files holds a page rfpii can read").with("inventory", json!(counted))
        );
    }
    if counted.forms > s.cfg.max_forms_per_job {
        return Err(ApiError::validation(format!("the job holds {} forms; at most {} are accepted in one job", counted.forms, s.cfg.max_forms_per_job)));
    }
    write_job_files(s, account, req, &counted, paths).map_err(|e| ApiError::internal(&e))?;
    let now = s.clock.now();
    let warnings = warnings(s, account, req, now);
    let stored =
        json!({"file_ids": req.file_ids, "instructions": req.instructions, "policy": req.policy, "per_page": per_page.map(|p| p.name()), "held_pages": account.held_pages});
    let (s2, account, job_id, key, hash) = (s.clone(), account.clone(), job_id.to_string(), key.map(String::from), hash.to_string());
    let (webhook, metadata) = (req.webhook_url.clone(), req.metadata.clone());
    db(s, move |c| {
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        if let Some(k) = &key
            && let Some((h, id)) = jobs::idempotent(&tx, &account.id, k, now)? {
                return Ok(if h == hash { jobs::load(&tx, &account.id, &id)?.map(Inserted::Replay).ok_or_else(key_reused) } else { Err(key_reused()) });
            }
        let period = crate::clock::period_of(now);
        if let Err(e) = crate::billing::admit(&tx, &account, counted.forms, &period, &s2.prices, &s2.cfg) {
            return Ok(Err(e));
        }
        tx.execute(
            "INSERT INTO jobs (id, account_id, status, request_json, plan, unit_price_microusd, period, forms_requested, reserved_forms, inventory_json, max_attempts, webhook_url, metadata_json, created_at)
             VALUES (?1, ?2, 'queued', ?3, ?4, ?5, ?6, ?7, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![job_id, account.id, stored.to_string(), account.plan, if account.plan == "paid" { s2.prices.unit_microusd } else { 0 }, period, counted.forms,
                    json!(counted).to_string(), s2.cfg.max_attempts, webhook, metadata.map(|m| m.to_string()), now],
        )?;
        for (file, name) in &names {
            tx.execute("INSERT INTO job_files (job_id, file_id, name) VALUES (?1, ?2, ?3)", params![job_id, file, name])?;
        }
        if let Some(k) = &key {
            tx.execute("INSERT INTO idempotency_keys (account_id, key, request_hash, job_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5)", params![account.id, k, hash, job_id, now])?;
        }
        let job = jobs::load(&tx, &account.id, &job_id)?.ok_or_else(|| anyhow::anyhow!("job vanished"))?;
        tx.commit()?;
        Ok(Ok(Inserted::New(job, warnings)))
    })
    .await?
}

/// Link each file into the job's `in/` folder, under the client's name when
/// it is safe and not taken, else under its id; (file id, name) pairs.
fn link_inputs(s: &AppState, account: &str, files: &[FileRow], paths: &JobPaths) -> ApiResult<Vec<(String, String)>> {
    let dir = paths.inputs();
    std::fs::create_dir_all(&dir).map_err(|e| ApiError::internal(&e.into()))?;
    let mut taken = std::collections::BTreeSet::new();
    let mut names = Vec::new();
    for f in files {
        let kind = Kind::parse(&f.kind).unwrap_or(Kind::Pdf);
        let name = f.name.clone().filter(|n| taken.insert(n.to_lowercase())).unwrap_or_else(|| format!("{}.{}", f.id, kind.ext()));
        taken.insert(name.to_lowercase());
        let src = storage::file_path(&s.cfg.data_dir, account, &f.id);
        let dst = dir.join(&name);
        if std::fs::hard_link(&src, &dst).is_err() {
            std::fs::copy(&src, &dst).map_err(|e| ApiError::internal(&anyhow::Error::from(e).context(format!("copying file {}", f.id))))?;
        }
        names.push((f.id.clone(), name));
    }
    Ok(names)
}

/// Write `controller.json` and `job.yaml` into the job's folder. The job
/// runs under the account's records and settings as they are now.
fn write_job_files(s: &AppState, account: &Account, req: &JobRequest, counted: &Counted, paths: &JobPaths) -> anyhow::Result<()> {
    let controller = accounts::effective_controller(account.controller.as_ref().unwrap_or(&Value::Null), &account.held_pages, account.dpia.as_ref(), &s.cfg);
    std::fs::write(paths.controller(), serde_json::to_vec_pretty(&controller)?)?;
    let cfg = &s.cfg;
    let jf = JobFile {
        inputs: vec![paths.inputs()],
        instructions: req.instructions.clone().unwrap_or_default(),
        policy: jobs::policy_value(req.policy.as_ref(), req.per_page.map(|p| p.value())),
        outputs: paths.out(),
        sign_key: std::path::absolute(&cfg.sign_key)?,
        controller: paths.controller(),
        dpia: account.dpia.clone().unwrap_or(Value::Null),
        templates: std::path::absolute(&cfg.rfpii_templates)?,
        held_pages: account.held_pages.clone(),
        models: rfpii::models_of(&cfg.orchestrator_models, &cfg.vision_models, cfg.ollama_host.as_deref()),
        budget: json!({"model_requests": (counted.forms as u64).saturating_mul(cfg.model_requests_per_form).max(1), "hours": cfg.job_timeout_secs as f64 / 3600.0}),
        certificate: cfg.certificate.as_ref().map(std::path::absolute).transpose()?,
        kenya_hosts: cfg.kenya_hosts.clone(),
    };
    std::fs::write(paths.job_file(), serde_yaml::to_string(&jf)?)?;
    Ok(())
}

/// What the client should know before the job runs: reasons the release
/// gate will hold everything.
fn warnings(s: &AppState, account: &Account, req: &JobRequest, now: i64) -> Vec<String> {
    let mut w = Vec::new();
    if let Some(why) = account.dpia.as_ref().and_then(|d| accounts::dpia_bars(d, now)) {
        w.push(format!("the release gate will hold every form: {why}"));
    }
    if s.cfg.vision_models.is_empty() {
        w.push("no vision model is configured on this server: without the visual review the release gate holds every form".into());
    }
    let instructions = req.instructions.as_deref().is_some_and(|i| !i.trim().is_empty());
    if instructions && s.cfg.orchestrator_models.is_empty() && req.policy.is_none() && req.per_page.is_none() {
        w.push("no orchestrator model is configured to compile the instructions and no policy is given: nothing will be released".into());
    }
    w
}

/// `GET /v1/jobs`.
pub async fn list(State(s): State<AppState>, Extension(caller): Extension<Caller>, ApiQuery(q): ApiQuery<PageQuery>) -> ApiResult<Response> {
    caller.require(Scope::JobsRead)?;
    let limit = q.limit()?;
    let (account, after) = (caller.account_id.clone(), q.after.clone());
    let (rows, more) = db(&s, move |c| jobs::list(c, &account, limit, after.as_deref())).await?;
    Ok(page("/v1/jobs", rows.iter().map(JobRow::resource).collect(), more, limit))
}

async fn load_job(s: &AppState, caller: &Caller, id: String) -> ApiResult<JobRow> {
    let account = caller.account_id.clone();
    db(s, move |c| jobs::load(c, &account, &id)).await?.ok_or_else(|| ApiError::not_found("job"))
}

/// `GET /v1/jobs/{id}`.
pub async fn get(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::JobsRead)?;
    Ok(axum::Json(load_job(&s, &caller, id).await?.resource()))
}

/// `POST /v1/jobs/{id}/cancel`: a queued job is canceled at once; a
/// running one is stopped by its worker (its process group is killed), so
/// the answer is `202` until it has.
pub async fn cancel(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult<Response> {
    caller.require(Scope::JobsWrite)?;
    let job = load_job(&s, &caller, id.clone()).await?;
    if job.is_terminal() {
        return Err(ApiError::conflict(format!("the job has already ended ({})", job.status)).with("job_status", job.status));
    }
    let jid = id.clone();
    db(&s, move |c| Ok(c.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?1 AND status IN ('queued', 'running')", [jid])?)).await?;
    if job.status == "queued" {
        crate::worker::finalize(&s, &id, None, crate::worker::Final::stopped("canceled", "canceled at the client's request")).await.map_err(|e| ApiError::internal(&e))?;
    }
    let job = load_job(&s, &caller, id).await?;
    let status = if job.is_terminal() { StatusCode::OK } else { StatusCode::ACCEPTED };
    Ok((status, axum::Json(job.resource())).into_response())
}

/// `DELETE /v1/jobs/{id}`: erase the job's files and outputs; its billing
/// record is kept.
pub async fn remove(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::JobsWrite)?;
    let job = load_job(&s, &caller, id.clone()).await?;
    if !job.is_terminal() {
        return Err(ApiError::conflict("the job is still queued or running: cancel it first").with("job_status", job.status));
    }
    if job.deleted_at.is_none() {
        crate::sweeper::delete_job_folder(&s, &job, "deleted at the client's request").await.map_err(|e| ApiError::internal(&e))?;
        let (now, jid) = (s.clock.now(), id.clone());
        db(&s, move |c| {
            // the instructions and metadata go too; the billing fields stay
            c.execute("UPDATE jobs SET deleted_at = ?2, metadata_json = NULL, request_json = json_set(request_json, '$.instructions', NULL) WHERE id = ?1", params![jid, now])?;
            Ok(())
        })
        .await?;
    }
    Ok(axum::Json(load_job(&s, &caller, id).await?.resource()))
}

/// The job, if its outputs can be read: ended, and not deleted or expired.
async fn finished_job(s: &AppState, caller: &Caller, id: String) -> ApiResult<(JobRow, JobPaths)> {
    let job = load_job(s, caller, id).await?;
    if !job.outputs_available() {
        return Err(ApiError::gone("the job's outputs were deleted (on request, or at the end of their retention period)"));
    }
    if !job.is_terminal() {
        return Err(ApiError::conflict(format!("the job has not ended yet ({})", job.status)).with("job_status", job.status.clone()));
    }
    let paths = JobPaths::new(&s.cfg.data_dir, &job.account_id, &job.id);
    Ok((job, paths))
}

/// `GET /v1/jobs/{id}/report`: the job's `report.json`.
pub async fn report(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult<Response> {
    caller.require(Scope::JobsRead)?;
    let (_, paths) = finished_job(&s, &caller, id).await?;
    let bytes = tokio::fs::read(paths.report())
        .await
        .map_err(|_| ApiError::new(StatusCode::NOT_FOUND, Some("not-found"), "Not found", "this job has no report: it ended before rfpii wrote one"))?;
    Ok(([(header::CONTENT_TYPE, "application/json")], bytes).into_response())
}

/// One released file: its path in the outputs, where it is, its size.
pub struct Output {
    pub path: String,
    pub file: PathBuf,
    pub bytes: u64,
}

/// What a job released: every file under `public/` (symbolic links are
/// not followed), and the transfers record if `public/` has none.
pub fn outputs_of(paths: &JobPaths) -> Vec<Output> {
    let mut out = Vec::new();
    walk(&paths.public(), "", &mut out);
    let transfers = paths.records().join("transfers.jsonl");
    if !out.iter().any(|o| o.path == "transfers.jsonl")
        && let Ok(m) = std::fs::symlink_metadata(&transfers)
        && m.is_file()
    {
        out.push(Output { path: "transfers.jsonl".into(), file: transfers, bytes: m.len() });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

fn walk(dir: &std::path::Path, prefix: &str, out: &mut Vec<Output>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let Ok(m) = std::fs::symlink_metadata(e.path()) else { continue };
        let name = e.file_name().to_string_lossy().into_owned();
        let rel = if prefix.is_empty() { name } else { format!("{prefix}/{name}") };
        if m.is_dir() {
            walk(&e.path(), &rel, out);
        } else if m.is_file() {
            out.push(Output { path: rel, file: e.path(), bytes: m.len() });
        }
    }
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "pdf" => "application/pdf",
        "json" => "application/json",
        "jsonl" => "application/x-ndjson",
        "pem" => "application/x-pem-file",
        "md" => "text/markdown; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// `GET /v1/jobs/{id}/outputs`.
pub async fn outputs(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::JobsRead)?;
    let (job, paths) = finished_job(&s, &caller, id).await?;
    let list = tokio::task::spawn_blocking(move || outputs_of(&paths)).await.map_err(|e| ApiError::internal(&e.into()))?;
    let data: Vec<Value> =
        list.iter().map(|o| json!({"path": o.path, "bytes": o.bytes, "content_type": content_type(&o.path), "url": format!("/v1/jobs/{}/outputs/{}", job.id, o.path)})).collect();
    Ok(axum::Json(json!({"object": "list", "data": data, "has_more": false})))
}

/// Whether `p` is a plain relative path: no empty, `.` or `..` component,
/// no backslash, NUL or leading slash.
pub fn plain_path(p: &str) -> bool {
    !p.is_empty() && p.len() <= 512 && !p.contains(['\\', '\0']) && p.split('/').all(|c| !c.is_empty() && c != "." && c != "..")
}

/// `GET /v1/jobs/{id}/outputs/{path}`: one released file. The path must be
/// one the listing gives; nothing else under the job's folder is reachable.
pub async fn output(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path((id, path)): Path<(String, String)>) -> ApiResult<Response> {
    caller.require(Scope::JobsRead)?;
    if !plain_path(&path) {
        return Err(ApiError::bad_request("the output path must be a plain relative path, as the outputs listing gives it"));
    }
    let (_, paths) = finished_job(&s, &caller, id).await?;
    let wanted = path.clone();
    let found = tokio::task::spawn_blocking(move || outputs_of(&paths).into_iter().find(|o| o.path == wanted)).await.map_err(|e| ApiError::internal(&e.into()))?;
    let Some(o) = found else { return Err(ApiError::new(StatusCode::NOT_FOUND, Some("not-found"), "Not found", "the job released no file at that path")) };
    let file = tokio::fs::File::open(&o.file).await.map_err(|e| ApiError::internal(&e.into()))?;
    let name = path.rsplit('/').next().unwrap_or("output");
    Ok(stream(file, content_type(&path), name, Some(o.bytes)))
}

fn stream(file: tokio::fs::File, ctype: &str, name: &str, len: Option<u64>) -> Response {
    let mut res = Response::new(Body::from_stream(tokio_util::io::ReaderStream::new(file)));
    let h = res.headers_mut();
    if let Ok(v) = HeaderValue::from_str(ctype) {
        h.insert(header::CONTENT_TYPE, v);
    }
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{name}\"")) {
        h.insert(header::CONTENT_DISPOSITION, v);
    }
    if let Some(n) = len {
        h.insert(header::CONTENT_LENGTH, HeaderValue::from(n));
    }
    res
}

/// `GET /v1/jobs/{id}/archive`: a zip of what the job released (under
/// `public/`) and its report.
pub async fn archive(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult<Response> {
    caller.require(Scope::JobsRead)?;
    let (job, paths) = finished_job(&s, &caller, id).await?;
    let tmp = storage::tmp_dir(&s.cfg.data_dir).join(crate::ids::new_id("archive"));
    let at = tmp.clone();
    let built = tokio::task::spawn_blocking(move || build_archive(&paths, &at)).await.map_err(|e| ApiError::internal(&e.into()))?;
    if let Err(e) = built {
        let _ = std::fs::remove_file(&tmp);
        return Err(ApiError::internal(&e));
    }
    let file = tokio::fs::File::open(&tmp).await.map_err(|e| ApiError::internal(&e.into()))?;
    let len = file.metadata().await.ok().map(|m| m.len());
    // the open file is read to the end; its name goes now
    let _ = tokio::fs::remove_file(&tmp).await;
    Ok(stream(file, "application/zip", &format!("{}.zip", job.id), len))
}

fn build_archive(paths: &JobPaths, to: &std::path::Path) -> anyhow::Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(to)?;
    let mut z = zip::ZipWriter::new(f);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for o in outputs_of(paths) {
        z.start_file(format!("public/{}", o.path), opts)?;
        std::io::copy(&mut std::fs::File::open(&o.file)?, &mut z)?;
    }
    for name in ["report.json", "report.md"] {
        if let Ok(bytes) = std::fs::read(paths.out().join(name)) {
            z.start_file(name, opts)?;
            z.write_all(&bytes)?;
        }
    }
    z.finish()?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_plain_relative_paths_name_an_output() {
        assert!(plain_path("x.pdf") && plain_path("batches/batch-1.json") && plain_path("Nyeri/scan.p001.pdf"));
        for bad in ["", "../x", "a/../../b", "/etc/passwd", "a//b", "./x", "a\\b", "a/./b", "a\0b"] {
            assert!(!plain_path(bad), "{bad:?}");
        }
    }
}
