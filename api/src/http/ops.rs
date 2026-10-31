//! Operations: liveness, readiness, metrics and the OpenAPI description.

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::OPENAPI;
use crate::metrics::DbFigures;
use crate::problem::{ApiError, ApiResult};
use crate::server::AppState;

/// `GET /healthz`: the process answers.
pub async fn healthz() -> axum::Json<Value> {
    axum::Json(json!({"status": "ok"}))
}

/// `GET /readyz`: the database answers, the data folder can be written,
/// and the rfpii binary runs. 503 with what failed otherwise.
pub async fn readyz(State(s): State<AppState>) -> Response {
    let db = s.db.call(|c| Ok(c.query_row("SELECT 1", [], |r| r.get::<_, i64>(0))?)).await.map(|_| "ok".to_string()).unwrap_or_else(|e| format!("{e:#}"));
    let probe = crate::storage::tmp_dir(&s.cfg.data_dir).join(crate::ids::new_id("ready"));
    let disk = tokio::fs::write(&probe, b"ok").await.and_then(|_| std::fs::remove_file(&probe)).map(|_| "ok".to_string()).unwrap_or_else(|e| format!("{e}"));
    let rfpii = crate::rfpii::version(&s.cfg.rfpii_bin).await;
    let checks = json!({
        "database": db,
        "data_dir": disk,
        "rfpii": rfpii.as_ref().map(|_| "ok".to_string()).unwrap_or_else(|e| format!("{e:#}")),
    });
    let ok = db == "ok" && disk == "ok" && rfpii.is_ok();
    if ok {
        axum::Json(json!({"status": "ready", "checks": checks, "rfpii_version": rfpii.ok()})).into_response()
    } else {
        ApiError::unavailable("not ready: a check failed").with("checks", checks).into_response()
    }
}

/// `GET /metrics`: Prometheus text; with METRICS_TOKEN set, only for a
/// bearer of that token.
pub async fn metrics(State(s): State<AppState>, headers: HeaderMap) -> ApiResult<Response> {
    if let Some(token) = &s.cfg.metrics_token {
        let given = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).unwrap_or("");
        if !crate::ids::ct_eq(given.as_bytes(), token.as_bytes()) {
            return Err(ApiError::unauthorized("the metrics token is required", None));
        }
    }
    let figures = s.db.call(db_figures).await.map_err(|e| ApiError::internal(&e))?;
    Ok((StatusCode::OK, [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], s.metrics.render(&figures)).into_response())
}

fn db_figures(c: &mut rusqlite::Connection) -> anyhow::Result<DbFigures> {
    let mut f = DbFigures::default();
    let mut st = c.prepare("SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY status")?;
    f.jobs_by_status = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
    f.queue_depth = c.query_row("SELECT COUNT(*) FROM jobs WHERE status = 'queued'", [], |r| r.get(0))?;
    f.forms_released = c.query_row("SELECT COALESCE(SUM(forms_released), 0) FROM jobs", [], |r| r.get(0))?;
    let mut st = c.prepare("SELECT model, SUM(input_tokens), SUM(output_tokens) FROM job_model_usage GROUP BY model ORDER BY model")?;
    f.tokens = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<rusqlite::Result<_>>()?;
    f.webhooks_pending = c.query_row("SELECT COUNT(*) FROM webhook_deliveries WHERE status = 'pending'", [], |r| r.get(0))?;
    f.usage_unreported = c.query_row("SELECT COUNT(*) FROM usage_records WHERE plan = 'paid' AND forms > 0 AND stripe_reported_at IS NULL", [], |r| r.get(0))?;
    Ok(f)
}

/// `GET /v1/openapi.json`.
pub async fn openapi() -> Response {
    ([(header::CONTENT_TYPE, "application/json")], OPENAPI).into_response()
}
