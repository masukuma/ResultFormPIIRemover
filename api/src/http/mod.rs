//! The HTTP API: routes, and the middleware every request passes through
//! (request id, trace, panic catcher, problem details, timeout, body
//! limit, metrics; authentication and rate limits where they apply).

use axum::Router;
use axum::body::Body;
use axum::extract::{ConnectInfo, FromRequest, FromRequestParts, MatchedPath, Request, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::middleware::{Next, from_fn, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodRouter, delete, get, patch, post, put};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tower::ServiceBuilder;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::{DefaultOnResponse, TraceLayer};

use crate::auth::{self, Caller};
use crate::problem::{self, ApiError, IsProblem};
use crate::server::AppState;

pub mod account;
pub mod admin;
pub mod billing;
pub mod files;
pub mod jobs;
pub mod keys;
pub mod ops;
pub mod verify;

/// The OpenAPI 3.1 description of every route.
pub const OPENAPI: &str = include_str!("../../openapi.json");

/// Routes, with a table of their methods and paths (checked against the
/// OpenAPI description by the tests).
struct Routes {
    router: Router<AppState>,
    table: Vec<(Method, &'static str)>,
}

impl Routes {
    fn new() -> Routes {
        Routes { router: Router::new(), table: Vec::new() }
    }

    fn add(mut self, method: Method, path: &'static str, mr: MethodRouter<AppState>) -> Routes {
        self.router = self.router.route(path, mr);
        self.table.push((method, path));
        self
    }
}

/// The application, and its route table.
pub fn router(state: AppState) -> (Router, Vec<(Method, &'static str)>) {
    let cfg = state.cfg.clone();
    let upload_limit = axum::extract::DefaultBodyLimit::max(cfg.max_upload_bytes as usize + 64 * 1024);
    let verify_limit = axum::extract::DefaultBodyLimit::max(cfg.max_verify_bytes as usize);

    let ops = Routes::new().add(Method::GET, "/healthz", get(ops::healthz)).add(Method::GET, "/readyz", get(ops::readyz)).add(Method::GET, "/metrics", get(ops::metrics)).add(
        Method::POST,
        "/v1/billing/stripe-webhook",
        post(billing::stripe_webhook),
    );

    let public = Routes::new().add(Method::GET, "/v1/openapi.json", get(ops::openapi)).add(Method::GET, "/v1/pricing", get(billing::pricing)).add(
        Method::POST,
        "/v1/verify",
        post(verify::verify).layer(verify_limit),
    );

    let private = Routes::new()
        .add(Method::POST, "/v1/files", post(files::create).layer(upload_limit))
        .add(Method::GET, "/v1/files", get(files::list))
        .add(Method::GET, "/v1/files/{id}", get(files::get))
        .add(Method::DELETE, "/v1/files/{id}", delete(files::remove))
        .add(Method::POST, "/v1/jobs", post(jobs::create))
        .add(Method::GET, "/v1/jobs", get(jobs::list))
        .add(Method::GET, "/v1/jobs/{id}", get(jobs::get))
        .add(Method::DELETE, "/v1/jobs/{id}", delete(jobs::remove))
        .add(Method::POST, "/v1/jobs/{id}/cancel", post(jobs::cancel))
        .add(Method::GET, "/v1/jobs/{id}/report", get(jobs::report))
        .add(Method::GET, "/v1/jobs/{id}/outputs", get(jobs::outputs))
        .add(Method::GET, "/v1/jobs/{id}/outputs/{*path}", get(jobs::output))
        .add(Method::GET, "/v1/jobs/{id}/archive", get(jobs::archive))
        .add(Method::GET, "/v1/account", get(account::get))
        .add(Method::PUT, "/v1/account/controller", put(account::put_controller))
        .add(Method::PUT, "/v1/account/dpia", put(account::put_dpia))
        .add(Method::PATCH, "/v1/account/settings", patch(account::patch_settings))
        .add(Method::POST, "/v1/account/webhook-secret", post(account::rotate_webhook_secret))
        .add(Method::GET, "/v1/keys", get(keys::list))
        .add(Method::POST, "/v1/keys", post(keys::create))
        .add(Method::DELETE, "/v1/keys/{id}", delete(keys::revoke))
        .add(Method::GET, "/v1/usage", get(billing::usage))
        .add(Method::POST, "/v1/billing/checkout", post(billing::checkout));

    let admin = Routes::new()
        .add(Method::GET, "/v1/admin/accounts", get(admin::list_accounts))
        .add(Method::POST, "/v1/admin/accounts", post(admin::create_account))
        .add(Method::PUT, "/v1/admin/accounts/{id}/plan", put(admin::set_plan))
        .add(Method::POST, "/v1/admin/accounts/{id}/keys", post(admin::create_key));

    let mut table = Vec::new();
    for r in [&ops, &public, &private, &admin] {
        table.extend(r.table.iter().cloned());
    }
    let cors = (!cfg.cors_allow_origins.is_empty()).then(|| {
        let origins: Vec<_> = cfg.cors_allow_origins.iter().filter_map(|o| o.parse().ok()).collect();
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods([Method::GET, Method::POST, Method::PUT, Method::PATCH, Method::DELETE])
            .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE, "idempotency-key".parse().expect("header name")])
            .expose_headers(["location", "ratelimit", "ratelimit-policy", "retry-after", "x-request-id"].map(|h| h.parse().expect("header name")))
    });
    let app = Router::new()
        .merge(ops.router)
        .merge(public.router.route_layer(from_fn_with_state(state.clone(), per_ip)))
        .merge(private.router.route_layer(from_fn_with_state(state.clone(), authenticate)))
        .merge(admin.router.route_layer(from_fn_with_state(state.clone(), admin::require_token)))
        .fallback(not_found)
        .layer(from_fn_with_state(state.clone(), track))
        .layer(
            ServiceBuilder::new()
                .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
                .layer(PropagateRequestIdLayer::x_request_id())
                .layer(TraceLayer::new_for_http().make_span_with(span).on_response(DefaultOnResponse::new().level(tracing::Level::INFO)))
                .layer(CatchPanicLayer::new())
                .layer(from_fn(problems))
                .layer(TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, Duration::from_secs(cfg.request_timeout_secs)))
                .layer(axum::extract::DefaultBodyLimit::max(cfg.max_json_bytes))
                .option_layer(cors),
        )
        .with_state(state);
    (app, table)
}

/// A request's span: its id, method and path (paths hold ids, never
/// personal data; query strings and headers are not logged).
fn span(req: &Request) -> tracing::Span {
    let rid = req.headers().get("x-request-id").and_then(|v| v.to_str().ok()).unwrap_or("");
    tracing::info_span!("http", request_id = %rid, method = %req.method(), path = %req.uri().path())
}

async fn not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, None, "Not Found", "no such route; see /v1/openapi.json")
}

/// Count requests by route template and status, with their latency.
async fn track(State(s): State<AppState>, req: Request, next: Next) -> Response {
    let route = req.extensions().get::<MatchedPath>().map(|m| m.as_str().to_string()).unwrap_or_else(|| "unmatched".into());
    let method = req.method().clone();
    let t = Instant::now();
    let res = next.run(req).await;
    s.metrics.observe(method.as_str(), &route, res.status().as_u16(), t.elapsed().as_secs_f64());
    res
}

/// Every error response as problem details (RFC 9457), with `instance`
/// and `request_id` naming the request (its X-Request-Id).
async fn problems(req: Request, next: Next) -> Response {
    let rid = req.headers().get("x-request-id").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let res = next.run(req).await;
    if res.status().as_u16() < 400 {
        return res;
    }
    let is_problem = res.extensions().get::<IsProblem>().is_some();
    let (mut parts, body) = res.into_parts();
    let bytes = axum::body::to_bytes(body, 64 * 1024).await.unwrap_or_default();
    let mut doc: Value = if is_problem {
        serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({}))
    } else {
        let title = parts.status.canonical_reason().unwrap_or("Error");
        let text = String::from_utf8_lossy(&bytes);
        let detail = if text.trim().is_empty() || text.len() > 1000 { title.to_string() } else { text.trim().to_string() };
        json!({"type": problem::type_uri(None), "title": title, "status": parts.status.as_u16(), "detail": detail})
    };
    doc["instance"] = json!(format!("urn:rfpii:request:{rid}"));
    doc["request_id"] = json!(rid);
    parts.headers.insert(header::CONTENT_TYPE, problem::CONTENT_TYPE.parse().expect("content type"));
    parts.headers.remove(header::CONTENT_LENGTH);
    Response::from_parts(parts, Body::from(serde_json::to_vec(&doc).unwrap_or_default()))
}

/// Authenticate the API key, apply its rate limit, and hand the caller to
/// the handler.
async fn authenticate(State(s): State<AppState>, mut req: Request, next: Next) -> Response {
    let header = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).map(String::from);
    let now = s.clock.now();
    let caller: Caller = match s.db.call(move |c| Ok(auth::authenticate(c, now, header.as_deref()))).await {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => return e.into_response(),
        Err(e) => return ApiError::internal(&e).into_response(),
    };
    let d = s.key_limits.check(&caller.key_id);
    let mut res = if d.allowed {
        req.extensions_mut().insert(caller);
        next.run(req).await
    } else {
        too_many_requests(d.retry_after).into_response()
    };
    d.apply(res.headers_mut());
    res
}

/// The client's address: the connection's, or the last X-Forwarded-For
/// entry when the server is told it sits behind a proxy that writes it.
pub fn client_ip(trust_forwarded: bool, headers: &HeaderMap, peer: Option<SocketAddr>) -> String {
    if trust_forwarded
        && let Some(last) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()).and_then(|v| v.rsplit(',').next())
        && !last.trim().is_empty()
    {
        return last.trim().to_string();
    }
    peer.map(|p| p.ip().to_string()).unwrap_or_else(|| "unknown".into())
}

/// The per-IP rate limit of the public routes.
async fn per_ip(State(s): State<AppState>, req: Request, next: Next) -> Response {
    let peer = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let ip = client_ip(s.cfg.trust_forwarded_for, req.headers(), peer);
    let d = s.ip_limits.check(&ip);
    let mut res = if d.allowed { next.run(req).await } else { too_many_requests(d.retry_after).into_response() };
    d.apply(res.headers_mut());
    res
}

fn too_many_requests(retry_after: u64) -> ApiError {
    ApiError::new(StatusCode::TOO_MANY_REQUESTS, Some("rate-limited"), "Too many requests", format!("the rate limit is reached; retry in {retry_after} s"))
        .with("retry_after", retry_after)
}

/// A JSON body, with errors as problem details.
pub struct ApiJson<T>(pub T);

impl<S: Send + Sync, T: DeserializeOwned> FromRequest<S> for ApiJson<T> {
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, ApiError> {
        use axum::extract::rejection::JsonRejection;
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(v)) => Ok(ApiJson(v)),
            Err(JsonRejection::JsonDataError(e)) => Err(ApiError::validation(e.body_text())),
            Err(JsonRejection::MissingJsonContentType(_)) => {
                Err(ApiError::new(StatusCode::UNSUPPORTED_MEDIA_TYPE, Some("invalid-request"), "Unsupported media type", "expected Content-Type: application/json"))
            }
            Err(JsonRejection::BytesRejection(e)) if e.status() == StatusCode::PAYLOAD_TOO_LARGE => {
                Err(ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, Some("too-large"), "Content too large", e.body_text()))
            }
            Err(e) => Err(ApiError::bad_request(e.body_text())),
        }
    }
}

/// Query parameters, with errors as problem details.
pub struct ApiQuery<T>(pub T);

impl<S: Send + Sync, T: DeserializeOwned> FromRequestParts<S> for ApiQuery<T> {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut axum::http::request::Parts, state: &S) -> Result<Self, ApiError> {
        match axum::extract::Query::<T>::from_request_parts(parts, state).await {
            Ok(axum::extract::Query(v)) => Ok(ApiQuery(v)),
            Err(e) => Err(ApiError::bad_request(e.body_text())),
        }
    }
}

/// `limit` and `after` of a list.
#[derive(serde::Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct PageQuery {
    pub limit: Option<i64>,
    pub after: Option<String>,
}

impl PageQuery {
    /// The page size: 20 unless asked, between 1 and 100.
    pub fn limit(&self) -> Result<i64, ApiError> {
        match self.limit {
            None => Ok(20),
            Some(n) if (1..=100).contains(&n) => Ok(n),
            Some(_) => Err(ApiError::bad_request("limit must be between 1 and 100")),
        }
    }
}

/// A page of a list, with the cursor of the next page in the body and in a
/// `Link` header (RFC 8288).
pub fn page(path: &str, data: Vec<Value>, more: bool, limit: i64) -> Response {
    let next = if more { data.last().and_then(|v| v["id"].as_str()).map(String::from) } else { None };
    let mut res = axum::Json(json!({"object": "list", "data": data, "has_more": more, "next_after": next})).into_response();
    if let Some(n) = next
        && let Ok(v) = format!("<{path}?limit={limit}&after={n}>; rel=\"next\"").parse()
    {
        res.headers_mut().insert(header::LINK, v);
    }
    res
}

/// Run blocking database work for a handler.
pub async fn db<R, F>(s: &AppState, f: F) -> Result<R, ApiError>
where
    R: Send + 'static,
    F: FnOnce(&mut rusqlite::Connection) -> anyhow::Result<R> + Send + 'static,
{
    s.db.call(f).await.map_err(|e| ApiError::internal(&e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_client_address_is_the_peer_unless_a_proxy_is_trusted() {
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "203.0.113.9, 198.51.100.7".parse().unwrap());
        let peer: SocketAddr = "10.0.0.2:5555".parse().unwrap();
        assert_eq!(client_ip(false, &h, Some(peer)), "10.0.0.2");
        assert_eq!(client_ip(true, &h, Some(peer)), "198.51.100.7");
        assert_eq!(client_ip(true, &HeaderMap::new(), Some(peer)), "10.0.0.2");
    }
}
