//! Errors as RFC 9457 problem details (`application/problem+json`): type,
//! title, status, detail and instance, plus extension members. Every error
//! the API returns has this shape; `http::problems` turns the rest (a
//! timeout, a body too large, a route not found) into it as well and adds
//! `instance` and `request_id`.

use axum::http::{HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};
use std::sync::OnceLock;

pub const CONTENT_TYPE: &str = "application/problem+json";

/// The members RFC 9457 defines, and the request id this API adds.
const RESERVED: [&str; 6] = ["type", "title", "status", "detail", "instance", "request_id"];

/// The base of problem type URIs: `<base>#problem-<slug>`.
static TYPE_BASE: OnceLock<String> = OnceLock::new();

/// Set the base of problem type URIs, once, at start.
pub fn set_type_base(base: &str) {
    let _ = TYPE_BASE.set(base.trim_end_matches('/').to_string());
}

/// The type URI of a problem: its section of API.md, or `about:blank` for
/// a plain HTTP status (RFC 9457 §4.2.1).
pub fn type_uri(slug: Option<&str>) -> String {
    match slug {
        Some(s) => format!("{}#problem-{}", TYPE_BASE.get().map(String::as_str).unwrap_or("https://github.com/masukuma/ResultFormPIIRemover/blob/main/API.md"), s),
        None => "about:blank".into(),
    }
}

/// An error to return to the client.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub slug: Option<&'static str>,
    pub title: String,
    pub detail: String,
    pub extensions: Map<String, Value>,
    pub headers: Vec<(HeaderName, HeaderValue)>,
}

/// Marks a response whose body is already a problem document.
#[derive(Clone, Copy, Debug)]
pub struct IsProblem;

impl ApiError {
    pub fn new(status: StatusCode, slug: Option<&'static str>, title: &str, detail: impl Into<String>) -> ApiError {
        ApiError { status, slug, title: title.into(), detail: detail.into(), extensions: Map::new(), headers: Vec::new() }
    }

    /// Add an extension member. The members RFC 9457 defines can't be
    /// overwritten this way.
    pub fn with(mut self, key: &str, value: impl Into<Value>) -> ApiError {
        debug_assert!(!RESERVED.contains(&key), "{key} is a member RFC 9457 defines");
        if !RESERVED.contains(&key) {
            self.extensions.insert(key.into(), value.into());
        }
        self
    }

    /// Add a response header.
    pub fn header(mut self, name: HeaderName, value: &str) -> ApiError {
        if let Ok(v) = HeaderValue::from_str(value) {
            self.headers.push((name, v));
        }
        self
    }

    pub fn bad_request(detail: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::BAD_REQUEST, Some("invalid-request"), "Invalid request", detail)
    }

    pub fn validation(detail: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, Some("validation-failed"), "Validation failed", detail)
    }

    pub fn not_found(what: &str) -> ApiError {
        ApiError::new(StatusCode::NOT_FOUND, Some("not-found"), "Not found", format!("no {what} with that id in this account"))
    }

    pub fn conflict(detail: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::CONFLICT, Some("conflict"), "Conflict", detail)
    }

    pub fn gone(detail: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::GONE, Some("gone"), "Gone", detail)
    }

    pub fn unauthorized(detail: impl Into<String>, error: Option<&str>) -> ApiError {
        let challenge = match error {
            Some(e) => format!("Bearer realm=\"rfpii\", error=\"{e}\""),
            None => "Bearer realm=\"rfpii\"".into(),
        };
        ApiError::new(StatusCode::UNAUTHORIZED, Some("unauthorized"), "Unauthorized", detail).header(header::WWW_AUTHENTICATE, &challenge)
    }

    pub fn forbidden_scope(scope: &str) -> ApiError {
        ApiError::new(StatusCode::FORBIDDEN, Some("insufficient-scope"), "Insufficient scope", format!("this API key lacks the scope {scope}"))
            .with("required_scope", scope)
            .header(header::WWW_AUTHENTICATE, &format!("Bearer realm=\"rfpii\", error=\"insufficient_scope\", scope=\"{scope}\""))
    }

    pub fn unsupported_media(detail: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::UNSUPPORTED_MEDIA_TYPE, Some("unsupported-file-type"), "Unsupported file type", detail)
    }

    pub fn too_large(detail: impl Into<String>, limit: u64) -> ApiError {
        ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, Some("too-large"), "Content too large", detail).with("limit_bytes", limit)
    }

    pub fn unavailable(detail: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::SERVICE_UNAVAILABLE, Some("unavailable"), "Service unavailable", detail)
    }

    /// An internal error: logged in full, returned without details.
    pub fn internal(err: &anyhow::Error) -> ApiError {
        tracing::error!(error = %format!("{err:#}"), "internal error");
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, None, "Internal Server Error", "the server could not complete the request; it has been logged")
    }

    /// The problem document.
    pub fn body(&self) -> Value {
        let mut v = json!({"type": type_uri(self.slug), "title": self.title, "status": self.status.as_u16(), "detail": self.detail});
        if let Some(m) = v.as_object_mut() {
            for (k, x) in &self.extensions {
                m.insert(k.clone(), x.clone());
            }
        }
        v
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> ApiError {
        ApiError::internal(&e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::to_vec(&self.body()).unwrap_or_default();
        let mut res = (self.status, [(header::CONTENT_TYPE, CONTENT_TYPE)], body).into_response();
        for (k, v) in self.headers {
            res.headers_mut().append(k, v);
        }
        res.extensions_mut().insert(IsProblem);
        res
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_problem_has_the_rfc_9457_members_and_its_extensions() {
        let e = ApiError::new(StatusCode::PAYMENT_REQUIRED, Some("free-allowance-exceeded"), "Free allowance exceeded", "6 forms requested, 2 left").with("forms_left", 2);
        let b = e.body();
        assert_eq!(b["status"], 402);
        assert!(b["type"].as_str().unwrap().ends_with("#problem-free-allowance-exceeded"));
        assert_eq!(b["forms_left"], 2);
        assert_eq!(ApiError::new(StatusCode::NOT_FOUND, None, "Not Found", "").body()["type"], "about:blank");
    }
}
