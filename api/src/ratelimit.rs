//! Rate limits: a token bucket per API key, and per IP address on the
//! public routes. Responses carry the `RateLimit-Policy` and `RateLimit`
//! fields of draft-ietf-httpapi-ratelimit-headers-11; a request over the
//! limit gets `429 Too Many Requests` with `Retry-After`.

use axum::http::{HeaderMap, HeaderValue};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Buckets kept before full ones are dropped.
const MAX_BUCKETS: usize = 100_000;

struct Bucket {
    tokens: f64,
    at: Instant,
}

/// Token buckets of one policy: `burst` tokens, refilled at `rate` a second.
pub struct Limiter {
    name: &'static str,
    rate: f64,
    burst: f64,
    buckets: Mutex<HashMap<String, Bucket>>,
}

/// What a request was allowed, for the response's header fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub allowed: bool,
    /// whole tokens left after this request
    pub remaining: u64,
    /// seconds until the bucket is full again
    pub reset: u64,
    /// seconds until a token is free (for Retry-After)
    pub retry_after: u64,
    policy: String,
    name: &'static str,
}

impl Limiter {
    pub fn new(name: &'static str, rate: f64, burst: u32) -> Limiter {
        Limiter { name, rate, burst: f64::from(burst.max(1)), buckets: Mutex::new(HashMap::new()) }
    }

    /// The quota policy, as `RateLimit-Policy` states it: `burst` requests
    /// in the time the bucket takes to fill (`w`, seconds).
    pub fn policy(&self) -> String {
        format!("\"{}\";q={};w={}", self.name, self.burst as u64, (self.burst / self.rate).ceil().max(1.0) as u64)
    }

    /// Take a token from `key`'s bucket, if one is left.
    pub fn check(&self, key: &str) -> Decision {
        self.check_at(key, Instant::now())
    }

    fn check_at(&self, key: &str, now: Instant) -> Decision {
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        if buckets.len() > MAX_BUCKETS {
            let (rate, burst) = (self.rate, self.burst);
            buckets.retain(|_, b| b.tokens + now.duration_since(b.at).as_secs_f64() * rate < burst);
        }
        let b = buckets.entry(key.to_string()).or_insert(Bucket { tokens: self.burst, at: now });
        b.tokens = (b.tokens + now.duration_since(b.at).as_secs_f64() * self.rate).min(self.burst);
        b.at = now;
        let allowed = b.tokens >= 1.0;
        if allowed {
            b.tokens -= 1.0;
        }
        Decision {
            allowed,
            remaining: b.tokens.floor() as u64,
            reset: ((self.burst - b.tokens) / self.rate).ceil() as u64,
            retry_after: if allowed { 0 } else { ((1.0 - b.tokens) / self.rate).ceil().max(1.0) as u64 },
            policy: self.policy(),
            name: self.name,
        }
    }
}

impl Decision {
    /// Add `RateLimit-Policy` and `RateLimit` (and `Retry-After` when the
    /// request was refused) to `headers`.
    pub fn apply(&self, headers: &mut HeaderMap) {
        let set = |h: &mut HeaderMap, k: &'static str, v: String| {
            if let Ok(v) = HeaderValue::from_str(&v) {
                h.insert(k, v);
            }
        };
        set(headers, "ratelimit-policy", self.policy.clone());
        set(headers, "ratelimit", format!("\"{}\";r={};t={}", self.name, self.remaining, self.reset));
        if !self.allowed {
            set(headers, "retry-after", self.retry_after.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_bucket_empties_refills_and_says_when() {
        let l = Limiter::new("per-key", 1.0, 3);
        let t0 = Instant::now();
        let ds: Vec<Decision> = (0..4).map(|_| l.check_at("k", t0)).collect();
        assert!(ds[..3].iter().all(|d| d.allowed) && !ds[3].allowed);
        assert_eq!((ds[2].remaining, ds[3].retry_after, ds[3].reset), (0, 1, 3));
        // another key has its own bucket
        assert!(l.check_at("other", t0).allowed);
        // a second later one token is back
        assert!(l.check_at("k", t0 + Duration::from_millis(1001)).allowed);
        let mut h = HeaderMap::new();
        ds[3].apply(&mut h);
        assert_eq!(h["ratelimit-policy"], "\"per-key\";q=3;w=3");
        assert_eq!(h["ratelimit"], "\"per-key\";r=0;t=3");
        assert_eq!(h["retry-after"], "1");
    }
}
