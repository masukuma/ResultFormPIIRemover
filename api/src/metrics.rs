//! Metrics in the Prometheus text format: requests by route and status,
//! their latency, and, read from the database when scraped, jobs by
//! status, the queue, forms released and model tokens.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::Mutex;

/// Upper bounds of the latency histogram's buckets, in seconds.
const BUCKETS: [f64; 12] = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0];

#[derive(Default)]
struct Histogram {
    counts: [u64; BUCKETS.len()],
    sum: f64,
    total: u64,
}

#[derive(Default)]
struct Inner {
    requests: BTreeMap<(String, String, u16), u64>,
    latency: BTreeMap<String, Histogram>,
}

#[derive(Default)]
pub struct Metrics {
    inner: Mutex<Inner>,
}

/// What the database says at scrape time.
#[derive(Default, Debug)]
pub struct DbFigures {
    pub jobs_by_status: Vec<(String, i64)>,
    pub queue_depth: i64,
    pub forms_released: i64,
    /// model, input tokens, output tokens
    pub tokens: Vec<(String, i64, i64)>,
    pub webhooks_pending: i64,
    pub usage_unreported: i64,
}

impl Metrics {
    /// Count one request to `route` (its template, never the raw path).
    pub fn observe(&self, method: &str, route: &str, status: u16, seconds: f64) {
        let mut m = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *m.requests.entry((method.to_string(), route.to_string(), status)).or_insert(0) += 1;
        let h = m.latency.entry(route.to_string()).or_default();
        for (i, le) in BUCKETS.iter().enumerate() {
            if seconds <= *le {
                h.counts[i] += 1;
            }
        }
        h.sum += seconds;
        h.total += 1;
    }

    /// Everything, as Prometheus text.
    pub fn render(&self, db: &DbFigures) -> String {
        let m = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = String::new();
        s += "# HELP rfpii_api_http_requests_total HTTP requests by method, route and status.\n# TYPE rfpii_api_http_requests_total counter\n";
        for ((method, route, status), n) in &m.requests {
            let _ = writeln!(s, "rfpii_api_http_requests_total{{method=\"{method}\",route=\"{}\",status=\"{status}\"}} {n}", escape(route));
        }
        s += "# HELP rfpii_api_http_request_duration_seconds HTTP request latency by route.\n# TYPE rfpii_api_http_request_duration_seconds histogram\n";
        for (route, h) in &m.latency {
            let r = escape(route);
            for (i, le) in BUCKETS.iter().enumerate() {
                let _ = writeln!(s, "rfpii_api_http_request_duration_seconds_bucket{{route=\"{r}\",le=\"{le}\"}} {}", h.counts[i]);
            }
            let _ = writeln!(s, "rfpii_api_http_request_duration_seconds_bucket{{route=\"{r}\",le=\"+Inf\"}} {}", h.total);
            let _ = writeln!(s, "rfpii_api_http_request_duration_seconds_sum{{route=\"{r}\"}} {}", h.sum);
            let _ = writeln!(s, "rfpii_api_http_request_duration_seconds_count{{route=\"{r}\"}} {}", h.total);
        }
        s += "# HELP rfpii_api_jobs Jobs by status.\n# TYPE rfpii_api_jobs gauge\n";
        for (status, n) in &db.jobs_by_status {
            let _ = writeln!(s, "rfpii_api_jobs{{status=\"{status}\"}} {n}");
        }
        let _ = writeln!(s, "# HELP rfpii_api_queue_depth Jobs waiting for a worker.\n# TYPE rfpii_api_queue_depth gauge\nrfpii_api_queue_depth {}", db.queue_depth);
        let _ = writeln!(
            s,
            "# HELP rfpii_api_forms_released_total Forms released by all jobs.\n# TYPE rfpii_api_forms_released_total counter\nrfpii_api_forms_released_total {}",
            db.forms_released
        );
        s += "# HELP rfpii_api_model_tokens_total Model tokens used by jobs, as their reports counted them.\n# TYPE rfpii_api_model_tokens_total counter\n";
        for (model, i, o) in &db.tokens {
            let _ = writeln!(s, "rfpii_api_model_tokens_total{{model=\"{}\",direction=\"input\"}} {i}", escape(model));
            let _ = writeln!(s, "rfpii_api_model_tokens_total{{model=\"{}\",direction=\"output\"}} {o}", escape(model));
        }
        let _ = writeln!(
            s,
            "# HELP rfpii_api_webhooks_pending Webhook deliveries not yet made.\n# TYPE rfpii_api_webhooks_pending gauge\nrfpii_api_webhooks_pending {}",
            db.webhooks_pending
        );
        let _ = writeln!(
            s,
            "# HELP rfpii_api_usage_unreported Paid usage records not yet reported to Stripe.\n# TYPE rfpii_api_usage_unreported gauge\nrfpii_api_usage_unreported {}",
            db.usage_unreported
        );
        s
    }
}

/// A label value with `\`, `"` and newlines escaped.
fn escape(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_are_counted_by_route_and_status_with_a_histogram() {
        let m = Metrics::default();
        m.observe("GET", "/v1/jobs/{id}", 200, 0.02);
        m.observe("GET", "/v1/jobs/{id}", 200, 3.0);
        m.observe("GET", "/v1/jobs/{id}", 404, 0.001);
        let s = m.render(&DbFigures { queue_depth: 2, tokens: vec![("gemma4:31b".into(), 10, 5)], ..Default::default() });
        assert!(s.contains("rfpii_api_http_requests_total{method=\"GET\",route=\"/v1/jobs/{id}\",status=\"200\"} 2"));
        assert!(s.contains("rfpii_api_http_request_duration_seconds_bucket{route=\"/v1/jobs/{id}\",le=\"0.025\"} 2"));
        assert!(s.contains("rfpii_api_http_request_duration_seconds_bucket{route=\"/v1/jobs/{id}\",le=\"+Inf\"} 3"));
        assert!(s.contains("rfpii_api_queue_depth 2"));
        assert!(s.contains("rfpii_api_model_tokens_total{model=\"gemma4:31b\",direction=\"output\"} 5"));
    }
}
