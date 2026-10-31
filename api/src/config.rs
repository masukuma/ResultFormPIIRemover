//! The server's configuration: each setting is a flag and an environment
//! variable of the same name (the flag wins). `api/README.md` lists them.

use anyhow::{Context, Result, bail};
use clap::Args;
use std::collections::BTreeMap;
use std::path::PathBuf;

const MIB: u64 = 1024 * 1024;

#[derive(Args, Clone, Debug)]
pub struct Config {
    /// Where the database, uploaded files and job folders are kept
    #[arg(long, env = "DATA_DIR", default_value = "./data")]
    pub data_dir: PathBuf,
    /// Address the HTTP server listens on (behind a TLS-terminating proxy)
    #[arg(long, env = "BIND", default_value = "127.0.0.1:8080")]
    pub bind: String,
    /// The API's public URL, for links in responses and Stripe redirects
    #[arg(long, env = "PUBLIC_BASE_URL", default_value = "http://localhost:8080")]
    pub public_base_url: String,

    /// The rfpii binary
    #[arg(long, env = "RFPII_BIN", default_value = "rfpii")]
    pub rfpii_bin: PathBuf,
    /// The templates folder given to every job
    #[arg(long, env = "RFPII_TEMPLATES", default_value = "/opt/rfpii/templates")]
    pub rfpii_templates: PathBuf,
    /// The platform's manifest signing key (its passphrase in $RFPII_SIGN_PASSWORD)
    #[arg(long, env = "SIGN_KEY", default_value = "/keys/signing_key.pem")]
    pub sign_key: PathBuf,
    /// A certificate to sign the PDFs with (its password in $RFPII_CERT_PASSWORD)
    #[arg(long, env = "CERTIFICATE")]
    pub certificate: Option<PathBuf>,
    /// The model host for jobs (Ollama Cloud when unset; key in $OLLAMA_API_KEY)
    #[arg(long, env = "OLLAMA_HOST")]
    pub ollama_host: Option<String>,
    /// Orchestrator models, in order of preference (empty: none)
    #[arg(long, env = "ORCHESTRATOR_MODELS", default_value = "gpt-oss:120b", value_delimiter = ',')]
    pub orchestrator_models: Vec<String>,
    /// Vision models, in order of preference (empty: none)
    #[arg(long, env = "VISION_MODELS", default_value = "gemma4:31b", value_delimiter = ',')]
    pub vision_models: Vec<String>,
    /// Model hosts in Kenya: calls to them are not transfers abroad
    #[arg(long, env = "KENYA_HOSTS", value_delimiter = ',')]
    pub kenya_hosts: Vec<String>,
    /// Model requests a job may make per form it is given (its budget)
    #[arg(long, env = "MODEL_REQUESTS_PER_FORM", default_value_t = 40)]
    pub model_requests_per_form: u64,
    /// Wall-clock limit of one job, in seconds
    #[arg(long, env = "JOB_TIMEOUT_SECS", default_value_t = 12 * 3600)]
    pub job_timeout_secs: u64,

    /// Forms a free account may have released each calendar month (UTC)
    #[arg(long, env = "FREE_FORMS_PER_MONTH", default_value_t = 5)]
    pub free_forms_per_month: i64,
    /// Price of one released form on the paid plan, in micro-dollars
    #[arg(long, env = "PRICE_PER_FORM_MICROUSD", default_value_t = 4900)]
    pub price_per_form_microusd: i64,
    /// Token prices per model, micro-dollars per million tokens: MODEL=IN/OUT,...
    #[arg(long, env = "MODEL_PRICES", default_value = "gemma4:31b=140000/400000,gpt-oss:120b=150000/600000")]
    pub model_prices: String,
    /// Server price per vCPU-hour, in micro-dollars, for each job's actual cost
    #[arg(long, env = "SERVER_PRICE_PER_VCPU_HOUR_MICROUSD", default_value_t = 59850)]
    pub server_price_per_vcpu_hour_microusd: i64,

    /// Days a job's inputs are kept after it ends (0: deleted when it ends)
    #[arg(long, env = "RETENTION_INPUT_DAYS", default_value_t = 0)]
    pub retention_input_days: i64,
    /// Days a job's outputs are kept after it ends
    #[arg(long, env = "RETENTION_OUTPUT_DAYS", default_value_t = 30)]
    pub retention_output_days: i64,
    /// Hours an uploaded file no job has used is kept
    #[arg(long, env = "RETENTION_UNUSED_FILE_HOURS", default_value_t = 24)]
    pub retention_unused_file_hours: i64,
    /// Seconds between retention sweeps
    #[arg(long, env = "SWEEP_INTERVAL_SECS", default_value_t = 300)]
    pub sweep_interval_secs: u64,

    /// Largest file accepted by POST /v1/files, in bytes
    #[arg(long, env = "MAX_UPLOAD_BYTES", default_value_t = 512 * MIB)]
    pub max_upload_bytes: u64,
    /// Most entries a zip may hold
    #[arg(long, env = "MAX_ZIP_ENTRIES", default_value_t = 20_000)]
    pub max_zip_entries: u64,
    /// Most bytes a zip may expand to
    #[arg(long, env = "MAX_ZIP_UNCOMPRESSED_BYTES", default_value_t = 4096 * MIB)]
    pub max_zip_uncompressed_bytes: u64,
    /// Most files one job may name
    #[arg(long, env = "MAX_FILES_PER_JOB", default_value_t = 1000)]
    pub max_files_per_job: usize,
    /// Most forms one job may hold
    #[arg(long, env = "MAX_FORMS_PER_JOB", default_value_t = 100_000)]
    pub max_forms_per_job: i64,
    /// Largest JSON request body, in bytes
    #[arg(long, env = "MAX_JSON_BYTES", default_value_t = MIB as usize)]
    pub max_json_bytes: usize,
    /// Largest POST /v1/verify body, in bytes
    #[arg(long, env = "MAX_VERIFY_BYTES", default_value_t = 64 * MIB)]
    pub max_verify_bytes: u64,
    /// `rfpii verify` runs at once
    #[arg(long, env = "VERIFY_CONCURRENCY", default_value_t = 2)]
    pub verify_concurrency: usize,
    /// Time limit of one HTTP request, in seconds
    #[arg(long, env = "REQUEST_TIMEOUT_SECS", default_value_t = 300)]
    pub request_timeout_secs: u64,

    /// Requests per second per API key (token bucket refill)
    #[arg(long, env = "RATE_LIMIT_PER_SEC", default_value_t = 10.0)]
    pub rate_limit_per_sec: f64,
    /// Burst per API key (token bucket size)
    #[arg(long, env = "RATE_LIMIT_BURST", default_value_t = 60)]
    pub rate_limit_burst: u32,
    /// Requests per second per IP address on the public routes
    #[arg(long, env = "PUBLIC_RATE_LIMIT_PER_SEC", default_value_t = 0.2)]
    pub public_rate_limit_per_sec: f64,
    /// Burst per IP address on the public routes
    #[arg(long, env = "PUBLIC_RATE_LIMIT_BURST", default_value_t = 10)]
    pub public_rate_limit_burst: u32,
    /// Take the client's address from the last X-Forwarded-For entry (set it
    /// only behind a reverse proxy that writes that header)
    #[arg(long, env = "TRUST_FORWARDED_FOR", default_value_t = false)]
    pub trust_forwarded_for: bool,
    /// Origins allowed by CORS (none: CORS off)
    #[arg(long, env = "CORS_ALLOW_ORIGINS", value_delimiter = ',')]
    pub cors_allow_origins: Vec<String>,

    /// Token for /v1/admin/* (unset: those routes are off)
    #[arg(long, env = "ADMIN_TOKEN", hide_env_values = true)]
    pub admin_token: Option<String>,
    /// Token for /metrics (unset: /metrics is open; keep it off the public proxy)
    #[arg(long, env = "METRICS_TOKEN", hide_env_values = true)]
    pub metrics_token: Option<String>,

    /// Stripe secret key (unset: no Stripe; plans are set by the admin)
    #[arg(long, env = "STRIPE_SECRET_KEY", hide_env_values = true)]
    pub stripe_secret_key: Option<String>,
    /// The Stripe price of the metered subscription
    #[arg(long, env = "STRIPE_PRICE_ID")]
    pub stripe_price_id: Option<String>,
    /// The signing secret of the Stripe webhook endpoint
    #[arg(long, env = "STRIPE_WEBHOOK_SECRET", hide_env_values = true)]
    pub stripe_webhook_secret: Option<String>,
    /// The event name of the Stripe billing meter
    #[arg(long, env = "STRIPE_METER_EVENT_NAME", default_value = "rfpii_forms")]
    pub stripe_meter_event_name: String,
    /// Stripe's API base URL (a stub in tests)
    #[arg(long, env = "STRIPE_API_BASE", default_value = "https://api.stripe.com")]
    pub stripe_api_base: String,
    /// Oldest Stripe-Signature timestamp accepted, in seconds
    #[arg(long, env = "STRIPE_SIGNATURE_TOLERANCE_SECS", default_value_t = 300)]
    pub stripe_signature_tolerance_secs: i64,

    /// Seconds before each webhook retry
    #[arg(long, env = "WEBHOOK_RETRY_SCHEDULE", default_value = "5,30,120,600,3600,21600", value_delimiter = ',')]
    pub webhook_retry_schedule: Vec<u64>,
    /// Allow webhook URLs over plain HTTP and to private addresses
    #[arg(long, env = "WEBHOOK_ALLOW_PRIVATE", default_value_t = false)]
    pub webhook_allow_private: bool,
    /// Time limit of one webhook delivery, in seconds
    #[arg(long, env = "WEBHOOK_TIMEOUT_SECS", default_value_t = 10)]
    pub webhook_timeout_secs: u64,

    /// Seconds a worker's lease on a job lasts without a heartbeat
    #[arg(long, env = "LEASE_SECS", default_value_t = 60)]
    pub lease_secs: i64,
    /// Times a job is started before it is failed
    #[arg(long, env = "MAX_ATTEMPTS", default_value_t = 3)]
    pub max_attempts: i64,
    /// How often idle workers and background tasks look for work, in ms
    #[arg(long, env = "POLL_INTERVAL_MS", default_value_t = 1000)]
    pub poll_interval_ms: u64,
    /// Seconds running jobs get to finish at shutdown before they are re-queued
    #[arg(long, env = "SHUTDOWN_GRACE_SECS", default_value_t = 30)]
    pub shutdown_grace_secs: u64,

    /// The platform's name, as a processor in each controller record
    #[arg(long, env = "PLATFORM_NAME", default_value = "rfpii API")]
    pub platform_name: String,
    /// Country (ISO 3166) where the platform processes the forms
    #[arg(long, env = "PLATFORM_COUNTRY", default_value = "KE")]
    pub platform_country: String,
    /// The model host's operator, as a processor
    #[arg(long, env = "MODEL_HOST_NAME", default_value = "Ollama")]
    pub model_host_name: String,
    /// Country of the model host
    #[arg(long, env = "MODEL_HOST_COUNTRY", default_value = "US")]
    pub model_host_country: String,
    /// Where the price is explained
    #[arg(long, env = "PRICING_URL", default_value = "https://github.com/masukuma/ResultFormPIIRemover/blob/main/PRICING.md")]
    pub pricing_url: String,
    /// Base of the problem type URIs (API.md, whose anchors document them)
    #[arg(long, env = "PROBLEM_TYPE_BASE", default_value = "https://github.com/masukuma/ResultFormPIIRemover/blob/main/API.md")]
    pub problem_type_base: String,
}

/// A model's token prices, in micro-dollars per million tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenPrice {
    pub input: i64,
    pub output: i64,
}

impl Config {
    /// The token prices of `MODEL_PRICES`, by model.
    pub fn token_prices(&self) -> Result<BTreeMap<String, TokenPrice>> {
        parse_model_prices(&self.model_prices)
    }

    /// Trim the lists and drop their empty items: `VISION_MODELS=` means
    /// no vision model, not a model named "".
    pub fn normalize(&mut self) {
        for list in [&mut self.orchestrator_models, &mut self.vision_models, &mut self.kenya_hosts, &mut self.cors_allow_origins] {
            *list = list.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        }
        if self.ollama_host.as_deref().is_some_and(|h| h.trim().is_empty()) {
            self.ollama_host = None;
        }
    }

    /// Whether jobs use any model.
    pub fn has_models(&self) -> bool {
        !self.orchestrator_models.is_empty() || !self.vision_models.is_empty()
    }

    /// The checks every command runs before it starts.
    pub fn validate(&self) -> Result<()> {
        self.token_prices()?;
        if self.price_per_form_microusd < 0 || self.free_forms_per_month < 0 {
            bail!("PRICE_PER_FORM_MICROUSD and FREE_FORMS_PER_MONTH can't be negative");
        }
        if self.rate_limit_per_sec <= 0.0 || self.public_rate_limit_per_sec <= 0.0 {
            bail!("rate limits must be above zero");
        }
        if self.lease_secs < 3 {
            bail!("LEASE_SECS must be at least 3");
        }
        Ok(())
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("rfpii-api.sqlite3")
    }
}

/// `MODEL=IN/OUT,...`: model names may hold colons (gemma4:31b), so the
/// prices are split on the last `=`.
pub fn parse_model_prices(s: &str) -> Result<BTreeMap<String, TokenPrice>> {
    let mut out = BTreeMap::new();
    for item in s.split(',').map(str::trim).filter(|i| !i.is_empty()) {
        let (model, prices) = item.rsplit_once('=').with_context(|| format!("MODEL_PRICES: {item:?} is not MODEL=IN/OUT"))?;
        let (i, o) = prices.split_once('/').with_context(|| format!("MODEL_PRICES: {item:?} is not MODEL=IN/OUT"))?;
        let price = TokenPrice { input: i.trim().parse().context("MODEL_PRICES: input price")?, output: o.trim().parse().context("MODEL_PRICES: output price")? };
        out.insert(model.trim().to_string(), price);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_prices_are_read_with_colons_in_the_names() {
        let p = parse_model_prices("gemma4:31b=140000/400000, gpt-oss:120b=150000/600000").unwrap();
        assert_eq!(p["gemma4:31b"], TokenPrice { input: 140000, output: 400000 });
        assert_eq!(p["gpt-oss:120b"].output, 600000);
        assert!(parse_model_prices("gemma4:31b=140000").is_err());
        assert!(parse_model_prices("").unwrap().is_empty());
    }

    #[derive(clap::Parser)]
    struct Only {
        #[command(flatten)]
        c: Config,
    }

    #[test]
    fn an_empty_model_list_means_no_model() {
        use clap::Parser;
        let mut c = Only::try_parse_from(["x", "--vision-models", "", "--orchestrator-models", " gpt-oss:120b ,", "--ollama-host", ""]).unwrap().c;
        assert_eq!(c.vision_models, [""]);
        c.normalize();
        assert!(c.vision_models.is_empty() && c.orchestrator_models == ["gpt-oss:120b"] && c.ollama_host.is_none());
        assert!(c.has_models());
    }
}
