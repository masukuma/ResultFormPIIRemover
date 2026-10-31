//! The process: shared state, and starting and stopping the HTTP server,
//! the workers and the background tasks (webhook deliveries, Stripe meter
//! events, retention sweeps). Shutdown is graceful: no new job is leased,
//! running jobs get SHUTDOWN_GRACE_SECS to finish and are re-queued if
//! they don't, and the HTTP server finishes the requests it has.

use anyhow::{Context, Result};
use axum::http::Method;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Semaphore, watch};
use tokio::task::JoinHandle;

use crate::billing::Prices;
use crate::clock::Clock;
use crate::config::Config;
use crate::db::Db;
use crate::metrics::Metrics;
use crate::ratelimit::Limiter;

/// What every handler, worker and task shares.
#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<Config>,
    pub db: Arc<Db>,
    pub clock: Arc<Clock>,
    pub metrics: Arc<Metrics>,
    pub key_limits: Arc<Limiter>,
    pub ip_limits: Arc<Limiter>,
    pub http: reqwest::Client,
    pub verify_slots: Arc<Semaphore>,
    pub prices: Arc<Prices>,
}

impl AppState {
    /// The state for `cfg`: its folders made, its database migrated.
    pub fn new(mut cfg: Config) -> Result<AppState> {
        cfg.normalize();
        cfg.validate()?;
        // jobs run in their own folders: every path they are given is absolute
        cfg.data_dir = std::path::absolute(&cfg.data_dir)?;
        if cfg.rfpii_bin.components().count() > 1 {
            cfg.rfpii_bin = std::path::absolute(&cfg.rfpii_bin)?;
        }
        crate::problem::set_type_base(&cfg.problem_type_base);
        for d in ["files", "jobs", "tmp"] {
            std::fs::create_dir_all(cfg.data_dir.join(d)).with_context(|| format!("creating {}", cfg.data_dir.join(d).display()))?;
        }
        let db = Db::open(&cfg.db_path())?;
        db.migrate()?;
        let http = reqwest::Client::builder()
            .user_agent(concat!("rfpii-api/", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(AppState {
            prices: Arc::new(Prices::from_config(&cfg)?),
            key_limits: Arc::new(Limiter::new("per-key", cfg.rate_limit_per_sec, cfg.rate_limit_burst)),
            ip_limits: Arc::new(Limiter::new("per-ip", cfg.public_rate_limit_per_sec, cfg.public_rate_limit_burst)),
            verify_slots: Arc::new(Semaphore::new(cfg.verify_concurrency.max(1))),
            cfg: Arc::new(cfg),
            db,
            clock: Arc::new(Clock::default()),
            metrics: Arc::new(Metrics::default()),
            http,
        })
    }
}

/// What to run.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// serve the HTTP API
    pub http: bool,
    /// in-process workers
    pub workers: usize,
}

/// A running server.
pub struct Server {
    pub addr: Option<SocketAddr>,
    pub state: AppState,
    pub routes: Vec<(Method, &'static str)>,
    shutdown: watch::Sender<bool>,
    http: Option<JoinHandle<()>>,
    workers: Vec<JoinHandle<()>>,
    background: Vec<JoinHandle<()>>,
}

/// Start what `opts` asks for, on `cfg`.
pub async fn start(cfg: Config, opts: Options) -> Result<Server> {
    let state = AppState::new(cfg)?;
    start_with(state, opts).await
}

/// Start on an existing state (tests keep a handle on its clock).
pub async fn start_with(state: AppState, opts: Options) -> Result<Server> {
    let (tx, rx) = watch::channel(false);
    let (router, routes) = crate::http::router(state.clone());
    let (mut addr, mut http) = (None, None);
    if opts.http {
        let listener = tokio::net::TcpListener::bind(&state.cfg.bind).await.with_context(|| format!("binding {}", state.cfg.bind))?;
        addr = Some(listener.local_addr()?);
        let mut stop = rx.clone();
        http = Some(tokio::spawn(async move {
            let serve = axum::serve(listener, router.into_make_service_with_connect_info::<SocketAddr>()).with_graceful_shutdown(async move {
                let _ = stop.wait_for(|s| *s).await;
            });
            if let Err(e) = serve.await {
                tracing::error!(error = %e, "HTTP server failed");
            }
        }));
    }
    let workers = crate::worker::spawn(&state, opts.workers, &rx);
    let background = vec![
        every(&state, &rx, "webhook deliveries", |s| Box::pin(async move { crate::webhooks::deliver_due(&s).await })),
        every(&state, &rx, "Stripe meter events", |s| Box::pin(async move { crate::stripe::report_due(&s).await })),
        tokio::spawn(crate::sweeper::run(state.clone(), rx.clone())),
    ];
    tracing::info!(addr = ?addr, workers = opts.workers, "started");
    Ok(Server { addr, state, routes, shutdown: tx, http, workers, background })
}

type Task = fn(AppState) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize>> + Send>>;

/// Run `task` every poll interval until shutdown.
fn every(state: &AppState, shutdown: &watch::Receiver<bool>, name: &'static str, task: Task) -> JoinHandle<()> {
    let (state, mut shutdown) = (state.clone(), shutdown.clone());
    tokio::spawn(async move {
        while !*shutdown.borrow() {
            if let Err(e) = task(state.clone()).await {
                tracing::error!(task = name, error = %format!("{e:#}"), "background task failed");
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(state.cfg.poll_interval_ms)) => {}
                _ = shutdown.changed() => {}
            }
        }
    })
}

impl Server {
    /// Stop: no new jobs, running jobs finish or are re-queued after the
    /// grace period, open requests complete.
    pub async fn stop(self) {
        let _ = self.shutdown.send(true);
        for w in self.workers {
            let _ = w.await;
        }
        if let Some(h) = self.http {
            let _ = h.await;
        }
        for b in self.background {
            let _ = b.await;
        }
        tracing::info!("stopped");
    }

    /// Run until SIGINT or SIGTERM, then stop.
    pub async fn run_until_signal(self) {
        shutdown_signal().await;
        tracing::info!("shutting down");
        self.stop().await;
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
}
