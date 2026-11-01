//! The command line: `serve`, `worker`, `migrate` and `admin`.

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use crate::config::Config;
use crate::server::{self, Options};

#[derive(Parser, Debug)]
#[command(name = "rfpii-api", version, about = "The rfpii redaction harness as a paid HTTP API (API.md)")]
pub struct Cli {
    #[command(flatten)]
    pub config: Config,
    /// Log format: json (one object per line) or text
    #[arg(long, env = "LOG_FORMAT", default_value = "json")]
    pub log_format: String,
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Serve the HTTP API, with in-process workers (--workers 0: API only)
    Serve {
        #[arg(long, env = "WORKERS", default_value_t = 2)]
        workers: usize,
    },
    /// Run workers only, on the same database and data folder as the API
    Worker {
        #[arg(long, env = "WORKERS", default_value_t = 2)]
        workers: usize,
    },
    /// Apply the database migrations not yet applied
    Migrate,
    /// Accounts, API keys and plans
    Admin {
        #[command(subcommand)]
        cmd: AdminCmd,
    },
}

#[derive(Subcommand, Debug)]
pub enum AdminCmd {
    /// Create an account; prints its id
    CreateAccount {
        #[arg(long)]
        name: String,
        #[arg(long)]
        email: Option<String>,
        #[arg(long, default_value = "free")]
        plan: String,
    },
    /// Create an API key for an account; prints the key, once
    CreateKey {
        #[arg(long)]
        account: String,
        #[arg(long)]
        name: Option<String>,
        /// Space- or comma-separated: jobs:write, jobs:read, account:admin
        #[arg(long, default_value = "jobs:write jobs:read account:admin")]
        scopes: String,
    },
    /// Set an account's plan (free or paid) and monthly spend cap
    SetPlan {
        #[arg(long)]
        account: String,
        #[arg(long)]
        plan: String,
        /// Micro-dollars a month, or "none"; unchanged if left out
        #[arg(long)]
        spend_cap_microusd: Option<String>,
    },
    /// List the accounts
    ListAccounts {
        #[arg(long)]
        json: bool,
    },
}

/// JSON logs to stdout (or text), at RUST_LOG's level (info by default).
pub fn init_logging(cli: &Cli) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let level = if matches!(cli.cmd, Cmd::Admin { .. } | Cmd::Migrate) { "warn" } else { "info" };
        tracing_subscriber::EnvFilter::new(level)
    });
    let builder = tracing_subscriber::fmt().with_env_filter(filter).with_target(false);
    if cli.log_format == "text" {
        builder.with_writer(std::io::stderr).init();
    } else {
        builder.json().flatten_event(true).with_current_span(true).with_span_list(false).init();
    }
}

pub fn run(cli: Cli) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    match cli.cmd {
        Cmd::Serve { workers } => rt.block_on(async {
            let s = server::start(cli.config, Options { http: true, workers }).await?;
            s.run_until_signal().await;
            Ok(())
        }),
        Cmd::Worker { workers } => rt.block_on(async {
            if workers == 0 {
                bail!("a worker process needs --workers 1 or more");
            }
            let s = server::start(cli.config, Options { http: false, workers }).await?;
            s.run_until_signal().await;
            Ok(())
        }),
        Cmd::Migrate => {
            let db = crate::db::Db::open(&cli.config.db_path())?;
            let applied = db.migrate()?;
            println!("schema version {} ({} migration(s) applied now)", db.version()?, applied.len());
            Ok(())
        }
        Cmd::Admin { cmd } => {
            println!("{}", admin(&cli.config, cmd)?);
            Ok(())
        }
    }
}

/// Run an admin command; what it prints.
pub fn admin(cfg: &Config, cmd: AdminCmd) -> Result<String> {
    let db = crate::db::Db::open(&cfg.db_path())?;
    db.migrate()?;
    let now = crate::clock::Clock::default().now();
    let actor = "admin-cli";
    db.with(|c| match cmd {
        AdminCmd::CreateAccount { name, email, plan } => Ok(crate::accounts::create(c, now, &name, email.as_deref(), &plan, actor)?.id),
        AdminCmd::CreateKey { account, name, scopes } => {
            if crate::accounts::load(c, &account)?.is_none() {
                bail!("no account {account}");
            }
            let scopes = crate::auth::parse_scopes(&scopes).map_err(anyhow::Error::msg)?;
            let (row, secret) = crate::auth::create(c, now, &account, name.as_deref(), &scopes, actor)?;
            Ok(format!("{secret}\n(key {}; shown once, store it now)", row.id))
        }
        AdminCmd::SetPlan { account, plan, spend_cap_microusd } => {
            let cap = match spend_cap_microusd.as_deref() {
                None => None,
                Some("none") => Some(None),
                Some(n) => Some(Some(n.parse::<i64>().map_err(|_| anyhow::anyhow!("--spend-cap-microusd must be a number or none"))?)),
            };
            if !crate::accounts::set_plan(c, now, &account, &plan, cap, actor)? {
                bail!("no account {account}");
            }
            Ok(format!("{account}: plan {plan}"))
        }
        AdminCmd::ListAccounts { json } => {
            let all = crate::accounts::list(c)?;
            if json {
                return Ok(serde_json::to_string_pretty(&all.iter().map(crate::http::admin::account_value).collect::<Vec<_>>())?);
            }
            let mut out = format!("{:<38} {:<6} {:<12} {}", "ID", "PLAN", "READY", "NAME");
            for a in &all {
                out += &format!("\n{:<38} {:<6} {:<12} {}", a.id, a.plan, if crate::accounts::missing_for_jobs(a).is_empty() { "yes" } else { "no" }, a.name);
            }
            Ok(out)
        }
    })
}
