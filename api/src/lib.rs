//! rfpii-api: the `rfpii` redaction harness (HARNESS.md) served as a paid
//! HTTP API. Clients upload forms, start jobs and download what the
//! release gate let out; workers run `rfpii job` for each job; usage is
//! settled per released form (PRICING.md). See API.md.

pub mod accounts;
pub mod audit;
pub mod auth;
pub mod billing;
pub mod cli;
pub mod clock;
pub mod config;
pub mod db;
pub mod http;
pub mod ids;
pub mod jobs;
pub mod metrics;
pub mod problem;
pub mod ratelimit;
pub mod rfpii;
pub mod server;
pub mod storage;
pub mod stripe;
pub mod sweeper;
pub mod webhooks;
pub mod worker;
