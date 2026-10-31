//! The API end to end: a server on a random port with a temporary data
//! folder and a fake `rfpii` (tests/api/fake-rfpii.sh), and local
//! stand-ins for webhook receivers and Stripe. Nothing reaches the
//! network. `real.rs` runs one job with the real binary, if it is there.

mod auth;
mod billing;
mod common;
mod files;
mod jobs;
mod ops;
mod real;
mod stripe;
mod webhooks;
