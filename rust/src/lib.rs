//! Verifiable PII removal for photographed election result forms.
//!
//! A port of the Python `resultform_pii` package: the same templates,
//! masking rules, checks, block-level proof and manifest formats.

pub mod align;
pub mod generic;
pub mod io;
pub mod kiems;
pub mod masking;
pub mod pipeline;
pub mod manifest;
pub mod process;
pub mod proof;
pub mod pyjson;
pub mod pyrand;
pub mod stamps;
pub mod structure;
pub mod templates;
pub mod util;
pub mod verify;
