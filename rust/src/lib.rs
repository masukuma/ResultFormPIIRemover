//! Verifiable PII removal for photographed election result forms.
//!
//! It was first written in Python; this implementation makes the same
//! decisions on the same inputs (same templates, masking rules, checks,
//! block-level proof and manifest formats), on the same native libraries.
//! See DESIGN.md.

pub mod align;
pub mod generic;
pub mod io;
pub mod kiems;
pub mod legal;
pub mod learn;
pub mod masking;
pub mod ocr;
pub mod pdfsign;
pub mod pipeline;
pub mod manifest;
pub mod npla;
pub mod process;
pub mod proof;
pub mod pyjson;
pub mod pyrand;
pub mod stamps;
pub mod structure;
pub mod synthetic;
pub mod templates;
#[cfg(test)]
pub mod testutil;
pub mod util;
pub mod verify;
pub mod vlm;
