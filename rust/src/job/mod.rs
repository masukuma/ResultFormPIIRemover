//! The harness: whole redaction jobs, in fixed phases (HARNESS.md).
//!
//! The tools here take and return JSON and need no model: an inventory of
//! the inputs, statistics and risk signals over a run's manifests, the two
//! views a model may be shown (`crop` of a published page, `held_view` of a
//! held one), and text diagnostics of a held page. `rfpii job` runs them,
//! with `rfpii redact`, through intake, plan, pilot, run, QA, remediation,
//! triage, release and report.

pub mod diagnose;
pub mod inventory;
pub mod stats;
pub mod view;

use anyhow::{bail, Result};
use std::collections::BTreeSet;
use std::path::Path;

/// The templates in `dir`, prepared for alignment as `redact` prepares them.
pub fn templates(dir: &Path) -> Result<Vec<crate::templates::Template>> {
    let mut ts = crate::templates::load_templates(dir)?;
    for t in ts.iter_mut() {
        crate::align::prepare(t, &crate::align::AlignSettings::default())?;
    }
    Ok(ts)
}

/// Pages from a list such as "3,7-9" (from 1).
pub fn parse_pages(list: &str) -> Result<BTreeSet<usize>> {
    let mut out = BTreeSet::new();
    for part in list.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        let (a, b): (usize, usize) = match (a.trim().parse(), b.trim().parse()) {
            (Ok(a), Ok(b)) if a >= 1 && a <= b => (a, b),
            _ => bail!("pages: {:?} is not a page or a range such as 7-9", part),
        };
        out.extend(a..=b);
    }
    if out.is_empty() {
        bail!("pages: no page given");
    }
    Ok(out)
}

/// The list `parse_pages` reads, ranges joined: [3, 7, 8, 9] is "3,7-9".
pub fn format_pages(pages: &[usize]) -> String {
    let mut v = pages.to_vec();
    v.sort_unstable();
    v.dedup();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if j > i { format!("{}-{}", v[i], v[j]) } else { v[i].to_string() });
        i = j + 1;
    }
    out.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_lists_read_and_write_alike() {
        assert_eq!(parse_pages("3, 7-9").unwrap().into_iter().collect::<Vec<_>>(), vec![3, 7, 8, 9]);
        assert_eq!(format_pages(&[9, 3, 7, 8, 3]), "3,7-9");
        for bad in ["", "0", "4-2", "x"] {
            assert!(parse_pages(bad).is_err(), "{:?}", bad);
        }
    }
}
