//! Multi-page forms (Form 34B): the declaration block must be found on
//! exactly one page, whatever its layout, and all pages must be one form.

use super::*;
use crate::testutil::{bare_template, matched_page};

fn rules(pages: &[&str], ts: &[Template]) -> Vec<String> {
    let doc: Vec<PageResult> = pages.iter().map(|n| matched_page(n)).collect();
    document_rules(&doc, ts)
}

#[test]
fn block_must_be_on_exactly_one_page() {
    let ts = [bare_template("results", "34B", false, 0), bare_template("block", "34B", true, 1)];
    assert!(rules(&["results", "results", "block"], &ts).is_empty());
    assert!(!rules(&["results", "results"], &ts).is_empty()); // block not recognised: never publish
    assert!(!rules(&["results", "block", "block"], &ts).is_empty()); // split or merged document
}

#[test]
fn block_layouts_are_alternatives() {
    // one block template per layout: a document needs exactly one block page
    // of any layout, not one of each
    let ts = [
        bare_template("results", "34B", false, 0),
        bare_template("results_b", "34B", false, 0),
        bare_template("block", "34B", true, 1),
        bare_template("block_b", "34B", true, 1),
    ];
    assert!(rules(&["results_b", "block_b"], &ts).is_empty());
    assert!(rules(&["results", "results", "block"], &ts).is_empty());
    assert!(!rules(&["results_b", "results_b"], &ts).is_empty());
    assert!(!rules(&["results", "block", "block_b"], &ts).is_empty()); // two blocks, whatever their layouts
}

#[test]
fn mixed_forms_are_rejected() {
    let ts = [bare_template("a34", "34A", false, 0), bare_template("results", "34B", false, 0), bare_template("block", "34B", true, 0)];
    assert!(rules(&["a34", "block"], &ts)[0].contains("different forms"));
}
