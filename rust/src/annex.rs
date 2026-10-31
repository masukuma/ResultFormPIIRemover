//! The technical annex of a data protection impact assessment (Data
//! Protection Act s.31; the Third Schedule of the Data Protection (General)
//! Regulations 2021), written from what a run recorded: its manifests, the
//! review folder's report, the transfers record and the ledger. It supplies
//! the systematic description, the residual-risk figures and the measures;
//! the necessity and proportionality assessment and any consultation with
//! the Commissioner are the controller's to write (`legal/C-recommendations.md`
//! §2.8).

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::legal::Controller;

/// Every file under `dir` whose name ends with `suffix`.
pub fn files_under(dir: &Path, suffix: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.to_string_lossy().ends_with(suffix) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn jsonl(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path).map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()).unwrap_or_default()
}

/// Text in single quotes (what OCR read off a form, as reasons and notes
/// quote it) replaced by `'…'`. A quote opens at the start or after a
/// character that is not a letter or digit, and closes before one: an
/// apostrophe inside a word ("can't", "the column's edge", "the officers'
/// signatures") is not a quote.
pub fn unquote(s: &str) -> String {
    regex::Regex::new(r"(^|[^\p{L}\p{N}])'.*?'($|[^\p{L}\p{N}])").unwrap().replace_all(s, "$1'…'$2").into_owned()
}

/// A held reason with its numbers and quoted text taken out, so like ones
/// count together (and no text read off a form is repeated).
pub fn normalise_reason(r: &str) -> String {
    let r = regex::Regex::new(r"^page \d+: ").unwrap().replace(r, "");
    let r = unquote(&r);
    let r = regex::Regex::new(r"\d+(\.\d+)?").unwrap().replace_all(&r, "N");
    r.split("; ").next().unwrap_or("").chars().take(100).collect()
}

/// The figures of the annex, from a run's output folder (and its review
/// folder, if given).
pub fn figures(out: &Path, review: Option<&Path>) -> Result<Value> {
    let manifests: Vec<Value> = files_under(out, ".manifest.json").iter().filter_map(|p| crate::manifest::read_json(p).ok()).collect();
    let mut pages = 0usize;
    let mut sources = std::collections::BTreeSet::new();
    let (mut first, mut last) = (String::new(), String::new());
    let mut settings = BTreeMap::new();
    let mut tools = BTreeMap::new();
    let mut masks: BTreeMap<String, usize> = BTreeMap::new();
    let (mut slots, mut hidden, mut numbers) = (0usize, 0usize, 0usize);
    let (mut stamped_pages, mut banded_pages, mut lifted_pages) = (0usize, 0usize, 0usize);
    let (mut reviews, mut kept_spots) = (0usize, 0usize);
    let mut regions: BTreeMap<String, usize> = BTreeMap::new();
    let mut controllers = BTreeMap::new();
    for m in &manifests {
        sources.insert(m["source"]["sha256"].as_str().unwrap_or("").to_string());
        let c = m["created_at"].as_str().unwrap_or("").to_string();
        if first.is_empty() || c < first {
            first = c.clone();
        }
        if c > last {
            last = c;
        }
        let st = &m["settings"];
        let key = format!(
            "hide {}, phone tail {}, whole IDs {}, whole phones {}, stamps {}",
            st["hide"], st["phone_tail"], st["full_ids"].as_bool().unwrap_or(false), st["full_phones"].as_bool().unwrap_or(false),
            st["stamps"].as_str().unwrap_or(if st["mask_stamps"] == true { "masked" } else { "protected" })
        );
        *settings.entry(key).or_insert(0usize) += 1;
        *tools.entry(format!("{} {} ({})", m["tool"]["name"].as_str().unwrap_or("?"), m["tool"]["version"].as_str().unwrap_or("?"), m["tool"]["commit"].as_str().unwrap_or("no commit recorded"))).or_insert(0usize) += 1;
        if let Some(c) = m["controller"]["controller"].as_str() {
            *controllers.entry(format!("{} (record {})", c, &m["controller"]["sha256"].as_str().unwrap_or("?")[..12.min(m["controller"]["sha256"].as_str().unwrap_or("?").len())])).or_insert(0usize) += 1;
        }
        for p in m["pages"].as_array().cloned().unwrap_or_default() {
            pages += 1;
            for mk in p["masks"].as_array().cloned().unwrap_or_default() {
                let field = match (mk["rule"].as_str(), mk["mode"].as_str()) {
                    (Some(r), _) if !r.is_empty() => r.to_string(),
                    (_, Some("signature")) => "signature".into(),
                    (_, Some(mode)) if mk["zone"].as_str().is_some_and(|z| z.contains("signature")) => format!("signature ({})", mode),
                    (_, Some(mode)) => mode.to_string(),
                    _ => "?".into(),
                };
                let mode = mk["mode"].as_str().unwrap_or("?");
                *masks.entry(format!("{} | {} | {}", mk["role"].as_str().unwrap_or("not recorded"), field, mode)).or_insert(0) += 1;
                for l in mk["lines"].as_array().cloned().unwrap_or_default() {
                    numbers += 1;
                    slots += l["slots"].as_u64().unwrap_or(0) as usize;
                    hidden += l["hidden"].as_array().map(|h| h.len()).unwrap_or(0);
                }
            }
            let stamps: Vec<&Value> = p["protected"].as_array().map(|a| a.iter().filter(|x| x["kind"] == "stamp").collect()).unwrap_or_default();
            if !stamps.is_empty() {
                stamped_pages += 1;
            }
            if stamps.iter().any(|s| s.get("except_blocks").is_some()) {
                banded_pages += 1;
            }
            if p["notes"].as_array().is_some_and(|n| n.iter().any(|x| x.as_str().is_some_and(|s| s.starts_with("masks lifted off")))) {
                lifted_pages += 1;
            }
        }
        for r in m["vlm_review"].as_array().cloned().unwrap_or_default() {
            reviews += 1;
            kept_spots += r["spots"].as_array().map(|s| s.iter().filter(|x| x["kept"] == true).count()).unwrap_or(0);
            *regions.entry(format!("review on {} ({})", r["host"].as_str().unwrap_or("?"), r["host_region"].as_str().unwrap_or("region not recorded"))).or_insert(0) += 1;
        }
    }
    // held pages, and why
    let mut held = 0usize;
    let mut why: BTreeMap<String, usize> = BTreeMap::new();
    if let Some(rv) = review {
        let mut last_of: BTreeMap<(String, String), Value> = BTreeMap::new();
        for r in jsonl(&rv.join("report.jsonl")) {
            last_of.insert((r["file"].as_str().unwrap_or("").to_string(), r["page"].to_string()), r);
        }
        for r in last_of.values().filter(|r| r["ok"] == false) {
            held += 1;
            *why.entry(normalise_reason(r["reason"].as_str().unwrap_or(""))).or_insert(0) += 1;
        }
    }
    // transfers abroad
    let transfers = jsonl(&out.join("transfers.jsonl"));
    let mut by_recipient: BTreeMap<String, usize> = BTreeMap::new();
    for t in &transfers {
        *by_recipient.entry(format!("{} {} ({})", t["recipient"]["host"].as_str().unwrap_or("?"), t["recipient"]["model"].as_str().unwrap_or("?"), t["purpose"].as_str().unwrap_or("?"))).or_insert(0) += 1;
    }
    let tr_dates: Vec<&str> = transfers.iter().filter_map(|t| t["at"].as_str()).collect();
    // the ledger's events
    let ledger = jsonl(&out.join("ledger.jsonl"));
    let count = |k: &str| ledger.iter().filter(|e| e["type"] == k).count();
    Ok(json!({
        "folder": out.to_string_lossy(),
        "documents": manifests.len(),
        "pages_published": pages,
        "sources": sources.len(),
        "first": first, "last": last,
        "settings": settings, "tools": tools, "controllers": controllers,
        "masks": masks,
        "numbers_masked": numbers, "characters": slots, "characters_hidden": hidden,
        "pages_with_stamps": stamped_pages, "pages_with_bands_across_stamps": banded_pages, "pages_with_writing_left_on_stamps": lifted_pages,
        "reviews": reviews, "review_spots_masked": kept_spots, "review_hosts": regions,
        "pages_held": held, "held_reasons": why,
        "transfers_abroad": transfers.len(), "transfers_by_recipient": by_recipient,
        "transfers_first": tr_dates.iter().min(), "transfers_last": tr_dates.iter().max(),
        "withdrawals": count("withdrawal"), "key_rotations": count("key_rotation"), "key_revocations": count("key_revocation"),
    }))
}

fn table(rows: &Value, head: (&str, &str)) -> String {
    let mut s = format!("| {} | {} |\n|---|---|\n", head.0, head.1);
    let mut any = false;
    for (k, v) in rows.as_object().cloned().unwrap_or_default() {
        s += &format!("| {} | {} |\n", k, v);
        any = true;
    }
    if any { s } else { "None recorded.\n".into() }
}

/// The annex as Markdown, in the order of s.31(2) and the Third Schedule.
pub fn markdown(f: &Value, controller: Option<&Controller>, retention: Option<&str>) -> String {
    let n = |k: &str| f[k].as_u64().unwrap_or(0);
    let mut s = String::new();
    s += "# Data protection impact assessment: technical annex\n\n";
    s += &format!(
        "Written by `rfpii dpia-annex` on {} from {} manifests in `{}`. It gives the systematic description of the processing, the residual-risk figures and the measures (Data Protection Act s.31(2)(a), (c), (d)). The assessment of necessity and proportionality (s.31(2)(b)), the risk judgement and any consultation with the Data Protection Commissioner (s.31(3)) are for the controller to write. The DPIA is to be submitted 60 days before the processing (s.31(5)).\n\n",
        crate::process::utc_now(), n("documents"), f["folder"].as_str().unwrap_or("")
    );
    s += "## 1. Controller and purpose\n\n";
    match controller {
        Some(c) => {
            let r = &c.raw;
            s += &format!("- Controller: {} (registration: {})\n", r["controller"]["name"].as_str().unwrap_or("?"), r["controller"]["registration_no"].as_str().unwrap_or("not given"));
            s += &format!("- Operator's role: {}\n- Legal basis: {}\n- Purpose: {}\n- Commercial use: {}\n", r["operator_role"].as_str().unwrap_or("?"), r["legal_basis"].as_str().unwrap_or("?"), r["purpose"].as_str().unwrap_or("?"), r["commercial"]);
            if let Some(d) = r["dpo"]["name"].as_str() {
                s += &format!("- Data protection officer: {}\n", d);
            }
            for p in r["processors"].as_array().cloned().unwrap_or_default() {
                s += &format!("- Processor: {} ({}), {}{}\n", p["name"].as_str().unwrap_or("?"), p["country"].as_str().unwrap_or("?"), p["service"].as_str().unwrap_or("?"), p["contract"].as_str().map(|c| format!("; contract {}", c)).unwrap_or_default());
            }
            s += &format!("- Controller record SHA-256: {}\n\n", c.sha256);
        }
        None => s += "No controller record given (`--controller`). The manifests name:\n\n",
    }
    s += &table(&f["controllers"], ("Controller named in the manifests", "Documents"));
    s += "\n## 2. Systematic description of the processing\n\n";
    s += &format!(
        "Published polling-station result forms are redacted before publication: {} documents ({} pages published) from {} source files, between {} and {}.\n\n",
        n("documents"), n("pages_published"), n("sources"), f["first"].as_str().unwrap_or("?"), f["last"].as_str().unwrap_or("?")
    );
    s += "Personal data on the forms, and what is done to each:\n\n";
    s += "- Names of officers and agents, and the agents' parties: left visible (prescribed fields of Forms 34A, 34B, 34C, LN 72/2017).\n";
    s += "- ID and passport numbers, phone numbers: characters hidden by the rule in the settings below.\n";
    s += "- Signatures (biometric data under the Elections Act s.2 and the Elections (Technology) Regulations reg. 2): a black band through the middle of each; that a line is signed stays visible.\n";
    s += "- Writing under the official stamp: as the stamp rule below says.\n\n";
    s += &table(&f["settings"], ("Settings", "Documents"));
    s += "\nMasks applied (whose data | field | how):\n\n";
    s += &table(&f["masks"], ("Role | field | mode", "Count"));
    if n("numbers_masked") > 0 {
        s += &format!("\nNumbers masked character by character: {}; {} of their {} characters hidden ({:.1} per number).\n", n("numbers_masked"), n("characters_hidden"), n("characters"), n("characters_hidden") as f64 / n("numbers_masked") as f64);
    }
    s += "\nSoftware:\n\n";
    s += &table(&f["tools"], ("Tool version (commit)", "Documents"));
    s += "\n## 3. Necessity and proportionality (to be written by the controller)\n\n";
    s += "Points the assessment must answer, from `legal/C-recommendations.md`:\n\n";
    s += "- The agents' ID/passport and phone columns are prescribed fields of the form, and the IEBC publishes the forms unredacted on its portal. Why the republisher's purpose does not need them, and what a second, permanent public copy adds to the risk.\n";
    s += "- Why names and parties stay visible (prescribed, and what readers check), and whether the party column reveals belief (s.2 \"conscience, belief\").\n";
    s += "- Why the masking chosen is the less restrictive means (Constitution Art. 24(1)(e)) against hiding whole numbers, or whole forms.\n\n";
    s += "## 4. Risks and residual-risk figures\n\n";
    s += &format!("- An 8-digit ID with 3 characters hidden leaves 1,000 candidate numbers beside a visible name; a phone number with 3 of its last 6 digits hidden, 1,000. Whole numbers hidden (`--full-ids`, `--full-phones`) leave none from the form.\n");
    s += &format!("- Pages with a stamp: {}; with a field's line hidden across one: {}; with writing left visible on one (a stamp kept whole, or writing off the fields' lines): {}.\n", n("pages_with_stamps"), n("pages_with_bands_across_stamps"), n("pages_with_writing_left_on_stamps"));
    s += &format!("- Vision reviews of redacted pages: {}; spots they found showing and had masked: {}.\n", n("reviews"), n("review_spots_masked"));
    s += &format!("- Pages held back rather than published: {}.\n", n("pages_held"));
    s += &format!("- Forms withdrawn after release: {}; signing keys rotated: {}, revoked: {}.\n\n", n("withdrawals"), n("key_rotations"), n("key_revocations"));
    s += &table(&f["held_reasons"], ("Why pages were held", "Pages"));
    s += "\n## 5. Measures and safeguards\n\n";
    s += "- Only 8×8 blocks are blacked; every other block is the original's data, and each form's manifest, signed with an encrypted Ed25519 key, lists them. A hash-chained ledger and a Merkle batch root fix the record; anyone can check a published form against its original with `rfpii verify`.\n";
    s += "- Results regions and stamps are protected regions no mask may touch (a stamp except along a number's line across it, or a line through the middle of a signature's row), checked before publishing and by `verify`.\n";
    s += "- Gates before publishing: page alignment, table placement, the column edge check, the leak check (written characters in a row left unmasked), signature coverage. A page failing any is held, never published.\n";
    s += "- Each page carries a notice that it is a redacted copy, naming the original by its hash.\n";
    s += "- Forms found leaking after release are withdrawn (`rfpii withdraw`) and the withdrawal is recorded in the ledger; signing keys can be rotated and revoked there.\n";
    s += "- Unredacted pages go to a model host abroad only with `--allow-remote-unredacted`; every page sent abroad is recorded (section 6).\n\n";
    s += "## 6. Safeguards prior to and cross-border transfers\n\n";
    s += &format!(
        "Transfers of pages to model hosts outside Kenya: {} (first {}, last {}), each recorded in `transfers.jsonl` with what reg. 41(2) asks. Reg. 26 requires processing for elections to run in Kenya or keep a serving copy there: originals and outputs stay on the operator's machine.\n\n",
        n("transfers_abroad"), f["transfers_first"].as_str().unwrap_or("-"), f["transfers_last"].as_str().unwrap_or("-")
    );
    s += &table(&f["transfers_by_recipient"], ("Recipient (purpose)", "Transfers"));
    s += "\n";
    s += &table(&f["review_hosts"], ("Review host (region)", "Reviews"));
    s += "\n## 7. Retention\n\n";
    match retention {
        Some(r) => s += &format!("The retention schedule applied (`rfpii retain`):\n\n```yaml\n{}\n```\n", r.trim_end()),
        None => s += "No retention schedule given (`--retention`).\n",
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasons_lose_page_numbers_figures_and_quoted_text_but_keep_their_words() {
        assert_eq!(
            normalise_reason("page 1: agent_id: printed rule found off the column's left edge (table placed wrong); agent_phone: can't see the printed rule"),
            "agent_id: printed rule found off the column's left edge (table placed wrong)"
        );
        assert_eq!(normalise_reason("'JANE DOE' header at y=812: no table rows found under it"), "'…' header at y=N: no table rows found under it");
    }
}
