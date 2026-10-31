//! The plan (HARNESS.md §4.2): what is redacted, in which groups, page by
//! page or whole, with how many workers; and whether the machine and the
//! budget can take it.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::policy::{PerPage, Policy};
use crate::process::output_rel;

/// A document `redact` makes: a file, or one page of it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Unit {
    pub src: String,
    /// its path under the output folders
    pub rel: String,
    /// the folder `rel` is relative to (`redact --base`)
    pub base: String,
    pub page: Option<usize>,
    pub group: String,
    pub folder: String,
    /// the source file's
    pub sha256: String,
}

impl Unit {
    /// What the ledger calls it, as `redact` does.
    pub fn name(&self) -> String {
        match self.page {
            None => self.rel.clone(),
            Some(_) => output_rel(&self.rel, self.page, "").to_string_lossy().into_owned(),
        }
    }

    /// Its file under `root` with `ext` (".pdf", ".manifest.json", ".json").
    pub fn at(&self, root: &Path, ext: &str) -> PathBuf {
        root.join(output_rel(&self.rel, self.page, ext))
    }

    /// The page of the source it is, or its first.
    pub fn page_no(&self) -> usize {
        self.page.unwrap_or(1)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Group {
    pub id: String,
    pub kind: String,
    pub per_page: bool,
    pub units: Vec<Unit>,
    /// pages (or documents) per folder
    pub folders: BTreeMap<String, usize>,
}

/// What a page costs, measured on the 2013 run (HARNESS.md §4.2): about
/// 0.4 pages a second and 0.55 MB of output a page; a worker on a scan of
/// many forms 2.2 GB, on anything else 1.2 GB.
pub const PAGES_PER_SECOND: f64 = 0.4;
pub const BYTES_PER_PAGE: f64 = 0.55e6;
const WORKER_BYTES: [(&str, f64); 2] = [("scan", 2.2e9), ("other", 1.2e9)];

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Plan {
    /// the seed every sample of the job is drawn from
    pub seed: u64,
    pub groups: Vec<Group>,
    pub workers: usize,
    pub reader: String,
    pub pages: usize,
    pub hours: f64,
    pub disk_bytes: f64,
    pub free_disk_bytes: Option<u64>,
    pub memory_available: Option<u64>,
    /// files left out (copies, unreadable), and why
    pub left_out: Vec<Value>,
}

/// Bytes free on the filesystem holding `dir` (`df`); None if unknown.
pub fn free_disk(dir: &Path) -> Option<u64> {
    let out = std::process::Command::new("df").arg("-Pk").arg(dir).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let kb: u64 = text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
    Some(kb * 1024)
}

/// Memory available, bytes (`/proc/meminfo`); None if unknown.
pub fn memory_available() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = text.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Workers for groups of these kinds, from the memory available: four
/// fifths of it, shared by the kind that needs most per worker, and no
/// more than the cores.
pub fn workers(kinds: &[&str], available: Option<u64>) -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let per = if kinds.contains(&"scan") { WORKER_BYTES[0].1 } else { WORKER_BYTES[1].1 };
    match available {
        Some(m) => ((m as f64 * 0.8 / per) as usize).clamp(1, cores),
        None => 1,
    }
}

/// Whether a file of `kind` and `pages` is redacted page by page.
fn per_page(policy: &Policy, kind: &str, pages: usize) -> bool {
    pages > 1
        && match policy.per_page {
            PerPage::Auto => kind == "scan",
            PerPage::Yes => true,
            PerPage::No => false,
        }
}

impl Plan {
    /// The plan for an inventory under `policy`: each group's documents
    /// (copies and unreadable files left out), per page or whole.
    pub fn new(inv: &Value, policy: &Policy, templates: &Path, outputs: &Path) -> Plan {
        let mut groups: BTreeMap<String, Group> = BTreeMap::new();
        let mut left_out = Vec::new();
        for f in inv["files"].as_array().into_iter().flatten() {
            let path = f["path"].as_str().unwrap_or("").to_string();
            if let Some(why) = f["broken"].as_str() {
                left_out.push(json!({"path": path, "why": format!("can't be read: {}", why)}));
                continue;
            }
            if let Some(orig) = f["duplicate_of"].as_str() {
                left_out.push(json!({"path": path, "why": format!("a copy of {}", orig)}));
                continue;
            }
            let (kind, pages) = (f["kind"].as_str().unwrap_or(""), f["pages"].as_u64().unwrap_or(1) as usize);
            let by_page = per_page(policy, kind, pages);
            let g = groups.entry(f["group"].as_str().unwrap_or("").to_string()).or_insert_with(|| Group { id: f["group"].as_str().unwrap_or("").into(), kind: kind.into(), per_page: false, units: vec![], folders: BTreeMap::new() });
            g.per_page |= by_page;
            let unit = |page| Unit {
                src: f["source"].as_str().unwrap_or("").into(),
                rel: path.clone(),
                base: f["base"].as_str().unwrap_or("").into(),
                page,
                group: g.id.clone(),
                folder: f["folder"].as_str().unwrap_or(".").into(),
                sha256: f["sha256"].as_str().unwrap_or("").into(),
            };
            let units: Vec<Unit> = if by_page { (1..=pages).map(|k| unit(Some(k))).collect() } else { vec![unit(None)] };
            *g.folders.entry(units[0].folder.clone()).or_insert(0) += units.len();
            g.units.extend(units);
        }
        let pages = inv["totals"]["pages"].as_u64().unwrap_or(0) as usize;
        let kinds: Vec<&str> = groups.values().map(|g| g.kind.as_str()).collect();
        let available = memory_available();
        let mut sums: Vec<&str> = inv["files"].as_array().into_iter().flatten().filter_map(|f| f["sha256"].as_str()).collect();
        sums.sort_unstable();
        Plan {
            seed: super::sample::seed(0, &sums.join(",")),
            workers: workers(&kinds, available),
            groups: groups.into_values().collect(),
            reader: format!("the templates in {}; a page no template matches is read without one (OCR)", templates.display()),
            pages,
            hours: pages as f64 / PAGES_PER_SECOND / 3600.0,
            disk_bytes: pages as f64 * BYTES_PER_PAGE,
            free_disk_bytes: free_disk(outputs),
            memory_available: available,
            left_out,
        }
    }

    /// Estimates again from what a pilot measured: seconds and output bytes
    /// per page.
    pub fn remeasure(&mut self, seconds_per_page: f64, bytes_per_page: f64, outputs: &Path) {
        self.hours = self.pages as f64 * seconds_per_page / 3600.0;
        self.disk_bytes = self.pages as f64 * bytes_per_page;
        self.free_disk_bytes = free_disk(outputs);
    }

    /// Why the machine or the budget can't take the plan, if they can't:
    /// the output won't fit on the disk (with a tenth to spare), or the run
    /// won't fit the hours.
    pub fn refusal(&self, budget_hours: Option<f64>) -> Option<String> {
        if let Some(free) = self.free_disk_bytes {
            if self.disk_bytes * 1.1 > free as f64 {
                return Some(format!("the output needs about {:.1} GB and {:.1} GB is free", self.disk_bytes / 1e9, free as f64 / 1e9));
            }
        }
        match budget_hours {
            Some(h) if self.hours > h => Some(format!("the run needs about {:.1} hours and the budget is {} hours", self.hours, h)),
            _ => None,
        }
    }

    /// Every unit of the plan.
    pub fn units(&self) -> impl Iterator<Item = &Unit> {
        self.groups.iter().flat_map(|g| g.units.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inv() -> Value {
        json!({"totals": {"pages": 43}, "files": [
            {"source": "/d/A/scan.pdf", "path": "A/scan.pdf", "base": "/d", "folder": "A", "sha256": "aa", "pages": 40, "kind": "scan", "group": "scan"},
            {"source": "/d/B/p.jpg", "path": "B/p.jpg", "base": "/d", "folder": "B", "sha256": "bb", "pages": 1, "kind": "photo", "group": "photo"},
            {"source": "/d/B/q.jpg", "path": "B/q.jpg", "base": "/d", "folder": "B", "sha256": "bb", "pages": 1, "kind": "photo", "group": "photo", "duplicate_of": "B/p.jpg"},
            {"source": "/d/B/x.pdf", "path": "B/x.pdf", "base": "/d", "folder": "B", "broken": "no pages can be read"},
            {"source": "/d/C/f.pdf", "path": "C/f.pdf", "base": "/d", "folder": "C", "sha256": "cc", "pages": 2, "kind": "pdf", "group": "pdf"}]})
    }

    #[test]
    fn scans_are_planned_page_by_page_and_copies_left_out() {
        let p = Plan::new(&inv(), &Policy::default(), Path::new("t"), Path::new("/tmp"));
        let g: BTreeMap<&str, &Group> = p.groups.iter().map(|g| (g.id.as_str(), g)).collect();
        assert!(g["scan"].per_page && g["scan"].units.len() == 40 && g["scan"].units[39].name() == "A/scan.p040");
        assert!(!g["photo"].per_page && g["photo"].units.len() == 1 && g["photo"].units[0].name() == "B/p.jpg");
        assert!(!g["pdf"].per_page);
        assert_eq!(p.left_out.len(), 2);
        assert_eq!(p.units().count(), 42);
        let all = Plan::new(&inv(), &Policy::from_value(&json!({"per_page": "yes"})).unwrap(), Path::new("t"), Path::new("/tmp"));
        assert_eq!(all.units().count(), 43);
    }

    #[test]
    fn a_plan_the_disk_or_budget_cant_take_is_refused() {
        let mut p = Plan::new(&inv(), &Policy::default(), Path::new("t"), Path::new("/tmp"));
        p.free_disk_bytes = Some(10_000_000);
        assert!(p.refusal(None).unwrap().contains("GB is free"));
        p.free_disk_bytes = Some(1 << 40);
        assert!(p.refusal(Some(1.0)).is_none());
        p.remeasure(3600.0, 1e3, Path::new("/tmp"));
        assert!(p.refusal(Some(12.0)).unwrap().contains("hours"));
        assert_eq!(workers(&["scan"], Some(9_000_000_000)).min(3), 3.min(std::thread::available_parallelism().unwrap().get()));
        assert_eq!(workers(&["photo"], None), 1);
    }
}
