//! `rfpii dataset ink`: the pipeline's own ink masks of any page, for
//! scoring a network against the rules (NEURAL.md §5, the rule stem of M3).
//!
//! Each page is made at most `max_side` px on its long side (`INTER_AREA`,
//! as `stamps::ink_maps` makes its working copy) and stored as a PNG, so a
//! network reads exactly the pixels the masks were computed from. Its mask
//! has three bits, each the pipeline's own function on that image:
//!
//! - 1: dark ink, `stamps::dark_ink` (Gaussian 3x3, mean threshold over
//!   25x25 with C = 12, inverted, a 2x2 opening);
//! - 2: rules, `masking::rule_lines` over the dark ink: what an opening with
//!   a horizontal or a vertical line `rule_len` px long keeps, `rule_len`
//!   being a fortieth of the long side (the length `ink_mask` opens a
//!   number's strip with, at the templates' scale);
//! - 4: stamp ink, `stamps::color_ink` (Lab, each channel's 31x31 median
//!   as the paper, chroma difference over 9, a and L gates, a 2x2 opening).
//!
//! ```text
//! inkset.json     rfpii/nn-inkset/1: how it was made, what each bit is
//! pages/<id>.png  the page, at most max_side px long
//! masks/<id>.png  its mask bits
//! index.jsonl     one line per page: source, page, size, split key, split
//! SHA256SUMS
//! ```

use anyhow::{bail, Context, Result};
use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::*;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use super::{created_now, read_index, sha256_hex, split_of, write_sums};
use crate::util;

pub const INKSET_SCHEMA: &str = "rfpii/nn-inkset/1";
pub const BIT_DARK: u8 = 1;
pub const BIT_RULE: u8 = 2;
pub const BIT_STAMP: u8 = 4;

/// One page to write: its source (a file or zip::member), its page number
/// (from 1), its split key and a free-form group (where it came from).
#[derive(Clone, Debug)]
pub struct InkJob {
    pub source: String,
    pub page: usize,
    pub key: String,
    pub group: String,
}

pub struct InkOptions {
    pub out: PathBuf,
    pub max_side: i32,
    pub append: bool,
    pub personal_data: bool,
    pub workers: usize,
}

/// The rules' length on a page `w` x `h` px: a fortieth of its long side.
pub fn rule_len(w: i32, h: i32) -> i32 {
    ((w.max(h) as f64 / 40.0).round() as i32).max(15)
}

/// The page at most `max_side` px long (`INTER_AREA`), as `ink_maps` makes
/// its working copy.
pub fn working(img: &Mat, max_side: i32) -> Result<Mat> {
    let s = (max_side as f64 / img.rows().max(img.cols()) as f64).min(1.0);
    if s < 1.0 {
        util::resize_by(img, s, imgproc::INTER_AREA)
    } else {
        Ok(img.try_clone()?)
    }
}

/// The three masks of a colour (BGR) page, as bits (`BIT_*`).
pub fn ink_bits(img: &Mat) -> Result<Vec<u8>> {
    if img.channels() != 3 {
        bail!("a colour page is needed ({} channels)", img.channels());
    }
    let dark_m = crate::stamps::dark_ink(&util::gray(img)?)?;
    let l = rule_len(img.cols(), img.rows());
    let rule = util::bytes(&crate::masking::rule_lines(&dark_m, l, l)?)?;
    let dark = util::bytes(&dark_m)?;
    let stamp = util::bytes(&crate::stamps::color_ink(img)?)?;
    Ok((0..dark.len())
        .map(|i| (if dark[i] != 0 { BIT_DARK } else { 0 }) | (if rule[i] != 0 { BIT_RULE } else { 0 }) | (if stamp[i] != 0 { BIT_STAMP } else { 0 }))
        .collect())
}

fn page_id(source: &str, page: usize) -> String {
    sha256_hex(format!("{}#{}", source, page).as_bytes())[..16].to_string()
}

fn one(job: &InkJob, o: &InkOptions) -> Result<Value> {
    let data = crate::io::read_bytes(&job.source)?;
    let (sp, _) = crate::io::source_page(&job.source, &data, job.page)?;
    let mut img = sp.pixels;
    if img.channels() == 1 {
        let mut c = Mat::default();
        imgproc::cvt_color_def(&img, &mut c, imgproc::COLOR_GRAY2BGR)?;
        img = c;
    }
    let (ow, oh) = (img.cols(), img.rows());
    let page = working(&img, o.max_side)?;
    let bits = ink_bits(&page)?;
    let id = page_id(&job.source, job.page);
    let png = crate::io::encode(&page, "png", 0)?;
    let mask = crate::io::encode(&util::from_bytes(page.rows(), page.cols(), &bits)?, "png", 0)?;
    let (image, mpath) = (format!("pages/{}.png", id), format!("masks/{}.png", id));
    std::fs::write(o.out.join(&image), &png)?;
    std::fs::write(o.out.join(&mpath), &mask)?;
    let count = |b: u8| bits.iter().filter(|&&v| v & b != 0).count();
    Ok(json!({
        "id": id, "source": job.source, "page": job.page, "group": job.group,
        "split_key": job.key, "split": split_of(&job.key),
        "width": page.cols(), "height": page.rows(), "original": [ow, oh],
        "rule_len": rule_len(page.cols(), page.rows()),
        "image": image, "image_sha256": sha256_hex(&png), "mask": mpath, "mask_sha256": sha256_hex(&mask),
        "ink": {"dark": count(BIT_DARK), "rule": count(BIT_RULE), "stamp": count(BIT_STAMP)},
    }))
}

/// Write the pages of `jobs` and their masks to `o.out`; returns
/// (inkset.json, the sources that failed with why).
pub fn run(jobs: &[InkJob], o: &InkOptions) -> Result<(Value, Vec<String>)> {
    let t0 = std::time::Instant::now();
    let mut existing = Vec::new();
    let mut created = None;
    if o.out.exists() && std::fs::read_dir(&o.out)?.next().is_some() {
        if !o.append {
            bail!("{} is not empty (use --append to add to an ink set)", o.out.display());
        }
        let m = crate::manifest::read_json(&o.out.join("inkset.json")).with_context(|| format!("{} is not an ink set", o.out.display()))?;
        if m["max_side"].as_i64() != Some(o.max_side as i64) {
            bail!("{} was made at --max-side {}, not {}", o.out.display(), m["max_side"], o.max_side);
        }
        created = Some(m["created"].clone());
        existing = read_index(&o.out)?;
    }
    for sub in ["pages", "masks"] {
        std::fs::create_dir_all(o.out.join(sub))?;
    }
    let todo: Vec<&InkJob> = jobs.iter().filter(|j| !existing.iter().any(|l| l["id"].as_str() == Some(page_id(&j.source, j.page).as_str()))).collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(o.workers.max(1)).build()?;
    let results: Vec<(String, Result<Value>)> = pool.install(|| todo.par_iter().map(|j| (format!("{} page {}", j.source, j.page), one(j, o))).collect());
    let mut lines = existing;
    let mut failed = Vec::new();
    let n_new = results.iter().filter(|r| r.1.is_ok()).count();
    for (what, r) in results {
        match r {
            Ok(l) => lines.push(l),
            Err(e) => failed.push(format!("{}: {:#}", what, e)),
        }
    }
    lines.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let mut index = String::new();
    for l in &lines {
        index += &serde_json::to_string(l)?;
        index.push('\n');
    }
    std::fs::write(o.out.join("index.jsonl"), index)?;
    let now = created_now();
    let count = |s: &str| lines.iter().filter(|l| l["split"] == s).count();
    let meta = json!({
        "schema": INKSET_SCHEMA,
        "created": created.unwrap_or(json!(now)),
        "updated": now,
        "rfpii": crate::process::tool_info(),
        "max_side": o.max_side,
        "personal_data": o.personal_data,
        "bits": {
            "1": "dark ink: stamps::dark_ink of the page's grey",
            "2": "rules: masking::rule_lines(dark ink, rule_len, rule_len), rule_len = round(long side / 40), at least 15",
            "4": "stamp ink: stamps::color_ink of the page",
        },
        "pages": lines.len(),
        "splits": {"train": count("train"), "val": count("val"), "test": count("test")},
        "this_run": {"pages": n_new, "failed": failed.len(), "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0},
    });
    std::fs::write(o.out.join("inkset.json"), format!("{}\n", serde_json::to_string_pretty(&meta)?))?;
    write_sums(&o.out)?;
    Ok((meta, failed))
}

/// Jobs from a JSON-lines list: `{"source", "page", "key", "group"}` each
/// (page defaults to 1, key to the source's folder, group to "").
pub fn read_list(path: &Path) -> Result<Vec<InkJob>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (i, l) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(l).with_context(|| format!("{} line {}", path.display(), i + 1))?;
        let Some(source) = v["source"].as_str() else { bail!("{} line {}: no source", path.display(), i + 1) };
        out.push(InkJob {
            source: source.to_string(),
            page: v["page"].as_u64().unwrap_or(1).max(1) as usize,
            key: v["key"].as_str().map(str::to_string).unwrap_or_else(|| super::teacher::folder_key(source)),
            group: v["group"].as_str().unwrap_or("").to_string(),
        });
    }
    Ok(out)
}

/// Jobs for every page (at most `pages` each, when given) of the sources
/// under `paths`, keyed by the folder each is in.
pub fn jobs_for(paths: &[String], pages: Option<usize>, group: &str) -> Result<Vec<InkJob>> {
    let mut out = Vec::new();
    for src in crate::io::list_inputs(paths)? {
        let data = crate::io::read_bytes(&src)?;
        let n = crate::io::page_count(&src, &data)?;
        for k in 1..=pages.map_or(n, |p| p.min(n)) {
            out.push(InkJob { source: src.clone(), page: k, key: super::teacher::folder_key(&src), group: group.to_string() });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Point, Scalar, CV_8UC3};

    #[test]
    fn the_three_masks_find_print_a_rule_and_a_stamp() {
        let mut img = Mat::new_rows_cols_with_default(400, 600, CV_8UC3, Scalar::new(235.0, 238.0, 240.0, 0.0)).unwrap();
        // a long printed rule, a short dark stroke, and a violet stroke
        imgproc::line(&mut img, Point::new(20, 100), Point::new(560, 100), Scalar::all(30.0), 3, imgproc::LINE_8, 0).unwrap();
        imgproc::line(&mut img, Point::new(100, 200), Point::new(110, 210), Scalar::all(30.0), 3, imgproc::LINE_8, 0).unwrap();
        imgproc::line(&mut img, Point::new(300, 300), Point::new(340, 300), Scalar::new(170.0, 60.0, 120.0, 0.0), 5, imgproc::LINE_8, 0).unwrap();
        let bits = ink_bits(&img).unwrap();
        let at = |x: usize, y: usize| bits[y * 600 + x];
        assert_eq!(at(300, 100) & (BIT_DARK | BIT_RULE), BIT_DARK | BIT_RULE);
        assert_eq!(at(105, 205) & (BIT_DARK | BIT_RULE), BIT_DARK);
        assert_ne!(at(320, 300) & BIT_STAMP, 0);
        assert_eq!(at(320, 100) & BIT_STAMP, 0);
        assert_eq!(at(50, 50), 0);
        assert_eq!(rule_len(600, 400), 15);
        assert_eq!(rule_len(1280, 905), 32);
    }

    #[test]
    fn an_ink_set_is_written_and_appended_to() {
        let d = crate::testutil::tmpdir("inkset");
        let src = d.join("p.png");
        let img = Mat::new_rows_cols_with_default(300, 500, CV_8UC3, Scalar::all(230.0)).unwrap();
        std::fs::write(&src, crate::io::encode(&img, "png", 0).unwrap()).unwrap();
        let out = d.join("set");
        let o = InkOptions { out: out.clone(), max_side: 250, append: false, personal_data: false, workers: 1 };
        let jobs = vec![InkJob { source: src.display().to_string(), page: 1, key: "k".into(), group: "g".into() }];
        let (meta, failed) = run(&jobs, &o).unwrap();
        assert!(failed.is_empty());
        assert_eq!(meta["pages"], 1);
        let idx = read_index(&out).unwrap();
        assert_eq!((idx[0]["width"].as_i64(), idx[0]["height"].as_i64()), (Some(250), Some(150)));
        assert!(run(&jobs, &o).is_err());
        let (meta, _) = run(&jobs, &InkOptions { append: true, ..o }).unwrap();
        assert_eq!((meta["pages"].as_i64(), meta["this_run"]["pages"].as_i64()), (Some(1), Some(0)));
        let _ = std::fs::remove_dir_all(&d);
    }
}
