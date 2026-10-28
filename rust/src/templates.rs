//! Form templates: a reference image plus the zones that hold PII.
//!
//! A template lives in its own directory:
//!
//! ```text
//! templates/<name>/
//!     template.yaml
//!     reference.png      # clean scan or photo of the form type
//! ```
//!
//! template.yaml:
//!
//! ```yaml
//! name: form34a_2022
//! form: 34A              # matched against the form code in the QR / KIEMS file name
//! reference: reference.png
//! padding: 8             # px (template space) added around every zone
//! protect:               # results regions: no mask may touch them
//!   - name: votes
//!     rect: [x, y, w, h]
//! structure:             # printed tables whose rules re-position their zones
//!   - name: agents
//!     rect: [x, y, w, h]
//! zones:
//!   - name: officer_signatures
//!     action: signature  # a band through the middle of each signature
//!     rect: [x, y, w, h]
//!     search: [12, 24]   # px searched above/below for strokes that stray
//!   - name: agent_id
//!     action: partial    # mask random digits, keep the rest visible
//!     digits: 8          # expected digit count (ID 8, phone 10)
//!     rect: [x, y, w, h] # one cell...
//!     rows: 11           # ...repeated down a table
//!     row_step: 36.4
//!     region: agents     # placed by the table's rules; page fails if they can't be matched
//! ```
//!
//! Names are deliberately not zones: they stay visible. All coordinates are
//! pixels in the reference image.

use anyhow::{anyhow, bail, Context, Result};
use opencv::core::Mat;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde_yaml::Value;
use sha2::{Digest, Sha256};
use std::path::Path;

pub const ACTIONS: &[&str] = &["blackout", "partial", "signature"];

#[derive(Clone, Debug)]
pub struct Zone {
    pub name: String,
    pub rect: [f64; 4], // x, y, w, h in template pixels
    pub action: String,
    pub digits: usize,
    pub group: String,  // table column this cell belongs to ("" = standalone)
    pub row: usize,
    pub region: String, // structure region whose printed rules position this zone
    pub rule: String,   // partial zones: "phone" or "id"
    pub search: Option<[f64; 2]>, // px searched above / below the cells
    pub anchor: Option<[f64; 4]>, // printed label (x, y, w, h) the zone moves with
}

impl Zone {
    pub fn new(name: &str, rect: [f64; 4], action: &str) -> Self {
        Zone {
            name: name.into(),
            rect,
            action: action.into(),
            digits: 0,
            group: String::new(),
            row: 0,
            region: String::new(),
            rule: String::new(),
            search: None,
            anchor: None,
        }
    }

    pub fn polygon(&self) -> [[f64; 2]; 4] {
        let [x, y, w, h] = self.rect;
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    }

    pub fn padded(&self, pad: f64) -> [[f64; 2]; 4] {
        let [x, y, w, h] = self.rect;
        Zone::new("", [x - pad, y - pad, w + 2.0 * pad, h + 2.0 * pad], "").polygon()
    }

    pub fn key(&self) -> &str {
        if self.group.is_empty() {
            &self.name
        } else {
            &self.group
        }
    }
}

pub struct Template {
    pub name: String,
    pub image: Mat, // grayscale reference
    pub zones: Vec<Zone>,
    pub padding: i32,
    pub form: String,
    pub structure: Vec<(String, [f64; 4])>, // in the YAML's order, as Python's dict
    pub protect: Vec<(String, [f64; 4])>,
    pub fingerprint: String,
    pub min_area_frac: Option<f64>,
    pub priority: i64,
    pub once: bool,
    pub page_long_side: Option<f64>,
    pub min_inliers: Option<i64>,
    pub min_score: Option<f64>,
    pub feature_dim: Option<i64>,
    pub n_features: Option<i64>,
    pub open_top: Vec<String>,
    pub min_glyph_h: Option<f64>,
    // filled in by align::prepare
    pub regions: Vec<(String, crate::structure::Region)>,
    pub mask: Mat,
    pub keypoints: Vec<[f32; 2]>,
    pub descriptors: Mat,
}

impl Template {
    pub fn size(&self) -> (i32, i32) {
        (self.image.cols(), self.image.rows())
    }

    /// Template-space mask (255 = any zone) with padding applied.
    pub fn build_mask(&self) -> Result<Mat> {
        let mut m = crate::util::zeros_u8(self.image.rows(), self.image.cols())?;
        for z in &self.zones {
            crate::util::fill_poly(&mut m, &z.padded(self.padding as f64), 255.0)?;
        }
        Ok(m)
    }
}

fn f(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|i| i as f64))
}

fn rect4(v: &Value, what: &str) -> Result<[f64; 4]> {
    let a = v.as_sequence().ok_or_else(|| anyhow!("{}: rect must be a list", what))?;
    if a.len() != 4 {
        bail!("{}: rect must have 4 numbers", what);
    }
    let mut r = [0.0; 4];
    for (i, x) in a.iter().enumerate() {
        r[i] = f(x).ok_or_else(|| anyhow!("{}: rect values must be numbers", what))?;
    }
    Ok(r)
}

fn zones_from_spec(spec: &Value) -> Result<Vec<Zone>> {
    let name = spec.get("name").and_then(|v| v.as_str()).unwrap_or("zone").to_string();
    let action = spec.get("action").and_then(|v| v.as_str()).unwrap_or("blackout").to_string();
    if !ACTIONS.contains(&action.as_str()) {
        bail!("zone {:?}: action must be one of {:?}", name, ACTIONS);
    }
    let rect = rect4(spec.get("rect").ok_or_else(|| anyhow!("zone {:?} needs 'rect'", name))?, &name)?;
    let digits = spec.get("digits").and_then(f).unwrap_or(0.0) as usize;
    if action == "partial" && digits < 2 {
        bail!("zone {:?}: partial zones need 'digits' >= 2", name);
    }
    let rule = if action == "partial" {
        spec.get("rule").and_then(|v| v.as_str()).map(String::from).unwrap_or_else(|| {
            if digits >= 10 { "phone".into() } else { "id".into() }
        })
    } else {
        String::new()
    };
    if !["", "phone", "id"].contains(&rule.as_str()) {
        bail!("zone {:?}: rule must be 'phone' or 'id'", name);
    }
    let rows = spec.get("rows").and_then(f).unwrap_or(1.0) as usize;
    let step = spec.get("row_step").and_then(f).unwrap_or(rect[3]);
    let search = match spec.get("search") {
        Some(v) => {
            let a = v.as_sequence().ok_or_else(|| anyhow!("zone {:?}: search must be [above, below]", name))?;
            Some([f(&a[0]).unwrap_or(0.0), f(&a[1]).unwrap_or(0.0)])
        }
        None => None,
    };
    let anchor = match spec.get("anchor") {
        Some(v) => Some(rect4(v, &name)?),
        None => None,
    };
    let region = spec.get("region").and_then(|v| v.as_str()).unwrap_or("").to_string();
    Ok((0..rows)
        .map(|r| Zone {
            name: if rows > 1 { format!("{}[{}]", name, r) } else { name.clone() },
            rect: [rect[0], rect[1] + r as f64 * step, rect[2], rect[3]],
            action: action.clone(),
            digits,
            group: if rows > 1 { name.clone() } else { String::new() },
            row: r,
            region: region.clone(),
            rule: rule.clone(),
            search,
            anchor,
        })
        .collect())
}

pub fn load_template(dir: &Path) -> Result<Template> {
    let yaml_path = dir.join("template.yaml");
    let yaml_bytes = std::fs::read(&yaml_path).with_context(|| format!("reading {}", yaml_path.display()))?;
    let spec: Value = serde_yaml::from_slice(&yaml_bytes).with_context(|| format!("parsing {}", yaml_path.display()))?;
    let reference = spec.get("reference").and_then(|v| v.as_str()).unwrap_or("reference.png");
    let ref_path = dir.join(reference);
    let image = imgcodecs::imread(&ref_path, imgcodecs::IMREAD_GRAYSCALE)?;
    if image.empty() {
        bail!("reference image missing for template in {}", dir.display());
    }
    let mut zones = Vec::new();
    if let Some(zs) = spec.get("zones").and_then(|v| v.as_sequence()) {
        for z in zs {
            zones.extend(zones_from_spec(z)?);
        }
    }
    let mut structure: Vec<(String, [f64; 4])> = Vec::new();
    let mut open_top = Vec::new();
    if let Some(ss) = spec.get("structure").and_then(|v| v.as_sequence()) {
        for r in ss {
            let n = r.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let rect = rect4(r.get("rect").unwrap_or(&Value::Null), &n)?;
            match structure.iter_mut().find(|(k, _)| *k == n) {
                Some(e) => e.1 = rect,
                None => structure.push((n.clone(), rect)),
            }
            if r.get("open_top").and_then(|v| v.as_bool()).unwrap_or(false) {
                open_top.push(n);
            }
        }
    }
    for z in &zones {
        if !z.region.is_empty() && !structure.iter().any(|(k, _)| *k == z.region) {
            bail!("zone {:?} refers to unknown structure region {:?}", z.name, z.region);
        }
    }
    let mut protect = Vec::new();
    if let Some(ps) = spec.get("protect").and_then(|v| v.as_sequence()) {
        for r in ps {
            let n = r.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            protect.push((n.clone(), rect4(r.get("rect").unwrap_or(&Value::Null), &n)?));
        }
    }
    let mut h = Sha256::new();
    h.update(&yaml_bytes);
    h.update(std::fs::read(&ref_path)?);
    let getf = |k: &str| spec.get(k).and_then(f);
    let geti = |k: &str| spec.get(k).and_then(|v| v.as_i64());
    let image_rows = image.rows();
    let image_cols = image.cols();
    let mut t = Template {
        name: spec.get("name").and_then(|v| v.as_str()).map(String::from).unwrap_or_else(|| {
            dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
        }),
        image,
        zones,
        padding: geti("padding").unwrap_or(8) as i32,
        form: spec.get("form").and_then(|v| v.as_str().map(String::from).or_else(|| v.as_i64().map(|i| i.to_string()))).unwrap_or_default(),
        structure,
        protect,
        fingerprint: hex::encode(h.finalize()),
        min_area_frac: getf("min_area_frac"),
        priority: geti("priority").unwrap_or(0),
        once: spec.get("once").and_then(|v| v.as_bool()).unwrap_or(false),
        page_long_side: getf("page_long_side"),
        min_inliers: geti("min_inliers"),
        min_score: getf("min_score"),
        feature_dim: geti("feature_dim"),
        n_features: geti("n_features"),
        open_top,
        min_glyph_h: getf("min_glyph_h"),
        regions: Vec::new(),
        mask: crate::util::zeros_u8(image_rows, image_cols)?,
        keypoints: Vec::new(),
        descriptors: Mat::default(),
    };
    t.mask = t.build_mask()?;
    Ok(t)
}

pub fn load_templates(root: &Path) -> Result<Vec<Template>> {
    if root.join("template.yaml").exists() {
        return Ok(vec![load_template(root)?]);
    }
    let mut dirs: Vec<_> = std::fs::read_dir(root)
        .with_context(|| format!("reading {}", root.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("template.yaml").exists())
        .collect();
    dirs.sort();
    if dirs.is_empty() {
        bail!("no templates found under {}", root.display());
    }
    dirs.iter().map(|d| load_template(d)).collect()
}
