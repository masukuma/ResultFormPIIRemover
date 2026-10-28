//! Pages no template matches: read their layout from the page itself.
//! (Being ported from the Python version; until then such pages go to review.)

use anyhow::Result;
use opencv::core::Mat;

use crate::masking::MaskSettings;
use crate::pipeline::PageResult;
use crate::pyrand::PyRandom;

pub fn redact_generic(_img: &Mat, _ms: &MaskSettings, _rng: &mut PyRandom, _verify: bool, why: &str) -> Result<PageResult> {
    let prefix = if why.is_empty() { String::new() } else { format!("no template matched ({}); ", why) };
    Ok(PageResult {
        ok: false,
        image: None,
        template: "generic".into(),
        inliers: 0,
        score: 0.0,
        reason: prefix + "reading unknown layouts is not available in this build",
        ms: 0.0,
        masks: vec![],
        notes: vec![],
        blocks: None,
        protected: vec![],
        h: None,
        layout: None,
    })
}
