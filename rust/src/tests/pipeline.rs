//! Whole pages: synthetic forms through alignment, masking and checks.

use super::*;
use crate::synthetic::{blank_form, filled_form, simulate_scan, write_template, ZONES};
use crate::templates::load_templates;
use crate::testutil::*;
use opencv::core::{Point, Scalar};
use opencv::imgproc;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn templates() -> Vec<Template> {
    static DIR: once_cell::sync::Lazy<std::path::PathBuf> = once_cell::sync::Lazy::new(|| {
        let d = tmpdir("tpl").join("sample_rs1");
        write_template(&d).unwrap();
        d
    });
    let mut ts = load_templates(&DIR).unwrap();
    for t in ts.iter_mut() {
        crate::align::prepare(t, &AlignSettings::default()).unwrap();
    }
    ts
}

fn run(scan: &Mat, ts: &[Template]) -> PageResult {
    redact_page(scan, ts, &AlignSettings::default(), &MaskSettings::default(), &mut PyRandom::new(0), &PageOptions::default()).unwrap()
}

/// A template rectangle as it lies on the scan (shrunk by `shrink`), as an
/// integer polygon.
fn project(rect: [f64; 4], h: &[[f64; 3]; 3], shrink: f64) -> Vec<[f64; 2]> {
    let [x, y, w, hh] = rect;
    let (x, y, w, hh) = (x + shrink, y + shrink, w - 2.0 * shrink, hh - 2.0 * shrink);
    [[x, y], [x + w, y], [x + w, y + hh], [x, y + hh]].iter().map(|p| util::project(h, *p)).map(|p| [p[0].trunc(), p[1].trunc()]).collect()
}

/// The pixels (all channels) of `img` inside `poly`.
fn region(img: &Mat, poly: &[[f64; 2]]) -> Vec<u8> {
    let mut m = gray(img.rows(), img.cols(), 0);
    util::fill_poly(&mut m, poly, 255.0).unwrap();
    let (mk, px) = (util::bytes(&m).unwrap(), util::bytes(img).unwrap());
    let ch = img.channels() as usize;
    mk.iter().enumerate().filter(|&(_, &v)| v != 0).flat_map(|(i, _)| px[i * ch..(i + 1) * ch].to_vec()).collect()
}

#[test]
fn pii_zones_fully_blacked_out() {
    let ts = templates();
    for (seed, angle) in [(0, 0.0), (1, 4.0), (2, -7.0), (3, 12.0), (4, 180.0)] {
        let (scan, h) = simulate_scan(&filled_form(seed).unwrap(), seed, angle, 1.3).unwrap();
        let r = run(&scan, &ts);
        assert!(r.ok, "seed {}: {}", seed, r.reason);
        let img = r.image.unwrap();
        for (name, rect) in ZONES {
            // the true zone (from the known warp) must be entirely black
            let px = region(&img, &project(rect.map(|v| v as f64), &h, 0.0));
            assert!(px.iter().all(|&v| v == 0), "seed {} zone {}", seed, name);
        }
    }
}

#[test]
fn results_table_survives() {
    let ts = templates();
    let (scan, h) = simulate_scan(&filled_form(5).unwrap(), 5, 3.0, 1.3).unwrap();
    let r = run(&scan, &ts);
    assert!(r.ok, "{}", r.reason);
    let table = project([60.0, 520.0, 1120.0, 540.0], &h, 4.0);
    assert_eq!(region(r.image.as_ref().unwrap(), &table), region(&scan, &table));
}

#[test]
fn unrecognised_pages_go_to_review() {
    let ts = templates();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let noise: Vec<u8> = (0..1800 * 1300 * 3).map(|_| rng.random_range(0..255u32) as u8).collect();
    let mut noise_img = bgr(1800, 1300, [0, 0, 0]);
    noise_img.data_bytes_mut().unwrap().copy_from_slice(&noise);
    let mut mirrored = Mat::default();
    opencv::core::flip(&blank_form().unwrap(), &mut mirrored, 1).unwrap(); // not a homography we accept
    for bad in [noise_img, bgr(1800, 1300, [255, 255, 255]), to_bgr(&mirrored)] {
        let r = run(&bad, &ts);
        assert!(!r.ok && r.image.is_none());
    }
}

const STAMP: (i32, i32, i32) = (700, 1262, 70); // template px: across the presiding officer's box, over the signature

/// A round stamp: thick ring, inner ring, lettering.
fn stamped(g: &Mat) -> Mat {
    let mut img = to_bgr(g);
    let c = Scalar::new(150.0, 40.0, 110.0, 0.0);
    let (cx, cy, r) = STAMP;
    imgproc::circle(&mut img, Point::new(cx, cy), r, c, 5, imgproc::LINE_AA, 0).unwrap();
    imgproc::circle(&mut img, Point::new(cx, cy), (r as f64 * 0.78) as i32, c, 2, imgproc::LINE_AA, 0).unwrap();
    text(&mut img, "I.E.B.C.", (cx - 42, cy - 8), 0.6, c, 2, true);
    text(&mut img, "OFFICER", (cx - 40, cy + 22), 0.5, c, 2, true);
    img
}

#[test]
fn stamp_over_a_signature_is_never_masked() {
    let ts = templates();
    let form = to_gray(&stamped(&filled_form(3).unwrap()));
    let (scan, h) = simulate_scan(&form, 3, 2.0, 1.3).unwrap();
    let r = run(&scan, &ts);
    assert!(r.ok, "{}", r.reason);
    assert!(r.protected.iter().any(|p| p["kind"] == "stamp"), "stamp not recorded as protected");
    // the stamp's disc, as it lies on the scan: not one blacked block may touch it
    let (cx, cy, rad) = (STAMP.0 as f64, STAMP.1 as f64, STAMP.2 as f64);
    let ring: Vec<[f64; 2]> = (0..72)
        .map(|k| {
            let t = 2.0 * std::f64::consts::PI * k as f64 / 72.0;
            let p = util::project(&h, [cx + rad * t.cos(), cy + rad * t.sin()]);
            [util::round_even(p[0]), util::round_even(p[1])]
        })
        .collect();
    let mut disc = gray(scan.rows(), scan.cols(), 0);
    util::fill_poly(&mut disc, &ring, 255.0).unwrap();
    let blocks = r.blocks.as_ref().unwrap();
    assert!(!blocks.and(&proof::to_blocks(&disc).unwrap()).any());
    let img = r.image.unwrap();
    assert_eq!(region(&img, &ring), region(&scan, &ring));
    // ... while the rest of the officer's box is still masked
    let q: Vec<[f64; 2]> = [[340.0, 1220.0], [600.0, 1220.0], [600.0, 1310.0], [340.0, 1310.0]]
        .iter()
        .map(|p| {
            let p = util::project(&h, *p);
            [util::round_even(p[0]), util::round_even(p[1])]
        })
        .collect();
    assert!(region(&img, &q).iter().all(|&v| v == 0));
}

#[test]
fn with_stamps_masked_over_the_officers_box_is_masked_stamp_and_all() {
    let ts = templates();
    let form = to_gray(&stamped(&filled_form(3).unwrap()));
    let (scan, h) = simulate_scan(&form, 3, 2.0, 1.3).unwrap();
    let ms = MaskSettings { mask_stamps: true, ..MaskSettings::default() };
    let r = redact_page(&scan, &ts, &AlignSettings::default(), &ms, &mut PyRandom::new(0), &PageOptions::default()).unwrap();
    assert!(r.ok, "{}", r.reason);
    assert!(!r.protected.iter().any(|p| p["kind"] == "stamp"), "no stamp is protected");
    let img = r.image.unwrap();
    for (name, rect) in ZONES {
        let px = region(&img, &project(rect.map(|v| v as f64), &h, 0.0));
        assert!(px.iter().all(|&v| v == 0), "zone {}", name);
    }
}

#[test]
fn with_stamps_masked_over_a_fields_band_across_a_stamp_is_hidden() {
    let stamp = Stamp::new(500.0, 500.0, 100.0, 1.0, "ring");
    let ident = masking::homography(util::IDENTITY);
    let mut id = Zone::new("id@300,480", [300.0, 480.0, 400.0, 40.0], "partial");
    id.search = Some([10.0, 10.0]);
    let sig = Zone::new("signature@300,560", [300.0, 560.0, 400.0, 30.0], "signature");
    let mut bands = gray(1000, 1000, 255);
    for z in [&id, &sig] {
        masking::fill(&mut bands, &ident, &masking::band(z)).unwrap();
    }
    let mut canvas = gray(1000, 1000, 255);
    assert!(hide_under_stamps(&mut canvas, &bands, std::slice::from_ref(&stamp)).unwrap() > 0);
    let px = |c: &Mat, x: i32, y: i32| *c.at_2d::<u8>(y, x).unwrap();
    // the number's line and the signature's line, across the stamp
    assert_eq!((px(&canvas, 500, 500), px(&canvas, 450, 575)), (0, 0));
    // the rest of the stamp (a name on the line above), and the fields
    // beyond it (theirs to mask), untouched
    assert_eq!((px(&canvas, 500, 420), px(&canvas, 500, 473), px(&canvas, 500, 545), px(&canvas, 680, 500)), (255, 255, 255, 255));

    // nothing is lifted off the stamp; by default the same mask is
    let mut lifted = canvas.clone();
    let page = bgr(1000, 1000, [255, 255, 255]);
    let ms = MaskSettings { mask_stamps: true, ..MaskSettings::default() };
    let (found, keep, notes) = protect_stamps(&page, &mut canvas, std::slice::from_ref(&stamp), (60.0, 140.0), None, &ms).unwrap();
    assert!(found.is_empty() && opencv::core::count_non_zero(&keep).unwrap() == 0);
    assert_eq!(px(&canvas, 500, 500), 0);
    assert!(notes.iter().any(|n| n.contains("--mask-stamps")), "{:?}", notes);
    let (found, _, _) = protect_stamps(&page, &mut lifted, std::slice::from_ref(&stamp), (60.0, 140.0), None, &MaskSettings::default()).unwrap();
    assert_eq!((found.len(), px(&lifted, 500, 500)), (1, 255));
}

#[test]
fn redact_boxes_masks_signature() {
    let mut page = bgr(400, 600, [230, 230, 230]);
    imgproc::put_text(&mut page, "Sig", Point::new(100, 200), imgproc::FONT_HERSHEY_SCRIPT_SIMPLEX, 2.0, Scalar::all(0.0), 3, imgproc::LINE_8, false).unwrap();
    let r = redact_boxes(&page, &[(90.0, 140.0, 260.0, 220.0, "signature".into())], &MaskSettings::default(), &mut PyRandom::new(0), 10.0).unwrap();
    let img = r.image.unwrap();
    let poly = [[100.0, 150.0], [249.0, 150.0], [249.0, 209.0], [100.0, 209.0]];
    assert!(region(&img, &poly).iter().all(|&v| v == 0));
}

#[test]
fn a_tables_columns_move_together_or_not_at_all() {
    // rows 35 px: headers that agree within half a row move the table
    assert_eq!(table_shift(&[-90.0, -89.0], 35.0), Some(-89.5));
    assert_eq!(table_shift(&[-22.0, -24.0, -21.0], 35.0), Some(-22.0));
    // a column's header matched a row or more off the others': no telling
    assert_eq!(table_shift(&[121.0, -63.0], 35.0), None);
    assert_eq!(table_shift(&[-63.0, -88.0], 35.0), None);
}
