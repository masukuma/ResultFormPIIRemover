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
    run_with(scan, ts, &MaskSettings::default())
}

fn run_with(scan: &Mat, ts: &[Template], ms: &MaskSettings) -> PageResult {
    redact_page(scan, ts, &AlignSettings::default(), ms, &mut PyRandom::new(0), &PageOptions::default()).unwrap()
}

fn kept() -> MaskSettings {
    MaskSettings { stamps: masking::StampRule::Protected, ..MaskSettings::default() }
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

/// The stamp's disc as it lies on the scan, as a polygon.
fn stamp_ring(h: &[[f64; 3]; 3]) -> Vec<[f64; 2]> {
    let (cx, cy, rad) = (STAMP.0 as f64, STAMP.1 as f64, STAMP.2 as f64);
    (0..72)
        .map(|k| {
            let t = 2.0 * std::f64::consts::PI * k as f64 / 72.0;
            let p = util::project(h, [cx + rad * t.cos(), cy + rad * t.sin()]);
            [util::round_even(p[0]), util::round_even(p[1])]
        })
        .collect()
}

#[test]
fn with_stamps_kept_a_stamp_over_a_signature_is_never_masked() {
    let ts = templates();
    let form = to_gray(&stamped(&filled_form(3).unwrap()));
    let (scan, h) = simulate_scan(&form, 3, 2.0, 1.3).unwrap();
    let r = run_with(&scan, &ts, &kept());
    assert!(r.ok, "{}", r.reason);
    assert!(r.protected.iter().any(|p| p["kind"] == "stamp"), "stamp not recorded as protected");
    // the stamp's disc, as it lies on the scan: not one blacked block may touch it
    let ring = stamp_ring(&h);
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
fn by_default_the_band_across_a_stamp_is_hidden_and_the_rest_of_the_stamp_kept() {
    let ts = templates();
    let form = to_gray(&stamped(&filled_form(3).unwrap()));
    let (scan, h) = simulate_scan(&form, 3, 2.0, 1.3).unwrap();
    let r = run(&scan, &ts);
    assert!(r.ok, "{}", r.reason);
    // the officers' boxes are masked whole, across the stamp too
    let img = r.image.as_ref().unwrap();
    for (name, rect) in ZONES {
        let px = region(img, &project(rect.map(|v| v as f64), &h, 0.0));
        assert!(px.iter().all(|&v| v == 0), "zone {}", name);
    }
    // the stamp is still protected, less the blocks of the bands across it
    let st = r.protected.iter().find(|p| p["kind"] == "stamp").expect("the stamp is protected");
    let except = st["except_blocks"].as_array().expect("the bands across it are listed");
    assert!(!except.is_empty());
    let mut disc = gray(scan.rows(), scan.cols(), 0);
    util::fill_poly(&mut disc, &stamp_ring(&h), 255.0).unwrap();
    for e in except {
        let v: Vec<i32> = e.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        assert!(v[0] % 8 == 0 && v[1] % 8 == 0 && v[2] % 8 == 0 && v[3] == 8, "whole blocks: {:?}", v);
        fill_rect(&mut disc, v[0], v[1], v[0] + v[2], v[1] + v[3], 0);
    }
    // ... and the rest of its disc is untouched: no blacked block, the same pixels
    assert!(opencv::core::count_non_zero(&disc).unwrap() > 0, "part of the stamp lies off the bands");
    let blocks = r.blocks.as_ref().unwrap();
    assert!(!blocks.and(&proof::to_blocks(&disc).unwrap()).any());
    let (d, a, b) = (util::bytes(&disc).unwrap(), util::bytes(img).unwrap(), util::bytes(&scan).unwrap());
    assert!(d.iter().enumerate().filter(|&(_, &v)| v != 0).all(|(i, _)| a[i * 3..i * 3 + 3] == b[i * 3..i * 3 + 3]));
}

#[test]
fn the_bands_across_a_stamp_are_listed_out_to_its_edge_as_verify_draws_it() {
    // the stamp's outline ends 0.3 px short of a block's edge (x = 600), so
    // drawn in whole px it reaches into that block: the band blocks there
    // are listed too, and the page passes verify's check
    for (cx, cy) in [(500.0, 500.0), (499.7, 500.3), (500.3, 499.7)] {
        let stamp = Stamp::new(cx, cy, 99.7, 1.0, "ring");
        let ident = masking::homography(util::IDENTITY);
        let mut bands = gray(1000, 1000, 255);
        for z in [Zone::new("id", [300.0, 480.0, 400.0, 40.0], "partial"), Zone::new("sig", [480.0, 300.0, 40.0, 400.0], "signature")] {
            masking::fill(&mut bands, &ident, &masking::band(&z)).unwrap();
        }
        let (img, mut canvas) = (bgr(1000, 1000, [255, 255, 255]), gray(1000, 1000, 255));
        let (_, _, _, records) = protect_stamps(&img, &mut canvas, std::slice::from_ref(&stamp), (40.0, 200.0), None, &MaskSettings::default(), Some(&bands)).unwrap();
        let hidden: Vec<u8> = util::bytes(&canvas).unwrap().iter().map(|&v| if v == 0 { 255 } else { 0 }).collect();
        let blocks = proof::to_blocks(&util::from_bytes(1000, 1000, &hidden).unwrap()).unwrap();
        assert!(blocks.any(), "the bands across the stamp are hidden");
        assert_eq!(crate::verify::protected_hits(&records, &blocks, 1000, 1000).unwrap(), 0, "stamp at {}, {}", cx, cy);
    }
}

#[test]
fn across_a_stamp_a_signature_row_gets_a_line_through_its_middle_and_a_number_its_whole_line() {
    let sig = Zone::new("sig", [300.0, 460.0, 400.0, 80.0], "signature");
    let id = Zone::new("id", [300.0, 600.0, 400.0, 40.0], "partial");
    // a signature: the middle half of its row; a number: all of it
    assert_eq!(masking::stamp_band(&sig, masking::StampRule::Bands), [[300.0, 480.0], [700.0, 480.0], [700.0, 520.0], [300.0, 520.0]]);
    assert_eq!(masking::stamp_band(&id, masking::StampRule::Bands), masking::band(&id));
    // the older rule, as its manifests recorded: the whole row
    assert_eq!(masking::stamp_band(&sig, masking::StampRule::Masked), masking::band(&sig));
    // a short row: at least SIG_BAND_MIN, never more than the row
    let thin = Zone::new("thin", [0.0, 100.0, 50.0, 12.0], "signature");
    let b = masking::stamp_band(&thin, masking::StampRule::Bands);
    assert!((b[2][1] - b[0][1] - masking::SIG_BAND_MIN).abs() < 1e-9);

    // on a page: across the stamp only the middle of the row is hidden;
    // above and below it the stamp is kept, and the page passes verify's check
    let stamp = Stamp::new(500.0, 500.0, 100.0, 1.0, "ring");
    let ident = masking::homography(util::IDENTITY);
    let mut bands = gray(1000, 1000, 255);
    masking::fill(&mut bands, &ident, &masking::stamp_band(&sig, masking::StampRule::Bands)).unwrap();
    let (img, mut canvas) = (bgr(1000, 1000, [255, 255, 255]), gray(1000, 1000, 255));
    let (_, _, _, records) = protect_stamps(&img, &mut canvas, std::slice::from_ref(&stamp), (40.0, 200.0), None, &MaskSettings::default(), Some(&bands)).unwrap();
    let c = util::bytes(&canvas).unwrap();
    let at = |x: usize, y: usize| c[y * 1000 + x];
    assert_eq!(at(500, 500), 0, "the middle of the row is hidden across the stamp");
    assert_eq!(at(500, 465), 255, "the top of the row is kept on the stamp");
    assert_eq!(at(500, 535), 255, "the bottom of the row is kept on the stamp");
    let hidden: Vec<u8> = c.iter().map(|&v| if v == 0 { 255 } else { 0 }).collect();
    let blocks = proof::to_blocks(&util::from_bytes(1000, 1000, &hidden).unwrap()).unwrap();
    assert_eq!(crate::verify::protected_hits(&records, &blocks, 1000, 1000).unwrap(), 0);
}

#[test]
fn a_signature_a_review_found_on_a_stamp_is_hidden_across_it_as_any_signature_is() {
    let found = vec![
        Found { label: "signature".into(), rect: [420.0, 470.0, 580.0, 530.0] },
        Found { label: "id_number".into(), rect: [100.0, 100.0, 300.0, 130.0] },
    ];
    let sigs = found_signatures(&found, &MaskSettings::default());
    assert_eq!(sigs.len(), 1);
    assert!(sigs[0].action == "signature" && sigs[0].rect == [420.0, 470.0, 160.0, 60.0]);
    // the stamp's band rule takes it; the older rules keep it as they did
    assert!(found_signatures(&found, &kept()).is_empty());
    assert!(found_signatures(&found, &MaskSettings { stamps: masking::StampRule::Masked, ..MaskSettings::default() }).is_empty());
}

#[test]
fn with_stamps_masked_over_as_before_the_officers_box_is_masked_stamp_and_all() {
    // the rule older manifests recorded as "mask_stamps": masks stay on the stamp
    let ts = templates();
    let form = to_gray(&stamped(&filled_form(3).unwrap()));
    let (scan, h) = simulate_scan(&form, 3, 2.0, 1.3).unwrap();
    let ms = MaskSettings { stamps: masking::StampRule::Masked, ..MaskSettings::default() };
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
    let ms = MaskSettings { stamps: masking::StampRule::Masked, ..MaskSettings::default() };
    let (found, keep, notes, _) = protect_stamps(&page, &mut canvas, std::slice::from_ref(&stamp), (60.0, 140.0), None, &ms, Some(&bands)).unwrap();
    assert!(found.is_empty() && opencv::core::count_non_zero(&keep).unwrap() == 0);
    assert_eq!(px(&canvas, 500, 500), 0);
    assert!(notes.iter().any(|n| n.contains("masked over")), "{:?}", notes);
    let mut kept_canvas = lifted.clone();
    let (found, _, _, _) = protect_stamps(&page, &mut lifted, std::slice::from_ref(&stamp), (60.0, 140.0), None, &kept(), None).unwrap();
    assert_eq!((found.len(), px(&lifted, 500, 500)), (1, 255));
    // by default the bands stay hidden and the rest of the stamp is kept
    let (found, keep, _, records) = protect_stamps(&page, &mut kept_canvas, std::slice::from_ref(&stamp), (60.0, 140.0), None, &MaskSettings::default(), Some(&bands)).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!((px(&kept_canvas, 500, 500), px(&kept_canvas, 500, 420)), (0, 255));
    assert_eq!((px(&keep, 500, 500), px(&keep, 500, 420)), (0, 255));
    assert!(records[0]["except_blocks"].as_array().is_some_and(|e| !e.is_empty()));
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
