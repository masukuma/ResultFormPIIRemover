//! Masking: hidden characters, signature bands, and the print, rules, strike
//! lines and backgrounds that must not be taken for writing.

use super::*;
use crate::testutil::*;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::Value;

const ROW_H: i32 = 36;
const ROWS: i32 = 6;
const X: i32 = 100;
const Y: i32 = 200;
const W: i32 = 220;

fn none() -> SigOptions<'static> {
    SigOptions { reach: Reach::default(), reference: None, printed: None, centered: false, strikes: None, exclude: None }
}

fn centered() -> SigOptions<'static> {
    SigOptions { centered: true, ..none() }
}

fn partial(canvas: &mut Mat, img: &Mat, cells: &[Zone], s: &MaskSettings, seed: u64) -> Vec<Value> {
    partial_column(canvas, img, &identity(), cells, s, &mut PyRandom::new(seed), Reach::default(), None, None, None).unwrap()
}

fn sigs(canvas: &mut Mat, img: &Mat, cells: &[Zone], o: &SigOptions) -> Vec<Value> {
    signature_column(canvas, img, &identity(), cells, &MaskSettings::default(), o).unwrap()
}

fn verify_sig(img: &Mat, out: &Mat, cells: &[Zone], o: &SigOptions) -> String {
    verify_signatures(img, out, &identity(), cells, &MaskSettings::default(), None, o).unwrap()
}

fn modes(rep: &[Value]) -> Vec<&str> {
    rep.iter().map(|r| r["mode"].as_str().unwrap()).collect()
}

/// A ruled column with handwriting-like digits in some rows.
fn table(entries: &[(i32, &str)], strike: bool) -> Mat {
    let mut img = gray(600, 500, 235);
    for r in 0..=ROWS {
        line(&mut img, (X - 60, Y + r * ROW_H), (X + W + 60, Y + r * ROW_H), colour(60.0), 2);
    }
    for x in [X, X + W] {
        line(&mut img, (x, Y - 40), (x, Y + ROWS * ROW_H), colour(60.0), 2);
    }
    for (r, t) in entries {
        text(&mut img, t, (X + 8, Y + r * ROW_H + 28), 0.85, colour(20.0), 2, true);
    }
    if strike {
        // the diagonal officials draw through unused rows
        line(&mut img, (X, Y + ROWS * ROW_H), (X + W, Y + 2 * ROW_H), colour(40.0), 2);
    }
    img
}

fn zone(name: &str, rect: [i32; 4], action: &str) -> Zone {
    Zone::new(name, rect.map(|v| v as f64), action)
}

fn cells_rule(group: &str, rule: &str, digits: usize) -> Vec<Zone> {
    (0..ROWS)
        .map(|r| Zone { digits, group: group.into(), row: r as usize, rule: rule.into(), ..zone(&format!("{}[{}]", group, r), [X, Y + r * ROW_H, W, ROW_H], "partial") })
        .collect()
}

fn cells(digits: usize) -> Vec<Zone> {
    cells_rule("id", if digits >= 10 { "phone" } else { "id" }, digits)
}

/// Which characters (by position) ended up fully under black.
fn hidden_chars(out: &Mat, g: &Mat, row: i32, text_: &str) -> Vec<usize> {
    let (o, gi) = (img(out), img(g));
    let (y0, y1) = ((Y + row * ROW_H + 4) as usize, (Y + (row + 1) * ROW_H - 4) as usize);
    let (x0, x1) = ((X + 4) as usize, (X + W - 4) as usize); // inside the column rules
    let cols: Vec<usize> = (x0..x1).filter(|&x| (y0..y1).any(|y| gi.at(x, y) < 120)).collect();
    let mut runs: Vec<Vec<usize>> = vec![];
    for x in cols {
        match runs.last_mut() {
            Some(r) if *r.last().unwrap() + 1 == x => r.push(x),
            _ => runs.push(vec![x]),
        }
    }
    assert_eq!(runs.len(), text_.len(), "{}", text_);
    runs.iter()
        .enumerate()
        .filter(|(_, r)| {
            let ink: Vec<(usize, usize)> = r.iter().flat_map(|&x| (y0..y1).map(move |y| (x, y))).filter(|&(x, y)| gi.at(x, y) < 120).collect();
            ink.iter().filter(|&&(x, y)| o.at(x, y) == 0).count() as f64 >= 0.9 * ink.len() as f64 // antialiased edges
        })
        .map(|(i, _)| i)
        .collect()
}

fn hidden_of(rep: &[Value], zone: &str) -> Vec<i64> {
    let r = rep.iter().find(|r| r["zone"] == zone).unwrap();
    r["lines"][0]["hidden"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect()
}

#[test]
fn id_rule_hides_exactly_three_characters() {
    let img = table(&[(0, "36676945"), (2, "24571109")], false);
    for seed in 0..20 {
        let mut canvas = gray(600, 500, 255);
        let rep = partial(&mut canvas, &img, &cells_rule("c", "id", 8), &MaskSettings::default(), seed);
        let by = |z: &str| rep.iter().find(|r| r["zone"] == z).unwrap()["mode"].clone();
        assert!(by("c[0]") == "partial" && by("c[1]") == "empty");
        assert_eq!(hidden_of(&rep, "c[0]").len(), 3);
        for (row, t) in [(0, "36676945"), (2, "24571109")] {
            assert_eq!(hidden_chars(&canvas, &img, row, t).len(), 3, "seed {}", seed);
        }
    }
}

#[test]
fn phone_rule_keeps_prefix_and_hides_three_of_last_six() {
    let img = table(&[(1, "0734471897")], false);
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..30 {
        let mut canvas = gray(600, 500, 255);
        let rep = partial(&mut canvas, &img, &cells_rule("c", "phone", 10), &MaskSettings::default(), seed);
        let l = &rep.iter().find(|r| r["zone"] == "c[1]").unwrap()["lines"][0];
        let hidden = hidden_of(&rep, "c[1]");
        assert!(l["slots"] == 10 && hidden.len() == 3 && *hidden.iter().min().unwrap() >= 4);
        let hid = hidden_chars(&canvas, &img, 1, "0734471897");
        assert!(!hid.iter().any(|i| *i < 3), "{:?}", hid); // 073 always readable
        seen.insert(hidden);
    }
    assert!(seen.len() > 3); // random, not a fixed pattern
}

#[test]
fn partial_masks_pass_the_leak_check() {
    let img = table(&[(0, "36676945"), (2, "0719108076")], false);
    let mut out = to_bgr(&img);
    partial(&mut out, &img, &cells(8), &MaskSettings::default(), 1);
    assert_eq!(verify_column(&out, &identity(), &cells(8), None, &MaskSettings::default(), Reach::default(), None, None).unwrap(), "");
}

#[test]
fn strike_line_through_unused_rows_is_not_masked() {
    let img = table(&[(0, "12345678")], true);
    let mut out = to_bgr(&img);
    let rep = partial(&mut out, &img, &cells(8), &MaskSettings::default(), 0);
    assert_eq!(modes(&rep)[2..], ["empty"; 4]);
    assert!(!has_zero(&out, X, Y + 3 * ROW_H, X + W, Y + 6 * ROW_H));
}

/// Any channel of any pixel in the rectangle is 0 (BGR image).
fn has_zero(out: &Mat, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
    let d = crate::util::bytes(out).unwrap();
    let w = out.cols() as usize;
    (y0 as usize..y1 as usize).any(|y| (x0 as usize..x1 as usize).any(|x| d[(y * w + x) * 3..(y * w + x) * 3 + 3].contains(&0)))
}

#[test]
fn writing_off_the_expected_rows_is_still_masked() {
    // table mapped a row low: row 0 writing sits above the first band
    let img = table(&[(0, "36676945")], false);
    let shifted: Vec<Zone> = cells(8).into_iter().map(|z| Zone { rect: [z.rect[0], z.rect[1] + ROW_H as f64, z.rect[2], z.rect[3]], ..z }).collect();
    let mut out = to_bgr(&img);
    partial(&mut out, &img, &shifted, &MaskSettings::default(), 2);
    let o = img_channel0(&out);
    let n = (X..X + W).filter(|&x| (Y + 4..Y + ROW_H - 4).any(|y| o.at(x as usize, y as usize) == 0)).count();
    assert!(n > 30, "{}", n); // three characters' worth
}

fn img_channel0(out: &Mat) -> Img {
    let mut ch = Mat::default();
    opencv::core::extract_channel(out, &mut ch, 0).unwrap();
    img(&ch)
}

#[test]
fn verify_flags_unmasked_number() {
    let img = table(&[(1, "0719108076")], false);
    let out = to_bgr(&img); // nothing masked
    assert!(verify_column(&out, &identity(), &cells(10), None, &MaskSettings::default(), Reach::default(), None, None).unwrap().contains("unmasked"));
}

#[test]
fn pick_hidden_never_leaves_long_visible_runs() {
    for (n, k) in [(8, 5), (10, 6), (6, 4), (12, 8)] {
        let mut rng = PyRandom::new(0);
        for _ in 0..50 {
            let hidden: std::collections::BTreeSet<usize> = pick_hidden(n, k, &mut rng, 2).into_iter().collect();
            let (mut run, mut best) = (0, 0);
            for i in 0..n {
                run = if hidden.contains(&i) { 0 } else { run + 1 };
                best = best.max(run);
            }
            assert!(best <= 2 && hidden.len() >= k);
        }
    }
}

#[test]
fn column_edge_check_catches_shifted_column() {
    let img = table(&[(0, "36676945")], false);
    assert_eq!(check_column_edges(&img, &identity(), &cells(8), false).unwrap(), "");
    let shift = homography([[1.0, 0.0, 60.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(check_column_edges(&img, &shift, &cells(8), false).unwrap().contains("placed wrong"));
}

/// Ruled column; rows -> x offset of a scribbled signature (cursive: up and
/// down strokes, 20 px tall).
fn signature_table(rows: &[(i32, i32)]) -> Mat {
    let mut img = gray(600, 500, 235);
    for r in 0..=ROWS {
        line(&mut img, (X - 60, Y + r * ROW_H), (X + W + 60, Y + r * ROW_H), colour(60.0), 2);
    }
    for x in [X, X + W] {
        line(&mut img, (x, Y - 40), (x, Y + ROWS * ROW_H), colour(60.0), 2);
    }
    for &(r, dx) in rows {
        let xs = linspace((X + dx) as f64, (X + dx + 70) as f64, 40);
        let pts: Vec<(i32, i32)> = xs
            .iter()
            .zip(JITTER_1)
            .map(|(&x, j)| (x as i32, ((Y + r * ROW_H) as f64 + 18.0 + 9.0 * ((x - X as f64 - dx as f64) / 4.0).sin() + j as f64) as i32))
            .collect();
        polyline(&mut img, &pts, colour(20.0), 2);
    }
    img
}

/// `np.random.default_rng(0).integers(-1, 2, 40)`, as the Python tests drew.
const JITTER_1: [i32; 40] = [1, 0, 0, -1, -1, -1, -1, -1, -1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, -1, 1, 1, -1, 0, 1, 0, -1, 1, 1, 1, -1, -1, 1, -1, 0, -1, -1, 0, 0];

/// `np.random.default_rng(1).integers(-2, 3, 120)`.
const JITTER_2: [i32; 120] = [
    0, 0, 1, 2, -2, -2, 2, 2, -1, -1, 2, 0, -1, 2, -1, 0, 1, 0, -2, -2, 2, 1, 2, 0, 2, -1, 0, 1, -2, -1, -2, 0, 2, -2, -1, 0, 2, -1, 0, -1, -2, 1, -2, -1, 0, 0, -2, 2, 1, 2, -2, 1, -1, 0, 2, -1, 1, -2, -1, 2, 0, 0, -1, -2, 0, 1, 0, 1, -1, 1, 1, 2, 0, -2, 1, 0, 2, 0, -1, -2, 0, 1, 1, 2, -1, 0, 2, -1, -1, 2,
    0, 0, 1, 0, 2, 1, -2, -2, 0, 2, -2, 1, 1, 1, 2, -2, 0, 2, -1, -2, 0, -2, -1, 2, 1, 2, 2, 2, -1, 0,
];

fn sig_cells() -> Vec<Zone> {
    (0..ROWS).map(|r| Zone { group: "s".into(), row: r as usize, ..zone(&format!("s[{}]", r), [X, Y + r * ROW_H, W, ROW_H], "signature") }).collect()
}

#[test]
fn each_signature_is_masked_through_its_middle() {
    // One band per signature, through its middle: that it is signed stays
    // visible, the signature can't be made out. Empty rows are untouched.
    let img = signature_table(&[(0, 20), (3, 110)]);
    let cs = sig_cells();
    let mut canvas = gray(600, 500, 255);
    let rep = sigs(&mut canvas, &img, &cs, &none());
    assert_eq!(modes(&rep), ["signature", "empty", "empty", "signature", "empty", "empty"]);
    let gi = img_of(&img);
    let (w, h) = (500usize, 600usize);
    let mut sig_ink = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let in_col = ((x as f64) - (X as f64 + W as f64 / 2.0)).abs() < W as f64 / 2.0 - 4.0;
            let on_rule = (0..=ROWS).any(|r| (y as i32) >= Y + r * ROW_H - 3 && (y as i32) < Y + r * ROW_H + 3);
            sig_ink[y * w + x] = gi.at(x, y) < 120 && in_col && (y as i32) >= Y + 2 && !on_rule;
        }
    }
    for r in [0, 3] {
        let one: Vec<bool> = sig_ink.iter().enumerate().map(|(i, &s)| s && ((i / w) as i32 - (Y + r * ROW_H + 18)).abs() < 18).collect();
        assert!(middle_hidden(&canvas, &one, 24, 0.85), "row {}", r);
    }
    let c = img_of(&canvas);
    let hidden = |y0: i32, y1: i32| (y0..y1).flat_map(|y| (X..X + W).map(move |x| (x, y))).filter(|&(x, y)| c.at(x as usize, y as usize) == 0).count();
    assert!((hidden(Y, Y + ROWS * ROW_H) as f64) < 0.35 * (W * ROWS * ROW_H) as f64); // far from the whole column
    assert_eq!(hidden(Y + ROW_H + 8, Y + 3 * ROW_H - 8), 0); // empty rows untouched
    assert_eq!(verify_sig(&img, &apply(&canvas, &img), &cs, &none()), "");
    assert!(verify_sig(&img, &to_bgr(&img), &cs, &none()).contains("not masked through the middle"));
    let writing = crate::util::dilate(&from_mask(&sig_ink, 600, 500), &crate::util::ones_kernel(7, 7).unwrap()).unwrap();
    let full: Vec<u8> = crate::util::bytes(&writing).unwrap().iter().map(|&v| if v > 0 { 0 } else { 255 }).collect();
    let full = apply(&crate::util::from_bytes(600, 500, &full).unwrap(), &img);
    assert_eq!(verify_sig(&img, &full, &cs, &none()), ""); // more is fine
}

fn img_of(m: &Mat) -> Img {
    img(m)
}

#[test]
fn dotted_leaders_are_print_not_writing() {
    let mut strip = gray(60, 700, 235);
    for x in (20..680).step_by(14) {
        // "ID Number: ..........."
        fill_rect(&mut strip, x, 44, x + 6, 48, 30);
    }
    let count = |m: &Mat, x0: i32, y0: i32, x1: i32, y1: i32| {
        let i = img(m);
        (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).filter(|&(x, y)| i.at(x as usize, y as usize) != 0).count()
    };
    assert_eq!(count(&ink_mask(&strip, 56.0, false).unwrap(), 0, 0, 700, 60), 0); // an empty field
    text(&mut strip, "2345678", (120, 40), 1.2, colour(20.0), 3, false);
    let ink = ink_mask(&strip, 56.0, false).unwrap();
    assert!(count(&ink, 120, 0, 330, 60) > 0); // the digits stay
    assert_eq!(count(&ink, 400, 44, 680, 48), 0); // the leader beyond them goes
}

#[test]
fn a_line_drawn_through_empty_rows_is_not_writing() {
    // Officials strike out unused rows with a long diagonal line. Found over
    // the whole table, it is left alone; digits it passes near are still masked.
    let (row_h, rows) = (60, 4);
    let mut img = gray(60 + rows * row_h + 40, 1400, 235);
    for i in 0..=rows {
        line(&mut img, (20, 60 + i * row_h), (1380, 60 + i * row_h), colour(60.0), 2);
    }
    for x in [20, 400, 700, 1000, 1380] {
        line(&mut img, (x, 60), (x, 60 + rows * row_h), colour(60.0), 2);
    }
    text(&mut img, "23456781", (420, 60 + row_h - 14), 1.2, colour(20.0), 3, false); // row 1 filled
    line(&mut img, (30, 60 + rows * row_h - 4), (1370, 60 + row_h + 4), colour(25.0), 9); // struck through rows 2-4
    let strikes = find_strikes(&img, &identity(), [10.0, 50.0, 1380.0, (rows * row_h + 20) as f64], row_h as f64).unwrap();
    assert_eq!(strikes.len(), 1);
    let cs: Vec<Zone> = (0..rows).map(|i| Zone { digits: 8, rule: "id".into(), ..zone(&format!("id[{}]", i), [403, 60 + i * row_h + 3, 294, row_h - 6], "partial") }).collect();
    let (h, w) = (img.rows(), img.cols());
    let mut canvas = gray(h, w, 255);
    let rep = partial_column(&mut canvas, &img, &identity(), &cs, &MaskSettings::default(), &mut PyRandom::new(1), Reach::default(), None, Some(&strikes), None).unwrap();
    assert!(rep[0]["mode"] == "partial" && rep[0]["lines"].as_array().unwrap().len() == 1); // the written ID
    assert!(rep[1..].iter().all(|r| r["mode"] == "empty"), "{:?}", rep); // struck rows: nothing masked
    let mut canvas2 = gray(h, w, 255);
    let rep2 = partial(&mut canvas2, &img, &cs, &MaskSettings::default(), 1);
    assert!(rep2[1..].iter().any(|r| r["mode"] != "empty")); // without it, the line was masked
}

#[test]
fn the_date_below_an_underlined_signature_stays_visible() {
    // A loop of the signature running down beside the date on the line
    // below: strokes mostly on the next line are not the signature's, and
    // neither the underline nor the loop stretches the mask over the date.
    for underline in [720, 420] {
        let mut img = gray(320, 900, 235);
        line(&mut img, (60, 200), (60 + underline, 200), colour(60.0), 2); // "Signature: ____"
        line(&mut img, (60, 250), (780, 250), colour(60.0), 2); // "Date: ____"
        let pts: Vec<(i32, i32)> = linspace(150.0, 400.0, 120) // cursive, 30 px tall
            .iter()
            .map(|&x| (x as i32, (177.0 + 14.0 * ((x - 150.0) / 5.0).sin()) as i32))
            .collect();
        polyline(&mut img, &pts, colour(20.0), 2);
        polyline(&mut img, &[(300, 185), (310, 228), (330, 230), (338, 186)], colour(20.0), 2); // the loop
        text(&mut img, "11-08-2022", (420, 242), 1.0, colour(20.0), 2, false);
        let field = [Zone { search: Some([0.0, 10.0]), ..zone("ro_signature", [100, 150, 600, 55], "signature") }];
        let gi = crate::testutil::img(&img);
        let at = |x: usize, y: usize| gi.at(x, y) < 120;
        let date_ink: Vec<bool> = (0..320 * 900).map(|i| { let (x, y) = (i % 900, i / 900); (210..244).contains(&y) && (415..700).contains(&x) && at(x, y) }).collect();
        let main: Vec<bool> = (0..320 * 900).map(|i| { let (x, y) = (i % 900, i / 900); at(x, y) && (140..410).contains(&x) && y < 196 }).collect();
        let mut canvas = gray(320, 900, 255);
        sigs(&mut canvas, &img, &field, &centered());
        assert!(middle_hidden(&canvas, &main, 24, 0.85), "underline {}", underline);
        assert_eq!(covered(&canvas, &date_ink), 0.0);
        assert!(all_eq(&canvas, 415, 150, 900, 205, 255)); // the underline beyond the signature
        assert!(all_eq(&canvas, 0, 205, 290, 320, 255) && all_eq(&canvas, 350, 205, 900, 320, 255)); // below the line: the loop only
        assert_eq!(verify_sig(&img, &apply(&canvas, &img), &field, &centered()), "");
        assert!(verify_sig(&img, &to_bgr(&img), &field, &centered()).contains("not masked through the middle"));
    }
}

/// A signature on its line, 150-400 px; `draw` draws the line. Returns the
/// image, the signature's own ink and the field.
fn underlined_signature(draw: impl Fn(&mut Mat)) -> (Mat, Vec<bool>, Vec<Zone>) {
    let mut img = gray(320, 900, 235);
    draw(&mut img);
    let pts: Vec<(i32, i32)> = linspace(150.0, 400.0, 120) // cursive: up and down strokes, 30 px tall
        .iter()
        .zip(JITTER_2)
        .map(|(&x, j)| (x as i32, (178.0 + 14.0 * ((x - 150.0) / 5.0).sin() + j as f64).clamp(160.0, 196.0) as i32))
        .collect();
    polyline(&mut img, &pts, colour(20.0), 2);
    let mut sig = gray(320, 900, 0);
    polyline(&mut sig, &pts, colour(1.0), 2);
    let field = vec![Zone { search: Some([0.0, 10.0]), ..zone("ro_signature", [100, 150, 600, 55], "signature") }];
    (img, mask(&sig, |v| v > 0), field)
}

fn plain_line(img: &mut Mat) {
    line(img, (60, 200), (780, 200), colour(60.0), 2);
}

#[test]
fn a_broken_underline_does_not_stretch_the_signature_mask() {
    // A scan breaks the line into dashes; joined through the line, they
    // stretched the mask along all of it, up to a stamp beyond.
    let (img, sig, field) = underlined_signature(|img| {
        plain_line(img);
        for x in (450..780).step_by(40) {
            // gaps the line runs through less cleanly
            fill_rect(img, x, 197, x + 20, 204, 235);
            fill_rect(img, x + 4, 201, x + 12, 203, 60);
        }
    });
    let mut canvas = gray(320, 900, 255);
    sigs(&mut canvas, &img, &field, &centered());
    assert!(middle_hidden(&canvas, &sig, 24, 0.85));
    assert!(all_eq(&canvas, 420, 0, 900, 320, 255));
}

#[test]
fn a_striped_scan_is_no_underline() {
    // Some scans are striped with long horizontal lines all over. They are
    // not the field's line: taking them out would take the signature with them.
    let (img, sig, field) = underlined_signature(|img| {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for y in (140..260).step_by(5) {
            // broken and uneven, as scanned
            let mut x = rng.random_range(0..40);
            while x < 900 {
                let n = rng.random_range(60..250);
                fill_rect(img, x, y, x + n, y + 2, 110);
                x += n + rng.random_range(8..30);
            }
        }
    });
    let mut canvas = gray(320, 900, 255);
    sigs(&mut canvas, &img, &field, &centered());
    assert!(middle_hidden(&canvas, &sig, 24, 0.95)); // stripes add to its height: a little more is hidden
    assert!(verify_sig(&img, &to_bgr(&img), &field, &centered()).contains("not masked through the middle"));
}

#[test]
fn a_flat_flourish_is_not_the_printed_line() {
    // A signature's long straight tail above its line is a long flat line
    // too; taken for the printed line, it was left visible.
    for with_ref in [false, true] {
        let (mut img, sig, field) = underlined_signature(plain_line);
        // the signature's rightmost point (first in row order)
        let (mut ex, mut ey) = (0usize, 0usize);
        for (i, &s) in sig.iter().enumerate() {
            if s && i % 900 > ex {
                (ex, ey) = (i % 900, i / 900);
            }
        }
        let end = (ex as i32, ey as i32);
        line(&mut img, end, (end.0 + 10, 186), colour(20.0), 2); // the tail, drawn on from the signature
        line(&mut img, (end.0 + 10, 186), (720, 186), colour(20.0), 2);
        let mut tail = gray(320, 900, 0);
        line(&mut tail, (end.0 + 10, 186), (720, 186), colour(1.0), 2);
        let reference = with_ref.then(|| {
            let mut r = gray(320, 900, 235);
            line(&mut r, (60, 208), (780, 208), colour(60.0), 2); // a layout variant: 8 px off
            r
        });
        let mut canvas = gray(320, 900, 255);
        sigs(&mut canvas, &img, &field, &SigOptions { reference: reference.as_ref(), ..centered() });
        assert!(middle_hidden(&canvas, &sig, 24, 0.85), "with_ref {}", with_ref);
        assert!(covered(&canvas, &mask(&tail, |v| v > 0)) > 0.97); // here the tail is the middle
        assert!(all_eq(&canvas, 740, 150, 900, 205, 255)); // the line beyond the tail
    }
}

/// An officer's ID field: from just below the name's line (y 115) to just
/// below its own (y 200), on a line running 60-780.
fn id_field(img: &mut Mat) -> Vec<Zone> {
    line(img, (60, 115), (780, 115), colour(60.0), 2);
    line(img, (60, 197), (780, 197), colour(60.0), 2);
    vec![Zone { digits: 8, rule: "id".into(), search: Some([0.0, 6.0]), ..zone("ro_id", [100, 118, 600, 85], "partial") }]
}

#[test]
fn writing_too_small_to_read_is_hidden_at_its_own_width() {
    // An ID written too small to read as characters is hidden whole, but only
    // as wide as the writing.
    let mut img = gray(320, 900, 235);
    let field = id_field(&mut img);
    text(&mut img, "2442587", (130, 190), 1.2, colour(20.0), 4, false); // ~28 px: under 40% of 85
    let mut canvas = gray(320, 900, 255);
    let rep = partial(&mut canvas, &img, &field, &MaskSettings::default(), 1);
    assert_eq!(rep[0]["mode"], "full");
    let digits: Vec<bool> = mask(&img, |v| v < 120).iter().enumerate().map(|(i, &d)| d && (150..196).contains(&(i / 900))).collect();
    assert_eq!(covered(&canvas, &digits), 1.0);
    assert!(all_eq(&canvas, 400, 0, 900, 320, 255));
}

#[test]
fn the_name_above_reaching_into_an_id_field_is_not_part_of_the_id() {
    // A name written large and low reaches into the ID field below it: its
    // foot must not be taken for an eighth character.
    let mut img = gray(320, 900, 235);
    let field = id_field(&mut img);
    text(&mut img, "KIP", (420, 145), 2.0, colour(20.0), 4, false); // the name, low
    text(&mut img, "6665112", (120, 190), 1.3, colour(20.0), 3, false);
    let s = MaskSettings { min_glyph_h: 0.25, ..MaskSettings::default() };
    let mut canvas = gray(320, 900, 255);
    let rep = partial(&mut canvas, &img, &field, &s, 3);
    assert!(rep[0]["mode"] == "partial" && rep[0]["lines"][0]["slots"] == 7, "{:?}", rep);
    assert!(all_eq(&canvas, 410, 100, 560, 150, 255)); // the name stays visible
    assert_eq!(verify_column(&apply(&canvas, &img), &identity(), &field, None, &s, Reach::default(), None, None).unwrap(), "");
}

#[test]
fn print_sitting_on_the_line_is_not_joined_to_the_signature() {
    let (img, sig, field) = underlined_signature(|img| {
        plain_line(img);
        text(img, "ELECTORAL", (560, 198), 0.9, colour(90.0), 2, false);
    });
    let mut canvas = gray(320, 900, 255);
    sigs(&mut canvas, &img, &field, &centered());
    assert!(middle_hidden(&canvas, &sig, 24, 0.85));
    assert!(all_eq(&canvas, 420, 150, 550, 205, 255));
}

#[test]
fn the_next_line_cut_off_below_a_signature_column_is_not_a_signature() {
    let mut img = signature_table(&[(0, 20)]);
    let bottom = Y + ROWS * ROW_H;
    text(&mut img, "NA", (X + 40, bottom + 120), 2.0, colour(20.0), 4, false); // top ~ bottom + 75
    let mut canvas = gray(600, 500, 255);
    let rep = sigs(&mut canvas, &img, &sig_cells(), &none());
    assert_eq!(modes(&rep), ["signature", "empty", "empty", "empty", "empty", "empty"]);
    assert!(all_eq(&canvas, 0, bottom + 10, 500, 600, 255));
}

#[test]
fn crumbs_of_print_in_an_empty_field_are_not_a_signature() {
    let (mut img, _, field) = underlined_signature(plain_line);
    fill_rect(&mut img, 0, 150, 900, 197, 235); // no signature: an empty field...
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    for x in (420..600).step_by(12) {
        // ...with crumbs of print in it, close together
        for y in (160..192).step_by(7) {
            let (x0, y0) = (x + rng.random_range(0..3), y + rng.random_range(0..3));
            let n = rng.random_range(4..9);
            fill_rect(&mut img, x0, y0, x0 + n, y0 + 2, 40);
        }
    }
    let mut canvas = gray(320, 900, 255);
    let rep = sigs(&mut canvas, &img, &field, &centered());
    assert!(rep[0]["mode"] == "empty" && all_eq(&canvas, 0, 0, 900, 320, 255));
}

/// Embossed forms scan with a background of thin dark horizontal lines every
/// few pixels, its lettering kinks in them.
fn embossed(img: &mut Mat) {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let (h, w) = (img.rows(), img.cols());
    for y in (0..h).step_by(5) {
        let mut x = 0;
        while x < w {
            // runs of line, each kinked a pixel or two up or down
            let n = rng.random_range(20..120);
            let dy = rng.random_range(-1..2);
            if y + dy >= 0 {
                fill_rect(img, x, y + dy, x + n, y + dy + 2, 25);
            }
            x += n;
        }
    }
    for _ in 0..(h * w) / 900 {
        // the lettering: kinks bridging neighbouring lines
        let (x, y) = (rng.random_range(0..w - 4), rng.random_range(0..h - 6));
        fill_rect(img, x, y, x + 2, y + 6, 25);
    }
}

#[test]
fn an_embossed_background_in_empty_fields_is_not_writing() {
    let (img, sig, field) = underlined_signature(|im| {
        embossed(im);
        line(im, (60, 200), (780, 200), colour(30.0), 2);
    });
    let mut canvas = gray(320, 900, 255);
    let rep = sigs(&mut canvas, &img, &field, &centered());
    assert!(rep[0]["mode"] == "signature" && middle_hidden(&canvas, &sig, 24, 0.85));
    assert!(all_eq(&canvas, 430, 0, 900, 320, 255)); // the empty rest of the field
    let mut empty = gray(320, 900, 235);
    embossed(&mut empty);
    let ids = id_field(&mut empty);
    let mut canvas = gray(320, 900, 255);
    assert_eq!(partial(&mut canvas, &empty, &ids, &MaskSettings::default(), 1)[0]["mode"], "empty");
    let s = [Zone { search: Some([0.0, 10.0]), ..zone("s", [100, 210, 600, 55], "signature") }];
    assert_eq!(sigs(&mut canvas, &empty, &s, &centered())[0]["mode"], "empty");
    assert!(all_eq(&canvas, 0, 0, 900, 320, 255));
}

#[test]
fn strokes_reaching_down_cover_only_their_own_width_of_the_date() {
    // Long flat strokes along the field and two downstrokes into the date
    // line: the printed line is still found where the reference has it, and
    // only the downstrokes' own width of the date line is masked.
    let (mut img, sig, field) = underlined_signature(plain_line);
    for y in [160, 188] {
        line(&mut img, (150, y), (620, y), colour(20.0), 2);
    }
    for x in [152, 398] {
        line(&mut img, (x, 188), (x - 6, 240), colour(20.0), 2);
    }
    text(&mut img, "11-08-22", (200, 242), 1.0, colour(20.0), 2, false);
    let gi = img_of(&img);
    let date_ink: Vec<bool> = (0..320 * 900).map(|i| { let (x, y) = (i % 900, i / 900); (214..244).contains(&y) && (200..360).contains(&x) && gi.at(x, y) < 120 }).collect();
    let mut reference = gray(320, 900, 235);
    plain_line(&mut reference);
    let mut canvas = gray(320, 900, 255);
    sigs(&mut canvas, &img, &field, &SigOptions { reference: Some(&reference), ..centered() });
    assert!(covered(&canvas, &sig) > 0.6); // between the long strokes, most of it is the middle
    assert!(covered(&canvas, &date_ink) < 0.05);
}

fn label_page(dy: i32) -> Mat {
    let mut img = gray(400, 900, 240);
    text(&mut img, "ID Number:", (40, 150 + dy), 1.0, colour(30.0), 2, true);
    line(&mut img, (230, 158 + dy), (800, 158 + dy), colour(60.0), 2);
    img
}

#[test]
fn anchor_follows_a_shifted_label() {
    let reference = label_page(0);
    let anchor = [36.0, 120.0, 190.0, 42.0];
    let (d, ncc) = find_anchor(&label_page(70), &identity(), &reference, anchor).unwrap();
    let d = d.unwrap();
    assert!((d[1] - 70.0).abs() <= 3.0 && d[0].abs() <= 3.0 && ncc > 0.8);
    let (d, _) = find_anchor(&gray(400, 900, 240), &identity(), &reference, anchor).unwrap();
    assert!(d.is_none()); // label absent: the page must fail, not guess
}

#[test]
fn digits_on_an_underline_are_not_mistaken_for_print() {
    // A digit written on a printed underline must count as writing.
    let mut reference = gray(90, 600, 240);
    line(&mut reference, (0, 70), (600, 70), colour(60.0), 2);
    let mut im = reference.clone();
    text(&mut im, "23029912", (20, 66), 1.6, colour(20.0), 3, true);
    let cell = [Zone { digits: 8, rule: "id".into(), search: Some([0.0, 6.0]), ..zone("id", [0, 10, 600, 70], "partial") }];
    let mut canvas = gray(90, 600, 255);
    let rep = partial_column(&mut canvas, &im, &identity(), &cell, &MaskSettings::default(), &mut PyRandom::new(0), Reach::default(), Some(&reference), None, None).unwrap();
    assert!(rep[0]["mode"] == "partial" && rep[0]["lines"][0]["hidden"].as_array().unwrap().len() == 3);
}

#[test]
fn reach_clips_at_fences() {
    let r = Reach { margin: (50.0, 200.0), top: Some(90.0), bottom: Some(330.0) };
    assert_eq!(r.clip(100.0, 300.0, 60.0, 60.0), (10.0, 30.0)); // stopped at the fences
    assert_eq!(Reach::margin(50.0, 200.0).clip(100.0, 300.0, 60.0, 60.0), (110.0, 260.0)); // no fences: wide
    assert_eq!(Reach { top: Some(110.0), ..r }.clip(100.0, 300.0, 60.0, 60.0).0, -10.0); // a label inside the cells as placed
    assert_eq!(r.fences().clip(100.0, 300.0, 60.0, 110.0), (10.0, 30.0));
}

#[test]
fn a_widened_search_stops_at_the_label_below_the_table() {
    // A table that couldn't be snapped is searched with a wide margin below
    // it; there, the comments written under the table are not the column's.
    let bottom = Y + ROWS * ROW_H;
    let mut img = table(&[(0, "36676945"), (5, "24571109")], false);
    text(&mut img, "WRONG", (X + 8, bottom + 80), 0.85, colour(20.0), 2, true); // a comment under the table
    let comment = |canvas: &Mat| (bottom + 50..bottom + 90).any(|y| (X..X + W).any(|x| img_of(canvas).at(x as usize, y as usize) == 0));
    let cs = cells(8);
    let wide = Reach::margin(50.0, 200.0);
    let mut canvas = gray(600, 500, 255);
    partial_column(&mut canvas, &img, &identity(), &cs, &MaskSettings::default(), &mut PyRandom::new(1), wide, None, None, None).unwrap();
    assert!(comment(&canvas)); // taken for digits of the last row
    let fenced = Reach { bottom: Some((bottom + 20) as f64), ..wide };
    let mut canvas = gray(600, 500, 255);
    let rep = partial_column(&mut canvas, &img, &identity(), &cs, &MaskSettings::default(), &mut PyRandom::new(1), fenced, None, None, None).unwrap();
    assert!(!comment(&canvas));
    assert!(rep[0]["lines"][0]["hidden"].as_array().unwrap().len() == 3 && rep[5]["lines"][0]["hidden"].as_array().unwrap().len() == 3);
    assert_eq!(hidden_chars(&canvas, &img, 5, "24571109").len(), 3);
    assert_eq!(verify_column(&apply(&canvas, &img), &identity(), &cs, None, &MaskSettings::default(), fenced, None, None).unwrap(), "");
    // the signatures' search stops there too
    let mut sig = signature_table(&[(5, 20)]);
    text(&mut sig, "WRONG", (X + 8, bottom + 80), 0.85, colour(20.0), 2, true);
    let mut canvas = gray(600, 500, 255);
    sigs(&mut canvas, &sig, &sig_cells(), &SigOptions { reach: wide, ..none() });
    assert!(comment(&canvas));
    let mut canvas = gray(600, 500, 255);
    let rep = sigs(&mut canvas, &sig, &sig_cells(), &SigOptions { reach: fenced, ..none() });
    assert!(!comment(&canvas) && rep[5]["mode"] == "signature");
}

#[test]
fn down_a_table_whose_rules_were_not_matched_signatures_are_hidden_whole() {
    // Its rows are then the template's, and down the column the writing has
    // drifted most of a row below them: cut at those rows, the bands would
    // miss. Each signature is hidden whole, the last one too, below the last
    // cell; two signature lines a line apart keep their bands.
    let img = signature_table(&[(1, 20), (2, 110), (5, 40)]);
    let cs: Vec<Zone> = sig_cells().into_iter().map(|z| Zone { rect: [z.rect[0], z.rect[1] - 22.0, z.rect[2], z.rect[3]], ..z }).collect();
    let share = |canvas: &Mat, rows: std::ops::Range<i32>| {
        let (gi, c) = (img_of(&img), img_of(canvas));
        let (mut ink, mut hidden) = (0, 0);
        for y in Y + rows.start * ROW_H..Y + rows.end * ROW_H {
            for x in X + 4..X + W - 4 {
                let on_rule = (0..=ROWS).any(|r| y >= Y + r * ROW_H - 3 && y < Y + r * ROW_H + 3);
                if gi.at(x as usize, y as usize) < 120 && !on_rule {
                    ink += 1;
                    hidden += (c.at(x as usize, y as usize) == 0) as usize;
                }
            }
        }
        hidden as f64 / ink as f64
    };
    let wide = SigOptions { reach: Reach::margin(50.0, 200.0), ..none() };
    let mut canvas = gray(600, 500, 255);
    sigs(&mut canvas, &img, &cs, &wide);
    assert!(share(&canvas, 0..ROWS) >= 0.98, "{}", share(&canvas, 0..ROWS));
    assert_eq!(verify_sig(&img, &apply(&canvas, &img), &cs, &wide), "");
    let mut banded = gray(600, 500, 255);
    sigs(&mut banded, &img, &cs[..2], &wide);
    assert!(share(&banded, 1..2) < 0.9);
}

#[test]
fn a_signature_hidden_whole_is_covered_along_its_ink() {
    // A long slanting signature across two rows: a box round it would black
    // out the paper above and below the stroke, up to the number written
    // before it. Covered along its ink instead, the paper more than a few px
    // from any stroke stays visible.
    let mut img = signature_table(&[]);
    let pts: Vec<(i32, i32)> = (0..90).map(|k| (X + 20 + 2 * k, Y + 2 * ROW_H + 24 - 2 * k / 3 + (9.0 * (k as f64 / 2.0).sin()) as i32 + JITTER_2[k as usize])).collect();
    polyline(&mut img, &pts, colour(20.0), 2);
    let mut canvas = gray(600, 500, 255);
    sigs(&mut canvas, &img, &sig_cells(), &SigOptions { reach: Reach::margin(50.0, 200.0), ..none() });
    let (gi, c) = (img_of(&img), img_of(&canvas));
    let ink: Vec<bool> = (0..600 * 500).map(|i| gi.d[i] < 120 && (X + 4..X + W - 4).contains(&((i % 500) as i32)) && !(0..=ROWS).any(|r| ((i / 500) as i32 - (Y + r * ROW_H)).abs() < 3)).collect();
    let near = crate::util::dilate(&from_mask(&ink, 600, 500), &crate::util::ones_kernel(33, 33).unwrap()).unwrap();
    let near = img_of(&near);
    let (mut black, mut far) = (0, 0);
    for i in 0..600 * 500 {
        if c.d[i] == 0 {
            black += 1;
            far += (near.d[i] == 0) as usize;
        }
    }
    assert!(black > 0 && far == 0, "{far} of {black} black px more than 16px from the ink");
    let hidden = (0..600 * 500).filter(|&i| ink[i] && c.d[i] == 0).count();
    assert!(hidden as f64 >= 0.98 * ink.iter().filter(|&&v| v).count() as f64, "{hidden} hidden");
}

/// Two signature lines (officers'), 56 px apart, with a printed "Date" at the
/// end of each; above the first, the line before with a constituency written
/// just over it. The page and its blank reference.
fn signature_lines() -> (Mat, Mat, Vec<Zone>) {
    let mut reference = gray(400, 700, 235);
    for k in 0..2 {
        let base = 150 + 56 * k + 45;
        line(&mut reference, (100, base), (420, base), colour(60.0), 2);
        text(&mut reference, "Date", (440, base), 0.8, colour(30.0), 2, true);
    }
    line(&mut reference, (100, 146), (420, 146), colour(60.0), 2); // the line before
    let mut page = reference.clone();
    text(&mut page, "LAMU WEST", (200, 146), 0.9, colour(20.0), 2, true); // its foot in the search
    for k in 0..2 {
        let base = 150 + 56 * k + 45;
        let pts: Vec<(i32, i32)> = (0..40).map(|i| (150 + 4 * i, base - 14 + (9.0 * (i as f64 / 1.5).sin()) as i32 + JITTER_1[i as usize])).collect();
        polyline(&mut page, &pts, colour(20.0), 2);
    }
    let cells = (0..2).map(|k| Zone { group: "o".into(), row: k, search: Some([12.0, 24.0]), region: "t".into(), ..zone(&format!("o[{}]", k), [100, 150 + 56 * k as i32, 400, 56], "signature") }).collect();
    (page, reference, cells)
}

#[test]
fn a_signature_line_keeps_its_printed_label_and_the_line_above() {
    // "Date" at the end of the line lies in the search; so does the bottom of
    // a constituency written just over the line before. Neither is the
    // signature: the label is found on the page and its print set aside, and
    // writing reaching in from the top and stopping above the first line's
    // middle is the line above's. The signatures still get their bands.
    let (page, reference, cells) = signature_lines();
    let printed = printed_labels(&page, &identity(), &reference, &[[436.0, 175.0, 64.0, 26.0], [436.0, 231.0, 64.0, 26.0]]).unwrap();
    let o = SigOptions { printed: Some(&printed), ..none() };
    let mut canvas = gray(400, 700, 255);
    let rep = sigs(&mut canvas, &page, &cells, &o);
    assert_eq!(modes(&rep), ["signature", "signature"]);
    let c = img_of(&canvas);
    let black = |x0: i32, y0: i32, x1: i32, y1: i32| (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).filter(|&(x, y)| c.at(x as usize, y as usize) == 0).count();
    assert_eq!(black(436, 175, 500, 257), 0); // both "Date"s
    assert_eq!(black(190, 110, 420, 147), 0); // the constituency
    assert!(black(150, 170, 310, 200) > 0 && black(150, 226, 310, 256) > 0); // the signatures
    assert_eq!(verify_sig(&page, &apply(&canvas, &page), &cells, &o), "");
    // without the label set aside, the end of the line was taken for writing
    let mut canvas = gray(400, 700, 255);
    sigs(&mut canvas, &page, &cells, &none());
    assert!((175..201).any(|y| (436..500).any(|x| img_of(&canvas).at(x, y) == 0)));
}

#[test]
fn faint_broken_rules_of_a_column_are_not_signatures() {
    // A photo where the table's printed rules are faint breaks them into
    // dashes the rule removal leaves; at the column's edges they cross in
    // small "+"s, tall enough to pass for writing. Their upright strokes
    // along the column's edges are set aside, the flat rest is taken for
    // rule: only the signatures are masked.
    let mut img = gray(600, 500, 235);
    let dashed = |img: &mut Mat, a: (i32, i32), b: (i32, i32)| {
        let n = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) / 12).max(1);
        for k in 0..n {
            let (t0, t1) = (k as f64 / n as f64, (k as f64 + 0.6) / n as f64);
            let p = |t: f64| (a.0 + ((b.0 - a.0) as f64 * t) as i32, a.1 + ((b.1 - a.1) as f64 * t) as i32);
            line(img, p(t0), p(t1), colour(70.0), 2);
        }
    };
    for r in 0..=ROWS {
        dashed(&mut img, (X - 60, Y + r * ROW_H), (X + W + 60, Y + r * ROW_H));
    }
    for x in [X, X + W] {
        dashed(&mut img, (x, Y - 40), (x, Y + ROWS * ROW_H));
        for r in 0..=ROWS {
            line(&mut img, (x - 6, Y + r * ROW_H), (x + 6, Y + r * ROW_H), colour(70.0), 2); // the crossings
        }
    }
    for &(r, dx) in &[(0, 20), (3, 110)] {
        let pts: Vec<(i32, i32)> = linspace((X + dx) as f64, (X + dx + 70) as f64, 40)
            .iter()
            .zip(JITTER_1)
            .map(|(&x, j)| (x as i32, ((Y + r * ROW_H) as f64 + 18.0 + 9.0 * ((x - X as f64 - dx as f64) / 4.0).sin() + j as f64) as i32))
            .collect();
        polyline(&mut img, &pts, colour(20.0), 2);
    }
    let cs: Vec<Zone> = sig_cells().into_iter().map(|z| Zone { region: "t".into(), ..z }).collect();
    let mut canvas = gray(600, 500, 255);
    let rep = sigs(&mut canvas, &img, &cs, &none());
    assert_eq!(modes(&rep), ["signature", "empty", "empty", "signature", "empty", "empty"]);
}

/// A ruled column under its printed header, `dx` px across and `k` times as
/// wide as at x = 300..500.
fn headed_column(dx: i32, k: f64) -> Mat {
    let mut g = gray(700, 900, 255);
    let (l, r) = (300 + dx, 300 + dx + (200.0 * k) as i32);
    for x in [l, r] {
        line(&mut g, (x, 180), (x, 620), colour(0.0), 2);
    }
    for k in 0..=12 {
        line(&mut g, (l - 150, 220 + 36 * k), (r + 150, 220 + 36 * k), colour(0.0), 1);
    }
    text(&mut g, "ID/Passport No.", (l + 20, 205), 0.7, colour(0.0), 2, true);
    g
}

#[test]
fn a_column_off_its_rules_is_placed_again_by_its_header() {
    let reference = headed_column(0, 1.0);
    let page = headed_column(20, 1.04); // the table 20 px across and 4% wider
    let cells: Vec<Zone> = (0..12).map(|k| zone(&format!("id_{}", k), [300, 220 + 36 * k, 200, 36], "partial")).collect();
    let f = identity();
    assert!(!check_column_edges(&page, &f, &cells, false).unwrap().is_empty()); // placed as in the reference: off
    let (g, [a, b], ncc) = place_by_header(&page, &f, &reference, [310.0, 182.0, 200.0, 30.0], &cells).unwrap().unwrap();
    assert!(ncc > 0.8);
    assert!((a * 300.0 + b - 320.0).abs() <= 2.0 && (a * 500.0 + b - 528.0).abs() <= 2.0, "{} {}", a, b);
    assert!(check_column_edges(&page, &g, &cells, true).unwrap().is_empty());
}
