//! Learning a printing's layout: grouping, aligning bent pages, the blank
//! form and its emptied fields. Reading the blank form is `generic`'s.

use super::*;
use crate::testutil::{colour, gray, line, text, tmpdir};

const W: i32 = 1000;
const H: i32 = 1400;

/// A printed form: rules, labels, and a fine background pattern like the
/// security print, the same on every page.
fn form() -> Mat {
    let mut g = gray(H, W, 235);
    let mut s: u64 = 7;
    let mut next = |m: u64| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) % m) as i32
    };
    for _ in 0..1500 {
        let (x, y) = (next(W as u64 - 20), next(H as u64 - 10) + 10);
        text(&mut g, ["x", "o", "+", "v"][next(4) as usize], (x, y), 0.4, colour(200.0), 1, false);
    }
    for (k, y) in (100..1350).step_by(50).enumerate() {
        line(&mut g, (50, y), (950, y), colour(40.0), 2);
        text(&mut g, &format!("Item {} code {}", k + 1, 3 * k + 7), (60, y - 14), 0.7, colour(40.0), 2, false);
    }
    for x in [50, 400, 700, 950] {
        line(&mut g, (x, 100), (x, 1300), colour(40.0), 2);
    }
    g
}

/// `seed`'s writing: different on every page.
fn write(g: &mut Mat, k: i32) {
    for r in 0..6 {
        let y = 140 + 50 * (3 * r + k % 3);
        text(g, &format!("{}{}{}", 7 * k + r, k, r), (420 + 13 * k, y), 0.9, colour(70.0), 2, true);
    }
}

/// `g` bent like a photo of a curled page: a smooth displacement growing
/// towards the bottom (up to `bend` px), then shifted.
fn bent(g: &Mat, bend: f64) -> Mat {
    let mut mx = Mat::new_rows_cols_with_default(H, W, core::CV_32F, Scalar::all(0.0)).unwrap();
    let mut my = mx.clone();
    for y in 0..H {
        for x in 0..W {
            let t = y as f64 / H as f64;
            *mx.at_2d_mut::<f32>(y, x).unwrap() = (x as f64 - 12.0 - 0.3 * bend * t * t * (x as f64 / W as f64)) as f32;
            *my.at_2d_mut::<f32>(y, x).unwrap() = (y as f64 + 8.0 - bend * t * t) as f32;
        }
    }
    let mut out = Mat::default();
    imgproc::remap(g, &mut out, &mx, &my, imgproc::INTER_LINEAR, core::BORDER_CONSTANT, Scalar::all(235.0), core::AlgorithmHint::ALGO_HINT_DEFAULT).unwrap();
    out
}

/// Row of the darkest horizontal run in columns x0..x1 between rows y0..y1.
fn rule_row(g: &Mat, x0: i32, x1: i32, y0: i32, y1: i32) -> i32 {
    (y0..y1).min_by_key(|&y| (x0..x1).map(|x| *g.at_2d::<u8>(y, x).unwrap() as i64).sum::<i64>()).unwrap()
}

#[test]
fn printing_groups_kiems_names_by_where_the_form_is_printed() {
    assert_eq!(printing("x.zip::1_34_A_005021010100101_Y3KK12CQ5IA00V1_20220809_200956.PDF", None), ("34A".into(), "34A national".into()));
    assert_eq!(printing("1_36_A_005021010100101_Y3KK12CQ5IA00V1_20220809_200956.PDF", None), ("36A".into(), "36A ward 0050210101".into()));
    assert_eq!(printing("1_35_A_005021010100101_Y3KK12CQ5IA00V1_20220809_200956.jpg", None), ("35A".into(), "35A constituency 005021".into()));
    assert_eq!(printing("1_38_A_005021010100101_Y3KK12CQ5IA00V1_20220809_200956.jpg", None), ("38A".into(), "38A county 005".into()));
    assert_eq!(printing("scans/p-3.jpg", Some("36A")), ("36A".into(), "36A".into()));
    assert_eq!(slug("36A ward 0050210101"), "learned_36a_ward_0050210101");
}

#[test]
fn a_curled_page_is_aligned_where_one_homography_is_not_enough() {
    let mut seed = form();
    write(&mut seed, 0);
    let mut page = form();
    write(&mut page, 1);
    // 60 px at the bottom, as on a folded page photographed
    let page = bent(&page, 60.0);
    let o = orb(&seed).unwrap();
    let (out, _) = onto(&o, &util::bytes(&seed).unwrap(), Size::new(W, H), &page).unwrap().expect("pages of one printing align");
    for (y, x0) in [(300, 100), (800, 100), (1250, 100), (1250, 760)] {
        let r = rule_row(&out, x0, x0 + 180, y - 20, y + 20);
        assert!((r - y).abs() <= 2, "rule at y={} found at {} (x from {})", y, r, x0);
    }
    for x in [400, 700] {
        let c = (x - 20..x + 20).min_by_key(|&c| (1150..1290).map(|y| *out.at_2d::<u8>(y, c).unwrap() as i64).sum::<i64>()).unwrap();
        assert!((c - x).abs() <= 2, "column rule at x={} found at {}", x, c);
    }
}

#[test]
fn a_page_of_another_form_does_not_align() {
    let seed = form();
    let mut other = gray(H, W, 235);
    for k in 0..30 {
        text(&mut other, &format!("Unrelated text {}", k * 37), (40 + 9 * (k % 7), 60 + 44 * k), 0.9, colour(40.0), 2, false);
    }
    let o = orb(&seed).unwrap();
    assert!(onto(&o, &util::bytes(&seed).unwrap(), Size::new(W, H), &other).unwrap().is_none());
}

#[test]
fn the_blank_form_keeps_the_print_and_drops_the_writing() {
    let stack: Vec<Mat> = (0..9)
        .map(|k| {
            let mut g = form();
            write(&mut g, k);
            g
        })
        .collect();
    let Blank { median, print, freq, writing } = blank(&stack, 0.45).unwrap();
    let px = |m: &Mat, x: i32, y: i32| *m.at_2d::<u8>(y, x).unwrap();
    // rules and labels are print, and nobody's writing
    assert!(px(&median, 200, 300) < 90 && px(&print, 200, 300) != 0);
    assert!(px(&median, 400, 700) < 90);
    assert!(writing.iter().all(|wr| wr[(300 * W + 200) as usize] == 0 && wr[(700 * W + 400) as usize] == 0));
    // no page's writing is left in the blank form: wherever any page wrote
    let mut written = 0;
    for y in 0..H {
        for x in 0..W {
            if freq[(y * W + x) as usize] > 0.0 {
                written += 1;
                assert!(px(&median, x, y) > 150, "writing left in the blank form at ({}, {}): {}", x, y, px(&median, x, y));
            }
        }
    }
    assert!(written > 1000, "the pages' writing is found ({} px)", written);
}

#[test]
fn writing_most_pages_have_in_one_place_is_still_writing() {
    // the first agent's ID: on 7 of 9 pages, starting alike
    let stack: Vec<Mat> = (0..9)
        .map(|k| {
            let mut g = form();
            if k < 7 {
                text(&mut g, &format!("{}", 31_400_000 + 123_457 * k), (420, 240), 0.9, colour(70.0), 2, true);
            }
            g
        })
        .collect();
    let Blank { writing, freq, .. } = blank(&stack, 0.45).unwrap();
    let cell = |wr: &Vec<u8>| (215..245).flat_map(|y| (420..600).map(move |x| (y * W + x) as usize)).filter(|&i| wr[i] != 0).count();
    assert!(cell(&writing[0]) > 150, "the figures on a page are writing ({} px)", cell(&writing[0]));
    assert!(cell(&writing[8]) < 10, "a page that left it empty has none there");
    let freq = &freq;
    let most = (215..245).flat_map(|y| (420..600).map(move |x| freq[(y * W + x) as usize])).fold(0.0f32, f32::max);
    assert!(most >= 0.6, "where most pages wrote, most pages have writing ({})", most);
}

#[test]
fn clean_empties_fields_but_their_rules_and_kept_labels() {
    let mut g = gray(400, 600, 235);
    line(&mut g, (20, 200), (580, 200), colour(30.0), 2);
    line(&mut g, (20, 300), (580, 300), colour(218.0), 2);
    text(&mut g, "Signature", (40, 100), 0.8, colour(30.0), 2, false);
    text(&mut g, "ghost", (300, 100), 0.8, colour(120.0), 2, false);
    let c = clean(&g, &[[0.0, 0.0, 600.0, 400.0]], &[[30.0, 70.0, 170.0, 40.0]]).unwrap();
    let px = |x: i32, y: i32| *c.at_2d::<u8>(y, x).unwrap();
    assert!(px(300, 200) < 60, "the rule stays");
    assert!(px(300, 300) < 225, "a light grey rule stays");
    assert!((70..110).any(|y| (40..200).any(|x| px(x, y) < 60)), "the kept label stays");
    assert!((70..110).all(|y| (300..420).all(|x| px(x, y) >= 230)), "the ghost goes");
}

#[test]
fn learning_a_batch_leaves_printings_already_learned() {
    let out = tmpdir("learned");
    std::fs::create_dir_all(out.join("learned_34a_national")).unwrap();
    std::fs::write(out.join("learned_34a_national/template.yaml"), "name: learned_34a_national\n").unwrap();
    let sources: Vec<String> = (0..6).map(|k| format!("1_34_A_00502101010010{}_Y3KK12CQ5IA00V1_20220809_200956.PDF", k)).collect();
    let r = learn_all(&sources, None, &out, &LearnOptions::default(), true).unwrap();
    assert!(r.is_empty());
    // 34B pages are tally forms: not learned here
    let tally: Vec<String> = (0..6).map(|k| format!("1_34_B_00502101010010{}_Y3KK12CQ5IA00V1_20220809_200956.PDF", k)).collect();
    assert!(learn_all(&tally, None, &out, &LearnOptions::default(), true).is_err());
}

/// A blank form's agents' table, 7 rows of 40 px under a heading row, and
/// what a reader that missed it made of it: its "Signature" heading taken
/// for an officer's signature line, as on the 2017 printings.
#[test]
fn an_officer_line_inside_the_agents_table_is_its_heading() {
    let mut g = gray(900, 1000, 235);
    for y in [380, 440, 480, 520, 560, 600, 640, 680] {
        line(&mut g, (250, y), (850, y), colour(40.0), 2);
    }
    line(&mut g, (250, 720), (850, 720), colour(40.0), 4);
    for x in [250, 450, 650, 850] {
        line(&mut g, (x, 380), (x, 720), colour(40.0), 2);
    }
    let head = |t: &str, x0: f32, x1: f32| {
        let n = t.chars().count();
        OLine::new([[x0, 395.0], [x1, 395.0], [x1, 425.0], [x0, 425.0]], t.into(), 0.95, (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect(), vec![0.95; n])
    };
    let officer = |y: f64| Field { kind: "signature".into(), zones: vec![Zone::new("signature", [660.0, y, 180.0, 34.0], "signature")], source: "declaration".into(), label: "Signature".into(), label_box: (600.0, y, 650.0, y + 34.0) };
    let mut l = Layout {
        frame: generic::Frame { k: 0, skew: 0.0, scale: 1.0, m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], size: (1000, 900), text_h: TEXT_H },
        image: crate::testutil::to_bgr(&g),
        gray: g.clone(),
        lines: vec![head("ID/Passport No.", 260.0, 440.0), head("Tel. Contact", 460.0, 640.0), head("Signature", 660.0, 840.0)],
        fields: vec![officer(315.0), officer(395.0)],
        protected: vec![("results".into(), vec![[0.0, 0.0], [1000.0, 0.0], [1000.0, 300.0], [0.0, 300.0]])],
        stamps: vec![],
        problems: vec![],
        notes: vec![],
        officer_names: BTreeSet::new(),
    };
    with_profile_table(&mut l, &g).unwrap();
    let kinds: BTreeSet<&str> = l.fields.iter().filter(|f| f.source == "table").map(|f| f.kind.as_str()).collect();
    assert_eq!(kinds, BTreeSet::from(["id", "phone", "signature"]), "the table is read from its rules");
    let officers: Vec<f64> = l.fields.iter().filter(|f| f.source == "declaration").map(|f| f.zones[0].rect[1]).collect();
    assert_eq!(officers, vec![315.0], "the line above the table stays, its heading goes");
    assert!(l.notes.iter().any(|n| n.contains("inside the agents' table dropped")), "{:?}", l.notes);
}
