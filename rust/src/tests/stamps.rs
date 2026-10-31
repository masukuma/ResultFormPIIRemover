//! Stamps are found and never masked (the pipeline side is in pipeline's tests).

use super::*;
use crate::testutil::{bgr, fill_rect, gray, line, text};
use opencv::core::{Point, Scalar, Size};

fn bgr_colour(c: (f64, f64, f64)) -> Scalar {
    Scalar::new(c.0, c.1, c.2, 0.0)
}

fn circle(img: &mut Mat, c: (i32, i32), r: i32, colour: Scalar, thick: i32) {
    imgproc::circle(img, Point::new(c.0, c.1), r, colour, thick, imgproc::LINE_AA, 0).unwrap();
}

fn arc(img: &mut Mat, c: (i32, i32), r: i32, a0: f64, a1: f64, colour: Scalar, thick: i32) {
    imgproc::ellipse(img, Point::new(c.0, c.1), Size::new(r, r), 0.0, a0, a1, colour, thick, imgproc::LINE_AA, 0).unwrap();
}

#[test]
fn finds_a_stamp_and_not_a_table() {
    let mut page = bgr(1400, 1000, [255, 255, 255]);
    for y in (100..700).step_by(40) {
        // a table with writing in it
        line(&mut page, (50, y), (950, y), Scalar::all(0.0), 2);
        text(&mut page, "0712 345 678  KAMAU W.", (70, y + 30), 0.8, bgr_colour((90.0, 30.0, 10.0)), 2, false);
    }
    for x in [50, 400, 950] {
        line(&mut page, (x, 100), (x, 660), Scalar::all(0.0), 2);
    }
    let (cx, cy, r) = (600, 1000, 90);
    circle(&mut page, (cx, cy), r, bgr_colour((150.0, 40.0, 110.0)), 5);
    text(&mut page, "I.E.B.C.", (cx - 55, cy), 0.8, bgr_colour((150.0, 40.0, 110.0)), 2, false);
    let found = find(&page, (40.0, 140.0), &[], None).unwrap();
    assert_eq!(found.len(), 1);
    let st = &found[0];
    assert!((st.cx - cx as f64).abs() < 6.0 && (st.cy - cy as f64).abs() < 6.0 && (st.r - r as f64).abs() < 8.0);
}

#[test]
fn lettering_read_with_the_writing_beside_it_is_the_rings() {
    // Gatundu South p127: "LUJMOFFICER", the name written through the
    // stamp and then its "OFFICER", boxed far outside the ring
    let mut page = bgr(1400, 1000, [255, 255, 255]);
    let (cx, cy, r) = (600, 1000, 90);
    circle(&mut page, (cx, cy), r, bgr_colour((150.0, 40.0, 110.0)), 5);
    let read = TextLine { text: "LUJMOFFICER".into(), bbox: [380.0, 1050.0, 690.0, 1090.0] };
    let found = find(&page, (40.0, 140.0), std::slice::from_ref(&read), None).unwrap();
    assert_eq!(found.len(), 1);
    assert!((found[0].cx - cx as f64).abs() < 6.0 && (found[0].r - r as f64).abs() < 8.0, "{:?}", found[0]);
    // with no ring round it, the lettering still stands for a stamp
    let found = find(&bgr(1400, 1000, [255, 255, 255]), (40.0, 140.0), &[read], None).unwrap();
    assert_eq!((found.len(), found[0].how.as_str()), (1, "text"));
}

#[test]
fn faint_coloured_stamp_is_found() {
    let mut page = bgr(1200, 900, [235, 235, 235]);
    circle(&mut page, (450, 600), 100, bgr_colour((200.0, 170.0, 205.0)), 3); // pale violet on grey paper
    let found = find(&page, (50.0, 150.0), &[], None).unwrap();
    assert!(!found.is_empty() && (found[0].r - 100.0).abs() < 10.0);
}

fn hide_band() -> Mat {
    let mut hide = gray(800, 800, 0);
    fill_rect(&mut hide, 300, 380, 500, 420, 255);
    hide
}

/// A broken coloured ring: arcs covering about half the way round.
fn half_ring(page: &mut Mat, c: (i32, i32), r: i32, colour: Scalar) {
    for a0 in (0..360).step_by(60) {
        arc(page, c, r, a0 as f64, a0 as f64 + 32.0, colour, 3);
    }
}

#[test]
fn a_small_broken_ring_near_a_mask_is_not_a_stamp() {
    // A handwritten "0" and the strokes around it, under a signature's mask,
    // fit a small, broken ring. Near a mask, a ring smaller than any stamp
    // must run all the way round; one in the stamp size range is still taken
    // when broken.
    let pen = bgr_colour((160.0, 60.0, 20.0));
    let mut small = bgr(800, 800, [245, 245, 245]);
    half_ring(&mut small, (400, 400), 47, pen); // under the smallest stamp, as seen (92%)
    assert!(near(&small, &hide_band(), (50.0, 150.0), &[], None).unwrap().is_empty());
    let mut big = bgr(800, 800, [245, 245, 245]);
    half_ring(&mut big, (400, 400), 90, pen);
    let found = near(&big, &hide_band(), (50.0, 150.0), &[], None).unwrap();
    assert!(found.len() == 1 && (found[0].r - 90.0).abs() < 10.0);
}

#[test]
fn writing_crossing_a_circle_near_a_mask_is_not_a_stamp() {
    // A column of IDs in blue pen can put short strokes all round a circle. A
    // stamp's ring, even broken, runs in long arcs; crossings are short.
    let pen = bgr_colour((160.0, 60.0, 20.0));
    let mut page = bgr(800, 800, [245, 245, 245]);
    for a0 in (0..360).step_by(10) {
        arc(&mut page, (400, 400), 90, a0 as f64, a0 as f64 + 5.0, pen, 3);
    }
    assert!(near(&page, &hide_band(), (50.0, 150.0), &[], None).unwrap().is_empty());
}
