use super::*;
use crate::testutil::*;
use crate::util::{self, Grid};
use opencv::core::{Mat, Vec3b};
use opencv::imgproc;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn jpeg(seed: u64, w: i32, h: i32) -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let noise: Vec<u8> = (0..w * h).map(|_| rng.random_range(0..255u32) as u8).collect();
    let mut blurred = Mat::default();
    imgproc::gaussian_blur_def(&util::from_bytes(h, w, &noise).unwrap(), &mut blurred, opencv::core::Size::new(0, 0), 3.0).unwrap();
    let mut img = to_bgr(&blurred);
    text(&mut img, "123", (40, 120), 2.0, colour(20.0), 4, false);
    crate::io::encode(&img, "jpg", 85).unwrap()
}

fn decode_gray(data: &[u8]) -> Mat {
    to_gray(&crate::io::decode_bytes(data).unwrap())
}

#[test]
fn block_packing_roundtrip() {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut b = Grid::new(37, 29);
    for v in b.data.iter_mut() {
        *v = rng.random::<f64>() > 0.7;
    }
    assert_eq!(unpack(&pack(&b)).unwrap().data, b.data);
}

#[test]
fn jpeg_redaction_changes_only_listed_blocks_to_black() {
    let data = jpeg(0, 320, 240);
    let mut m = gray(240, 320, 0);
    fill_rect(&mut m, 40, 90, 100, 130, 255);
    let blocks = to_blocks(&m).unwrap();
    let red = redact_jpeg(&data, &blocks).unwrap();
    let cmp = compare_jpeg(&data, &red.data, Some(&red.luma_blocks)).unwrap();
    assert!(cmp.ok, "{:?}", cmp.problems);
    assert!(cmp.changed_blocks.count() <= blocks.count());
    assert_eq!(cmp.identical_blocks, cmp.total_blocks - cmp.changed_blocks.count());
    let dec = decode_gray(&red.data);
    let under = to_pixels(&blocks, dec.rows(), dec.cols()).unwrap();
    let (d, u) = (util::bytes(&dec).unwrap(), util::bytes(&under).unwrap());
    assert!(d.iter().zip(&u).filter(|&(_, &m)| m != 0).all(|(&v, _)| v <= 8)); // decodes to black
}

#[test]
fn jpeg_comparison_catches_a_single_changed_coefficient() {
    let data = jpeg(1, 320, 240);
    let red = redact_jpeg(&data, &Grid::new(30, 40)).unwrap(); // no masks: re-encoded only
    assert!(compare_jpeg(&data, &red.data, None).unwrap().ok);
    let bad = nudge_jpeg(&red.data, 0, 5, 5, 1, 3).unwrap(); // a nudge to one coefficient, e.g. to alter a digit
    let cmp = compare_jpeg(&data, &bad, None).unwrap();
    assert!(!cmp.ok && cmp.problems[0].contains("other than black"), "{:?}", cmp.problems);
}

#[test]
fn undeclared_black_block_is_caught() {
    let data = jpeg(2, 320, 240);
    let mut b = Grid::new(30, 40);
    b.set(3, 3, true);
    let red = redact_jpeg(&data, &b).unwrap();
    assert!(!compare_jpeg(&data, &red.data, Some(&Grid::new(30, 40))).unwrap().ok);
}

#[test]
fn pixel_path() {
    let img = crate::io::decode_bytes(&jpeg(3, 320, 240)).unwrap();
    let g = to_blocks(&gray(img.rows(), img.cols(), 0)).unwrap();
    let mut b = Grid::new(g.rows, g.cols);
    for r in 10..12 {
        for c in 10..15 {
            b.set(r, c, true);
        }
    }
    let red = redact_pixels(&img, &b).unwrap();
    let mut out = crate::io::decode_bytes(&red.data).unwrap();
    assert!(compare_pixels(&img, &out, Some(&b)).unwrap().ok);
    *out.at_2d_mut::<Vec3b>(0, 0).unwrap() = Vec3b::from([1, 2, 3]);
    assert!(!compare_pixels(&img, &out, Some(&b)).unwrap().ok);
}
