use super::*;
use crate::testutil::{colour, gray, line};

const V_LINES: [i32; 8] = [100, 150, 500, 680, 1050, 1220, 1450, 1560];

fn rows() -> Vec<f64> {
    (0..11).map(|i| (500 + 50 + 36 * i) as f64).collect() // header bottom + row rules
}

fn h_lines() -> Vec<f64> {
    let mut h = vec![500.0];
    h.extend(rows());
    h
}

fn grid(dy: f64, scale: f64) -> Mat {
    let mut img = gray(1200, 1700, 235);
    let hl = h_lines();
    let (top, bot) = (500.0 + dy, 500.0 + dy + (hl[hl.len() - 1] - 500.0) * scale);
    for y in &hl {
        let yy = util::round_even(500.0 + dy + (y - 500.0) * scale) as i32;
        line(&mut img, (100, yy), (1560, yy), colour(50.0), 2);
    }
    for x in V_LINES {
        line(&mut img, (x, top as i32), (x, bot as i32), colour(50.0), 2);
    }
    img
}

const EYE: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[test]
fn match_1d_ignores_spurious_lines_and_repetition() {
    let hl = h_lines();
    let mut found: Vec<f64> = hl.iter().map(|h| h + 7.0).collect();
    found.extend([430.0, 455.0, 760.0, 1040.0]); // extra underlines etc.
    let (_, _, pairs) = match_1d(&hl, &found, 10.0, (0.75, 1.35), None, None, Some((507.0, 15.0, hl[hl.len() - 1] + 7.0, 25.0)));
    assert_eq!(pairs.len(), hl.len());
    assert!(pairs.iter().all(|&(i, j)| found[j] - hl[i] == 7.0));
}

#[test]
fn snap_finds_shifted_table() {
    let region = prepare_region(&grid(0.0, 1.0), "agents", [80.0, 480.0, 1500.0, 480.0], false).unwrap();
    assert!(region.vlines.len() == V_LINES.len() && region.hlines.len() >= 10);
    let scan = grid(30.0, 1.05); // table 30 px lower and stretched
    let sn = snap(&scan, &EYE, &region).unwrap();
    assert!(sn.ok, "{}", sn.reason);
    let r = rows();
    let p = sn.map(&[[500.0, r[5]], [1220.0, r[10]]]);
    let expect = [[500.0, 530.0 + (r[5] - 500.0) * 1.05], [1220.0, 530.0 + (r[10] - 500.0) * 1.05]];
    for (a, b) in p.iter().zip(&expect) {
        assert!((a[0] - b[0]).abs() < 4.0 && (a[1] - b[1]).abs() < 4.0, "{:?} vs {:?}", a, b);
    }
}

#[test]
fn snap_a_table_hung_under_another() {
    // Layout C: the agents table hangs under the results table, with the same
    // columns, so the column rules run on up into it. Only their lower end
    // marks the table; without `open_top` the span check rejects it.
    let page = |dy: i32| {
        let mut img = grid(dy as f64, 1.0);
        for x in V_LINES {
            line(&mut img, (x, 120 + dy), (x, 500 + dy), colour(50.0), 2);
        }
        for y in (120..500).step_by(40) {
            line(&mut img, (100, y + dy), (1560, y + dy), colour(50.0), 2);
        }
        img
    };
    let (reference, scan) = (grid(0.0, 1.0), page(40));
    let closed = prepare_region(&reference, "agents", [80.0, 480.0, 1500.0, 480.0], false).unwrap();
    assert!(!snap(&scan, &EYE, &closed).unwrap().ok);
    let region = prepare_region(&reference, "agents", [80.0, 480.0, 1500.0, 480.0], true).unwrap();
    let sn = snap(&scan, &EYE, &region).unwrap();
    assert!(sn.ok, "{}", sn.reason);
    let r = rows();
    let p = sn.map(&[[500.0, r[5]]]);
    assert!((p[0][1] - (r[5] + 40.0)).abs() < 4.0);
}

#[test]
fn rules_running_up_past_a_table_that_ends_where_expected() {
    // A scan's dark band above the table can make its column rules seem to
    // run on up the page; where their lower ends are where the table ends,
    // the table is placed by them.
    let mut scan = grid(6.0, 1.0);
    for x in V_LINES {
        line(&mut scan, (x, 250), (x, 506), colour(50.0), 2);
    }
    let region = prepare_region(&grid(0.0, 1.0), "agents", [80.0, 480.0, 1500.0, 480.0], false).unwrap();
    let sn = snap(&scan, &EYE, &region).unwrap();
    assert!(sn.ok, "{}", sn.reason);
    let r = rows();
    let p = sn.map(&[[500.0, r[5]]]);
    assert!((p[0][1] - (r[5] + 6.0)).abs() < 4.0);
}
