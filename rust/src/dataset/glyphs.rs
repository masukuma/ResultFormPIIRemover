//! A small stroke font for handwriting-like digits and capitals.
//!
//! Each glyph is a few polylines in a unit box: x from 0 (left) to its
//! advance width, y from 0 (top of a capital) to 1 (the baseline). Curves are
//! arcs of ellipses. `write` lays a string out with a given height, slant,
//! spacing and jitter, so the same glyphs look written by many hands; the
//! stamp text uses them with no jitter, along a ring.

pub type Pt = [f64; 2];
pub type Poly = Vec<Pt>;

/// Points along an ellipse arc centred on (cx, cy), from angle `a0` to `a1`
/// degrees (0 = right, 90 = down, as on the page), `a1` may be less than
/// `a0` (drawn the other way round).
fn arc(cx: f64, cy: f64, rx: f64, ry: f64, a0: f64, a1: f64) -> Poly {
    let n = (((a1 - a0).abs() / 22.5).ceil() as usize).max(2);
    (0..=n)
        .map(|k| {
            let a = (a0 + (a1 - a0) * k as f64 / n as f64).to_radians();
            [cx + rx * a.cos(), cy + ry * a.sin()]
        })
        .collect()
}

fn pts(p: &[(f64, f64)]) -> Poly {
    p.iter().map(|&(x, y)| [x, y]).collect()
}

fn join(mut a: Poly, b: Poly) -> Poly {
    a.extend(b);
    a
}

/// The glyph of `c` (capitals, digits, `.`, `/`, `-`, `:`; anything else is
/// a space) and its advance width.
pub fn glyph(c: char) -> (Vec<Poly>, f64) {
    let g: Vec<Poly> = match c {
        '0' => vec![arc(0.5, 0.5, 0.42, 0.5, 270.0, 630.0)],
        '1' => vec![pts(&[(0.15, 0.22), (0.45, 0.0), (0.45, 1.0)])],
        '2' => vec![join(arc(0.5, 0.3, 0.4, 0.3, 200.0, 380.0), pts(&[(0.08, 1.0), (0.95, 1.0)]))],
        '3' => vec![join(arc(0.48, 0.26, 0.38, 0.26, 200.0, 450.0), arc(0.48, 0.74, 0.42, 0.26, 270.0, 520.0))],
        '4' => vec![pts(&[(0.72, 1.0), (0.72, 0.0), (0.05, 0.68), (0.95, 0.68)])],
        '5' => vec![join(pts(&[(0.88, 0.0), (0.22, 0.0), (0.19, 0.45)]), arc(0.47, 0.68, 0.4, 0.32, 225.0, 520.0))],
        '6' => vec![join(arc(0.55, 0.55, 0.45, 0.55, 290.0, 180.0), arc(0.5, 0.72, 0.4, 0.28, 180.0, 540.0))],
        '7' => vec![pts(&[(0.08, 0.0), (0.92, 0.0), (0.38, 1.0)])],
        '8' => vec![join(arc(0.5, 0.25, 0.33, 0.25, 90.0, 450.0), arc(0.5, 0.75, 0.4, 0.25, 270.0, 630.0))],
        '9' => vec![join(arc(0.5, 0.3, 0.4, 0.3, 0.0, 360.0), pts(&[(0.88, 0.6), (0.62, 1.0)]))],
        'A' => vec![pts(&[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)]), pts(&[(0.22, 0.6), (0.78, 0.6)])],
        'B' => vec![
            pts(&[(0.1, 1.0), (0.1, 0.0)]),
            join(join(pts(&[(0.1, 0.0), (0.5, 0.0)]), arc(0.5, 0.25, 0.35, 0.25, 270.0, 450.0)), pts(&[(0.1, 0.5)])),
            join(join(pts(&[(0.1, 0.5), (0.5, 0.5)]), arc(0.5, 0.75, 0.42, 0.25, 270.0, 450.0)), pts(&[(0.1, 1.0)])),
        ],
        'C' => vec![arc(0.55, 0.5, 0.45, 0.5, 320.0, 40.0)],
        'D' => vec![pts(&[(0.1, 0.0), (0.1, 1.0)]), join(join(pts(&[(0.1, 0.0), (0.4, 0.0)]), arc(0.4, 0.5, 0.52, 0.5, 270.0, 450.0)), pts(&[(0.1, 1.0)]))],
        'E' => vec![pts(&[(0.9, 0.0), (0.1, 0.0), (0.1, 1.0), (0.9, 1.0)]), pts(&[(0.1, 0.5), (0.7, 0.5)])],
        'F' => vec![pts(&[(0.9, 0.0), (0.1, 0.0), (0.1, 1.0)]), pts(&[(0.1, 0.5), (0.7, 0.5)])],
        'G' => vec![join(arc(0.52, 0.5, 0.45, 0.5, 320.0, 30.0), pts(&[(0.95, 0.75), (0.95, 0.55), (0.6, 0.55)]))],
        'H' => vec![pts(&[(0.1, 0.0), (0.1, 1.0)]), pts(&[(0.9, 0.0), (0.9, 1.0)]), pts(&[(0.1, 0.5), (0.9, 0.5)])],
        'I' => vec![pts(&[(0.3, 0.0), (0.3, 1.0)]), pts(&[(0.05, 0.0), (0.55, 0.0)]), pts(&[(0.05, 1.0), (0.55, 1.0)])],
        'J' => vec![join(pts(&[(0.85, 0.0), (0.85, 0.7)]), arc(0.5, 0.7, 0.35, 0.3, 0.0, 160.0))],
        'K' => vec![pts(&[(0.1, 0.0), (0.1, 1.0)]), pts(&[(0.9, 0.0), (0.1, 0.6)]), pts(&[(0.35, 0.45), (0.9, 1.0)])],
        'L' => vec![pts(&[(0.1, 0.0), (0.1, 1.0), (0.85, 1.0)])],
        'M' => vec![pts(&[(0.05, 1.0), (0.12, 0.0), (0.6, 0.7), (1.08, 0.0), (1.15, 1.0)])],
        'N' => vec![pts(&[(0.1, 1.0), (0.1, 0.0), (0.9, 1.0), (0.9, 0.0)])],
        'O' => vec![arc(0.5, 0.5, 0.47, 0.5, 270.0, 630.0)],
        'P' => vec![join(join(pts(&[(0.1, 1.0), (0.1, 0.0), (0.5, 0.0)]), arc(0.5, 0.27, 0.4, 0.27, 270.0, 450.0)), pts(&[(0.1, 0.54)]))],
        'Q' => vec![arc(0.5, 0.5, 0.47, 0.5, 270.0, 630.0), pts(&[(0.6, 0.7), (0.98, 1.05)])],
        'R' => vec![join(join(pts(&[(0.1, 1.0), (0.1, 0.0), (0.5, 0.0)]), arc(0.5, 0.27, 0.4, 0.27, 270.0, 450.0)), pts(&[(0.1, 0.54)])), pts(&[(0.45, 0.54), (0.9, 1.0)])],
        'S' => vec![join(arc(0.5, 0.26, 0.4, 0.26, 330.0, 90.0), arc(0.5, 0.74, 0.42, 0.26, 270.0, 510.0))],
        'T' => vec![pts(&[(0.0, 0.0), (1.0, 0.0)]), pts(&[(0.5, 0.0), (0.5, 1.0)])],
        'U' => vec![join(join(pts(&[(0.1, 0.0), (0.1, 0.65)]), arc(0.5, 0.65, 0.4, 0.35, 180.0, 0.0)), pts(&[(0.9, 0.0)]))],
        'V' => vec![pts(&[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)])],
        'W' => vec![pts(&[(0.0, 0.0), (0.3, 1.0), (0.6, 0.35), (0.9, 1.0), (1.2, 0.0)])],
        'X' => vec![pts(&[(0.05, 0.0), (0.95, 1.0)]), pts(&[(0.95, 0.0), (0.05, 1.0)])],
        'Y' => vec![pts(&[(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]), pts(&[(0.5, 0.5), (0.5, 1.0)])],
        'Z' => vec![pts(&[(0.05, 0.0), (0.95, 0.0), (0.05, 1.0), (0.95, 1.0)])],
        '.' => vec![pts(&[(0.05, 0.9), (0.2, 0.9), (0.2, 1.0), (0.05, 1.0), (0.05, 0.9)])],
        ':' => vec![pts(&[(0.05, 0.3), (0.2, 0.3), (0.2, 0.4), (0.05, 0.4), (0.05, 0.3)]), pts(&[(0.05, 0.9), (0.2, 0.9), (0.2, 1.0), (0.05, 1.0), (0.05, 0.9)])],
        '/' => vec![pts(&[(0.8, 0.0), (0.1, 1.0)])],
        '-' => vec![pts(&[(0.1, 0.55), (0.7, 0.55)])],
        _ => vec![],
    };
    let adv = match c {
        '1' => 0.6,
        'I' => 0.6,
        'M' | 'W' => 1.25,
        '.' | ':' => 0.3,
        '-' | '/' => 0.85,
        ' ' => 0.6,
        _ => 1.0,
    };
    (g, adv)
}

/// How a hand writes: letter height and width (px), slant (x shift per px
/// of height, positive leaning right), gap between letters (share of the
/// width), jitter of each point (share of the height), stroke width (px).
#[derive(Clone, Debug)]
pub struct Hand {
    pub height: f64,
    pub width: f64,
    pub slant: f64,
    pub gap: f64,
    pub jitter: f64,
    pub thick: i32,
}

/// One character as written: its strokes on the page and its box
/// (x0, y0, x1, y1), the stroke width included.
#[derive(Clone, Debug)]
pub struct Written {
    pub ch: char,
    pub polys: Vec<Poly>,
    pub bbox: [f64; 4],
}

/// Corner cutting (Chaikin), once: rounds a jittered polyline the way a
/// pen rounds its turns, keeping the end points.
pub fn smooth(p: &Poly) -> Poly {
    if p.len() < 3 {
        return p.clone();
    }
    let mut out = vec![p[0]];
    for w in p.windows(2) {
        let (a, b) = (w[0], w[1]);
        out.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
        out.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
    }
    out.push(*p.last().unwrap());
    out
}

/// The box (x0, y0, x1, y1) of polylines drawn `thick` px wide.
pub fn bbox(polys: &[Poly], thick: i32) -> [f64; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in polys.iter().flatten() {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let r = thick as f64 / 2.0 + 1.5;
    [x0 - r, y0 - r, x1 + r, y1 + r]
}

/// Write `s` from (x, baseline) in `hand`. `noise` gives a value in
/// [-1, 1) each time it is called (the caller's seeded generator), so the
/// writing is repeatable. Spaces advance without strokes and are not
/// returned.
pub fn write(s: &str, x: f64, baseline: f64, hand: &Hand, noise: &mut dyn FnMut() -> f64) -> Vec<Written> {
    let mut out = Vec::new();
    let mut cx = x;
    for ch in s.chars() {
        let (g, adv) = glyph(ch);
        // each letter a little different: size, baseline, width
        let k = 1.0 + 0.08 * noise();
        let (h, w) = (hand.height * k, hand.width * k * (1.0 + 0.06 * noise()));
        let base = baseline + 0.04 * hand.height * noise();
        let j = hand.jitter * hand.height;
        let polys: Vec<Poly> = g
            .iter()
            .map(|p| {
                let moved: Poly = p
                    .iter()
                    .map(|&[u, v]| {
                        let (px, py) = (cx + u * w, base - h + v * h);
                        [px + hand.slant * (base - py) + j * noise(), py + j * noise()]
                    })
                    .collect();
                smooth(&moved)
            })
            .collect();
        if !polys.is_empty() {
            let b = bbox(&polys, hand.thick);
            out.push(Written { ch, polys, bbox: b });
        }
        cx += adv * w + hand.gap * hand.width;
    }
    out
}

/// The width `s` takes in `hand` (without jitter).
pub fn width(s: &str, hand: &Hand) -> f64 {
    s.chars().map(|c| glyph(c).1 * hand.width + hand.gap * hand.width).sum::<f64>() - hand.gap * hand.width
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_digit_and_capital_has_strokes_inside_its_box() {
        for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:/-".chars() {
            let (g, adv) = glyph(c);
            assert!(!g.is_empty(), "{c}");
            for p in g.iter().flatten() {
                assert!(p[0] >= -0.01 && p[0] <= adv + 0.21 && p[1] >= -0.01 && p[1] <= 1.06, "{c}: {p:?}");
            }
        }
        assert!(glyph(' ').0.is_empty());
    }

    #[test]
    fn written_characters_run_left_to_right_inside_their_boxes() {
        let hand = Hand { height: 30.0, width: 18.0, slant: 0.2, gap: 0.3, jitter: 0.03, thick: 2 };
        let mut k = 0.0f64;
        let mut noise = || {
            k += 0.37;
            (k.sin() * 1000.0).fract()
        };
        let w = write("07 12345678", 100.0, 200.0, &hand, &mut noise);
        assert_eq!(w.len(), 10);
        for pair in w.windows(2) {
            assert!(pair[0].bbox[0] < pair[1].bbox[0]);
        }
        for c in &w {
            for p in c.polys.iter().flatten() {
                assert!(p[0] >= c.bbox[0] && p[0] <= c.bbox[2] && p[1] >= c.bbox[1] && p[1] <= c.bbox[3]);
            }
            assert!(c.bbox[3] > 195.0 && c.bbox[1] < 175.0);
        }
        assert!((width("12", &hand) - (0.6 * 18.0 + 0.3 * 18.0 + 18.0)).abs() < 1e-9);
    }
}
