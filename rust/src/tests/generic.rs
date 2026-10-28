//! Reading forms without a template: label matching and free-text rules.

use super::*;
use crate::testutil::{colour, gray, text};

fn line_probs(t: &str, x0: f32, x1: f32, probs: Option<Vec<f64>>) -> OLine {
    let q = [[x0, 0.0], [x1, 0.0], [x1, 10.0], [x0, 10.0]];
    let n = t.chars().count();
    let pos = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    OLine::new(q, t.into(), 1.0, pos, probs.unwrap_or_else(|| vec![1.0; n]))
}

fn line(t: &str) -> OLine {
    line_probs(t, 0.0, 100.0, None)
}

#[test]
fn date_labels_found() {
    for (t, found) in [
        ("Pate 18-05-2021", true), // misread, but a date follows
        ("bate 17-10-2019", true),
        ("Date", true),
        ("Name of Candidate", false), // "...date"
        ("UNIQUE ESTATE GATE 02 Polling Station", false), // GATE without a date
    ] {
        assert_eq!(!date_labels(&line(t), false).is_empty(), found, "{}", t);
    }
}

#[test]
fn misread_date_on_an_officer_line() {
    assert!(!date_labels(&line("PRESiD Dat"), true).is_empty());
    assert!(date_labels(&line("PRESiD Dat"), false).is_empty());
}

#[test]
fn signature_labels() {
    for (t, found) in [
        ("Signature:", true),
        ("AGNES KADURAsinature AK.", true),
        ("ESTHERAZOI MA Shature", true),
        ("Signatu", true), // cut off by a stamp
        ("Reasons for Refusal to Sign (if any)", false),
        ("THE COUNTING AND SIGNING WAS SMOOTH", false),
    ] {
        assert_eq!(!sig_labels(&line(t)).is_empty(), found, "{}", t);
    }
}

#[test]
fn dates_written_without_separators() {
    for (d, is_date) in [("15122020", true), ("1512020", true), ("26012017", true), ("25039403", false), ("33086703", false), ("2748235", false)] {
        assert_eq!(date_digits(d), is_date, "{}", d);
    }
}

fn group1(r: &Regex, t: &str) -> Vec<String> {
    r.captures_iter(t).map(|c| c.unwrap().get(1).unwrap().as_str().to_string()).collect()
}

#[test]
fn free_text_numbers() {
    assert_eq!(group1(&FREE_ID_RE, "PO NAME 33086703 15-12-2020 total 231"), ["33086703"]);
    let pre = ID_PREFIX_RE.captures("1D. 2≤03qy0 7").unwrap().unwrap();
    assert!(pre.get(1).unwrap().as_str().chars().filter(|c| c.is_ascii_digit()).count() >= 4);
    assert_eq!(group1(&FREE_PHONE_RE, "call 0724 441656 or +254 712345678"), ["0724 441656", "+254 712345678"]);
}

#[test]
fn char_positions_and_confidence() {
    let ln = line_probs("ABC Kaad", 0.0, 80.0, Some(vec![1.0, 1.0, 1.0, 1.0, 0.7, 1.0, 0.4, 0.8]));
    let (x0, x1) = ln.find(&Regex::new("Kaad").unwrap()).unwrap();
    assert!(35.0 < x0 && x0 < 45.0 && x1 > 75.0);
    assert_eq!(ln.chars_in(40.0, 80.0).iter().map(|c| c.0).collect::<String>(), "Kaad");
    assert!(ln.fuzzy("signature", 3).is_empty());
}

#[test]
fn officer_name_matching() {
    assert!(name_match("MWICHEE", "MWISHEE"));
    assert!(name_match("KIPLIMO", "KIPLI"));
    assert!(!name_match("VOTES", "MWISHEE"));
    assert!(!name_match("MAO", "MAO")); // too short to tell
}

#[test]
fn printed_fill_is_recognised() {
    assert!(x_fill("XXXXXXXXXXXXXX"));
    assert!(x_fill("xxxxxxxx xxxx"));
    assert!(!x_fill("XAVIER"));
    assert!(has(&PRINTED_LINE, "Tel. Contact"));
    assert!(!has(&PRINTED_LINE, "Signature: Jutiq"));
}

#[test]
fn ocr_reads_printed_labels() {
    if !ocr::available() {
        eprintln!("OCR models not installed: skipped");
        return;
    }
    let mut img = crate::testutil::bgr(120, 900, [255, 255, 255]);
    text(&mut img, "ID/Passport No.    Tel. Contact    Signature", (20, 70), 1.0, colour(0.0), 2, true);
    let lines = ocr::read(&img, 0.3).unwrap();
    let t = lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join(" ").to_lowercase();
    assert!(t.contains("passport") && t.contains("signature"), "{}", t);
}

fn read(t: &str, x0: f32, x1: f32, p: f64) -> OLine {
    let q = [[x0, 0.0], [x1, 0.0], [x1, 34.0], [x0, 34.0]];
    let n = t.chars().count();
    OLine::new(q, t.into(), p, (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect(), vec![p; n])
}

#[test]
fn signatures_are_what_ocr_cannot_read() {
    for (page, band, is_legible) in [
        ("MAPOA", "MAoA", true), // handwriting read alike at both scales
        ("BrLLoi", "BrLLoi", true),
        ("1015", "dol", false), // a signature: read differently each time
        ("uNGu中", "uNGu母", false), // strokes read as CJK
        ("", "", false), // not read at all
        ("ID", "", true), // one confident reading is enough
    ] {
        let a = if page.is_empty() { vec![] } else { vec![read(page, 0.0, 120.0, if page == "ID" { 0.95 } else { 0.5 })] };
        let b = if band.is_empty() { vec![] } else { vec![read(band, 0.0, 120.0, 0.5)] };
        let passes = vec![a.iter().collect::<Vec<_>>(), b.iter().collect::<Vec<_>>()];
        assert_eq!(legible(&readings(&passes, (0.0, 0.0, 120.0, 34.0)), 120.0), is_legible, "{} / {}", page, band);
    }
}

#[test]
fn cursive_reading() {
    let r = |t: &str, p: f64| (t.to_string(), vec![p; t.chars().count()]);
    assert!(cursive(&[r("Kaagd", 0.8), r("Kaad", 0.8)]));
    assert!(!cursive(&[r("MAPOA", 0.6), r("MAoA", 0.6)]));
    assert!(!cursive(&[r("15-12-2020", 0.9), (String::new(), vec![])]));
    assert!(!cursive(&[r("Four", 0.6), r("Fur", 0.7)])); // "FOUR" in capitals
}

#[test]
fn a_struck_line_is_not_a_signature() {
    let mut lab = Mat::new_rows_cols_with_default(60, 200, opencv::core::CV_32SC1, opencv::core::Scalar::all(0.0)).unwrap();
    imgproc::line(&mut lab, Point::new(5, 50), Point::new(190, 10), opencv::core::Scalar::all(1.0), 3, imgproc::LINE_8, 0).unwrap();
    assert!(straight(&util::labels_vec(&lab).unwrap(), 200, &[1], TEXT_H).unwrap());
    lab.set_to_def(&opencv::core::Scalar::all(0.0)).unwrap();
    imgproc::ellipse(&mut lab, Point::new(60, 30), opencv::core::Size::new(40, 18), 0.0, 0.0, 300.0, opencv::core::Scalar::all(2.0), 3, imgproc::LINE_8, 0).unwrap();
    imgproc::line(&mut lab, Point::new(20, 40), Point::new(180, 25), opencv::core::Scalar::all(2.0), 3, imgproc::LINE_8, 0).unwrap();
    assert!(!straight(&util::labels_vec(&lab).unwrap(), 200, &[2], TEXT_H).unwrap());
}

#[test]
fn results_end_only_at_the_declaration() {
    for (t, found) in [
        ("Declaration", true),
        ("We, the undersigned, being present when the results of the count were announced", true),
        ("COLLATION OF PRESIDENTIAL ELECTION RESULTS AT THE CONSTITUENCY TALLYING CENTRE", false),
        ("Total valid votes cast", false),
    ] {
        assert_eq!(has(&DECL_RE, t), found, "{}", t);
    }
}

#[test]
fn a_templated_form_is_never_read_without_a_template() {
    use crate::align::AlignSettings;
    use crate::pipeline::{redact_page, PageOptions};
    if !ocr::available() {
        eprintln!("OCR models not installed: skipped");
        return;
    }
    let t = crate::testutil::bare_template("form34b_block", "34B", false, 0);
    let img = gray(400, 300, 255);
    let run = |form: &str| {
        let o = PageOptions { form: Some(form.into()), generic: true, ..PageOptions::default() };
        redact_page(&img, std::slice::from_ref(&t), &AlignSettings::default(), &MaskSettings::default(), &mut PyRandom::new(0), &o).unwrap()
    };
    let r = run("34B");
    assert!(!r.ok && r.template != "generic");
    assert_eq!(run("35A").template, "generic");
}
