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
    // 2013's officer label, which a signature's search reaches from below
    assert!(has(&PRINTED_LINE, "ID No. / Passport:") && has(&PRINTED_LINE, "ID No./ Passport:"));
    assert!(!has(&PRINTED_LINE, "ID No. / Passport: 28' @rs0&n?"));
}

#[test]
fn a_mobile_polling_station_is_no_phone_label() {
    // Tiaty and Isiolo: the code after a mobile station's name was masked as
    // a phone number
    for t in ["LOMUKEMOBILE(039)", "KWOLOYOYMOBILE.(04O)", "KULA(MOBILE) (054)", "SITOT MOBILE (081)"] {
        assert!(!has(&TEL_RE, t), "{}", t);
    }
    for t in ["Mobile No.", "MOBILE NO:", "Mobile:", "Mobile Number", "Tel/Mobile", "Tel. Contact"] {
        assert!(has(&TEL_RE, t), "{}", t);
    }
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
        // 2013's Form 34: the title, read with and without its spaces, then the section
        ("DECLARATION OF PRESIDENTIAL ELECTION RESULTS AT A POLLING STATION", false),
        ("DECLARATIONOFPRESIDENTIALELECTIONRESULTSATAPOLLINGSTATION", false),
        ("8. Declaration", true),
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

#[test]
fn readings_agree_across_any_two_passes() {
    let r = |t: &str| (t.to_string(), vec![0.5; t.chars().count()]);
    assert!(legible(&[r(""), r("MAPOA"), r("MAoA")], 120.0));
    assert!(!legible(&[r(""), r("1015"), r("dol")], 120.0));
}

/// A comments area: a scribble signed before a date, a piece of it written
/// apart above, and "08" handwritten on the same line that OCR missed.
fn comments_area() -> Layout {
    let (w, h) = (900, 260);
    let mut g = gray(h, w, 255);
    let (x0, yc) = (200.0, 150.0);
    let pts: Vec<(i32, i32)> = (0..=400)
        .map(|k| k as f64 * 10.0 * std::f64::consts::PI / 400.0)
        .map(|t| ((x0 + 4.0 * t - 10.0 * t.sin()) as i32, (yc + 13.0 * t.cos() - t) as i32))
        .collect();
    crate::testutil::polyline(&mut g, &pts, colour(0.0), 3);
    let top = pts.iter().filter(|p| p.0 >= 280).map(|p| p.1).min().unwrap();
    imgproc::ellipse(&mut g, Point::new(300, top - 13), opencv::core::Size::new(20, 7), 0.0, 0.0, 360.0, colour(0.0), 2, imgproc::LINE_8, 0).unwrap();
    text(&mut g, "08", (420, 150), 1.0, colour(0.0), 3, false);
    let date = OLine::new([[600.0, 60.0], [800.0, 60.0], [800.0, 190.0], [600.0, 190.0]], "09-08-2022".into(), 0.95, (0..10).map(|i| (i as f64 + 0.5) / 10.0).collect(), vec![0.95; 10]);
    Layout {
        frame: Frame { k: 0, skew: 0.0, scale: 1.0, m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], size: (w, h), text_h: TEXT_H },
        image: crate::testutil::to_bgr(&g),
        gray: g,
        lines: vec![date],
        fields: vec![],
        protected: vec![],
        stamps: vec![],
        problems: vec![],
        notes: vec![],
        officer_names: BTreeSet::new(),
    }
}

fn signature_spans(l: &Layout) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = l.fields.iter().filter(|f| f.kind == "signature").map(|f| (f.label_box.0, f.label_box.2)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn in_comments_a_signature_needs_a_joined_stroke() {
    let mut l = comments_area();
    free_signatures(&mut l, 0.0, &[0, 1], false).unwrap();
    let loose = signature_spans(&l);
    assert!(loose.iter().any(|s| s.0 > 400.0), "digits OCR missed pass for a signature on other forms: {:?}", loose);
    let mut l = comments_area();
    free_signatures(&mut l, 0.0, &[0, 1], true).unwrap();
    let strict = signature_spans(&l);
    assert!(strict.iter().all(|s| s.1 < 400.0), "{:?}", strict); // the digits are left alone
    assert!(strict.iter().any(|s| s.0 < 210.0 && s.1 > 320.0), "{:?}", strict); // the scribble
    assert_eq!(strict.len(), 2, "{:?}", strict); // and the piece above it, too short alone
}

#[test]
fn a_letter_read_doubtfully_off_a_name_is_left_to_the_signature_beside_it() {
    // "OMAR", then the signature's first loop read as a "U"
    let mut l = comments_area();
    l.lines.push(OLine::new([[40.0, 90.0], [250.0, 90.0], [250.0, 180.0], [40.0, 180.0]], "OMARU".into(), 0.8, (0..5).map(|i| (i as f64 + 0.5) / 5.0).collect(), vec![0.95, 0.9, 0.95, 0.95, 0.25]));
    free_signatures(&mut l, 0.0, &[0, 1], true).unwrap();
    let found = signature_spans(&l);
    assert!(found.iter().any(|s| s.0 < 225.0 && s.1 > 320.0), "the whole scribble, from where the name ends: {:?}", found);
}

/// Printed text at (x0, y0)-(x1, y1), its characters evenly spaced.
fn at(t: &str, x0: f32, y0: f32, x1: f32, y1: f32) -> OLine {
    let n = t.chars().count();
    OLine::new([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], t.into(), 0.95, (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect(), vec![0.95; n])
}

fn page(lines: Vec<OLine>) -> Layout {
    let g = gray(2100, 1500, 255);
    Layout {
        frame: Frame { k: 0, skew: 0.0, scale: 1.0, m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], size: (1500, 2100), text_h: TEXT_H },
        image: crate::testutil::to_bgr(&g),
        gray: g,
        lines,
        fields: vec![],
        protected: vec![],
        stamps: vec![],
        problems: vec![],
        notes: vec![],
        officer_names: BTreeSet::new(),
    }
}

#[test]
fn a_label_of_several_words_is_read_to_its_end() {
    // "ID No. / Passport:", and the same with "Passport" misread
    for t in ["ID No. / Passport:", "ID No. / Passpdrt:"] {
        // the page's print runs on to the right, on the line above
        let mut l = page(vec![at(t, 300.0, 1955.0, 510.0, 1990.0), at("Polling Station,", 1135.0, 1795.0, 1313.0, 1827.0)]);
        labels(&mut l, &BTreeSet::new(), 0.0);
        assert!(l.problems.is_empty(), "{}: {:?}", t, l.problems);
        let ids: Vec<f64> = l.fields.iter().filter(|f| f.kind == "id").map(|f| f.zones[0].rect[0]).collect();
        assert_eq!(ids.len(), 1, "{}", t);
        assert!(ids[0] > 480.0, "{}: the field starts after the whole label, not at {}", t, ids[0]);
    }
}

#[test]
fn a_label_read_with_the_writing_after_it_reaches_no_further_than_print() {
    // Baringo North_B p29: OCR boxed the ID with its label, 50 px high; a
    // field that high reached the "Date:" below and masked it
    let mut l = page(vec![at("ID No. / Passport: 28' @rs0&n?", 303.0, 1956.0, 737.0, 2006.0), at("Date:", 924.0, 2008.0, 1000.0, 2039.0)]);
    labels(&mut l, &BTreeSet::new(), 0.0);
    let z = &l.fields.iter().find(|f| f.kind == "id").unwrap().zones[0];
    let reach = z.rect[1] + z.rect[3] + z.search.unwrap()[1];
    assert!(reach < 2016.0, "reaches to y = {}, into 'Date:'", reach);
}

#[test]
fn a_label_read_twice_makes_one_field() {
    // Eldama Ravine_B p5: the officers' lines, read again zoomed in, give the
    // label twice; the first read's garbled "Passport" ends the label at
    // "No.", and the second read's "Passport" was taken for the next label
    let title = at("PRESIDENTIAL ELECTION RESULTS AT THE POLLING STATION", 300.0, 270.0, 1300.0, 295.0);
    let mut l = page(vec![title, at("ID No. / Rasqp0rt", 322.0, 2048.0, 534.0, 2075.0), at("ID No. / Passport:", 324.0, 2049.8, 538.0, 2073.8)]);
    labels(&mut l, &BTreeSet::new(), 0.0);
    assert!(l.problems.is_empty(), "{:?}", l.problems);
    let ids: Vec<_> = l.fields.iter().filter(|f| f.kind == "id").collect();
    assert_eq!(ids.len(), 1);
    assert!(ids[0].zones[0].rect[0] > 515.0, "the field starts after the label, at x = {}", ids[0].zones[0].rect[0]);
}

#[test]
fn passport_misread_after_id_no_is_still_the_label() {
    let title = at("PRESIDENTIAL ELECTION RESULTS AT THE POLLING STATION", 300.0, 270.0, 1300.0, 295.0);
    for t in ["ID No, I Passport:", "IDNo.1Passport:", "ID No. ypassport:", "IDNo.Passoort:", "ID No. / Rassport:", "ID No./P&ssport:"] {
        let mut l = page(vec![title.clone(), at(t, 320.0, 1980.0, 530.0, 2004.0)]);
        labels(&mut l, &BTreeSet::new(), 0.0);
        assert!(l.problems.is_empty(), "{}: {:?}", t, l.problems);
        let z = &l.fields.iter().find(|f| f.kind == "id").unwrap().zones[0];
        assert!(z.rect[0] > 515.0, "{}: field from x = {}", t, z.rect[0]);
    }
    // an ID read as letters is no "Passport": the field starts after "ID No."
    let mut l = page(vec![title, at("ID No. Rkcohaa:", 320.0, 1980.0, 530.0, 2004.0)]);
    labels(&mut l, &BTreeSet::new(), 0.0);
    assert!(l.fields.iter().find(|f| f.kind == "id").unwrap().zones[0].rect[0] < 420.0);
}

#[test]
fn an_officers_stacked_lines_are_all_the_declarations() {
    // 2013's Form 34: the role, then name, ID, signature and date one under another
    let lines = vec![
        at("Presiding Officer:", 303.0, 1886.0, 506.0, 1912.0),
        at("Name:", 305.0, 1924.0, 397.0, 1948.0),
        at("ID No. / Passport:", 302.0, 1955.0, 512.0, 1992.0),
        at("Signature:", 303.0, 1998.0, 434.0, 2028.0),
        at("Date:", 909.0, 1982.0, 1085.0, 2033.0),
    ];
    // nine lines below the 'Declaration' label falls short of the signature
    let end = officer_lines_end(&lines, 1680.0 + 9.0 * TEXT_H);
    assert!(end > 2028.0, "the signature line is in: ends at {}", end);
    // and the run stops at a gap
    let mut far = lines.clone();
    far.push(at("Date:", 300.0, 2300.0, 400.0, 2330.0));
    assert!(officer_lines_end(&far, 1680.0 + 9.0 * TEXT_H) < 2100.0);
}

#[test]
fn a_page_opening_with_an_officers_lines_goes_on_with_the_declaration() {
    let top = at("Deputy Presiding Officer:", 283.0, 89.0, 573.0, 117.0);
    assert!(goes_on_with_declaration(&[top.clone()], 2040.0));
    // an officer named only lower down (a declaration on the page) doesn't
    assert!(!goes_on_with_declaration(&[at("Presiding Officer:", 303.0, 1886.0, 506.0, 1912.0)], 2125.0));
    // nor a stamp's lettering
    assert!(!goes_on_with_declaration(&[at("OFFICER", 283.0, 89.0, 400.0, 117.0)], 2040.0));
    // nor a blurred front page, its 'Declaration' and counts misread: the title
    // is no officer, and its results are still words of results
    let front = [
        at("PRESIDENTIALELECTION RESULTSAT THEPOLLINGSTATION", 463.0, 270.0, 1300.0, 295.0),
        at("Totan Nunb Ke ted BallotPapes", 314.0, 1025.0, 740.0, 1050.0),
        at("Declaatloa", 281.0, 1197.0, 420.0, 1222.0),
    ];
    assert!(!goes_on_with_declaration(&front, 2411.0));
    let mut back = vec![top, at("Totan Nunb Ke ted BallotPapes", 314.0, 1025.0, 740.0, 1050.0)];
    assert!(!goes_on_with_declaration(&back, 2040.0));
    // comments at the foot of a back page may speak of votes
    back[1] = at("There were no disputed votes", 230.0, 1700.0, 900.0, 1730.0);
    assert!(goes_on_with_declaration(&back, 2040.0));
    // nor is an agent's name read "VOTU" a word of results
    back[1] = at("VOTU", 200.0, 545.0, 330.0, 573.0);
    assert!(goes_on_with_declaration(&back, 2040.0));
    // the deputy's label garbled, or unread but for his ID line, or the
    // page opening with the agents (2013 back pages, Baringo)
    for opener in ["DeputyPresigingOfficer:", "ID No. / Passport:", "CandidatesorCandidates'Agents"] {
        assert!(goes_on_with_declaration(&[at(opener, 276.0, 114.0, 562.0, 139.0)], 2040.0), "{}", opener);
    }
}

/// The 2013 Form 34's agents' table, its heading row at y = 465: number,
/// name, signature, reasons for refusal; a scanner's streak through the
/// last heading and the page's edge as vertical "rules".
fn agents_2013(reasons: &str) -> (Layout, Vec<Line>, Vec<Line>) {
    let mut lines = vec![at("Name", 217.0, 451.0, 293.0, 479.0), at("Signature", 581.0, 452.0, 692.0, 479.0)];
    if !reasons.is_empty() {
        lines.push(at(reasons, 855.0, 450.0, 1153.0, 478.0));
    }
    lines.push(at("1.", 135.0, 490.0, 160.0, 515.0));
    let v = |x: f64, lo: f64, hi: f64| Line { pos: x, slope: 0.0, lo, hi };
    let hs = vec![Line { pos: 442.0, slope: 0.0, lo: 126.0, hi: 1317.0 }, Line { pos: 484.0, slope: 0.0, lo: 126.0, hi: 1317.0 }];
    let vs = vec![v(1.0, 0.0, 2000.0), v(126.0, 440.0, 1600.0), v(209.0, 440.0, 1600.0), v(571.0, 440.0, 1600.0), v(839.0, 440.0, 1600.0), v(1105.0, 300.0, 1700.0), v(1312.0, 440.0, 1600.0)];
    (page(lines), hs, vs)
}

#[test]
fn a_table_without_id_or_phone_columns_has_every_other_heading_read() {
    let sig: BTreeSet<String> = ["signature".to_string()].into();
    for reasons in ["Reasons for refusal to sign", "Reasonsforrefusaltoign"] {
        let (l, hs, vs) = agents_2013(reasons);
        assert!(headings_read(&l, &hs, &vs, 465.0, &sig), "{}", reasons);
    }
    // a column whose heading wasn't read may be the ID's or the phone's
    let (l, hs, vs) = agents_2013("");
    assert!(!headings_read(&l, &hs, &vs, 465.0, &sig));
    // a phone column's heading read, but no phone column found
    let (l, hs, vs) = agents_2013("Tel. Contact");
    assert!(!headings_read(&l, &hs, &vs, 465.0, &sig));
    // a heading of unknown words
    let (l, hs, vs) = agents_2013("Tei. Contaci");
    assert!(!headings_read(&l, &hs, &vs, 465.0, &sig));
    // a stamp over the headings may hide one (and the rule beside it)
    let (mut l, hs, vs) = agents_2013("Reasons for refusal to sign");
    l.stamps.push(crate::stamps::Stamp { cx: 900.0, cy: 480.0, r: 70.0, coverage: 0.8, how: "ring".into() });
    assert!(!headings_read(&l, &hs, &vs, 465.0, &sig));
    // a party column: the agents' table since 2017, which has ID and phone columns
    let (l, hs, vs) = agents_2013("Party Name/Independent Candidate");
    assert!(!headings_read(&l, &hs, &vs, 465.0, &sig));
}

#[test]
fn an_agents_section_is_found_below_the_results() {
    let lines = [at("Agents or Candidates (if present)", 280.0, 1360.0, 700.0, 1385.0)];
    assert!(agents_section(&lines, 1200.0));
    // 2013's back page heads it the other way round
    assert!(agents_section(&[at("Candidates or Candidates' Agents", 177.0, 352.0, 558.0, 381.0)], 0.0));
    // above the declaration it is someone else's
    assert!(!agents_section(&lines, 1500.0));
}

#[test]
fn a_stamps_lettering_read_into_a_name_names_no_officer() {
    // KIAMBU (117) p15: the stamp's "PRESIDING" read with the name under it
    let lines = vec![
        at("Presiding Officer:", 326.0, 1924.0, 531.0, 1952.0),
        at("PRESiDINGNSOZOGZ", 631.0, 1953.0, 933.0, 1987.0),
        at("Name:", 325.0, 1960.0, 410.0, 1991.0),
        at("ID No. / Passport:", 325.0, 1996.0, 534.0, 2031.0),
        at("Signature:.", 326.0, 2041.0, 452.0, 2068.0),
    ];
    let stamp = crate::stamps::Stamp::new(708.0, 1959.0, 93.0, 1.0, "ring");
    let mut l = page(lines.clone());
    l.stamps.push(stamp.clone());
    declaration(&mut l, 1900.0, 2100.0);
    assert!(l.problems.is_empty(), "{:?}", l.problems);
    assert_eq!(l.fields.iter().filter(|f| f.kind == "signature").count(), 1);
    // with no stamp found there, it is an officer's line without a signature line
    let mut l = page(lines.clone());
    declaration(&mut l, 1900.0, 2100.0);
    assert!(l.problems.iter().any(|p| p.contains("2 officer(s) named but 1")), "{:?}", l.problems);
    // the printed label, in small letters, names one even under a stamp
    let mut l = page(vec![at("Deputy Presiding Officer:", 326.0, 1924.0, 600.0, 1952.0), lines[4].clone()]);
    l.stamps.push(crate::stamps::Stamp::new(450.0, 1938.0, 93.0, 1.0, "ring"));
    l.lines.push(at("Presiding Officer:", 326.0, 1824.0, 531.0, 1852.0));
    declaration(&mut l, 1800.0, 2100.0);
    assert!(l.problems.iter().any(|p| p.contains("2 officer(s) named but 1")), "{:?}", l.problems);
}

#[test]
fn a_last_signature_line_takes_the_writing_under_it() {
    // MOYALE (2) p117: signed under the line, over the security stripe, with
    // nothing written below but the stamp's "OFFICER"
    let officer = vec![
        at("Presiding Officer:", 297.0, 1880.0, 500.0, 1906.0),
        at("Name:", 297.0, 1920.0, 375.0, 1946.0),
        at("ID No. / Passport:", 297.0, 1958.0, 500.0, 1986.0),
        at("Signature:", 297.0, 1994.0, 415.0, 2020.0),
        at("Date:", 915.0, 1994.0, 980.0, 2020.0),
    ];
    let reach = |l: &Layout| hang(l, l.fields.iter().find(|f| f.kind == "signature").unwrap());
    let mut l = page(officer.clone());
    l.lines.push(at("OFFICER", 600.0, 2025.0, 730.0, 2055.0));
    l.stamps.push(crate::stamps::Stamp::new(670.0, 1990.0, 100.0, 1.0, "ring"));
    declaration(&mut l, 1850.0, 2100.0);
    assert!(reach(&l) >= 0.9 * TEXT_H, "{}", reach(&l));
    // the next officer's line under it: its writing is that line's
    let mut l = page(officer.clone());
    l.lines.push(at("Deputy Presiding Officer:", 297.0, 2040.0, 600.0, 2066.0));
    declaration(&mut l, 1850.0, 2100.0);
    assert_eq!(reach(&l), 0.0);
}

#[test]
fn printed_text_is_masked_where_its_characters_are() {
    // Tiaty p19: OCR boxed "Signature:" with the signature after it (to
    // x = 595) and read the label alone
    let mut pos: Vec<f64> = (0..10).map(|i| 0.03 + 0.04 * i as f64).collect();
    pos.push(0.5);
    let label = OLine::new([[306.0, 2022.0], [595.0, 2022.0], [595.0, 2064.0], [306.0, 2064.0]], "Signature: ".into(), 0.96, pos, vec![0.96; 11]);
    let heading = at("Signature", 586.0, 448.0, 696.0, 475.0);
    let (mask, _) = printed(&page(vec![label, heading])).unwrap();
    let px = |x: i32, y: i32| *mask.at_2d::<u8>(y, x).unwrap();
    assert_eq!(px(330, 2043), 255);
    assert_eq!(px(520, 2043), 0, "the signature is no print");
    assert_eq!((px(590, 461), px(692, 461)), (255, 255));
    // Luanda p33: the signature's first stroke read as a bracket
    let mut pos: Vec<f64> = (0..10).map(|i| 0.03 + 0.04 * i as f64).collect();
    pos.push(0.8);
    let label = OLine::new([[295.0, 1914.0], [623.0, 1914.0], [623.0, 1972.0], [295.0, 1972.0]], "signature:)".into(), 0.93, pos, vec![0.93; 11]);
    let (mask, _) = printed(&page(vec![label])).unwrap();
    assert_eq!((*mask.at_2d::<u8>(1943, 330).unwrap(), *mask.at_2d::<u8>(1943, 520).unwrap()), (255, 0));
}

#[test]
fn a_word_read_on_an_officers_signature_line_was_written() {
    // Githunguri p140: the deputy's signature read as "No", the printed
    // heading of the agents' table
    let mut l = page(vec![at("Signature:", 287.0, 208.0, 407.0, 240.0), at("No", 491.0, 196.0, 637.0, 232.0), at("No.", 100.0, 448.0, 140.0, 475.0), at("Date:", 870.0, 207.0, 944.0, 237.0)]);
    l.fields.push(Field { kind: "signature".into(), zones: vec![Zone::new("signature@403,222", [403.0, 196.5, 462.0, 44.2], "signature")], source: "declaration".into(), label: "'Signature' label".into(), label_box: (403.0, 205.0, 865.0, 239.0) });
    let (mask, _) = printed(&l).unwrap();
    let px = |x: i32, y: i32| *mask.at_2d::<u8>(y, x).unwrap();
    assert_eq!(px(560, 214), 0, "the signature is no print");
    assert_eq!((px(300, 224), px(120, 461), px(900, 222)), (255, 255, 255));
}
