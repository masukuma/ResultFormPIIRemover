use super::*;
use crate::testutil::gray;

#[test]
fn parse_kiems_file_name() {
    let n = parse_name("1_34_A_007029014000102_F230450M00203609_20220810_132155").unwrap();
    assert!(n.form == "34A" && n.station == "007029014000102" && n.kit == "F230450M00203609");
    assert_eq!(n.captured_at, "2022-08-10T13:21:55");
    let d = n.to_value();
    assert_eq!((&d["county"], &d["constituency"], &d["ward"], &d["centre"], &d["stream"]), (&json!("007"), &json!("029"), &json!("0140"), &json!("001"), &json!("02")));
    assert_eq!(d["election_type"], "presidential");
    assert!(parse_name("p-3").is_none());
}

#[test]
fn parse_qr_payloads() {
    let q = parse_qr("00702901400010234");
    assert!(q["station"] == "007029014000102" && q["form"] == "34A");
    let p = parse_qr("GE2022040228FORM34B03");
    assert_eq!(p, json!({"election": "GE", "year": "2022", "county": "040", "constituency": "228", "form": "34B", "page": 3}));
    assert_eq!(parse_qr("hello"), json!({}));
}

#[test]
fn check_flags_mismatched_station() {
    let n = parse_name("1_34_A_007029014000102_F230450M00203609_20220810_132155").unwrap();
    assert_eq!(check(Some(&n), &parse_qr("00702901400010234")), "");
    assert!(check(Some(&n), &parse_qr("00703001490240134")).contains("station"));
    assert_eq!(check(Some(&n), &json!({})), "");
}

#[test]
fn decode_qr_from_header() {
    let mut enc = opencv::objdetect::QRCodeEncoder::create_def().unwrap();
    let mut qr = Mat::default();
    enc.encode("00702901400010234", &mut qr).unwrap();
    let mut big = Mat::default();
    opencv::imgproc::resize(&qr, &mut big, opencv::core::Size::new(0, 0), 8.0, 8.0, opencv::imgproc::INTER_NEAREST).unwrap();
    let mut page = gray(2340, 1654, 255);
    let mut roi = Mat::roi_mut(&mut page, opencv::core::Rect::new(1300, 60, big.cols(), big.rows())).unwrap();
    big.copy_to(&mut roi).unwrap();
    assert_eq!(decode_qr(&page).as_deref(), Some("00702901400010234"));
}
