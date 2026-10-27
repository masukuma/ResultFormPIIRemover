import cv2
import numpy as np

from resultform_pii import kiems


def test_parse_kiems_file_name():
    n = kiems.parse_name("1_34_A_007029014000102_F230450M00203609_20220810_132155")
    assert n.form == "34A" and n.station == "007029014000102" and n.kit == "F230450M00203609"
    assert n.captured_at == "2022-08-10T13:21:55"
    d = n.to_dict()
    assert (d["county"], d["constituency"], d["ward"], d["centre"], d["stream"]) == ("007", "029", "0140", "001", "02")
    assert d["election_type"] == "presidential"
    assert kiems.parse_name("p-3") is None


def test_parse_qr_payloads():
    q = kiems.parse_qr("00702901400010234")
    assert q["station"] == "007029014000102" and q["form"] == "34A"
    p = kiems.parse_qr("GE2022040228FORM34B03")
    assert p == {"election": "GE", "year": "2022", "county": "040", "constituency": "228", "form": "34B", "page": 3}
    assert kiems.parse_qr("hello") == {}


def test_check_flags_mismatched_station():
    n = kiems.parse_name("1_34_A_007029014000102_F230450M00203609_20220810_132155")
    assert kiems.check(n, kiems.parse_qr("00702901400010234")) == ""
    assert "station" in kiems.check(n, kiems.parse_qr("00703001490240134"))
    assert kiems.check(n, {}) == ""


def test_decode_qr_from_header():
    qr = cv2.QRCodeEncoder.create().encode("00702901400010234")
    qr = cv2.resize(qr, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    page = np.full((2340, 1654), 255, np.uint8)
    page[60:60 + qr.shape[0], 1300:1300 + qr.shape[1]] = qr
    assert kiems.decode_qr(page) == "00702901400010234"
