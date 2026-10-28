//! VLM plumbing against a stub Ollama server (no model needed).

use super::*;
use crate::testutil::bgr;
use std::io::{BufRead, BufReader, Read, Write as _};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// A local server answering /api/chat with the queued replies; records
/// every (path, body) it was sent.
struct Stub {
    url: String,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
}

fn stub(replies: Vec<Value>) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    std::thread::spawn(move || {
        let mut replies = replies.into_iter();
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { break };
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let mut first = String::new();
            r.read_line(&mut first).unwrap();
            let path = first.split_whitespace().nth(1).unwrap_or("").to_string();
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                if h.trim().is_empty() {
                    break;
                }
                if let Some((k, v)) = h.split_once(':') {
                    if k.eq_ignore_ascii_case("content-length") {
                        len = v.trim().parse().unwrap();
                    }
                }
            }
            let mut body = vec![0u8; len];
            r.read_exact(&mut body).unwrap();
            seen.lock().unwrap().push((path, serde_json::from_slice(&body).unwrap()));
            let content = replies.next().unwrap_or(json!({}));
            let data = serde_json::to_vec(&json!({"message": {"role": "assistant", "content": content.to_string()}})).unwrap();
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", data.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(&data).unwrap();
        }
    });
    Stub { url, requests }
}

fn settings(host: &str, model: &str) -> VlmSettings {
    VlmSettings { host: host.into(), model: model.into(), api_key: None, ..VlmSettings::default() }
}

fn page(h: i32, w: i32) -> Mat {
    bgr(h, w, [255, 255, 255])
}

#[test]
fn audit_request_and_verdict() {
    let s = stub(vec![json!({"id_number_visible": false, "phone_number_visible": true, "signature_visible": false, "details": "phone in comments"})]);
    let client = Ollama::new(settings(&s.url, "m")).unwrap();
    let a = client.audit(&page(2340, 1654)).unwrap();
    assert!(!a.ok && a.findings["details"] == "phone in comments");
    let (path, body) = s.requests.lock().unwrap()[0].clone();
    assert!(path == "/api/chat" && body["model"] == "m" && body["stream"] == false);
    assert!(!body["format"]["required"].as_array().unwrap().is_empty() && body["options"]["temperature"] == 0);
    let sent = base64::engine::general_purpose::STANDARD.decode(body["messages"][0]["images"][0].as_str().unwrap()).unwrap();
    let img = crate::io::decode_bytes(&sent).unwrap();
    assert!(img.rows().max(img.cols()) <= 1288 && img.rows() % 28 == 0 && img.cols() % 28 == 0);
}

#[test]
fn locate_scales_boxes_back() {
    let s = stub(vec![json!({"items": [{"label": "signature", "bbox": [100, 200, 50, 220]}, {"label": "name", "bbox": [0, 0, 10, 10]}]})]);
    let client = Ollama::new(settings(&s.url, "qwen2.5vl:7b")).unwrap(); // pixel boxes
    let boxes = client.locate(&page(2340, 1654)).unwrap();
    assert!(boxes.len() == 1 && boxes[0].label == "signature"); // unknown label dropped
    // sent at 896x1288 (multiples of 28 within 1288); x1 < x0 in the reply gets sorted
    let (sx, sy) = ((1654.0 * 1288.0 / 2340.0 / 28.0f64).floor() * 28.0 / 1654.0, 1288.0 / 2340.0);
    let b = &boxes[0];
    for (got, want) in [b.x0, b.y0, b.x1, b.y1].iter().zip([50.0 / sx, 200.0 / sy, 100.0 / sx, 220.0 / sy]) {
        assert!((got - want).abs() < 1e-6, "{} vs {}", got, want);
    }
}

#[test]
fn qwen3_coords_and_audit_record() {
    let s = stub(vec![json!({"id_number_visible": false, "phone_number_visible": false, "signature_visible": false, "details": ""})]);
    let client = Ollama::new(settings(&s.url, "qwen3-vl:235b")).unwrap();
    assert_eq!(client.s.coords, "norm1000"); // Qwen3-VL boxes are 0..1000
    let a = client.audit(&page(500, 400)).unwrap();
    assert!(a.ok && a.record["model"] == "qwen3-vl:235b" && a.record["image_sent_sha256"].as_str().unwrap().len() == 64);
}

#[test]
fn remote_host_refuses_unredacted_and_sends_key() {
    let client = Ollama::new(VlmSettings { host: CLOUD_HOST.into(), model: String::new(), api_key: Some("k123".into()), ..VlmSettings::default() }).unwrap();
    assert!(!client.local() && client.s.model == DEFAULT_CLOUD_MODEL);
    let e = client.locate(&page(500, 400)).unwrap_err();
    assert!(e.to_string().contains("refusing to send an unredacted form"));
    // redacted pages may go to the cloud, with the key
    assert_eq!((client.chat_url(), client.authorization()), ("https://ollama.com/api/chat".to_string(), Some("Bearer k123".to_string())));
    let local = Ollama::new(VlmSettings { host: "localhost:11434".into(), api_key: Some("k123".into()), ..VlmSettings::default() }).unwrap();
    assert_eq!(local.authorization(), None); // the key never goes to a local server
}

#[test]
fn cloud_without_key_is_rejected() {
    assert!(Ollama::new(VlmSettings { host: CLOUD_HOST.into(), api_key: None, ..VlmSettings::default() }).is_err());
}

#[test]
fn qwen3_boxes_are_on_a_1000_grid() {
    let s = stub(vec![json!({"items": [{"label": "signature", "bbox": [500, 500, 600, 550]}]})]);
    let client = Ollama::new(settings(&s.url, "")).unwrap(); // default local: latest Qwen
    assert!(client.s.model == DEFAULT_LOCAL_MODEL && client.s.coords == "norm1000");
    let b = client.locate(&page(2340, 1654)).unwrap().remove(0);
    for (got, want) in [b.x0, b.y0, b.x1, b.y1].iter().zip([827.0, 1170.0, 992.4, 1287.0]) {
        assert!((got - want).abs() < 2.0, "{} vs {}", got, want);
    }
}
