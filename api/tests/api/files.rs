//! Uploads: told by their first bytes, limited in size, zips checked,
//! stored by id, never served back.

use crate::common::{TestServer, jpeg, problem};
use serde_json::Value;
use std::io::Write;

fn zip_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, data) in entries {
        w.start_file(*name, opts).unwrap();
        w.write_all(data).unwrap();
    }
    w.finish().unwrap().into_inner()
}

fn sha256(b: &[u8]) -> String {
    rfpii_api::ids::sha256_hex(b)
}

#[tokio::test(flavor = "multi_thread")]
async fn uploads_are_told_by_their_first_bytes_and_stored_by_id() {
    let t = TestServer::start(0, |c| c.max_upload_bytes = 4096).await;
    let (account, key) = t.account("free", false).await;
    let pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n".to_vec();
    let r = t.upload_raw(&key, "../../etc/34A form.pdf", pdf.clone()).await;
    assert_eq!(r.status(), 201);
    let loc = r.headers()["location"].to_str().unwrap().to_string();
    let f: Value = r.json().await.unwrap();
    assert_eq!((f["kind"].as_str(), f["bytes"].as_u64(), f["sha256"].as_str()), (Some("pdf"), Some(pdf.len() as u64), Some(sha256(&pdf).as_str())));
    assert_eq!(loc, format!("/v1/files/{}", f["id"].as_str().unwrap()));
    // a name with a space is not kept; the file is stored by its id
    assert!(f["name"].is_null());
    let stored = t.dir.join("data/files").join(&account).join(f["id"].as_str().unwrap());
    assert_eq!(std::fs::read(&stored).unwrap(), pdf);
    assert_eq!(std::fs::read_dir(stored.parent().unwrap()).unwrap().count(), 1);

    // the raw body, named in the query
    let png = b"\x89PNG\r\n\x1a\n rest of a png".to_vec();
    let r = t.client.post(t.url("/v1/files?name=scan.png")).bearer_auth(&key).header("content-type", "image/png").body(png).send().await.unwrap();
    assert_eq!(r.status(), 201);
    let f2: Value = r.json().await.unwrap();
    assert_eq!((f2["kind"].as_str(), f2["name"].as_str()), (Some("png"), Some("scan.png")));
    let r = t.client.post(t.url("/v1/files")).bearer_auth(&key).body(b"MM\0*tiff".to_vec()).send().await.unwrap();
    assert_eq!(r.json::<Value>().await.unwrap()["kind"], "tiff");

    // what isn't a PDF, image or zip, whatever it is called
    let p = problem(t.upload_raw(&key, "form.pdf", b"<html>a web page</html>".to_vec()).await, 415).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-unsupported-file-type"));
    // too large, in either form
    let mut big = jpeg();
    big.resize(5000, 0);
    let p = problem(t.upload_raw(&key, "big.jpg", big.clone()).await, 413).await;
    assert_eq!(p["limit_bytes"], 4096);
    problem(t.client.post(t.url("/v1/files")).bearer_auth(&key).body(big).send().await.unwrap(), 413).await;
    problem(t.client.post(t.url("/v1/files")).bearer_auth(&key).body(Vec::new()).send().await.unwrap(), 400).await;
    // nothing is left behind by refused uploads
    assert_eq!(std::fs::read_dir(t.dir.join("data/tmp")).unwrap().count(), 0);

    // described, never served; another account can't see it
    let id = f["id"].as_str().unwrap();
    let got: Value = t.get(&key, &format!("/v1/files/{id}")).await.json().await.unwrap();
    assert!(got["status"] == "available" && got.get("content").is_none() && got.get("url").is_none());
    let (_, other) = t.account("free", false).await;
    problem(t.get(&other, &format!("/v1/files/{id}")).await, 404).await;
    let list: Value = t.get(&key, "/v1/files").await.json().await.unwrap();
    assert_eq!(list["data"].as_array().unwrap().len(), 3);

    // deleted: gone from disk, and no job can use it
    let r = t.delete(&key, &format!("/v1/files/{id}")).await;
    assert_eq!(r.json::<Value>().await.unwrap()["status"], "deleted");
    assert!(!stored.exists());
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn zips_are_checked_for_entries_size_and_paths() {
    let t = TestServer::start(0, |c| {
        c.max_zip_entries = 3;
        c.max_zip_uncompressed_bytes = 10_000;
    })
    .await;
    let (_, key) = t.account("free", false).await;
    let ok = zip_of(&[("Nyeri/a.pdf", b"%PDF-1.4"), ("b.jpg", &jpeg())]);
    let r = t.upload_raw(&key, "county.zip", ok).await;
    assert_eq!(r.status(), 201);
    assert_eq!(r.json::<Value>().await.unwrap()["kind"], "zip");

    let p = problem(t.upload_raw(&key, "x.zip", zip_of(&[("../../etc/cron.d/x.pdf", b"%PDF")])).await, 422).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-unsafe-zip") && p["reason"].as_str().unwrap().contains("outside"));
    let p = problem(t.upload_raw(&key, "x.zip", zip_of(&[("a", b"1"), ("b", b"2"), ("c", b"3"), ("d", b"4")])).await, 422).await;
    assert!(p["reason"].as_str().unwrap().contains("4 entries"));
    // a small zip that expands past the limit
    let bomb = zip_of(&[("big.pdf", &vec![0u8; 50_000])]);
    assert!(bomb.len() < 1000);
    let p = problem(t.upload_raw(&key, "x.zip", bomb).await, 422).await;
    assert!(p["reason"].as_str().unwrap().contains("expands to more than 10000 bytes"));
    let list: Value = t.get(&key, "/v1/files").await.json().await.unwrap();
    assert_eq!(list["data"].as_array().unwrap().len(), 1);
    t.stop().await;
}
