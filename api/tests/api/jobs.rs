//! Jobs: created as long-running operations, run by a worker as one
//! `rfpii job` process, read, cancelled and deleted.

use crate::common::{TestServer, problem};
use rfpii_api::server::{self, Options};
use serde_json::{Value, json};
use std::io::Read;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread")]
async fn a_job_needs_a_controller_record_and_a_dpia() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account("free", false).await;
    let f = t.upload(&key, "a.jpg").await;
    let p = problem(t.create_job(&key, json!({"file_ids": [f]})).await, 422).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-account-not-ready"));
    assert_eq!(p["missing"], json!(["controller", "dpia"]));

    // checked as rfpii checks it; the operator is the customer's processor
    let mut bad = crate::common::controller();
    bad["commercial"] = json!(true);
    let p = problem(t.put(&key, "/v1/account/controller", &bad).await, 422).await;
    assert!(p["detail"].as_str().unwrap().contains("statutory_authority"));
    let mut role = crate::common::controller();
    role["operator_role"] = json!("controller");
    problem(t.put(&key, "/v1/account/controller", &role).await, 422).await;
    let r = t.put(&key, "/v1/account/controller", &crate::common::controller()).await;
    let a: Value = r.json().await.unwrap();
    assert_eq!(a["controller"]["operator_role"], "processor");
    assert_eq!(a["controller"]["processors"][0]["name"], "rfpii API");
    let p = problem(t.create_job(&key, json!({"file_ids": [f]})).await, 422).await;
    assert_eq!(p["missing"], json!(["dpia"]));

    problem(t.put(&key, "/v1/account/dpia", &json!({"reference": "ODPC/1", "submitted_on": "31-10-2026"})).await, 422).await;
    let r = t.put(&key, "/v1/account/dpia", &json!({"reference": "ODPC/DPIA/2026/14", "submitted_on": "2026-10-20", "status": "submitted"})).await;
    let a: Value = r.json().await.unwrap();
    assert!(a["ready_for_jobs"] == true && a["dpia"]["processing_may_start"] == false);
    let r = t.create_job(&key, json!({"file_ids": [f]})).await;
    assert_eq!(r.status(), 202);
    let j: Value = r.json().await.unwrap();
    // a DPIA submitted less than 60 days ago: the gate will hold everything
    assert!(j["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("60 days")));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_runs_to_its_outputs() {
    let t = TestServer::start(1, |_| {}).await;
    t.fake("FORMS=3\nRELEASED=2\n");
    let (account, key) = t.account("free", true).await;
    let f = t.upload(&key, "1_34_A_007029014000102_F230450M00203609_20220810_132155.jpg").await;
    let r = t.create_job(&key, json!({"file_ids": [f], "policy": {"full_ids": true}, "metadata": {"batch": "Nyeri"}})).await;
    assert_eq!(r.status(), 202);
    let loc = r.headers()["location"].to_str().unwrap().to_string();
    let j: Value = r.json().await.unwrap();
    let id = j["id"].as_str().unwrap().to_string();
    assert_eq!(loc, format!("/v1/jobs/{id}"));
    assert_eq!((j["status"].as_str(), j["forms_requested"].as_i64()), (Some("queued"), Some(3)));

    let j = t.wait_end(&key, &id).await;
    assert_eq!(j["status"], "succeeded", "{j}");
    assert_eq!((j["forms_released"].as_i64(), j["forms_held"].as_i64(), j["exit_code"].as_i64()), (Some(2), Some(1), Some(2)));
    assert_eq!(j["metadata"]["batch"], "Nyeri");
    assert_eq!(j["cost"]["models"][0]["model"], "gemma4:31b");
    assert!(j["outputs_expire_at"].is_string());
    // the inputs are deleted just after the job ends
    t.wait_job(&key, &id, |j| j["inputs_deleted_at"].is_string()).await;

    // what rfpii job was given: the client's name, the policy, no secret
    assert_eq!(t.fake_file("last-inputs.txt").unwrap().trim(), "1_34_A_007029014000102_F230450M00203609_20220810_132155.jpg");
    let yaml = t.fake_file("last-job.yaml").unwrap();
    assert!(yaml.contains("held_pages: none") && yaml.contains("ids: whole") && !yaml.contains("models:"), "{yaml}");
    let controller: Value = serde_json::from_str(&t.fake_file("last-controller.json").unwrap()).unwrap();
    assert_eq!((controller["operator_role"].as_str(), controller["dpia"]["reference"].as_str()), (Some("processor"), Some("ODPC/DPIA/2026/14")));

    let report: Value = t.get(&key, &format!("/v1/jobs/{id}/report")).await.json().await.unwrap();
    assert_eq!(report["schema"], "rfpii/job-report/1");
    let outs: Value = t.get(&key, &format!("/v1/jobs/{id}/outputs")).await.json().await.unwrap();
    let paths: Vec<&str> = outs["data"].as_array().unwrap().iter().map(|o| o["path"].as_str().unwrap()).collect();
    assert_eq!(paths, ["batches/batch-1.json", "form1.manifest.json", "form1.pdf", "form2.manifest.json", "form2.pdf", "ledger.jsonl", "signing_key.pub.pem", "transfers.jsonl"]);
    let r = t.get(&key, &format!("/v1/jobs/{id}/outputs/form1.pdf")).await;
    assert_eq!(r.headers()["content-type"], "application/pdf");
    assert_eq!(r.text().await.unwrap(), "%PDF-1.4 fake form 1\n");
    let r = t.get(&key, &format!("/v1/jobs/{id}/outputs/batches/batch-1.json")).await;
    assert_eq!(r.status(), 200);

    // nothing but the listed files: no traversal, nothing else of the job
    for bad in ["..%2Fjob.yaml", "batches%2F..%2F..%2Fjob.yaml", "%2Fetc%2Fpasswd"] {
        problem(t.get(&key, &format!("/v1/jobs/{id}/outputs/{bad}")).await, 400).await;
    }
    for absent in ["report.json", "staging/transfers.jsonl", "nope.pdf"] {
        problem(t.get(&key, &format!("/v1/jobs/{id}/outputs/{absent}")).await, 404).await;
    }

    // the archive: what was released, and the report
    let r = t.get(&key, &format!("/v1/jobs/{id}/archive")).await;
    assert_eq!(r.headers()["content-type"], "application/zip");
    let bytes = r.bytes().await.unwrap();
    let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes.to_vec())).unwrap();
    let mut names: Vec<String> = z.file_names().map(|n| n.unwrap().to_string()).collect();
    names.sort();
    assert!(
        names.contains(&"public/form2.pdf".into())
            && names.contains(&"public/transfers.jsonl".into())
            && names.contains(&"report.json".into())
            && names.contains(&"report.md".into()),
        "{names:?}"
    );
    let mut s = String::new();
    z.by_name("public/form1.pdf").unwrap().read_to_string(&mut s).unwrap();
    assert_eq!(s, "%PDF-1.4 fake form 1\n");

    // the originals are deleted when the job ends, and what wasn't released
    let dir = t.dir.join("data/jobs").join(&account).join(&id);
    assert!(!dir.join("in").exists() && !dir.join("out/staging").exists() && !dir.join("out/review").exists() && dir.join("out/public/form1.pdf").exists());
    assert!(!t.dir.join("data/files").join(&account).join(&f).exists());
    let file: Value = t.get(&key, &format!("/v1/files/{f}")).await.json().await.unwrap();
    assert_eq!(file["status"], "deleted");
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_that_released_nothing_keeps_its_transfers_record() {
    let t = TestServer::start(1, |_| {}).await;
    t.fake("FORMS=2\nRELEASED=0\n");
    let (_, key) = t.account("free", true).await;
    let j = t.start_job(&key).await;
    let id = j["id"].as_str().unwrap();
    let j = t.wait_end(&key, id).await;
    assert_eq!((j["status"].as_str(), j["forms_released"].as_i64(), j["forms_held"].as_i64()), (Some("succeeded"), Some(0), Some(2)));
    let outs: Value = t.get(&key, &format!("/v1/jobs/{id}/outputs")).await.json().await.unwrap();
    assert_eq!(outs["data"].as_array().unwrap().len(), 1);
    assert_eq!(outs["data"][0]["path"], "transfers.jsonl");
    let r = t.get(&key, &format!("/v1/jobs/{id}/outputs/transfers.jsonl")).await;
    assert!(r.text().await.unwrap().contains("ollama.com"));
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!(a["free_forms_left"], 5);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn idempotency_keys_replay_the_same_job() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account("paid", true).await;
    let f = t.upload(&key, "a.jpg").await;
    let send = |body: String, k: &'static str| {
        t.client.post(t.url("/v1/jobs")).bearer_auth(&key).header("idempotency-key", k).header("content-type", "application/json").body(body).send()
    };
    let r1 = send(format!(r#"{{"file_ids": ["{f}"], "instructions": "Hide ID numbers completely."}}"#), "k-1").await.unwrap();
    assert_eq!(r1.status(), 202);
    let j1: Value = r1.json().await.unwrap();
    // the same request, written differently
    let r2 = send(format!(r#"{{"instructions":"Hide ID numbers completely.","file_ids":["{f}"]}}"#), "k-1").await.unwrap();
    assert_eq!((r2.status().as_u16(), r2.headers()["idempotent-replayed"].to_str().unwrap()), (202, "true"));
    let j2: Value = r2.json().await.unwrap();
    assert_eq!(j1["id"], j2["id"]);
    // the same key with another request
    let p = problem(send(format!(r#"{{"file_ids": ["{f}"]}}"#), "k-1").await.unwrap(), 422).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-idempotency-key-reused"));
    // another key: another job
    let r3 = send(format!(r#"{{"file_ids": ["{f}"]}}"#), "k-2").await.unwrap();
    assert_ne!(r3.json::<Value>().await.unwrap()["id"], j1["id"]);
    let list: Value = t.get(&key, "/v1/jobs").await.json().await.unwrap();
    assert_eq!(list["data"].as_array().unwrap().len(), 2);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn job_requests_are_validated() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account("free", true).await;
    let f = t.upload(&key, "a.jpg").await;
    problem(t.create_job(&key, json!({"file_ids": []})).await, 422).await;
    problem(t.create_job(&key, json!({"file_ids": [f, f]})).await, 422).await;
    problem(t.create_job(&key, json!({"file_ids": [f], "colour": "red"})).await, 422).await;
    problem(t.create_job(&key, json!({"file_ids": [f], "policy": {"hide_names": true}})).await, 422).await;
    problem(t.create_job(&key, json!({"file_ids": [f], "per_page": "sometimes"})).await, 422).await;
    problem(t.create_job(&key, json!({"file_ids": [f], "metadata": [1, 2]})).await, 422).await;
    let p = problem(t.create_job(&key, json!({"file_ids": ["file_nope"]})).await, 422).await;
    assert_eq!(p["file_ids"], json!(["file_nope"]));
    // a webhook needs the account's signing secret first
    let p = problem(t.create_job(&key, json!({"file_ids": [f], "webhook_url": "https://hooks.example.org/x"})).await, 422).await;
    assert!(p["detail"].as_str().unwrap().contains("webhook-secret"));
    problem(t.client.post(t.url("/v1/jobs")).bearer_auth(&key).header("content-type", "application/json").body("{not json").send().await.unwrap(), 400).await;
    problem(t.client.post(t.url("/v1/jobs")).bearer_auth(&key).body("{}").send().await.unwrap(), 415).await;
    problem(t.client.post(t.url("/v1/jobs")).bearer_auth(&key).header("idempotency-key", "").json(&json!({"file_ids": [f]})).send().await.unwrap(), 400).await;
    // a file holding no form
    t.fake("FORMS=0\n");
    let p = problem(t.create_job(&key, json!({"file_ids": [f]})).await, 422).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-no-forms"));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn jobs_are_listed_newest_first_with_a_cursor() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account("paid", true).await;
    let mut ids = Vec::new();
    for _ in 0..3 {
        ids.push(t.start_job(&key).await["id"].as_str().unwrap().to_string());
    }
    let r = t.get(&key, "/v1/jobs?limit=2").await;
    let link = r.headers()["link"].to_str().unwrap().to_string();
    let p1: Value = r.json().await.unwrap();
    assert_eq!(p1["data"][0]["id"], ids[2].as_str());
    assert_eq!(p1["has_more"], true);
    let after = p1["next_after"].as_str().unwrap();
    assert!(link.contains(&format!("after={after}")) && link.contains("rel=\"next\""));
    let p2: Value = t.get(&key, &format!("/v1/jobs?limit=2&after={after}")).await.json().await.unwrap();
    assert_eq!(p2["data"].as_array().unwrap().len(), 1);
    assert_eq!((p2["data"][0]["id"].as_str(), p2["has_more"].as_bool()), (Some(ids[0].as_str()), Some(false)));
    problem(t.get(&key, "/v1/jobs?limit=500").await, 400).await;
    // another account sees none of them
    let (_, other) = t.account("free", false).await;
    let none: Value = t.get(&other, "/v1/jobs").await.json().await.unwrap();
    assert!(none["data"].as_array().unwrap().is_empty());
    problem(t.get(&other, &format!("/v1/jobs/{}", ids[0])).await, 404).await;
    t.stop().await;
}

fn alive(pid: i32) -> bool {
    // SAFETY: signal 0 only checks the process exists.
    unsafe { libc::kill(pid, 0) == 0 }
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_kills_the_process() {
    let t = TestServer::start(1, |_| {}).await;
    t.fake("FORMS=2\nRELEASED=2\nDELAY=30\n");
    let (_, key) = t.account("free", true).await;
    let id = t.start_job(&key).await["id"].as_str().unwrap().to_string();
    t.wait_job(&key, &id, |j| j["status"] == "running").await;
    let start = std::time::Instant::now();
    while t.fake_file("job.pid").is_none() {
        assert!(start.elapsed() < Duration::from_secs(10));
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let pid: i32 = t.fake_file("job.pid").unwrap().trim().parse().unwrap();
    assert!(alive(pid));
    let r = t.post(&key, &format!("/v1/jobs/{id}/cancel"), &json!({})).await;
    assert_eq!(r.status(), 202);
    assert_eq!(r.json::<Value>().await.unwrap()["cancel_requested"], true);
    let j = t.wait_end(&key, &id).await;
    assert_eq!(j["status"], "canceled");
    assert!(!alive(pid), "rfpii job still runs");
    assert_eq!((j["price"]["forms_charged"].as_i64(), j["forms_released"].as_i64()), (Some(0), Some(0)));
    problem(t.post(&key, &format!("/v1/jobs/{id}/cancel"), &json!({})).await, 409).await;
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!((a["free_forms_left"].as_i64(), a["forms_reserved"].as_i64()), (Some(5), Some(0)));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_queued_job_is_canceled_at_once() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account("free", true).await;
    let id = t.start_job(&key).await["id"].as_str().unwrap().to_string();
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!(a["free_forms_left"], 2);
    let r = t.post(&key, &format!("/v1/jobs/{id}/cancel"), &json!({})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["status"], "canceled");
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!(a["free_forms_left"], 5);
    // a canceled job's outputs: there are none, but it has ended
    let outs: Value = t.get(&key, &format!("/v1/jobs/{id}/outputs")).await.json().await.unwrap();
    assert!(outs["data"].as_array().unwrap().is_empty());
    problem(t.get(&key, &format!("/v1/jobs/{id}/report")).await, 404).await;
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_erases_the_files_and_keeps_the_billing_record() {
    let t = TestServer::start(1, |_| {}).await;
    t.fake("FORMS=3\nRELEASED=2\nDELAY=1\n");
    let (account, key) = t.account("paid", true).await;
    let id = t.start_job(&key).await["id"].as_str().unwrap().to_string();
    problem(t.delete(&key, &format!("/v1/jobs/{id}")).await, 409).await;
    t.wait_end(&key, &id).await;
    let r = t.delete(&key, &format!("/v1/jobs/{id}")).await;
    let j: Value = r.json().await.unwrap();
    assert!(j["deleted_at"].is_string() && j["instructions"].is_null());
    assert_eq!(j["price"]["amount_microusd"], 9800);
    assert!(!t.dir.join("data/jobs").join(&account).join(&id).exists());
    problem(t.get(&key, &format!("/v1/jobs/{id}/outputs")).await, 410).await;
    problem(t.get(&key, &format!("/v1/jobs/{id}/report")).await, 410).await;
    problem(t.get(&key, &format!("/v1/jobs/{id}/archive")).await, 410).await;
    let u: Value = t.get(&key, "/v1/usage").await.json().await.unwrap();
    assert_eq!((u["records"][0]["job_id"].as_str(), u["amount_due_microusd"].as_i64()), (Some(id.as_str()), Some(9800)));
    let id2 = id.clone();
    let deletions: i64 = t.state.db.call(move |c| Ok(c.query_row("SELECT COUNT(*) FROM deletions WHERE job_id = ?1", [id2], |r| r.get(0))?)).await.unwrap();
    assert!(deletions >= 2, "{deletions}");
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_job_is_not_charged() {
    let t = TestServer::start(1, |_| {}).await;
    t.fake("FORMS=2\nRELEASED=2\nEXIT=1\nNO_REPORT=1\n");
    let (_, key) = t.account("paid", true).await;
    let id = t.start_job(&key).await["id"].as_str().unwrap().to_string();
    let j = t.wait_end(&key, &id).await;
    assert_eq!(j["status"], "failed");
    let err = j["error"].as_str().unwrap();
    assert!(err.contains("exited with code 1") && err.contains("the fake was told to fail"), "{err}");
    assert_eq!((j["price"]["amount_microusd"].as_i64(), j["forms_released"].as_i64()), (Some(0), Some(0)));
    let u: Value = t.get(&key, "/v1/usage").await.json().await.unwrap();
    assert!(u["records"].as_array().unwrap().is_empty());
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expired_lease_requeues_the_job() {
    let t = TestServer::start(0, |c| c.max_attempts = 2).await;
    let (_, key) = t.account("paid", true).await;
    let id = t.start_job(&key).await["id"].as_str().unwrap().to_string();
    // a worker leased it, then vanished: its lease ran out a second ago
    let (jid, now) = (id.clone(), t.state.clock.now());
    t.state
        .db
        .call(move |c| {
            Ok(c.execute("UPDATE jobs SET status = 'running', lease_owner = 'ghost', lease_expires_at = ?2, attempts = 1 WHERE id = ?1", rusqlite::params![jid, now - 1])?)
        })
        .await
        .unwrap();
    rfpii_api::worker::reap(&t.state).await.unwrap();
    let j: Value = t.get(&key, &format!("/v1/jobs/{id}")).await.json().await.unwrap();
    assert_eq!(j["status"], "queued");
    // a worker picks it up and runs it: its second attempt
    let workers = server::start_with(t.state.clone(), Options { http: false, workers: 1 }).await.unwrap();
    let j = t.wait_end(&key, &id).await;
    assert_eq!((j["status"].as_str(), j["attempts"].as_i64()), (Some("succeeded"), Some(2)));
    workers.stop().await;

    // on its last attempt, an expired lease fails the job, uncharged
    let id = t.start_job(&key).await["id"].as_str().unwrap().to_string();
    let (jid, now) = (id.clone(), t.state.clock.now());
    t.state
        .db
        .call(move |c| {
            Ok(c.execute("UPDATE jobs SET status = 'running', lease_owner = 'ghost', lease_expires_at = ?2, attempts = 2 WHERE id = ?1", rusqlite::params![jid, now - 1])?)
        })
        .await
        .unwrap();
    rfpii_api::worker::reap(&t.state).await.unwrap();
    let j: Value = t.get(&key, &format!("/v1/jobs/{id}")).await.json().await.unwrap();
    assert_eq!(j["status"], "failed");
    assert!(j["error"].as_str().unwrap().contains("lease expired"));
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!(a["forms_reserved"], 0);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn shutdown_requeues_a_running_job() {
    let mut t = TestServer::start(1, |_| {}).await;
    t.fake("FORMS=1\nRELEASED=1\nDELAY=30\n");
    let (_, key) = t.account("free", true).await;
    let id = t.start_job(&key).await["id"].as_str().unwrap().to_string();
    t.wait_job(&key, &id, |j| j["status"] == "running").await;
    while t.fake_file("job.pid").is_none() {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let pid: i32 = t.fake_file("job.pid").unwrap().trim().parse().unwrap();
    t.server.take().unwrap().stop().await;
    assert!(!alive(pid));
    let jid = id.clone();
    let (status, attempts): (String, i64) =
        t.state.db.call(move |c| Ok(c.query_row("SELECT status, attempts FROM jobs WHERE id = ?1", [jid], |r| Ok((r.get(0)?, r.get(1)?)))?)).await.unwrap();
    assert_eq!((status.as_str(), attempts), ("queued", 0));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn held_pages_go_to_the_cloud_only_on_opt_in() {
    let t = TestServer::start(1, |c| c.vision_models = vec!["gemma4:31b".into()]).await;
    let (_, key) = t.account("paid", true).await;
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!(a["settings"]["held_pages"], "none");
    assert!(a["controller"]["processors"][1]["service"].as_str().unwrap().contains("redacted pages only"));
    let j = t.start_job(&key).await;
    t.wait_end(&key, j["id"].as_str().unwrap()).await;
    let yaml = t.fake_file("last-job.yaml").unwrap();
    assert!(yaml.contains("held_pages: none") && yaml.contains("gemma4:31b"), "{yaml}");

    problem(t.patch(&key, "/v1/account/settings", &json!({"held_pages": "sometimes"})).await, 422).await;
    let a: Value = t.patch(&key, "/v1/account/settings", &json!({"held_pages": "cloud"})).await.json().await.unwrap();
    assert_eq!(a["settings"]["held_pages"], "cloud");
    let j = t.start_job(&key).await;
    assert_eq!(j["held_pages"], "cloud");
    t.wait_end(&key, j["id"].as_str().unwrap()).await;
    assert!(t.fake_file("last-job.yaml").unwrap().contains("held_pages: cloud"));
    let c: Value = serde_json::from_str(&t.fake_file("last-controller.json").unwrap()).unwrap();
    assert_eq!(c["processors"][1]["name"], "Ollama");
    assert!(c["processors"][1]["service"].as_str().unwrap().contains("held pages"));
    t.stop().await;
}
