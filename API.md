# The rfpii API

An HTTP API that sells the `rfpii` redaction harness ([HARNESS.md](HARNESS.md))
as a service. A customer uploads election result forms, starts a job, and
downloads what the release gate let out: each published form with its signed
manifest, its ledger entry and a batch root, so anyone can check that nothing
but personal data was touched ([VERIFYING.md](VERIFYING.md)). A form costs
$0.0049 on the paid plan; 5 forms a month are free ([PRICING.md](PRICING.md)).

The server is `rfpii-api`, a Rust crate in [`api/`](api/README.md). It calls
the `rfpii` binary as a subprocess and does not link it.

Contents: [Architecture](#architecture) · [Resources](#resources) ·
[Authentication](#authentication) · [Endpoints](#endpoints) ·
[Job lifecycle](#job-lifecycle) · [Webhooks](#webhooks) ·
[Plans and billing](#plans-and-billing) · [Rate limits](#rate-limits) ·
[Errors](#errors) · [Security and privacy](#security-and-privacy) ·
[Legal points](#legal-points) · [Deployment](#deployment) ·
[Standards](#standards)

## Architecture

```
   clients ──HTTPS──► reverse proxy (TLS) ──HTTP──► rfpii-api serve
                                                     │  HTTP API (axum)
                                                     │  N workers ──► rfpii job (one process per job)
   Stripe ◄── checkout, meter events ────────────────┤  webhook deliveries ──► customers' URLs
   Stripe ──► /v1/billing/stripe-webhook             │  Stripe reporter, retention sweeper
                                                     │
                       rfpii-api worker (optional) ──┤  more workers, same database and folder
                                                     ▼
                       DATA_DIR/  rfpii-api.sqlite3 (WAL)
                                  files/<account>/<file id>           uploads, by id
                                  jobs/<account>/<job id>/            in/  job.yaml  controller.json
                                                                      out/public/  out/report.json
                                                                      records/  rfpii.log
                                  tmp/                                uploads in progress
```

- **The API** checks keys, uploads and requests, counts a job's forms with
  `rfpii inventory --json`, reserves them against the plan, and queues the
  job. It runs `rfpii verify` for anyone checking a published form.
- **Workers** lease queued jobs from SQLite (an atomic `UPDATE … RETURNING`
  with a lease; a heartbeat renews it). Each job runs as one `rfpii job`
  process in its own folder, in its own process group, with a time limit.
  When it ends, the worker reads `report.json`, settles the usage, records
  the job's actual cost, queues its webhook and deletes the originals.
- **Background tasks** in every process deliver webhooks, report paid usage
  to Stripe, and sweep files past their retention period.
- `rfpii-api serve --workers 0` serves the API alone; `rfpii-api worker`
  runs workers alone. Any number of each can share one database and data
  folder on one machine (SQLite in WAL mode with a busy timeout).

`rfpii job` makes every redaction and decides what is released (HARNESS.md
§2, §9). The API never edits a form and never overrides the gate: it only
serves what `rfpii job` put in `public/`.

## Resources

| Resource | Id | What it is |
|---|---|---|
| Account | `acct_…` | A customer: its plan, its controller record and DPIA, its settings |
| API key | `key_…` | `rfpii_live_…`, with scopes; only its SHA-256 and prefix are stored |
| File | `file_…` | An uploaded PDF, JPEG, PNG, TIFF or zip, stored by id; never served back |
| Job | `job_…` | One run of `rfpii job` on some files: status, forms, price, outputs |
| Usage record | `usage_…` | The forms a settled job released, at its plan's unit price |
| Webhook delivery | `msg_…` | One event to one URL, with every attempt |

Ids are a prefix and a UUIDv7, so they sort by creation time. Times are RFC
3339 in UTC; months are `YYYY-MM` (UTC); money is integer micro-dollars
(`4900` is $0.0049), with a `*_usd` string beside it where useful.

A **form** is a document as `rfpii inventory` counts it, and as `rfpii job`
plans it (`rust/src/job/plan.rs`): each readable file that is not a copy of
another is one form, or one per page when it is redacted page by page (a
scan of several pages, or any file with `per_page: yes`).

## Authentication

Every route under `/v1/` except `/v1/pricing`, `/v1/verify`,
`/v1/openapi.json` and the Stripe webhook needs an API key:

```
Authorization: Bearer rfpii_live_3kTq0sV9cA2bX7mN4pR8wY1zL5dF6gH0jK2eU9iO3tQ
```

A key is `rfpii_live_` and 32 random bytes in base62 (43 characters). The
server stores its SHA-256 and its first characters (`prefix`), never the key;
it is shown once, when it is made. A key has scopes:

| Scope | Allows |
|---|---|
| `jobs:write` | upload and delete files; create, cancel and delete jobs |
| `jobs:read` | read files, jobs, outputs and usage |
| `account:admin` | set the controller record, DPIA and settings; rotate the webhook secret; manage keys; start Checkout |

`GET /v1/account` takes any valid key. A missing, unknown or revoked key
gets `401` with `WWW-Authenticate: Bearer`; a key without the scope gets
`403`. Each key records when it was last used. The operator creates the
first key of an account with the admin command (or `/v1/admin/*`, see
[`api/README.md`](api/README.md)).

## Endpoints

The OpenAPI 3.1 description is served at `GET /v1/openapi.json`
(`api/openapi.json`); a test checks that every route is in it and that it
describes nothing else. In the examples, `$API` is the server's URL and
`$KEY` an API key.

### Account and legal records

A job is refused (`422 account-not-ready`) until the account has a
controller record and a DPIA: `rfpii job` won't run without the first, and
its release gate checks the second (HARNESS.md §9, conditions 8 and 9).

```bash
# The plan, this month's released forms, the free forms left, the price, the spend
curl -s $API/v1/account -H "Authorization: Bearer $KEY"

# The controller record: who publishes the forms and on what legal basis.
# Checked as rfpii checks controller.json; operator_role becomes "processor",
# and the platform (and the model host, when pages go to it) are added as
# processors.
curl -s -X PUT $API/v1/account/controller -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d '{
    "controller": {"name": "Example Observer Group", "registration_no": "ODPC/123"},
    "legal_basis": "s.30(1)(b)(vii) legitimate interests",
    "purpose": "republishing polling-station results forms",
    "commercial": false}'

# The DPIA (Data Protection Act s.31): approved, or submitted 60 days before
curl -s -X PUT $API/v1/account/dpia -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"reference": "ODPC/DPIA/2026/14", "submitted_on": "2026-08-01", "status": "submitted"}'

# Settings: held pages to the cloud model (default "none"), the webhook URL.
# Setting a webhook URL for the first time returns the signing secret, once.
curl -s -X PATCH $API/v1/account/settings -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"held_pages": "none", "webhook_url": "https://hooks.example.org/rfpii"}'

# A new webhook signing secret (whsec_…), returned once
curl -s -X POST $API/v1/account/webhook-secret -H "Authorization: Bearer $KEY"
```

### Files

```bash
# Multipart, field "file"; or the raw body, named in the query
curl -s $API/v1/files -H "Authorization: Bearer $KEY" -F file=@1_34_A_007029014000102_F230450M00203609_20220810_132155.PDF
curl -s "$API/v1/files?name=scan.pdf" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/pdf' --data-binary @scan.pdf

curl -s $API/v1/files -H "Authorization: Bearer $KEY"              # list, newest first
curl -s $API/v1/files/$FILE -H "Authorization: Bearer $KEY"        # describe
curl -s -X DELETE $API/v1/files/$FILE -H "Authorization: Bearer $KEY"
```

`201 Created` with a `Location` and `{id, kind, bytes, sha256, name, …}`.
The kind is told by the file's first bytes, not its name or declared type:
PDF, JPEG, PNG, TIFF or zip, else `415`. A file over `MAX_UPLOAD_BYTES` is
`413`. A zip is expanded once, counting, before it is accepted: more than
`MAX_ZIP_ENTRIES` entries, more than `MAX_ZIP_UNCOMPRESSED_BYTES` once
expanded (sizes a zip states can lie), an entry named with an absolute path,
a `..` component or a backslash, an encrypted entry or a symbolic link: `422
unsafe-zip`. Files are stored by id under the account. The client's name is
kept, reduced to safe characters, only to name the file in a job's input
folder (KIEMS names carry the form and station, which `rfpii` reads). There
is no route that returns a file's content: originals are never served back.
A file is deleted when a job that used it ends (see
[Security and privacy](#security-and-privacy)); deleting one a queued or
running job uses is `409`.

### Jobs

```bash
curl -s -i $API/v1/jobs -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -H "Idempotency-Key: $(uuidgen)" -d '{
    "file_ids": ["file_019a3c…", "file_019a3d…"],
    "instructions": "Hide ID numbers completely. Keep the phone prefixes.",
    "policy": {"full_ids": true, "full_phones": false, "keep_stamps": false},
    "per_page": "auto",
    "webhook_url": "https://hooks.example.org/rfpii/job",
    "metadata": {"batch": "Nyeri 2022"}}'
```

`202 Accepted`, `Location: /v1/jobs/job_…` and the job. Only `file_ids` is
required.

- `instructions`: free text, compiled into a policy by the orchestrator
  model (HARNESS.md §3). With no orchestrator configured and no `policy`,
  instructions can't be shown to be met, and `rfpii job` releases nothing.
- `policy`: `full_ids` (`ids: whole`), `full_phones` (`phones: whole`),
  `keep_stamps` (`stamps: protected`). When given, it is the job file's
  policy, used when no model compiles the instructions.
- `per_page`: `auto` (scans of several pages page by page), `yes`, `no`, or
  `true`/`false`.
- `webhook_url`: this job's events go here instead of the account's URL. It
  needs the account's signing secret.
- `metadata`: a JSON object of at most 8 KiB, kept with the job.

On creation the server counts the forms (`rfpii inventory --json`), checks
them against the plan (`402`, see [Plans and billing](#plans-and-billing)),
reserves them, and queues the job. The response lists `warnings`: reasons
the release gate will hold every form (a DPIA submitted less than 60 days
ago, no vision model configured).

**Idempotency.** With `Idempotency-Key`, the same key and the same body
(compared as canonical JSON) return the same job, with
`Idempotent-Replayed: true`; the same key with a different body is `422
idempotency-key-reused`. Keys are kept 24 hours.

```bash
curl -s "$API/v1/jobs?limit=20" -H "Authorization: Bearer $KEY"             # newest first
curl -s "$API/v1/jobs?limit=20&after=job_019a…" -H "Authorization: Bearer $KEY"  # next page
curl -s $API/v1/jobs/$JOB -H "Authorization: Bearer $KEY"
curl -s -X POST $API/v1/jobs/$JOB/cancel -H "Authorization: Bearer $KEY"
curl -s -X DELETE $API/v1/jobs/$JOB -H "Authorization: Bearer $KEY"
```

Lists are pages of `limit` (1 to 100, default 20) with `has_more` and
`next_after`, and a `Link: <…>; rel="next"` header. Cancelling a queued job
ends it at once (`200`); a running one is stopped by its worker, which kills
its process group (`202` until it has). Deleting a job (once it has ended;
`409` before) erases its folder (inputs, outputs, report, log) and its
instructions and metadata; its billing fields and usage record are kept.

The job resource:

```json
{
  "id": "job_019a3c…", "object": "job", "status": "succeeded",
  "file_ids": ["file_019a3c…"], "policy": {"full_ids": true, "full_phones": false, "keep_stamps": false},
  "per_page": "auto", "held_pages": "none",
  "forms_requested": 3, "forms_released": 2, "forms_held": 1,
  "inventory": {"files": 1, "pages": 3, "duplicates": 0, "broken": 0, "forms": 3},
  "price": {"currency": "usd", "plan": "paid", "unit_microusd": 4900, "estimate_microusd": 14700,
            "forms_charged": 2, "amount_microusd": 9800, "amount_usd": "0.0098"},
  "exit_code": 2, "error": null, "attempts": 1,
  "created_at": "2026-10-31T09:00:00Z", "started_at": "…", "finished_at": "…",
  "inputs_deleted_at": "…", "outputs_expire_at": "2026-11-30T09:01:10Z",
  "links": {"self": "/v1/jobs/job_019a3c…", "report": "…/report", "outputs": "…/outputs", "archive": "…/archive"}
}
```

Customers see the price, not what the job cost to run. The operator does:
`GET /v1/admin/jobs/{id}` (with ADMIN_TOKEN) returns the same resource with
`cost`, its CPU time (with its child processes', from `wait4`) at
`SERVER_PRICE_PER_VCPU_HOUR_MICROUSD`, and each model's tokens (from the
report's `model_usage`) at `MODEL_PRICES`, as PRICING.md computes it, so the
margin can be watched:

```json
"cost": {"currency": "usd", "cpu_seconds": 31.2, "server_microusd": 519,
         "models": [{"host": "ollama.com", "model": "gemma4:31b", "requests": 8, "input_tokens": 2041,
                     "output_tokens": 146, "cost_microusd": 344, "price_known": true}],
         "total_microusd": 863, "total_usd": "0.000863"}
```

### Outputs

```bash
curl -s $API/v1/jobs/$JOB/report -H "Authorization: Bearer $KEY"     # report.json (HARNESS.md §10)
curl -s $API/v1/jobs/$JOB/outputs -H "Authorization: Bearer $KEY"    # what was released
curl -s -O $API/v1/jobs/$JOB/outputs/1_34_A_007029014000102_F230450M00203609_20220810_132155.pdf \
  -H "Authorization: Bearer $KEY"
curl -s -o job.zip $API/v1/jobs/$JOB/archive -H "Authorization: Bearer $KEY"  # public/… and the report
```

The outputs are what `rfpii job` put in `public/`: the released PDFs and
their manifests, `ledger.jsonl`, `batches/`, `signing_key.pub.pem` and
`transfers.jsonl`. When nothing was released, the transfers record is kept
from `staging/` and listed as `transfers.jsonl` all the same. A path must be
one the listing gives (`400` for a path with `..`, a backslash or an empty
component; `404` for anything else); nothing else of a job's folder can be
reached. These routes are `409` until the job has ended and `410` once its
outputs were deleted.

### Verification (public)

```bash
curl -s $API/v1/verify -F pdf=@x.pdf -F manifest=@x.manifest.json \
  -F original=@x.PDF -F pubkey=@signing_key.pub.pem -F ledger=@ledger.jsonl
```

Runs `rfpii verify PDF --manifest M [--original O] [--pubkey K] [--ledger L]`
and returns `{verified, checks: [{ok, text}], notes, result}`. No key is
needed, it is free, and it is limited per IP address: anyone can check a
published form. The files are deleted when the check ends.

### Keys

```bash
curl -s $API/v1/keys -H "Authorization: Bearer $KEY"
curl -s $API/v1/keys -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"name": "ci", "scopes": ["jobs:read", "jobs:write"]}'     # the secret is in the answer, once
curl -s -X DELETE $API/v1/keys/$KEY_ID -H "Authorization: Bearer $KEY"   # revoke
```

### Usage and billing

```bash
curl -s "$API/v1/usage?period=2026-10" -H "Authorization: Bearer $KEY"   # records, forms, amount due
curl -s $API/v1/pricing                                                  # public
# Stripe Checkout for the paid plan (501 without Stripe); the body is optional:
# the pages Checkout returns to (else the API's /v1/pricing)
curl -s -X POST $API/v1/billing/checkout -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"success_url": "https://observers.example.org/billing/done", "cancel_url": "https://observers.example.org/billing"}'
```

The answer is `{id, url}`: send the customer to `url`.

`POST /v1/billing/stripe-webhook` is for Stripe; see
[Plans and billing](#plans-and-billing).

### Operations

```bash
curl -s $API/healthz    # the process answers
curl -s $API/readyz     # database, data folder writable, rfpii binary runs (503 otherwise)
curl -s $API/metrics -H "Authorization: Bearer $METRICS_TOKEN"   # Prometheus text
```

`/metrics` has requests by route template and status
(`rfpii_api_http_requests_total`), their latency
(`rfpii_api_http_request_duration_seconds`), jobs by status, the queue
depth, forms released, model tokens by model, webhooks pending and paid
usage not yet reported to Stripe. Keep it off the public proxy, or set
`METRICS_TOKEN`.

## Job lifecycle

```
            POST /v1/jobs                lease (worker)           rfpii job exits 0 or 2
  (none) ───────────────► queued ───────────────────► running ─────────────────────────► succeeded
                            │  ▲  lease expires,          │  exits otherwise, no report,
                    cancel  │  │  attempts left           │  time limit, or lease expires
                            │  └──────────────────────────┤  on the last attempt ────────► failed
                            ▼                             │
                         canceled ◄───────────────────────┘  cancel: process group killed
```

1. **Created**: the files are linked into `in/` (by their safe names, else
   by id), the forms counted and reserved, `controller.json` and `job.yaml`
   written from the account and the server's configuration (`held_pages`
   from the account's setting, the models, a model-request budget of
   `MODEL_REQUESTS_PER_FORM` per form, `JOB_TIMEOUT_SECS`, the platform's
   signing key). The job runs under the records and settings of that moment.
2. **Running**: one `rfpii job job.yaml` process, in the job's folder and its
   own process group, its output in `rfpii.log`, its environment limited to
   what `rfpii` needs (no Stripe or admin secret). The worker renews its
   lease every `LEASE_SECS / 3`. A lease that runs out (the worker died)
   puts the job back in the queue, until it has been tried `MAX_ATTEMPTS`
   times; then it fails.
3. **Ended**: `succeeded` when `rfpii job` exits 0 (everything released) or
   2 (something held) and wrote its report; `failed` otherwise. Forms
   released are the sum of the report's `groups[].released`; forms held are
   the plan's documents less those. The usage is settled, the cost recorded,
   the webhook queued, and the inputs deleted.

At shutdown (SIGTERM or SIGINT) workers stop leasing; running jobs get
`SHUTDOWN_GRACE_SECS` to finish, then are stopped and put back in the queue
(the attempt doesn't count). The HTTP server finishes the requests it has.

## Webhooks

Events: `job.succeeded`, `job.failed`, `job.canceled`, sent to the job's
`webhook_url`, else the account's. The body:

```json
{"type": "job.succeeded", "timestamp": "2026-10-31T09:01:10Z", "data": { …the job… }}
```

Deliveries follow [Standard Webhooks](https://www.standardwebhooks.com/):

```
webhook-id: msg_019a3c…
webhook-timestamp: 1793437270
webhook-signature: v1,K5oZfzN95Z9UVu1EsfQmfVNQhnkZ2pj9o9NDN/H/pI4=
```

The signature is the base64 HMAC-SHA256 of `{webhook-id}.{webhook-timestamp}.{body}`,
keyed with the base64-decoded part of the secret after `whsec_`. A delivery
keeps its `webhook-id` across retries, so a receiver can drop duplicates. A
2xx answer is success; anything else, or no answer in
`WEBHOOK_TIMEOUT_SECS`, is retried after 5 s, 30 s, 2 min, 10 min, 1 h and
6 h (`WEBHOOK_RETRY_SCHEDULE`), then given up. Every attempt is recorded
(status, error, time). Redirects are not followed, and a URL must be https
to a public address.

Verifying a delivery, in Python with no library:

```python
import base64, hashlib, hmac, time

def verify(secret: str, headers: dict, body: bytes, tolerance: int = 300) -> bool:
    msg_id, ts = headers["webhook-id"], headers["webhook-timestamp"]
    if abs(time.time() - int(ts)) > tolerance:
        return False                       # too old (or a replay)
    key = base64.b64decode(secret.removeprefix("whsec_"))
    mac = hmac.new(key, f"{msg_id}.{ts}.".encode() + body, hashlib.sha256).digest()
    expected = "v1," + base64.b64encode(mac).decode()
    return any(hmac.compare_digest(expected, s) for s in headers["webhook-signature"].split(" "))
```

The Standard Webhooks libraries (`standardwebhooks` for Python,
JavaScript, Go, Rust and others) do the same.

## Plans and billing

| Plan | Price | What it includes |
|---|---|---|
| Free | $0 | 5 forms a calendar month (UTC) |
| Paid | $0.0049 per form, billed monthly | every form released; optional monthly spend cap |

How the price was set is in [PRICING.md](PRICING.md); both values are
configuration (`FREE_FORMS_PER_MONTH`, `PRICE_PER_FORM_MICROUSD`).

- **Only released forms count.** A held form, or one in a group the gate
  held, is not charged and doesn't use the free allowance. A job is never
  charged for more forms than it reserved (if the orchestrator chose to read
  a file page by page where the count read it whole, the extra forms are
  free).
- **Reserve, then settle.** A job reserves its forms when it is created. A
  free account whose request is more than its month has left gets `402
  free-allowance-exceeded`, naming the forms requested, the forms left and
  how to upgrade. A paid account with a spend cap gets `402
  spend-cap-reached` if the job, at its estimate, could pass the cap. When
  the job ends, the forms it released are recorded (a usage record at the
  unit price of its creation) and the rest of the reservation is let go. A
  failed or cancelled job is not charged. A job's forms count in the month
  it was created.
- **With Stripe** (`STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`,
  `STRIPE_WEBHOOK_SECRET`): `POST /v1/billing/checkout` makes a Checkout
  Session subscribing the account to the metered price. Stripe's webhook
  (`POST /v1/billing/stripe-webhook`, its `Stripe-Signature` checked with a
  300-second tolerance, each event handled once) sets the plan:
  `checkout.session.completed` makes the account paid and records its
  customer; `customer.subscription.updated` keeps it paid while the
  subscription is active, trialing or past due; `customer.subscription.deleted`
  makes it free. A background task reports each paid usage record as a
  [billing meter event](https://docs.stripe.com/api/billing/meter-event/create)
  (`event_name`, `payload[stripe_customer_id]`, `payload[value]` = forms,
  `identifier` = the usage record's id, so a retry is never counted twice),
  retrying after 1, 5 and 30 minutes, then hourly. Stripe invoices monthly;
  a bill below its minimum charge is carried to the next month, as PRICING.md
  says.
- **Without Stripe** the operator sets plans (`rfpii-api admin set-plan`)
  and invoices from `GET /v1/usage?period=YYYY-MM`.

## Rate limits

A token bucket per API key (`RATE_LIMIT_BURST` requests, refilled at
`RATE_LIMIT_PER_SEC`), and one per client IP address on the public routes
(`/v1/verify`, `/v1/pricing`, `/v1/openapi.json`). Responses carry the
header fields of
[draft-ietf-httpapi-ratelimit-headers-11](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/):

```
RateLimit-Policy: "per-key";q=60;w=6
RateLimit: "per-key";r=41;t=2
```

`q` is the bucket's size and `w` the seconds it takes to fill; `r` the
requests left now and `t` the seconds until the bucket is full again. Over
the limit: `429 Too Many Requests` with `Retry-After` (seconds) and a
problem document. Behind a proxy, set `TRUST_FORWARDED_FOR=true` so the
client's address is taken from the last `X-Forwarded-For` entry.

## Errors

Every error is an RFC 9457 problem document, `Content-Type:
application/problem+json`:

```json
{
  "type": "https://github.com/masukuma/ResultFormPIIRemover/blob/main/API.md#problem-free-allowance-exceeded",
  "title": "Free allowance exceeded",
  "status": 402,
  "detail": "this job has 3 form(s) and the free plan has 2 of its 5 left in 2026-10; upgrade to the paid plan ($0.0049 per form) with POST /v1/billing/checkout, or ask the operator",
  "instance": "urn:rfpii:request:0b7e1c1e-5d1f-4a43-9a1e-2f3c1c9d7a10",
  "request_id": "0b7e1c1e-5d1f-4a43-9a1e-2f3c1c9d7a10",
  "forms_requested": 3, "forms_left": 2, "free_forms_per_month": 5, "period": "2026-10",
  "upgrade": {"checkout": "POST /v1/billing/checkout", "pricing": "https://…/PRICING.md", "price_per_form_microusd": 4900}
}
```

`instance` and `request_id` name the request: its `X-Request-Id`, taken from
the request or generated, returned on every response and written in the
logs. Errors raised outside the handlers (no such route, wrong method, body
too large, timeout, a panic) are problem documents too, of type
`about:blank`. The types:

| Type | Status | When |
|---|---|---|
| <a id="problem-invalid-request"></a>`invalid-request` | 400 | malformed JSON, query or header; 415 when the body isn't JSON |
| <a id="problem-unauthorized"></a>`unauthorized` | 401 | no API key, an unknown one, or a revoked one |
| <a id="problem-insufficient-scope"></a>`insufficient-scope` | 403 | the key lacks the scope (`required_scope`) |
| <a id="problem-not-found"></a>`not-found` | 404 | no such resource in this account |
| <a id="problem-conflict"></a>`conflict` | 409 | the job's state doesn't allow it (`job_status`): cancel an ended job, delete a running one, read outputs before the end |
| <a id="problem-gone"></a>`gone` | 410 | the job's outputs were deleted |
| <a id="problem-too-large"></a>`too-large` | 413 | over the upload or body limit (`limit_bytes`) |
| <a id="problem-unsupported-file-type"></a>`unsupported-file-type` | 415 | not a PDF, JPEG, PNG, TIFF or zip by its first bytes |
| <a id="problem-validation-failed"></a>`validation-failed` | 422 | a field is wrong (unknown fields are refused) |
| <a id="problem-unsafe-zip"></a>`unsafe-zip` | 422 | the zip failed its checks (`reason`) |
| <a id="problem-unknown-files"></a>`unknown-files` | 422 | file ids not of this account, or deleted (`file_ids`) |
| <a id="problem-no-forms"></a>`no-forms` | 422 | no file holds a page `rfpii` can read |
| <a id="problem-account-not-ready"></a>`account-not-ready` | 422 | no controller record or DPIA yet (`missing`) |
| <a id="problem-invalid-controller"></a>`invalid-controller` | 422 | the controller record lacks a required field |
| <a id="problem-idempotency-key-reused"></a>`idempotency-key-reused` | 422 | the key was used with another request in the last 24 hours |
| <a id="problem-free-allowance-exceeded"></a>`free-allowance-exceeded` | 402 | more forms than the free month has left |
| <a id="problem-spend-cap-reached"></a>`spend-cap-reached` | 402 | the job could pass the paid account's monthly cap |
| <a id="problem-rate-limited"></a>`rate-limited` | 429 | over the rate limit (`retry_after`) |
| <a id="problem-invalid-signature"></a>`invalid-signature` | 400 | a Stripe webhook whose signature doesn't verify |
| <a id="problem-billing-not-configured"></a>`billing-not-configured` | 501 | Stripe is not configured |
| <a id="problem-billing-unavailable"></a>`billing-unavailable` | 502 | Stripe refused or could not be reached |
| <a id="problem-unavailable"></a>`unavailable` | 503 | `/readyz`: a check failed (`checks`) |

## Security and privacy

- **Least data.** Originals are deleted when the job that used them ends
  (its `in/` folder and the uploaded files), with what was redacted but not
  released (`staging/`, `review/`, `quarantine/`, `defects/`). Uploads no job
  used are deleted after `RETENTION_UNUSED_FILE_HOURS` (24); outputs
  `RETENTION_OUTPUT_DAYS` (30) after the job ends. `RETENTION_INPUT_DAYS`
  keeps inputs longer, if the controller instructs it. A sweeper enforces
  this and records every deletion (what, when, why) in the database.
  `DELETE /v1/files/{id}` and `DELETE /v1/jobs/{id}` delete on request.
  Billing records are kept.
- **Never returned.** No route serves an uploaded file. Outputs are only
  what `rfpii job` released, by an exact path from the listing.
- **Per-account storage.** Files and jobs are kept under their account's
  folder, by id; every query is scoped to the caller's account.
- **Secrets.** API keys are stored as SHA-256 hashes; webhook and Stripe
  signatures, the admin and metrics tokens are compared in constant time. A
  job's process gets only the environment `rfpii` needs (its passphrases and
  the model host's key), never the API's own secrets, and its job file holds
  no secret.
- **Logs** are JSON lines with the request id, method, path (ids only),
  status and latency; never headers, keys, query strings, file contents or
  personal data.
- **The audit log** records every change to keys, plans, the controller
  record, the DPIA, settings and the webhook secret: who (key, admin
  command, admin API, Stripe), what and when.
- **Limits.** Body sizes (`MAX_JSON_BYTES`, `MAX_UPLOAD_BYTES`,
  `MAX_VERIFY_BYTES`), zip checks, a request timeout
  (`REQUEST_TIMEOUT_SECS`), a job time limit, `rfpii verify` runs at once
  (`VERIFY_CONCURRENCY`), rate limits. CORS is off unless
  `CORS_ALLOW_ORIGINS` names origins.
- **Webhooks** go only to https URLs on public addresses (checked when set
  and again, by resolving the host, before each delivery), without
  following redirects.
- **TLS** is terminated by a reverse proxy in front of the server (see
  [Deployment](#deployment)); the server itself speaks plain HTTP on
  `BIND`, which should not be reachable from outside.

## Legal points

What the platform does about the duties `legal/C-recommendations.md` sets
out (see also `legal/A-data-protection.md`). It is a summary for the
operator, not legal advice.

- **The customer is the controller; the operator is its processor**
  (§2.9). Each account's controller record names the customer as
  controller, with its registration, legal basis and purpose; the platform
  sets `operator_role: "processor"` and lists itself, and the model host
  when pages are sent to one, among the processors. The record's hash and
  main fields go into every manifest `rfpii` writes. The operator should
  have the written processing contract s.42(2)(b) and regulation 24 ask
  for, and the customer's authorisation for the model host as a
  sub-processor (regulation 25).
- **A DPIA and a controller record are required** before any job runs
  (s.31; §2.8): the API refuses a job without them, and `rfpii job`'s
  release gate holds every form until the DPIA is approved or 60 days have
  passed since it was submitted.
- **Held pages go to the cloud only on opt-in.** A held page may still show
  personal data. It is sent to the vision model only when the account sets
  `held_pages: "cloud"` (default `"none"`); published pages go to it only as
  verified crops (HARNESS.md §2, §7).
- **Transfers abroad are recorded.** Every call that sends a page to a
  model host outside Kenya is recorded in `transfers.jsonl` with what
  regulation 41(2) asks (date and time, recipient, justification,
  description of the data, hashes of the images); it is part of every job's
  outputs (§2.1). Name hosts in Kenya in `KENYA_HOSTS`.
- **Regulation 26: election data in Kenya.** Processing "for the purpose of
  … facilitating the conduct of elections" must go "through a server and
  data centre located in Kenya", or keep "at least one serving copy" in a
  data centre in Kenya (Data Protection (General) Regulations 2021, reg. 26;
  `legal/A-data-protection.md` §4.1.8). Host the platform in Kenya
  (`PLATFORM_COUNTRY=KE`), or keep a serving copy of the data there. A
  model host abroad is a transfer, and the ODPC is reported to read reg. 26
  strictly for election data; a model host in Kenya (`OLLAMA_HOST`,
  `KENYA_HOSTS`) avoids it.
- **Breaches and withdrawals** (§2.11): a leak found on a released form is
  handled with `rfpii job-incident` in the job's folder (withdrawal from
  `public/`, the incident recorded, the s.43 deadlines set: the
  Commissioner within 72 hours, and the controller within 48 hours when the
  operator is its processor). The operator runs it and notifies the
  customer.
- **Retention** (§2.10): see [Security and privacy](#security-and-privacy);
  the defaults keep no original and keep outputs 30 days, and every
  deletion is recorded.
- **Commercial use** (s.37; §2.16): a controller record with `commercial:
  true` needs `statutory_authority`, as `rfpii` requires.

## Deployment

The image needs the `rfpii` binary with its native libraries (OpenCV,
MuPDF, ONNX Runtime with the PP-OCR models, OpenBLAS, under `/opt`, built as
the main [README](README.md#install) says) and the templates. `api/Dockerfile`
builds `rfpii-api` and copies those in from a `dist/` folder you prepare:

```bash
# from the repository root, after building rfpii (rust/target/release/rfpii)
mkdir -p dist && cp rust/target/release/rfpii dist/ && cp -a /opt dist/opt
docker build -f api/Dockerfile -t rfpii-api .
```

`api/docker-compose.yml`, with a volume for the data folder and one for the
signing key:

```yaml
services:
  api:
    image: rfpii-api
    command: ["serve", "--workers", "2"]
    environment:
      BIND: 0.0.0.0:8080
      PUBLIC_BASE_URL: https://rfpii.example.co.ke
      SIGN_KEY: /keys/signing_key.pem
      RFPII_SIGN_PASSWORD: ${RFPII_SIGN_PASSWORD}
      OLLAMA_API_KEY: ${OLLAMA_API_KEY}
      TRUST_FORWARDED_FOR: "true"
      METRICS_TOKEN: ${METRICS_TOKEN}
      STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY:-}
      STRIPE_PRICE_ID: ${STRIPE_PRICE_ID:-}
      STRIPE_WEBHOOK_SECRET: ${STRIPE_WEBHOOK_SECRET:-}
    volumes:
      - rfpii-data:/data
      - ./keys:/keys:ro
    expose: ["8080"]
    stop_grace_period: 60s
  proxy:
    image: caddy:2
    command: caddy reverse-proxy --from rfpii.example.co.ke --to api:8080
    ports: ["80:80", "443:443"]
volumes:
  rfpii-data:
```

- **TLS** is the proxy's job (Caddy above obtains a certificate; nginx or a
  cloud load balancer do as well). Set `TRUST_FORWARDED_FOR=true` only
  behind such a proxy, and don't route `/metrics` through it unless
  `METRICS_TOKEN` is set.
- **Scaling** on one machine: `rfpii-api serve --workers 0` for the API and
  `rfpii-api worker --workers N` processes on the same volume. SQLite in WAL
  mode serves this; several machines would need a shared database, which
  this version doesn't have.
- **Migrations** run at start; `rfpii-api migrate` runs them alone.
- **Stopping**: `docker stop` sends SIGTERM; running jobs get
  `SHUTDOWN_GRACE_SECS` and are re-queued if they don't finish, so give the
  container a longer stop grace period.
- **Where**: in Kenya, or with a serving copy there (regulation 26, above).

## Standards

| Standard | Used for |
|---|---|
| [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110) HTTP Semantics | `202 Accepted` for jobs (§15.3.3), `Location`, `Retry-After` (§10.2.3), status codes |
| [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) Problem Details for HTTP APIs | every error |
| [draft-ietf-httpapi-ratelimit-headers-11](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/) | `RateLimit` and `RateLimit-Policy` |
| [draft-ietf-httpapi-idempotency-key-header](https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/) | `Idempotency-Key` on `POST /v1/jobs` |
| [Standard Webhooks](https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md) | webhook headers, signatures and retries |
| [Stripe billing meter events](https://docs.stripe.com/api/billing/meter-event/create) | paid usage, with `identifier` for idempotency |
| [Stripe webhook signatures](https://docs.stripe.com/webhooks#verify-manually) | `Stripe-Signature` |
| [RFC 6750](https://www.rfc-editor.org/rfc/rfc6750) Bearer tokens | `Authorization: Bearer`, `WWW-Authenticate` |
| [RFC 8288](https://www.rfc-editor.org/rfc/rfc8288) Web Linking | `Link: rel="next"` on lists |
| [OpenAPI 3.1](https://spec.openapis.org/oas/v3.1.0) | `/v1/openapi.json` |
| [Prometheus text format](https://prometheus.io/docs/instrumenting/exposition_formats/) | `/metrics` |
