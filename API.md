# The rfpii API

An HTTP API that sells the `rfpii` redaction harness ([HARNESS.md](HARNESS.md))
as a service. A customer uploads election result forms, starts a job, and
downloads what the release gate let out: each published form with its signed
manifest, its ledger entry and a batch root, so anyone can check that nothing
but personal data was touched ([VERIFYING.md](VERIFYING.md)). Every account
has 5 free forms a month; beyond them, forms are bought in packages of 100
for $35.00 ($0.35 a form), plus the card fee and VAT ([PRICING.md](PRICING.md)).

The server is `rfpii-api`, a Rust crate in [`api/`](api/README.md). It calls
the `rfpii` binary as a subprocess and does not link it.

Contents: [Architecture](#architecture) · [Resources](#resources) ·
[Authentication](#authentication) · [Endpoints](#endpoints) ·
[Job lifecycle](#job-lifecycle) · [Webhooks](#webhooks) ·
[Free forms, packages and billing](#free-forms-packages-and-billing) · [Rate limits](#rate-limits) ·
[Errors](#errors) · [Security and privacy](#security-and-privacy) ·
[Legal points](#legal-points) · [Deployment](#deployment) ·
[Standards](#standards)

## Architecture

```
   clients ──HTTPS──► reverse proxy (TLS) ──HTTP──► rfpii-api serve
                                                     │  HTTP API (axum)
                                                     │  N workers ──► rfpii job (one process per job)
   Stripe ◄── Checkout Sessions (payment mode) ──────┤  webhook deliveries ──► customers' URLs
   Stripe ──► /v1/billing/stripe-webhook             │  retention sweeper
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
  `rfpii inventory --json`, reserves them against the month's free forms
  and the account's credits, and queues the job. It sells packages through
  Stripe Checkout, and runs `rfpii verify` for anyone checking a published
  form.
- **Workers** lease queued jobs from SQLite (an atomic `UPDATE … RETURNING`
  with a lease; a heartbeat renews it). Each job runs as one `rfpii job`
  process in its own folder, in its own process group, with a time limit.
  When it ends, the worker reads `report.json`, settles the forms it
  released against the free forms and credits, records the job's actual
  cost, queues its webhook and deletes the originals.
- **Background tasks** in every process deliver webhooks and sweep files
  past their retention period.
- `rfpii-api serve --workers 0` serves the API alone; `rfpii-api worker`
  runs workers alone. Any number of each can share one database and data
  folder on one machine (SQLite in WAL mode with a busy timeout).

`rfpii job` makes every redaction and decides what is released (HARNESS.md
§2, §9). The API never edits a form and never overrides the gate: it only
serves what `rfpii job` put in `public/`.

## Resources

| Resource | Id | What it is |
|---|---|---|
| Account | `acct_…` | A customer: its free forms and form credits, its controller record and DPIA, its settings |
| API key | `key_…` | `rfpii_live_…`, with scopes; only its SHA-256 and prefix are stored |
| File | `file_…` | An uploaded PDF, JPEG, PNG, TIFF or zip, stored by id; never served back |
| Job | `job_…` | One run of `rfpii job` on some files: status, forms, what they counted against, outputs |
| Purchase | `pur_…` | Packages bought: by Stripe Checkout, or paid otherwise and recorded by the operator; its lines (packages, card fee, VAT on each), total and status |
| Usage record | `usage_…` | The forms a settled job counted: how many were free, how many credits |
| Webhook delivery | `msg_…` | One event to one URL, with every attempt |

Ids are a prefix and a UUIDv7, so they sort by creation time. Times are RFC
3339 in UTC; months are `YYYY-MM` (UTC). Prices, quotes and purchases are
integer cents (`4315` is $43.15), with a `*_usd` string beside them; rates
are basis points (`1600` is 16%); the operator's cost of a job is integer
micro-dollars.

A **form** is a document as `rfpii inventory` counts it, and as `rfpii job`
plans it (`rust/src/job/plan.rs`): each readable file that is not a copy of
another is one form, or one per page when it is redacted page by page (a
scan of several pages, or any file with `per_page: yes`).

## Authentication

Every route under `/v1/` except `/v1/pricing`, `/v1/billing/quote`,
`/v1/verify`, `/v1/openapi.json` and the Stripe webhook needs an API key:

```
Authorization: Bearer rfpii_live_3kTq0sV9cA2bX7mN4pR8wY1zL5dF6gH0jK2eU9iO3tQ
```

A key is `rfpii_live_` and 32 random bytes in base62 (43 characters). The
server stores its SHA-256 and its first characters (`prefix`), never the key;
it is shown once, when it is made. A key has scopes:

| Scope | Allows |
|---|---|
| `jobs:write` | upload and delete files; create, cancel and delete jobs |
| `jobs:read` | read files, jobs, outputs, usage and purchases |
| `account:admin` | set the controller record, DPIA and settings; rotate the webhook secret; manage keys; buy packages (Checkout); read purchases |

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
# This month's free forms, the form credits, the purchases, the legal records and settings
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
them against the month's free forms and the account's credits (`402`, see
[Free forms, packages and billing](#free-forms-packages-and-billing)),
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
  "billing": {"period": "2026-10", "reserved": {"free_forms": 1, "credits": 2}, "settled": true,
              "counted": {"forms": 2, "free_forms": 1, "credits": 1}},
  "exit_code": 2, "error": null, "attempts": 1,
  "created_at": "2026-10-31T09:00:00Z", "started_at": "…", "finished_at": "…",
  "inputs_deleted_at": "…", "outputs_expire_at": "2026-11-30T09:01:10Z",
  "links": {"self": "/v1/jobs/job_019a3c…", "report": "…/report", "outputs": "…/outputs", "archive": "…/archive"}
}
```

`billing` says what the job reserved when it was created and, once it has
ended, what its released forms counted against (`counted` is null until
then). Customers see that, not what the job cost to run. The operator does:
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
curl -s $API/v1/pricing                                # public: free forms, the package, the card fee, VAT, one package's quote
curl -s "$API/v1/billing/quote?packages=10"            # public: what 10 packages cost, line by line
curl -s "$API/v1/usage?period=2026-10" -H "Authorization: Bearer $KEY"     # the month's records; credits and purchases now
curl -s "$API/v1/billing/purchases?limit=20" -H "Authorization: Bearer $KEY"   # purchases, newest first

# Buy 10 packages by card (501 without Stripe). The return URLs are optional
# (else the API's /v1/pricing).
curl -s -X POST $API/v1/billing/checkout -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"packages": 10, "success_url": "https://observers.example.org/billing/done", "cancel_url": "https://observers.example.org/billing"}'
```

The quote, in cents:

```json
{
  "object": "quote", "currency": "usd", "packages": 10, "forms": 1000,
  "lines": [
    {"item": "packages", "description": "10 package(s) of 100 forms", "quantity": 10, "unit_amount_cents": 3500,
     "amount_cents": 35000, "vat_rate_basis_points": 1600, "vat_cents": 5600},
    {"item": "card_fee", "description": "card processing fee (the processor's 4.4% + $0.30 of the charge)", "quantity": 1,
     "unit_amount_cents": 1914, "amount_cents": 1914, "vat_rate_basis_points": 1600, "vat_cents": 306}
  ],
  "net_cents": 35000, "card_fee_cents": 1914, "vat_rate_basis_points": 1600,
  "vat_net_cents": 5600, "vat_card_fee_cents": 306, "vat_cents": 5906,
  "total_cents": 42820, "total_usd": "428.20", "effective_price_per_form_usd": "0.4282"
}
```

Checkout answers `{object: "checkout_session", id, url, purchase}`: send the
customer to `url`. The purchase is `pending` until Stripe's webhook says the
session was paid; then it is `paid` and its forms are the account's
credits. A purchase has the quote's lines and totals, its `method`
(`stripe`, or `manual` for one the operator recorded), its `status`
(`pending`, `paid`, `expired`, `mismatch`), the Checkout Session or the
operator's `reference`, and when it was made, paid or closed.

`GET /v1/account` and `GET /v1/usage` give the free forms (`per_month`,
`used`, `reserved`, `left`), the credits (`bought`, `used`, `expired`,
`adjusted`, `reserved`, `available`) and a summary of the purchases (how
many in each status, forms bought, total paid). `POST
/v1/billing/stripe-webhook` is for Stripe; see
[Free forms, packages and billing](#free-forms-packages-and-billing).

### Operations

```bash
curl -s $API/healthz    # the process answers
curl -s $API/readyz     # database, data folder writable, rfpii binary runs (503 otherwise)
curl -s $API/metrics -H "Authorization: Bearer $METRICS_TOKEN"   # Prometheus text
```

`/metrics` has requests by route template and status
(`rfpii_api_http_requests_total`), their latency
(`rfpii_api_http_request_duration_seconds`), jobs by status, the queue
depth, forms released, model tokens by model, webhooks pending and
purchases by status (`rfpii_api_purchases`; a `mismatch` is a payment
that wasn't the quoted amount and needs the operator). Keep it off the
public proxy, or set
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
   the plan's documents less those. The job is settled against the free
   forms and credits, the cost recorded, the webhook queued, and the inputs
   deleted.

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

## Free forms, packages and billing

| | Price | What it includes |
|---|---|---|
| Free | $0 | 5 forms a calendar month (UTC), for every account |
| Package | $35.00 ($0.35 a form), plus the card fee and VAT | 100 form credits |

Accounts have no plan. Every account has the month's free forms; beyond
them, a job's forms are drawn from the account's **form credits**, bought
in packages. A purchase of *n* packages costs the packages, a card-fee line
(the smallest amount with which the operator keeps the packages' price
after the card processor's 4.4% + $0.30), and 16% VAT on each line: $43.15
for one package, $428.20 for ten. How the price was set and the quote are
in [PRICING.md](PRICING.md); every figure is configuration
([`api/README.md`](api/README.md#configuration)).

- **Only released forms count.** A held form, or one in a group the gate
  held, uses neither a free form nor a credit. A job never counts more forms
  than it reserved (if the orchestrator chose to read a file page by page
  where the count read it whole, the extra forms are free).
- **Reserve, then settle.** A job reserves its forms when it is created:
  the free forms its month has left first, then credits that are available
  (bought, less used, expired and reserved). When both together are too
  few, the job is refused with `402 insufficient-credits`, naming the forms
  requested, the free forms left, the credits available, and how to buy
  (`packages_needed`, `GET /v1/billing/quote`, `POST /v1/billing/checkout`).
  When the job ends, the forms it released are counted: as free forms, as
  many as its month has left (its own reservation included), then as
  credits (never more than it reserved); one usage record says how many of
  each, and the rest of the reservation is let go. A failed or canceled job
  counts nothing. A job draws on the free forms of the month it was created
  in; a new month brings new free forms, and leaves credits as they are.
  Admission and settlement each run in one `IMMEDIATE` SQLite transaction,
  so jobs started at once never reserve more than there is.
- **Credits** are kept in a ledger of signed entries (a purchase adds its
  forms; a settled job, an expiry or the operator's adjustment takes
  some), so the balance is their sum and every change can be traced.
  Credits don't expire unless `CREDIT_EXPIRY_DAYS` is set; then a purchase's
  credits expire that many days after it, except what a running job holds,
  and jobs use the soonest-expiring credits first.
- **With Stripe** (`STRIPE_SECRET_KEY`, `STRIPE_TAX_RATE_ID`,
  `STRIPE_WEBHOOK_SECRET`): `POST /v1/billing/checkout` with `{packages}`
  records a pending purchase at today's quote and makes a Checkout Session
  in `payment` mode with two lines priced in the request (`price_data`): the
  packages (`unit_amount` the package price, `quantity` the packages) and
  the card fee (quantity 1), each with the tax rate `STRIPE_TAX_RATE_ID`;
  `client_reference_id` is the account and the metadata names the purchase
  and its packages. The operator creates that tax rate in Stripe once (in
  the Dashboard, or `POST /v1/tax_rates` with `percentage=16`,
  `inclusive=false`, `display_name=VAT`, `country=KE`): 16%, **exclusive**,
  so Stripe adds it to each line ([`api/README.md`](api/README.md#stripe)). Stripe's webhook
  (`POST /v1/billing/stripe-webhook`, its `Stripe-Signature` checked with a
  300-second tolerance, each event handled once) settles the purchase:
  `checkout.session.completed` (or `checkout.session.async_payment_succeeded`)
  with `payment_status: paid` marks it paid and adds `packages × 100` forms
  to the account's credits, once, if the session's `amount_total` is the
  quote's total and its currency `usd`; otherwise the purchase becomes
  `mismatch`, what Stripe charged is recorded, nothing is added, and the
  operator settles it by hand (a tax rate that isn't 16% exclusive does
  this). `checkout.session.expired` marks the purchase expired. Send those
  three events to the endpoint. Without `STRIPE_SECRET_KEY` and
  `STRIPE_TAX_RATE_ID`, Checkout answers `501`.
- **Without Stripe** the operator records packages paid otherwise (a bank
  transfer): `rfpii-api admin add-credits --account acct_… --packages 2
  --reference "bank transfer 123"` (or `POST /v1/admin/accounts/{id}/credits`)
  records a paid purchase with VAT and no card fee and adds its forms;
  `admin adjust-credits` adds credits or takes back available ones, with a
  reason. Both are in the audit log.
- **Tax invoices.** The platform issues none; each purchase carries what a
  KRA eTIMS invoice needs (buyer, date, lines with quantity, unit price and
  VAT, totals, reference), for the operator (PRICING.md).

## Rate limits

A token bucket per API key (`RATE_LIMIT_BURST` requests, refilled at
`RATE_LIMIT_PER_SEC`), and one per client IP address on the public routes
(`/v1/verify`, `/v1/pricing`, `/v1/billing/quote`, `/v1/openapi.json`).
Responses carry the
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
  "type": "https://github.com/masukuma/ResultFormPIIRemover/blob/main/API.md#problem-insufficient-credits",
  "title": "Not enough free forms or credits",
  "status": 402,
  "detail": "this job has 3 form(s); the account has 2 of its 5 free forms left in 2026-10 and 0 credit(s) available. Buy 1 package(s) of 100 forms ($35.00 each before the card fee and VAT): GET /v1/billing/quote?packages=1 gives the price, POST /v1/billing/checkout takes the payment; or ask the operator",
  "instance": "urn:rfpii:request:0b7e1c1e-5d1f-4a43-9a1e-2f3c1c9d7a10",
  "request_id": "0b7e1c1e-5d1f-4a43-9a1e-2f3c1c9d7a10",
  "forms_requested": 3, "free_forms_left": 2, "free_forms_per_month": 5, "credits_available": 0, "period": "2026-10",
  "buy": {"packages_needed": 1, "package_forms": 100, "package_price_cents": 3500, "quote": "GET /v1/billing/quote?packages=1",
          "checkout": "POST /v1/billing/checkout", "checkout_available": true, "pricing": "https://…/PRICING.md"}
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
| <a id="problem-insufficient-credits"></a>`insufficient-credits` | 402 | more forms than the month's free forms left and the credits available together (`forms_requested`, `free_forms_left`, `credits_available`, `buy`) |
| <a id="problem-rate-limited"></a>`rate-limited` | 429 | over the rate limit (`retry_after`) |
| <a id="problem-invalid-signature"></a>`invalid-signature` | 400 | a Stripe webhook whose signature doesn't verify |
| <a id="problem-billing-not-configured"></a>`billing-not-configured` | 501 | Stripe Checkout is not configured (`STRIPE_SECRET_KEY` and `STRIPE_TAX_RATE_ID`), or the webhook has no secret |
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
- **The audit log** records every change to keys, purchases and credits
  (checkout started, paid, mismatched or expired; credits added or
  adjusted by the operator), the controller record, the DPIA, settings
  and the webhook secret: who (key, admin
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
      STRIPE_TAX_RATE_ID: ${STRIPE_TAX_RATE_ID:-}
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
| [Stripe Checkout Sessions](https://docs.stripe.com/api/checkout/sessions/create) | purchases of packages: `mode=payment`, `line_items[].price_data`, `line_items[].tax_rates` |
| [Stripe tax rates](https://docs.stripe.com/api/tax_rates) | VAT (16%, exclusive) on each Checkout line |
| [Stripe webhook signatures](https://docs.stripe.com/webhooks#verify-manually) | `Stripe-Signature` |
| [RFC 6750](https://www.rfc-editor.org/rfc/rfc6750) Bearer tokens | `Authorization: Bearer`, `WWW-Authenticate` |
| [RFC 8288](https://www.rfc-editor.org/rfc/rfc8288) Web Linking | `Link: rel="next"` on lists |
| [OpenAPI 3.1](https://spec.openapis.org/oas/v3.1.0) | `/v1/openapi.json` |
| [Prometheus text format](https://prometheus.io/docs/instrumenting/exposition_formats/) | `/metrics` |
