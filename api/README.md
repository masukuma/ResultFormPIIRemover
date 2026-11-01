# rfpii-api

The HTTP API that serves the `rfpii` redaction harness as a paid product.
What it does, every endpoint, billing, webhooks, security, the legal points
and deployment are in [API.md](../API.md); the price in
[PRICING.md](../PRICING.md). This file is for building and running it.

It is a crate of its own (edition 2024), not part of `rust/`'s build: it runs
the `rfpii` binary as a subprocess (`RFPII_BIN`) and does not link the
`rfpii` crate, so it needs no OpenCV.

## Build

```bash
cd api
cargo build --release          # target/release/rfpii-api
```

SQLite is built in (`rusqlite`, bundled); HTTPS calls (Stripe, webhooks) use
the system's OpenSSL (`libssl-dev` to build, `libssl3` to run).

## Run

```bash
export DATA_DIR=/srv/rfpii-api RFPII_BIN=/usr/local/bin/rfpii RFPII_TEMPLATES=/opt/rfpii/templates
export SIGN_KEY=/keys/signing_key.pem RFPII_SIGN_PASSWORD=...   # the platform's key (rfpii keygen)
export OLLAMA_API_KEY=...                                       # the models (Ollama Cloud)

rfpii-api migrate                      # optional: serve and worker migrate at start
rfpii-api serve --workers 2            # the API and 2 workers on $BIND (127.0.0.1:8080)
rfpii-api serve --workers 0            # the API alone
rfpii-api worker --workers 4           # workers alone, on the same DATA_DIR
```

Put a TLS-terminating reverse proxy in front (API.md, Deployment). SIGTERM
or SIGINT stops it gracefully: no new job is leased, running jobs get
`SHUTDOWN_GRACE_SECS` to finish and are re-queued if they don't. Logs are
JSON lines on stdout (`LOG_FORMAT=text` for people), at `RUST_LOG`'s level
(`info` by default).

## Admin

The operator creates accounts and their first key, and records packages
paid outside Stripe:

```bash
rfpii-api admin create-account --name "Example Observer Group" --email ops@example.org   # prints acct_…
rfpii-api admin create-key --account acct_… --name ops                                    # prints rfpii_live_…, once
rfpii-api admin create-key --account acct_… --scopes "jobs:read jobs:write"
rfpii-api admin add-credits --account acct_… --packages 2 --reference "bank transfer 123" # 200 forms, $81.20 with VAT
rfpii-api admin adjust-credits --account acct_… --forms -100 --reason "refund of PR-17"  # or a positive grant
rfpii-api admin list-accounts [--json]                                                    # with credits available
```

`add-credits` records a paid purchase (method `manual`) at the package
price with VAT and no card fee, named by the payment's reference, and adds
its forms to the account's credits; `adjust-credits` adds credits or takes
back available ones. With `ADMIN_TOKEN` set, the same is served under
`/v1/admin/*` for a bearer of that token (`GET`/`POST /v1/admin/accounts`,
`POST /v1/admin/accounts/{id}/keys`, `POST /v1/admin/accounts/{id}/credits`,
`POST /v1/admin/accounts/{id}/credit-adjustments`), and
`GET /v1/admin/jobs/{id}` gives any job with what it cost to run. Every
change is written to the audit log. With Stripe configured, customers buy
packages themselves (`POST /v1/billing/checkout`), and Stripe's webhook adds
their credits once the payment is in.

### Stripe

1. Create a tax rate in Stripe, in the Dashboard or with
   `POST /v1/tax_rates` (`display_name=VAT`, `percentage=16`,
   `inclusive=false`, `country=KE`): 16%, **exclusive**. Its id (`txr_…`)
   is `STRIPE_TAX_RATE_ID`. The API puts it on both Checkout lines, so Stripe
   adds 16% to each; a webhook whose total differs from the quote credits
   nothing (the purchase is recorded as `mismatch`).
2. Add a webhook endpoint at `https://<host>/v1/billing/stripe-webhook`
   for `checkout.session.completed`, `checkout.session.async_payment_succeeded`
   and `checkout.session.expired`; its signing secret is
   `STRIPE_WEBHOOK_SECRET`.
3. Set `STRIPE_SECRET_KEY`. Charges are in US dollars; check the card fee
   (`CARD_FEE_BASIS_POINTS`, `CARD_FEE_FIXED_CENTS`) against the account's
   pricing (PRICING.md).

## Configuration

Each setting is an environment variable and a flag of the same name
(`--data-dir` for `DATA_DIR`; the flag wins). `rfpii-api --help` lists them.

| Variable | Default | What it sets |
|---|---|---|
| `DATA_DIR` | `./data` | the database (`rfpii-api.sqlite3`), uploads (`files/`), job folders (`jobs/`), `tmp/` |
| `BIND` | `127.0.0.1:8080` | where the HTTP server listens (plain HTTP, behind the proxy) |
| `PUBLIC_BASE_URL` | `http://localhost:8080` | the API's public URL (Stripe Checkout's return URLs) |
| `WORKERS` | `2` | workers of `serve` and `worker` (`--workers`) |
| `RFPII_BIN` | `rfpii` | the rfpii binary |
| `RFPII_TEMPLATES` | `/opt/rfpii/templates` | the templates every job gets |
| `SIGN_KEY` | `/keys/signing_key.pem` | the platform's manifest signing key; its passphrase in `RFPII_SIGN_PASSWORD`, passed to `rfpii` |
| `CERTIFICATE` | none | a certificate to sign the PDFs with; its password in `RFPII_CERT_PASSWORD` |
| `OLLAMA_HOST` | none (Ollama Cloud) | the model host jobs use; `OLLAMA_API_KEY` is passed to `rfpii` |
| `ORCHESTRATOR_MODELS` | `gpt-oss:120b` | orchestrator models, in order of preference; empty for none |
| `VISION_MODELS` | `gemma4:31b` | vision models, in order of preference; empty for none (then the release gate holds every form) |
| `KENYA_HOSTS` | none | model hosts in Kenya: calls to them are not transfers abroad |
| `MODEL_REQUESTS_PER_FORM` | `40` | a job's model-request budget, per form |
| `JOB_TIMEOUT_SECS` | `43200` | a job's wall-clock limit |
| `FREE_FORMS` | `5` | free forms every account has once (never renewed); after them, forms come from packages |
| `PACKAGE_FORMS` | `100` | forms in a package (beyond the free forms, forms are sold in packages) |
| `PACKAGE_PRICE_CENTS` | `3500` | a package's price, before the card fee and VAT ($35.00: $0.35 a form, PRICING.md) |
| `VAT_RATE_BASIS_POINTS` | `1600` | VAT on every purchase line (16%); Stripe's tax rate must match it |
| `CARD_FEE_BASIS_POINTS` | `440` | the card processor's rate of the whole charge (4.4%: 2.9% + 1.5% for international cards) |
| `CARD_FEE_FIXED_CENTS` | `30` | the card processor's fixed fee per charge ($0.30) |
| `MAX_PACKAGES_PER_PURCHASE` | `100` | the most packages one purchase may hold (1 to 10000) |
| `CREDIT_EXPIRY_DAYS` | `0` | days credits last once added (0: they never expire); a job keeps what it reserved |
| `MODEL_PRICES` | `gemma4:31b=140000/400000,gpt-oss:120b=150000/600000` | token prices for the actual cost, micro-dollars per million tokens, input/output |
| `SERVER_PRICE_PER_VCPU_HOUR_MICROUSD` | `59850` | the server's price per vCPU-hour, for the actual cost |
| `RETENTION_INPUT_DAYS` | `0` | days a job's inputs are kept after it ends (0: deleted when it ends) |
| `RETENTION_OUTPUT_DAYS` | `30` | days a job's outputs are kept after it ends |
| `RETENTION_UNUSED_FILE_HOURS` | `24` | hours an upload no job used is kept |
| `SWEEP_INTERVAL_SECS` | `300` | seconds between retention sweeps |
| `MAX_UPLOAD_BYTES` | `536870912` (512 MiB) | the largest upload |
| `MAX_ZIP_ENTRIES` | `20000` | the most entries a zip may hold |
| `MAX_ZIP_UNCOMPRESSED_BYTES` | `4294967296` (4 GiB) | the most a zip may expand to |
| `MAX_FILES_PER_JOB` | `1000` | the most files one job may name |
| `MAX_FORMS_PER_JOB` | `100000` | the most forms one job may hold |
| `MAX_JSON_BYTES` | `1048576` | the largest JSON body |
| `MAX_VERIFY_BYTES` | `67108864` | the largest `POST /v1/verify` body |
| `VERIFY_CONCURRENCY` | `2` | `rfpii verify` processes at once |
| `REQUEST_TIMEOUT_SECS` | `300` | an HTTP request's time limit |
| `RATE_LIMIT_PER_SEC`, `RATE_LIMIT_BURST` | `10`, `60` | the token bucket of each API key |
| `PUBLIC_RATE_LIMIT_PER_SEC`, `PUBLIC_RATE_LIMIT_BURST` | `0.2`, `10` | the token bucket of each IP address on the public routes |
| `TRUST_FORWARDED_FOR` | `false` | take the client's address from the last `X-Forwarded-For` entry (behind a proxy only) |
| `CORS_ALLOW_ORIGINS` | none (CORS off) | origins browsers may call the API from |
| `ADMIN_TOKEN` | none (routes off) | the token of `/v1/admin/*` |
| `METRICS_TOKEN` | none (open) | the token of `/metrics` |
| `STRIPE_SECRET_KEY` | none (no Checkout) | Stripe's secret key: Checkout Sessions for purchases |
| `STRIPE_TAX_RATE_ID` | none (no Checkout) | the Stripe tax rate (`txr_…`, VAT 16% exclusive) put on each Checkout line |
| `STRIPE_WEBHOOK_SECRET` | none | the signing secret of the Stripe webhook endpoint |
| `STRIPE_API_BASE` | `https://api.stripe.com` | Stripe's API (a stub in tests) |
| `STRIPE_SIGNATURE_TOLERANCE_SECS` | `300` | the oldest `Stripe-Signature` timestamp accepted |
| `WEBHOOK_RETRY_SCHEDULE` | `5,30,120,600,3600,21600` | seconds before each webhook retry |
| `WEBHOOK_ALLOW_PRIVATE` | `false` | allow webhooks over plain HTTP and to private addresses |
| `WEBHOOK_TIMEOUT_SECS` | `10` | one delivery's time limit |
| `LEASE_SECS` | `60` | a worker's lease on a job without a heartbeat |
| `MAX_ATTEMPTS` | `3` | times a job is started before it fails |
| `POLL_INTERVAL_MS` | `1000` | how often idle workers and background tasks look for work |
| `SHUTDOWN_GRACE_SECS` | `30` | how long running jobs get at shutdown |
| `PLATFORM_NAME`, `PLATFORM_COUNTRY` | `rfpii API`, `KE` | the platform, as a processor in every controller record |
| `MODEL_HOST_NAME`, `MODEL_HOST_COUNTRY` | `Ollama`, `US` | the model host, as a processor when pages go to it |
| `PRICING_URL` | PRICING.md on GitHub | where the price is explained (`/v1/pricing`, 402 answers) |
| `PROBLEM_TYPE_BASE` | API.md on GitHub | the base of problem type URIs |
| `LOG_FORMAT` | `json` | `json` or `text` |

`rfpii` reads its own variables (`RFPII_SIGN_PASSWORD`, `RFPII_CERT_PASSWORD`,
`RFPII_OCR_MODELS`, `RFPII_ORT_LIB`, `OLLAMA_API_KEY`, …): a job's process
gets those, the proxy and locale variables, and nothing else of the API's
environment.

## Layout

| Path | Holds |
|---|---|
| `src/http/` | routes and middleware: request id, trace, problem details, timeout, body limit, metrics, keys, rate limits |
| `src/worker.rs` | leasing, running and settling jobs |
| `src/billing.rs` | the free forms, the credit ledger, admission, settlement, usage, the actual cost |
| `src/quote.rs` | what a purchase of packages costs: the card fee and VAT, in cents |
| `src/purchases.rs`, `src/stripe.rs` | purchases; Stripe Checkout and its webhook |
| `src/webhooks.rs`, `src/sweeper.rs` | the background tasks |
| `src/accounts.rs`, `src/auth.rs` | accounts, the controller record and DPIA; API keys |
| `src/rfpii.rs` | the `rfpii` subprocess: inventory, verify, the job file |
| `migrations/` | the database schema, versioned |
| `openapi.json` | the OpenAPI 3.1 description, served at `/v1/openapi.json` |
| `tests/api/` | end-to-end tests with a fake `rfpii` (`fake-rfpii.sh`) and one with the real binary |

## Tests

```bash
cd api && cargo test && cargo clippy --all-targets
```

The tests start the server on a random port with a temporary data folder
and a fake `rfpii` (a shell script answering `inventory`, `job` and
`verify`), with local stand-ins for webhook receivers and Stripe; none
reaches the network. `tests/api/real.rs` runs a whole job and a
verification with the real binary at `/home/user/apiwork/bin/rfpii` (or
`$RFPII_REAL_BIN`), on synthetic forms from `rfpii demo`, and says it was
skipped if the binary isn't there.
