-- The first schema. Times are Unix seconds (UTC); months are 'YYYY-MM';
-- money is integer micro-dollars (1 USD = 1,000,000).

CREATE TABLE accounts (
    id                     TEXT PRIMARY KEY,
    name                   TEXT NOT NULL,
    email                  TEXT,
    plan                   TEXT NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'paid')),
    -- monthly spend cap of a paid account; NULL: none
    spend_cap_microusd     INTEGER,
    -- the controller record as the customer gave it (JSON), and its DPIA
    controller_json        TEXT,
    dpia_json              TEXT,
    held_pages             TEXT NOT NULL DEFAULT 'none' CHECK (held_pages IN ('none', 'cloud')),
    webhook_url            TEXT,
    webhook_secret         TEXT,
    stripe_customer_id     TEXT UNIQUE,
    stripe_subscription_id TEXT,
    created_at             INTEGER NOT NULL
);

CREATE TABLE api_keys (
    id           TEXT PRIMARY KEY,
    account_id   TEXT NOT NULL REFERENCES accounts(id),
    name         TEXT,
    -- the first characters of the key, to tell keys apart; never the key
    prefix       TEXT NOT NULL,
    sha256       TEXT NOT NULL UNIQUE,
    scopes       TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    last_used_at INTEGER,
    revoked_at   INTEGER
);
CREATE INDEX api_keys_account ON api_keys(account_id);

CREATE TABLE files (
    id         TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    -- the client's file name, reduced to safe characters; storage is by id
    name       TEXT,
    kind       TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    sha256     TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    deleted_at INTEGER,
    deleted_why TEXT
);
CREATE INDEX files_account ON files(account_id, id);

CREATE TABLE jobs (
    id                 TEXT PRIMARY KEY,
    account_id         TEXT NOT NULL REFERENCES accounts(id),
    status             TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'canceled')),
    request_json       TEXT NOT NULL,
    plan               TEXT NOT NULL,
    unit_price_microusd INTEGER NOT NULL,
    period             TEXT NOT NULL,
    forms_requested    INTEGER NOT NULL,
    -- forms held against the month's allowance (or spend cap) until settled
    reserved_forms     INTEGER NOT NULL DEFAULT 0,
    settled            INTEGER NOT NULL DEFAULT 0,
    forms_released     INTEGER,
    forms_held         INTEGER,
    forms_charged      INTEGER,
    amount_microusd    INTEGER,
    cost_json          TEXT,
    cost_microusd      INTEGER,
    inventory_json     TEXT,
    exit_code          INTEGER,
    error              TEXT,
    attempts           INTEGER NOT NULL DEFAULT 0,
    max_attempts       INTEGER NOT NULL,
    lease_owner        TEXT,
    lease_expires_at   INTEGER,
    cancel_requested   INTEGER NOT NULL DEFAULT 0,
    webhook_url        TEXT,
    metadata_json      TEXT,
    created_at         INTEGER NOT NULL,
    started_at         INTEGER,
    finished_at        INTEGER,
    inputs_deleted_at  INTEGER,
    outputs_expire_at  INTEGER,
    outputs_deleted_at INTEGER,
    deleted_at         INTEGER
);
CREATE INDEX jobs_account ON jobs(account_id, id);
CREATE INDEX jobs_queue ON jobs(status, created_at);

CREATE TABLE job_files (
    job_id  TEXT NOT NULL REFERENCES jobs(id),
    file_id TEXT NOT NULL REFERENCES files(id),
    -- the name the file has in the job's in/ folder
    name    TEXT NOT NULL,
    PRIMARY KEY (job_id, file_id)
);
CREATE INDEX job_files_file ON job_files(file_id);

-- tokens each job's models used, as its report counted them
CREATE TABLE job_model_usage (
    job_id         TEXT NOT NULL REFERENCES jobs(id),
    host           TEXT NOT NULL,
    model          TEXT NOT NULL,
    requests       INTEGER NOT NULL,
    input_tokens   INTEGER NOT NULL,
    output_tokens  INTEGER NOT NULL,
    cost_microusd  INTEGER NOT NULL
);
CREATE INDEX job_model_usage_job ON job_model_usage(job_id);

CREATE TABLE idempotency_keys (
    account_id   TEXT NOT NULL,
    key          TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    PRIMARY KEY (account_id, key)
);

-- one record per settled job with forms released; the bill is their sum
CREATE TABLE usage_records (
    id                  TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL REFERENCES accounts(id),
    job_id              TEXT NOT NULL UNIQUE REFERENCES jobs(id),
    period              TEXT NOT NULL,
    plan                TEXT NOT NULL,
    forms               INTEGER NOT NULL,
    unit_price_microusd INTEGER NOT NULL,
    amount_microusd     INTEGER NOT NULL,
    created_at          INTEGER NOT NULL,
    stripe_reported_at  INTEGER,
    stripe_attempts     INTEGER NOT NULL DEFAULT 0,
    stripe_next_at      INTEGER NOT NULL DEFAULT 0,
    stripe_claimed_until INTEGER,
    stripe_error        TEXT
);
CREATE INDEX usage_account_period ON usage_records(account_id, period);

CREATE TABLE webhook_deliveries (
    id             TEXT PRIMARY KEY,
    account_id     TEXT NOT NULL REFERENCES accounts(id),
    job_id         TEXT,
    event          TEXT NOT NULL,
    url            TEXT NOT NULL,
    payload        TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'delivered', 'failed')),
    attempts       INTEGER NOT NULL DEFAULT 0,
    next_attempt_at INTEGER NOT NULL,
    claimed_until  INTEGER,
    last_status    INTEGER,
    last_error     TEXT,
    created_at     INTEGER NOT NULL,
    delivered_at   INTEGER
);
CREATE INDEX webhook_due ON webhook_deliveries(status, next_attempt_at);

CREATE TABLE stripe_events (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    received_at INTEGER NOT NULL
);

-- changes to keys, plans and settings: who, what, when
CREATE TABLE audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         INTEGER NOT NULL,
    account_id TEXT,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    detail     TEXT NOT NULL
);

-- every file or folder deleted under the retention rules or on request
CREATE TABLE deletions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         INTEGER NOT NULL,
    account_id TEXT NOT NULL,
    job_id     TEXT,
    file_id    TEXT,
    what       TEXT NOT NULL,
    reason     TEXT NOT NULL
);
