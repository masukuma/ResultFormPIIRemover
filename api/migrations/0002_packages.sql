-- Prepaid packages replace the metered plan (PRICING.md). Accounts no
-- longer have a plan: every account has the month's free allowance and
-- draws any further forms from its form credits, bought in packages.
-- Purchases are in integer cents; the operator's cost stays in
-- micro-dollars.

-- A purchase of packages: by Stripe Checkout (pending until Stripe says
-- the session was paid), or recorded by the operator for a payment made
-- outside Stripe (method 'manual'). Its quote is kept line by line, as a
-- tax invoice needs it.
CREATE TABLE purchases (
    id                    TEXT PRIMARY KEY,
    account_id            TEXT NOT NULL REFERENCES accounts(id),
    method                TEXT NOT NULL CHECK (method IN ('stripe', 'manual')),
    status                TEXT NOT NULL CHECK (status IN ('pending', 'paid', 'expired', 'mismatch')),
    packages              INTEGER NOT NULL CHECK (packages > 0),
    forms                 INTEGER NOT NULL CHECK (forms > 0),
    package_price_cents   INTEGER NOT NULL,
    net_cents             INTEGER NOT NULL,
    card_fee_cents        INTEGER NOT NULL,
    vat_rate_basis_points INTEGER NOT NULL,
    vat_net_cents         INTEGER NOT NULL,
    vat_card_fee_cents    INTEGER NOT NULL,
    vat_cents             INTEGER NOT NULL,
    total_cents           INTEGER NOT NULL,
    currency              TEXT NOT NULL DEFAULT 'usd',
    stripe_session_id     TEXT UNIQUE,
    -- the operator's reference of a manual purchase (a bank transfer)
    reference             TEXT,
    -- what Stripe said was paid, when it differs from the quote ('mismatch')
    amount_received_cents INTEGER,
    currency_received     TEXT,
    created_at            INTEGER NOT NULL,
    paid_at               INTEGER,
    -- when it was found expired or mismatched
    closed_at             INTEGER
);
CREATE INDEX purchases_account ON purchases(account_id, id);

-- Every change to an account's form credits, signed: a purchase adds
-- forms (a "lot"), a settled job, an expiry or a negative adjustment takes
-- them from a lot (lot_id), a positive adjustment is a lot of its own.
-- An account's balance is the sum of its entries.
CREATE TABLE credit_ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id  TEXT NOT NULL REFERENCES accounts(id),
    at          INTEGER NOT NULL,
    forms       INTEGER NOT NULL CHECK (forms <> 0),
    kind        TEXT NOT NULL CHECK (kind IN ('purchase', 'job', 'expiry', 'adjustment')),
    purchase_id TEXT REFERENCES purchases(id),
    job_id      TEXT REFERENCES jobs(id),
    -- the lot a negative entry takes from; NULL for a lot
    lot_id      INTEGER REFERENCES credit_ledger(id),
    -- when a lot's remaining forms expire (CREDIT_EXPIRY_DAYS); NULL: never
    expires_at  INTEGER,
    actor       TEXT NOT NULL,
    note        TEXT,
    CHECK ((forms > 0) = (lot_id IS NULL))
);
CREATE INDEX credit_ledger_account ON credit_ledger(account_id, id);
CREATE INDEX credit_ledger_lot ON credit_ledger(lot_id);
-- a purchase adds its forms once
CREATE UNIQUE INDEX credit_ledger_purchase ON credit_ledger(purchase_id) WHERE purchase_id IS NOT NULL;

-- Jobs reserve free forms and credits, and settle into both. A free
-- job not yet settled keeps its reservation; a paid one's (metered) is let
-- go: it settles on what free forms its month has left, then nothing.
ALTER TABLE jobs ADD COLUMN reserved_free INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN reserved_credits INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN free_forms INTEGER;
ALTER TABLE jobs ADD COLUMN credit_forms INTEGER;
UPDATE jobs SET reserved_free = reserved_forms WHERE settled = 0 AND plan = 'free';
UPDATE jobs SET free_forms = CASE WHEN plan = 'free' THEN COALESCE(forms_charged, 0) ELSE 0 END, credit_forms = 0 WHERE settled = 1;
ALTER TABLE jobs RENAME COLUMN forms_charged TO forms_counted;
ALTER TABLE jobs DROP COLUMN reserved_forms;
ALTER TABLE jobs DROP COLUMN plan;
ALTER TABLE jobs DROP COLUMN unit_price_microusd;
ALTER TABLE jobs DROP COLUMN amount_microusd;

-- One usage record per settled job that counted forms: how many came
-- from the month's free allowance and how many from credits. Records made
-- under the metered plan keep its plan, unit price and amount (legacy_*);
-- their free forms still count against their month's allowance.
CREATE TABLE usage_records_v2 (
    id                         TEXT PRIMARY KEY,
    account_id                 TEXT NOT NULL REFERENCES accounts(id),
    job_id                     TEXT NOT NULL UNIQUE REFERENCES jobs(id),
    period                     TEXT NOT NULL,
    forms                      INTEGER NOT NULL,
    free_forms                 INTEGER NOT NULL,
    credit_forms               INTEGER NOT NULL,
    created_at                 INTEGER NOT NULL,
    legacy_plan                TEXT,
    legacy_unit_price_microusd INTEGER,
    legacy_amount_microusd     INTEGER
);
INSERT INTO usage_records_v2 (id, account_id, job_id, period, forms, free_forms, credit_forms, created_at, legacy_plan, legacy_unit_price_microusd, legacy_amount_microusd)
    SELECT id, account_id, job_id, period, forms, CASE WHEN plan = 'free' THEN forms ELSE 0 END, 0, created_at, plan, unit_price_microusd, amount_microusd
    FROM usage_records;
DROP TABLE usage_records;
ALTER TABLE usage_records_v2 RENAME TO usage_records;
CREATE INDEX usage_account_period ON usage_records(account_id, period);

-- No plan, spend cap or subscription. The Stripe customer, if Checkout
-- made one, is kept.
ALTER TABLE accounts DROP COLUMN plan;
ALTER TABLE accounts DROP COLUMN spend_cap_microusd;
ALTER TABLE accounts DROP COLUMN stripe_subscription_id;
