# Pricing

What one form costs to redact through the harness, measured on real 2022
Forms 34A against Ollama Cloud on 31 October 2026 (the operator's cost
basis), and the price the API charges: $0.35 a form, sold in packages of
100, with the card fee and VAT on top.

## The price

| | Price | What it includes |
|---|---|---|
| Free | $0 | 5 forms a calendar month (UTC), for every account |
| Package | **$35.00 for 100 forms** ($0.35 a form), before the card fee and VAT | 100 form credits |

The owner set the price at $0.35 a form, sold only in packages of 100. A
customer buys one or more packages at a time (up to 100 in one purchase).
Card processing fees are paid by the customer, as a line of their own, and
every purchase carries VAT at 16%, Kenya's standard rate, on the whole
purchase: the packages and the card fee.

A form is a document as `rfpii inventory` counts it: a file, or one page of
a scanned batch read page by page. Only forms a job **releases** are
counted: a held form, or one in a group the release gate holds back, uses
neither a free form nor a credit. A job's forms come from the month's free
forms first, then from credits; credits don't expire (an operator can set
an expiry, `CREDIT_EXPIRY_DAYS`). How jobs reserve and settle forms is in
[API.md](API.md#free-forms-packages-and-billing).

## What a purchase costs

For *n* packages, with VAT rate *v* (16%) and the card processor's rate *p*
and fixed fee *f* (4.4% and $0.30, below), in whole cents:

- the packages: *B* = *n* × $35.00;
- VAT on a line of *x* cents: VAT(*x*) = 16% of *x*, rounded half up to the
  cent;
- the card fee *S*: the smallest whole number of cents such that the
  operator keeps at least *B*, where the total is
  *T* = *B* + *S* + VAT(*B*) + VAT(*S*), the processor takes
  *F* = round(*p*·*T* + *f*), and the operator keeps
  *T* − VAT(*B*) − VAT(*S*) − *F*.

Leaving rounding aside, *S* ≥ *p*·(1 + *v*)·(*B* + *S*) + *f* gives
*S* = ⌈(1.16·*p*·*B* + *f*) / (1 − 1.16·*p*)⌉; the API computes that, then
moves it a cent at a time until it is the smallest *S* that works with the
rounding (for 10 packages the formula gives $19.15 and the smallest is
$19.14). At that *S* the operator keeps exactly the packages' price.

| | 1 package | 10 packages |
|---|---|---|
| Packages (100 / 1,000 forms) | $35.00 | $350.00 |
| Card fee | $2.20 | $19.14 |
| VAT 16% on the packages | $5.60 | $56.00 |
| VAT 16% on the card fee | $0.35 | $3.06 |
| **Total the customer pays** | **$43.15** | **$428.20** |
| Per form, all in | $0.4315 | $0.4282 |
| The processor's fee, 4.4% of the total + $0.30 | $1.8986 + $0.30 → $2.20 | $18.8408 + $0.30 → $19.14 |
| VAT the operator remits | $5.95 | $59.06 |
| **The operator keeps** | **$35.00** | **$350.00** |

`GET /v1/billing/quote?packages=n` gives these lines for any *n*, and
`GET /v1/pricing` the quote for one package.

### Why the card fee is 4.4% + $0.30

Stripe doesn't open merchant accounts for Kenyan businesses, so a Kenyan
operator's Stripe account is in another country, and Kenyan cards are
international cards to it. Stripe's US pricing, as third-party sources
give it, is 2.9% + $0.30 a successful card charge for domestic cards and
1.5% more for international cards: 4.4% + $0.30 (Xero's
[Stripe fee table](https://www.xero.com/us/pricing-plans/pricing-and-fees-for-stripe/)
lists "4.4% + 30c" for international cards; also
[Orb](https://www.withorb.com/blog/stripe-pricing); both found by web
search on 1 November 2026, their pages not opened). stripe.com was outside
this environment's network policy, so check
[stripe.com/pricing](https://stripe.com/pricing) for the account's country
before going live. Those sources also list 1% for currency conversion: a
charge in US dollars to an account that pays out in US dollars has none,
but an account that pays out in another currency does, and its rate
belongs in `CARD_FEE_BASIS_POINTS` (540).

### VAT on both lines

The card fee is charged by the seller as part of the price of what it
sells, so it is part of the consideration VAT is charged on, like the
packages. VAT is exclusive (added to the prices above) and is the same 16%
on each line, rounded per line. 16% of a whole number of cents never ends
in exactly half a cent, so any rounding rule gives the same VAT; Stripe
applies the tax rate to each Checkout line, and a payment whose total
differs from the quote adds no credits (API.md).

## The margin

The measured cost of the most involved path one form can take (below) is
$0.003247: $0.35 a form is about 108 times that. An ordinary published
form cost $0.0010, about a 350th of the price. The cost counts only server
time and model calls; storage, bandwidth, support, VAT administration and
the operator's time come out of the difference. The card fee covers the
processor, so the margin is the same however the customer pays.

## Notes before going live

- **Surcharging.** Card-network rules on adding a card fee to the price
  differ by country and by acquirer; confirm that a separate card-fee line
  is allowed for the account before charging one.
- **Tax invoices.** A VAT-registered seller in Kenya issues tax invoices
  through KRA's eTIMS. The platform doesn't: each purchase record carries
  what an invoice needs (the buyer, the date, each line with its quantity,
  unit price and VAT, the totals, the payment's reference), for the
  operator to issue it.
- **Processors.** Stripe isn't directly available to Kenyan businesses.
  Paystack, which Stripe owns, serves Kenya; another processor means
  another fee rate, which is configuration (`CARD_FEE_BASIS_POINTS`,
  `CARD_FEE_FIXED_CENTS`), though the API's checkout speaks Stripe's API
  only. Packages paid by bank transfer are recorded by the operator
  (`rfpii-api admin add-credits`), with VAT and no card fee.

## The most involved call

The most involved path one form can take through `rfpii job`:

1. It is redacted and **held** (its table couldn't be placed).
2. Its 512 px triage view goes to the vision model, which takes it for the
   job's form; the page goes again at full resolution to **locate** its
   signatures, IDs and phone numbers.
3. It is redacted again with those boxes, **verified**, **re-run** from its
   seed, and **reviewed** (the vision model finds, then checks each spot).
4. The review confirms a leak: the orchestrator chooses the **remedy**, the
   form is masked again, verified, re-run and reviewed again, with four
   more spots to check.
5. For a job of one form, the orchestrator also compiles the **policy** and
   writes the **pilot note**.

Steps 1, 2 and 5 were measured on a held Form 34A (Lamu, station
005022010905701). Step 3 was measured on a published one (station
005022010603301), which the job released. Step 4 is those measurements
again, plus a remedy call the size of the pilot note.

| | Held form (steps 1, 2, 5) | Published form (step 3) | Worst path (1–5) |
|---|---|---|---|
| CPU time (redaction, OCR, verify, re-run) | 43.1 s | 15.5 s | 74.1 s |
| Wall time | 19.3 s | 16.1 s | |
| `gemma4:31b` tokens in / out | 1,014 / 515 | 1,027 / 73 | 5,122 / 807 |
| `gpt-oss:120b` tokens in / out | 732 / 902 | 713 / 785 | 1,098 / 1,353 |

CPU time is the job's and its child processes', taken from `getrusage` on a
4-vCPU Intel Xeon; tokens are what Ollama Cloud's replies counted
(`prompt_eval_count`, `eval_count`), as the job report now lists them
(`model_usage`).

### Cost

| Item | Rate | Cost |
|---|---|---|
| Server time: 74.1 CPU-seconds | $0.05985 per vCPU-hour (AWS `c7i.xlarge`, Africa (Cape Town), $0.2394/hour for 4 vCPU, on demand) | $0.001231 |
| `gemma4:31b` (vision) | $0.14 input, $0.40 output per million tokens | $0.001040 |
| `gpt-oss:120b` (orchestrator) | $0.15 input, $0.60 output per million tokens | $0.000977 |
| **Total** | | **$0.003247** |

Most forms cost far less: an ordinary published form, reviewed and
released in a job of its own, cost $0.0010 (15.5 CPU-seconds and the four
model calls above).

### Where the rates come from

- **Ollama Cloud**, [ollama.com/pricing](https://ollama.com/pricing), read
  31 October 2026: since 31 August 2026 cloud models are billed per token;
  `gemma4` $0.14 input / $0.05 cached input / $0.40 output, `gpt-oss:120b`
  $0.15 / $0.014 / $0.60, per million tokens. Cached-input discounts are
  not counted. `qwen3.5:397b`, which HARNESS.md listed for the
  orchestrator, was retired on 25 September 2026.
- **Server**: AWS `c7i.xlarge` in `af-south-1` at $0.2394 an hour, from
  [aws-pricing.com](https://aws-pricing.com/c7i.xlarge.html) and
  [DevZero](https://www.devzero.io/instances/aws/c7i.xlarge) (aggregators;
  check the AWS price list before relying on it). A server in Kenya costs
  less: Lineserve's Nairobi data centre lists a 4-vCPU, 8 GB server at
  about KES 8,330 a month ([lineserve.co.ke](https://www.lineserve.co.ke/pricing)),
  about $0.022 per vCPU-hour at KES 129 to the dollar, which would make the
  worst path $0.00247. Where the forms are election data, regulation 26 of
  the Data Protection (General) Regulations 2021 bears on processing them
  outside Kenya (`legal/A-data-protection.md`).

## Changing the price

The price and rates are configuration (`api/README.md`): forms and price
of a package (`PACKAGE_FORMS`, `PACKAGE_PRICE_CENTS`), VAT
(`VAT_RATE_BASIS_POINTS`), the processor's fee (`CARD_FEE_BASIS_POINTS`,
`CARD_FEE_FIXED_CENTS`), the most packages in one purchase
(`MAX_PACKAGES_PER_PURCHASE`), free forms a month (`FREE_FORMS_PER_MONTH`),
credit expiry (`CREDIT_EXPIRY_DAYS`), and, for the cost, token prices per
model and the server's price per vCPU-hour. A purchase keeps the quote it
was made at. The API records each job's actual cost from its CPU time and
the tokens its report lists; the operator reads it at
`GET /v1/admin/jobs/{id}` (customers see only their free forms and
credits), so the margin can be watched as models and prices change.
