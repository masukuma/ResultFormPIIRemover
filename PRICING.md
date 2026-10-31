# Pricing

What one form costs to redact through the harness, measured on real 2022
Forms 34A against Ollama Cloud on 31 October 2026, and the price the API
charges for it: cost plus 50%.

## Plans

| Plan | Price | What it includes |
|---|---|---|
| Free | $0 | 5 forms a calendar month |
| Paid (metered subscription) | **$0.0049 per form** | every form released, billed monthly |

A form is a document as `rfpii inventory` counts it: a file, or one page of
a scanned batch read page by page. Only forms the job **releases** are
counted: a held form, or one in a group the release gate holds back, is
not charged and doesn't use the free allowance.

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
| **Price: total × 1.5** | | **$0.004871 → $0.0049** |

The price is rounded up to the next $0.0001. Most forms cost far less: an
ordinary published form, reviewed and released in a job of its own, cost
$0.0010 (15.5 CPU-seconds and the four model calls above).

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
  worst path $0.00247 and its price $0.0037. The price is set on the dearer
  host so it covers either. Where the forms are election data, regulation
  26 of the Data Protection (General) Regulations 2021 bears on processing
  them outside Kenya (`legal/A-data-protection.md`).

### What the price leaves out

Storage, bandwidth, payment fees and support are not in the cost, as asked
(server time and model calls only). Card payments carry a minimum charge
(Stripe's is $0.50), about 103 forms at this price: a smaller monthly bill
is carried to the next month rather than charged.

## Changing the price

The rates are configuration (`api/README.md`): price per form, free forms a
month, token prices per model and the server's price per vCPU-hour. The API
records each job's actual cost next to its price, from its CPU time and the
tokens its report lists; the operator reads it at `GET /v1/admin/jobs/{id}`
(customers see only the price), so the margin can be watched as models and
prices change.
