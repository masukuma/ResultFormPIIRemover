# ResultFormPIIRemover

Removes personal data from photographed election result forms (KIEMS Form
34A and others) before publication. Each published form comes with evidence
that **nothing but the personal data was touched**.

- **Signatures** are masked by a black band through the middle of each:
  the tops and bottoms of the strokes show that the form was signed, the
  signature itself can't be made out.
- **ID and passport numbers** have 3 random characters hidden, or all of
  them with `--full-ids`.
- **Phone numbers** keep their first digits; 3 of the last 6 are hidden, or
  every digit with `--full-phones`.
- **Names, results, stamps and QR codes** stay readable. Where a stamp lies
  over a number, that number's line across the stamp is blacked; over a
  signature, a line through the middle of its row. The rest of the stamp is
  kept, and checked unchanged.
  `--keep-stamps` never masks a stamp at all.

The only edit ever made is replacing 8×8 blocks with solid black. On KIEMS
JPEGs this is done at coefficient level, so every other block is the same
data as the original. Each form gets a signed manifest, an entry in a
hash-chained ledger, and a place in a published batch root. Each published
PDF can also carry a digital signature that PDF readers check (with your
certificate, or a self-signed one the tool offers to create). Anyone can then
run `rfpii verify` against the original from the IEBC portal and see that no
result was changed. See [VERIFYING.md](VERIFYING.md).

Templates: Form 34A (2022) and Form 34B (2022, multi-page: the
system-printed layout A and two other printings, B and C). Forms are aligned to a template, tables are snapped to their printed rules,
and each page passes geometric, protected-region and leak checks before it is
published. Forms with no template (2017 Forms 34A, 2021 by-election Forms
35A/36A/38A and other printings) are read without one: OCR finds the labels,
the printed rules give the columns, and everything above the declaration is
protected. Anything uncertain goes to a review folder. For runs too large for
people to look at the review folder, vision models on Ollama Cloud (or a local
Ollama) review every redacted page and mask what the rules missed, and pages
whose table columns can't be placed are published with those columns hidden
whole once the review passes them. See [DESIGN.md](DESIGN.md).

## Install

The tool is written in Rust (`rust/`). It runs on the same native libraries
the reference implementation used, so it produces the same output: OpenCV
5.0, MuPDF 1.28, ONNX Runtime with the PP-OCR models (for older forms read
without a template and the comments on Form 34A; on the CPU, nothing leaves
the machine) and numpy's OpenBLAS. Without the OCR models, those pages go to
review.

```bash
sudo rust/scripts/setup-deps.sh     # once: the native libraries, under /opt
cd rust && cargo build --release    # the tool: rust/target/release/rfpii
```

`setup-deps.sh` takes them from the packages they ship in (PyPI wheels, the
OpenCV source distribution) and needs Python with pip, CMake, a C/C++
compiler and clang.

## Run

```bash
# Once: a signing key for the manifests, its private key encrypted with a
# passphrase ($RFPII_SIGN_PASSWORD, or asked for). Keep it offline; publish
# the .pub.pem.
rfpii keygen keys/

# A key to retire, or lost: hand over to a new one, or revoke it, in the
# ledger. Forms keep verifying with the first published key; those a
# revoked key signs from then on (or since --since) don't.
rfpii rotate-key --key keys/signing_key.pem --new keys2/ --ledger public/ledger.jsonl
rfpii revoke-key --key keys2/signing_key.pem --ledger public/ledger.jsonl --reason "laptop stolen"

# A leak found after release: take the form down (out of public/, recorded
# in the ledger; verify then reports it withdrawn), and republish it
# corrected, naming what it replaces
rfpii withdraw public/x.pdf --reason "phone number readable in row 3" --sign-key keys/signing_key.pem
rfpii redact x.PDF --templates templates --out public --review review \
    --sign-key keys/signing_key.pem --supersedes public-withdrawn/x.manifest.json

# Once, unless you have a certificate: a self-signed certificate for the
# PDFs' digital signatures. Publish signer.crt (or its SHA-256 fingerprint).
rfpii certgen --name "Returning Officer, Kibwezi West" --org "..." --country KE

# Files, folders, or KIEMS county zips directly
rfpii redact Forms/34A/*.zip --templates templates --out public --review review \
    --sign-key keys/signing_key.pem

# Forms no one made a template for: each printing's layout is learned from
# the batch itself (GENERALIZING.md), kept in learned/, and used
rfpii redact Forms/36A/ --learn learned --out public --review review \
    --sign-key keys/signing_key.pem

# Scanned batches, one constituency to a PDF: each page published (or held
# back) on its own, as public/<name>.p012.pdf
rfpii redact Scans/ --templates templates --out public --review review \
    --sign-key keys/signing_key.pem --per-page

# Unattended, at scale: vision models on Ollama Cloud review every redacted
# page (only redacted pages are sent) and mask what the rules missed; pages
# whose columns can't be placed get them hidden whole if the review passes
export OLLAMA_API_KEY=...            # from ollama.com/settings/keys
rfpii redact Forms/ --templates templates --out public --review review \
    --sign-key keys/signing_key.pem --vlm-audit --vlm-rescue   # default: gemma4:31b
# ...or on a local Ollama: --vlm-host localhost:11434 (default qwen3.8:27b)

# Phone photos with no orientation tag, many stored on their side: Clef
# judges which way up each redacted page stands, and it is published upright
rfpii redact Photos/ --learn learned --out public --review review \
    --sign-key keys/signing_key.pem --orient

# Towards the DPIA (Data Protection Act s.31): the technical annex, from
# what the run recorded: the processing, residual-risk figures, measures,
# transfers abroad, retention
rfpii dpia-annex public --review review --controller controller.json \
    --retention retention.yaml -o dpia-annex.md

# Keep files no longer than their schedule (s.39): report what is due, or
# delete it, recording each disposal in the ledger by hash
rfpii retain --policy retention.yaml --apply --sign-key keys/signing_key.pem

# For a court, a petitioner or agents: the forms asked for, their manifests,
# each one's verify log against the original, the ledger, batch roots, key,
# a cover sheet and SHA256SUMS (originals only with --include-originals)
rfpii bundle public/x.pdf public/y.pdf --out bundle/ --public public \
    --pubkey signing_key.pub.pem --originals originals/

# A whole job, in phases, to a report (HARNESS.md): intake, plan, pilot,
# run, QA, remediation, triage, release through a gate, report. job.yaml
# names the inputs, the instructions (or a structured policy), the output
# folder, the signing key, the controller record, the DPIA, the templates,
# the models (an orchestrator and a vision model, on Ollama Cloud unless a
# host is given) and whether held pages may go to the cloud model. Models
# see only verified crops of published pages and, if allowed, views of held
# ones; nothing is released unless every condition of the gate holds
rfpii job job.yaml
# A leak found on a released form: withdrawn, the incident recorded with
# whose data it holds by class, the s.43 deadlines and a notice drafted
rfpii job-incident jobs/2013-34 jobs/2013-34/public/x.pdf --reason "ID readable in row 3" --kind id_number

# The harness's tools (HARNESS.md §6), each giving JSON (--json): what a
# job is given; what a run published and held, and its pages by risk; a
# verified region of a published page, and a held page's diagnostics
# (numbers and printed words only), for a model
rfpii inventory Forms/ --json
rfpii stats public --review review --risk
rfpii crop public/x.manifest.json --region declaration --templates templates --out x.png
rfpii held-view Scans/k.pdf --page 12 --mode triage --job jobs/2013-34 --out k12.jpg
rfpii diagnose Scans/k.pdf --page 12 --templates templates --json

# Anyone, later: prove a published form against its original
rfpii verify public/x.pdf --original x.PDF --pubkey signing_key.pub.pem \
    --ledger public/ledger.jsonl --overlay changes/ --trust signer.crt
```

**Digital signatures on the PDFs.** `redact` signs every published PDF with
`--certificate` (a PKCS#12 `.p12`/`.pfx`, or PEM with its key; the password
comes from `$RFPII_CERT_PASSWORD` or is asked for). Without one it uses
`~/.config/rfpii/signer.p12` if `certgen` made it, and otherwise offers to
create a self-signed certificate there. Run without a terminal, it warns and
publishes unsigned PDFs; `--no-pdf-signature` skips signing. PDF readers
(Acrobat, pdfsig) show the signature as valid and covering the whole file.
For a self-signed certificate they also say its issuer is unknown: readers
can trust it by comparing its fingerprint with the one you published.
A certificate from a CA your readers already trust avoids that step.

- `public/` holds what you publish:
  - the redacted PDFs;
  - each one's signed `.manifest.json` (source and output hashes, blocks
    blacked, protected regions, seed, tool and template versions; no PII);
  - `ledger.jsonl`;
  - `batches/*.json`, each with the batch's Merkle root. Announce that root
    widely when the batch is released.
- `review/` holds a JSON report for each withheld document, `report.jsonl` for
  all documents, and `vlm/` for VLM-proposed redactions awaiting approval.
- The exit code is `2` if anything was withheld.

Useful flags:

- `--hide`: characters hidden per ID or phone number (default 3).
- `--phone-tail`: phone numbers are masked only within their last N digits
  (default 6).
- `--vlm-audit`: review every redacted page with vision models before
  publishing; what they find still showing is masked and the page reviewed
  again (`--vlm-rounds`, default 2). `--vlm-model` finds, `--vlm-checker`
  confirms each spot on a zoomed crop (both default to `gemma4:31b` on Ollama
  Cloud). What was found is recorded in the manifest and replayed by
  `verify --rerun`.
- `--vlm-checker clef`: confirm spots with Clef, a decision model, which
  scores each spot instead of writing a reply: the same score every time,
  recorded in the manifest. A spot is masked at a score of
  `--vlm-check-threshold` (default 0.5) or more.
- `--vlm-rescue`: with `--vlm-audit`, publish pages whose ID/phone columns
  can't be placed, with those columns hidden whole, once the review passes.
- `--vlm-concurrency`: requests to the model host at once across all workers
  (default 10; Ollama Cloud answers 429 above an account's limit).
- `--orient`: photos with no EXIF orientation tag are shown as stored, often
  on their side. With this flag, Clef on Ollama Cloud is asked which way up
  each published page stands, and the page is turned to show it upright. It
  sees only the redacted page, gets one question per upright page and four
  per turned one, and the turn is recorded in the manifest
  (`--orient-model`, default `clef`).
- `--allow-remote-unredacted`: needed before `--vlm-fallback` or `propose` may
  send unredacted forms to a cloud model.
- `--strict-structure`: withhold pages whose table rules can't be matched,
  instead of masking them with widened zones.
- `--full-ids`: hide every character of each ID or passport number instead
  of `--hide` of them.
- `--full-phones`: hide every digit of each phone number, its first digits
  too. Both are recorded in the manifest and repeated by `verify --rerun`;
  with them the leak check withholds a page showing 3 written characters in
  a row of such a number outside a stamp.
- `--controller controller.json`: who publishes the forms and on what legal
  basis: the controller and its registration with the Data Protection
  Commissioner, whether the operator is the controller or a processor, the
  legal basis, the purpose, whether the use is commercial, the processors
  (model hosts with their country) and the DPIA reference. Checked when the
  run starts, its hash and main fields recorded in every manifest, and
  printed by `verify`. See `legal/C-recommendations.md` §2.9.
- `--no-notice`: don't write the redaction notice. By default each published
  page carries a line below it, outside the form image, saying it is a
  redacted copy under the Data Protection Act, that the results are
  unchanged, which original it was made from (its SHA-256) and how to check
  it. `verify` checks the notice is there and says what the manifest
  records.
- `--supersedes MANIFEST`: the manifest of a withdrawn form this run
  republishes; the new manifest names it.
- `--resume`: skip a document whose output and manifest exist from a run
  with the same settings and spot masks, and redo one whose differ.
- `--pages 3,7-9`: with `--per-page`, only these pages of each file.
- `--base DIR`: output paths are the inputs' paths relative to `DIR`, so
  files redacted one at a time land where a run over their folder puts them.
- `--spot-masks FILE`: boxes to mask as well (a JSON list: each box's source
  SHA-256, page, label, box, and where it came from), masked as the vision
  review's finds are, recorded in the manifest (`spot_masks`) and replayed by
  `verify --rerun`.
- `--kenya-host HOST`: a model host in Kenya. Every call that sends a page to
  any other remote host (the review, `--orient`, `--vlm-fallback`) is a
  transfer of personal data abroad, recorded in `<out>/transfers.jsonl` with
  what reg. 41(2) of the Data Protection (General) Regulations asks: date and
  time, recipient, justification, a description of the data, and hashes of
  the images sent.
- `--keep-stamps`: never mask a stamp; an ID, phone number or signature a
  stamp lies over stays visible. By default each number's line across a
  stamp is blacked whole, and a signature's row gets a line through its
  middle (half the row's height), covering that part of the stamp; the rest
  of the stamp is listed as protected and checked unchanged by `verify`
  (`--mask-stamps` asks for that default, for older scripts). Recorded in
  the manifest (`settings.stamps`), and repeated by `verify --rerun`.
- `--workers`: number of documents processed at once. With `--vlm-audit`,
  use about three per core: most of a document's time is spent waiting for
  the models.
- `--no-generic`: send forms with no template to review instead of reading
  them without one.
- `--learn DIR`: learn the layout of every printing in the batch that has
  no template in `DIR` yet (at least 5 pages of it), write it there, and
  redact with it. `rfpii learn FORMS --out DIR` only learns.
- `--per-page`: each page of a file of several pages is a document of its
  own, published as `NAME.p012.pdf` with its own manifest, or sent to review
  alone. For scanned batches of many forms in one PDF, where otherwise one
  page that fails holds back the whole file. Only that page is decoded, so
  memory stays at a page per worker. The manifest names the whole file (its
  SHA-256) and the page, and `verify` checks it against that page of the
  original.

## Adding a form type

1. **Reference image:** pick a flat, sharp photo of the form and **erase the
   handwriting from it**, keeping the printed lines. It is committed to the
   repo, so no personal data may remain.
2. **Draft zones:** `rfpii propose reference.png -o templates/<form>/template.yaml --form 34B`
   (needs Ollama), or write them by hand:

   ```yaml
   name: form34b_2022_p3
   form: 34B
   reference: reference.png
   structure:                    # printed tables that position their zones
     - {name: agents, rect: [x, y, w, h]}
   zones:
     - {name: signatures, action: signature, rect: [x, y, w, row_h],
        rows: 4, row_step: 36.4, region: agents}
     - {name: agent_id, action: partial, rule: id, digits: 8, rect: [x, y, w, row_h],
        rows: 4, row_step: 36.4, region: agents,
        fence: {above: [x, y, w, h], below: [x, y, w, h]}}  # printed labels a widened search stops at
   protect:                      # results regions no mask may ever touch
     - {name: votes, rect: [x, y, w, h]}
   text_h: 34.6                  # printed text height in the reference (OCR box, px)
   free_text:                    # free writing (comments): read with OCR for
     - {name: comments, rect: [x, y, w, h]}  # numbers and signatures written there
   ```

   For a section that floats or varies between layouts (like the Form 34B
   declaration block), see `templates/form34b_2022_block/template.yaml`:
   - `min_area_frac` and `page_long_side` for a reference cut from a page;
   - `priority` and `once` for the multi-page rules;
   - `anchor` to tie a field to its printed label.

3. **Check:** `rfpii preview templates/<form> -o preview.png`.
4. **Pilot:** run a pilot batch and look at `review/report.jsonl` and a sample of
   `public/` before a full run.

## Tests

```bash
cd rust && cargo test --release
```

The tests use synthetic forms and a stub Ollama server. No real forms are
needed, and none are in the repo.
