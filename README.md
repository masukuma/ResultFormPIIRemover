# ResultFormPIIRemover

Removes personal data from photographed election result forms (KIEMS Form
34A and others) before publication. Each published form comes with evidence
that **nothing but the personal data was touched**.

- **Signatures** are masked by a black band through the middle of each:
  the tops and bottoms of the strokes show that the form was signed, the
  signature itself can't be made out.
- **ID and passport numbers** have 3 random characters hidden.
- **Phone numbers** keep their first digits; 3 of the last 6 are hidden.
- **Names, results, stamps and QR codes** stay readable. A stamp is never
  masked, even where it lies over a signature or a number.

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
# Once: a signing key for the manifests. Keep the private key offline;
# publish the .pub.pem.
rfpii keygen keys/

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
- `--allow-remote-unredacted`: needed before `--vlm-fallback` or `propose` may
  send unredacted forms to a cloud model.
- `--strict-structure`: withhold pages whose table rules can't be matched,
  instead of masking them with widened zones.
- `--mask-stamps`: the stamp isn't sacred. An ID, phone number or signature
  a stamp lies over is hidden: each field's band across the stamp is
  blacked whole, covering that part of the stamp. By default stamps are
  never masked and writing under one stays visible. Recorded in the
  manifest, and repeated by `verify --rerun`.
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
