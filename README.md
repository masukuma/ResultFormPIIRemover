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
hash-chained ledger, and a place in a published batch root. Anyone can then
run `rfpii verify` against the original from the IEBC portal and see that no
result was changed. See [VERIFYING.md](VERIFYING.md).

Templates: Form 34A (2022) and Form 34B (2022, multi-page: the
system-printed layout A and two other printings, B and C). Forms are aligned to a template, tables are snapped to their printed rules,
and each page passes geometric, protected-region and leak checks before it is
published. Forms with no template (2017 Forms 34A, 2021 by-election Forms
35A/36A/38A and other printings) are read without one: OCR finds the labels,
the printed rules give the columns, and everything above the declaration is
protected. Anything uncertain goes to a review folder. An optional vision
model on Ollama Cloud or a local Ollama audits pages. See
[DESIGN.md](DESIGN.md).

## Install

```bash
pip install -e '.[test]'
pip install -e '.[generic]'   # older forms without a template (OCR runs locally)
```

## Run

```bash
# Once: a signing key. Keep the private key offline; publish the .pub.pem.
rfpii keygen keys/

# Files, folders, or KIEMS county zips directly
rfpii redact Forms/34A/*.zip --templates templates --out public --review review \
    --sign-key keys/signing_key.pem

# With a Qwen vision model on Ollama Cloud auditing every redacted page
export OLLAMA_API_KEY=...            # from ollama.com/settings/keys
rfpii redact Forms/ --templates templates --out public --review review \
    --sign-key keys/signing_key.pem --vlm-audit          # default: qwen3-vl:235b
# ...or on a local Ollama: --vlm-host localhost:11434 (default qwen3.8:27b)

# Anyone, later: prove a published form against its original
rfpii verify public/x.pdf --original x.PDF --pubkey signing_key.pub.pem \
    --ledger public/ledger.jsonl --overlay changes/
```

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
- `--allow-remote-unredacted`: needed before `--vlm-fallback` or `propose` may
  send unredacted forms to a cloud model.
- `--strict-structure`: withhold pages whose table rules can't be matched,
  instead of masking them with widened zones.
- `--workers`: number of worker processes.
- `--no-generic`: send forms with no template to review instead of reading
  them without one.

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
        rows: 4, row_step: 36.4, region: agents}
   protect:                      # results regions no mask may ever touch
     - {name: votes, rect: [x, y, w, h]}
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
pytest
```

The tests use synthetic forms and a stub Ollama server. No real forms are
needed, and none are in the repo.
