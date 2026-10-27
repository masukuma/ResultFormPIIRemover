# ResultFormPIIRemover

Removes personal data from photographed election result forms (KIEMS Form
34A and others) before publication. Signatures are blacked out, and ID and
phone numbers are partly masked at random so they can't be used but the form
visibly carries them. Names, results, stamps and QR codes stay readable.

Forms are aligned to a template, tables are snapped to their printed rules,
and every page passes geometric and leak checks before it is published.
Anything uncertain goes to a review folder. An optional local vision model
(Ollama) audits pages and helps with forms that have no template. See
[DESIGN.md](DESIGN.md).

## Install

```bash
pip install -e '.[test]'
```

## Run

```bash
# Files, folders, or KIEMS county zips directly
rfpii redact Forms/34A/*.zip --templates templates --out public --review review

# With a local vision model (run `ollama pull qwen2.5vl:7b` first)
rfpii redact Forms/ --templates templates --out public --review review \
    --vlm-audit --vlm-fallback --vlm-model qwen2.5vl:7b
```

- `public/` holds the redacted PDFs plus a `.json` provenance sidecar for each
  (station, KIEMS kit, capture time, source SHA-256; no PII). Publish only this
  folder.
- `review/` holds a JSON report for each withheld document, `report.jsonl` for
  all documents, and `vlm/` for VLM-proposed redactions awaiting approval.
- The exit code is `2` if anything was withheld.

Useful flags:

- `--hide-fraction`: share of each ID or phone number hidden (default 0.6).
- `--strict-structure`: withhold pages whose table rules can't be matched,
  instead of masking them with widened zones.
- `--workers`: number of worker processes.
- `--dpi`: render resolution for PDFs.

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
     - {name: signatures, action: blackout, rect: [x, y, w, h], region: agents}
     - {name: agent_id, action: partial, digits: 8, rect: [x, y, w, row_h],
        rows: 4, row_step: 36.4, region: agents}
   ```

3. **Check:** `rfpii preview templates/<form> -o preview.png`.
4. **Pilot:** run a pilot batch and look at `review/report.jsonl` and a sample of
   `public/` before a full run.

## Tests

```bash
pytest
```

The tests use synthetic forms and a stub Ollama server. No real forms are
needed, and none are in the repo.
