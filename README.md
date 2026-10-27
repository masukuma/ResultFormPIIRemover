# ResultFormPIIRemover

Fast, fail-closed removal of personal data (officials' and agents' names, ID
numbers, phone numbers, signatures) from scanned election result forms before
they are published.

Each scan is aligned to a template of its form type and the template's PII zones
are filled in solid black. Pages that can't be aligned confidently are sent for
review and never published. See [DESIGN.md](DESIGN.md).

## Install

```bash
pip install -e '.[test]'
```

## Quick start (synthetic demo)

```bash
rfpii demo demo/                     # sample template + 20 fake scans + 1 junk image
rfpii preview demo/templates/sample_rs1 -o zones.png
rfpii redact demo/scans --templates demo/templates --out demo/public --review demo/review
```

## Adding a real form type

1. Scan a blank form and save it as `templates/<form>/reference.png`.
2. Write `templates/<form>/template.yaml`:

   ```yaml
   name: form_x
   reference: reference.png
   padding: 12
   zones:
     - name: presiding_officer
       rect: [x, y, w, h]          # reference-image pixels
     - name: agents
       polygon: [[x, y], [x, y], [x, y], [x, y]]
   ```
3. Check the zones with `rfpii preview templates/<form>`.
4. Run a pilot batch and check `review/report.jsonl` (inliers, score) before
   bulk processing.

## Output

- `public/<name>.pdf` is an image-only PDF with zones blacked out and no text layer or metadata.
- `review/<name>.json` holds the reason a document was withheld.
- `review/report.jsonl` holds per-file timing and alignment metrics.

The exit code is `2` if any file was sent to review.

## Tests

```bash
pytest
```
