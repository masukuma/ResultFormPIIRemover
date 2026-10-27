# Design: PII removal for published election result forms

## Goal

Polling-station result forms (Form 34A and others) are photographed and
transmitted by KIEMS kits and then published. Their **results** (station,
candidates, votes) and the officials' and agents' **names** must stay readable,
and readers must still be able to see that a form is genuine. Each kind of
personal data gets a different treatment:

| Field | Treatment |
|---|---|
| Signatures (officers, agents) | **Blacked out** completely |
| National ID / passport numbers | **Random part masked**: about 60% of the characters, never leaving more than two neighbouring characters visible |
| Phone numbers | Same as IDs |
| Names | Left visible |
| QR code, stamp, station codes, votes | Left visible: these are what make the form verifiable |

## Inputs: KIEMS results transmission

The kits produce PDFs that wrap a single photo of the form, at 1654×2340 px.
Their file names encode the transmission:

```
1_34_A_007029014000102_F230450M00203609_20220810_132155.PDF
│ │  │ └ station (county 007, const. 029, ward 0140, centre 001, stream 02)
│ │  │                   └ KIEMS kit serial     └ capture date/time
│ │  └ form suffix
│ └ form number (34 = presidential)
└ election type (1 = presidential)
```

The printed form carries a QR code. On Form 34A it encodes the station code
followed by `34`, and on Form 34B election, county, constituency, form and
page (`GE2022040228FORM34B03`). `kiems.py` parses both:

- **Template choice:** the form type picks the template, so a 34B page is never
  redacted with a 34A layout. A form with no template goes to review.
- **Consistency check:** a QR code that disagrees with the file name sends the
  document to review rather than publishing it under the wrong station.
- **Provenance sidecar:** each published PDF gets `<name>.json` with station,
  kit, capture time and the SHA-256 of the original transmission. It contains
  no personal data. Readers can use it to tie the redacted form to the original.

On 215 real Forms 34A (Lamu and Diaspora), all file names parsed. The QR
decoded on 90% of them and agreed with the file name every time.

Inputs can be PDFs, images, directories, or the county zip archives directly
(`archive.zip::member.pdf`). PDFs are rendered at the embedded photo's native
resolution.

## Pipeline

```
photo ─► page alignment (ORB + MAGSAC) ─► table snapping (printed rules) ─► masking ─► gates ─► publish
               │ fails                         │ fails                                    │ fail
               ▼                               ▼                                          ▼
            review               widened zones (or review with --strict-structure)     review
```

### 1. Page alignment (`align.py`)
ORB features are matched to the template's reference image, and a homography is
fitted with USAC/MAGSAC. This handles rotation (including upside-down and
sideways photos), scale and perspective. The fit is checked by inlier count,
inlier ratio, plausible geometry, and normalised cross-correlation (NCC)
between the reference and the scan warped back. A page failing any check is
withheld.

### 2. Table snapping (`structure.py`)
KIEMS photos are of booklet pages that **curl**. All the distinctive features
are in the header (logo, QR code, title), while the PII is in the agents table
at the bottom. A single homography extrapolated that far can be several rows
off. On the real samples, before this step existed, it produced a form whose
last three agents' IDs, phones and signatures were completely unmasked.

Templates therefore declare `structure` regions, such as the agents table. For
each one:

1. **Find the rules.** Horizontal and vertical printed rules are detected in the
   scan within a generous search window.
2. **Column rules first.** Handwriting never makes long vertical lines, so the
   column rules give the x positions. Where they start and end gives the
   table's real top and bottom border.
3. **Row rules second.** Rows are matched by a 1-D scale/offset search pinned to
   that top and bottom, because rows repeat and an unpinned match can lock on
   one or two rows off. When the rows match cleanly, every row is placed
   exactly; otherwise rows are interpolated between the verified borders.
4. **Mesh mapping.** Zones are mapped through a bilinear mesh over the rule
   intersections rather than a homography, so they follow the curl. Points
   outside the table are extrapolated with the whole grid's average spacing.

If the table can't be matched, its zones fall back to the page alignment but
are widened 50 px up and 200 px down: more than the worst drift seen, which is
always downward. With `--strict-structure`, such pages go to review instead.

### 3. Masking (`masking.py`)

**Blackout zones** (signatures) are solid fills, mapped edge by edge so they
stay covered on curled pages.

**Partial zones** (ID and phone columns) work on a whole table column at once:

1. **Straighten the column.** It is extracted with a remap at template scale,
   extending at least 60 px above and below the table so that writing on a
   table placed a row off is still inside.
2. **Extract the ink.** Contrast is normalised, because KIEMS photos range from
   crisp to grey-on-grey. Then three things are removed:
   - table rules;
   - the long diagonal strike-through lines officials draw through unused rows.
     Only segments at least four rows long count, because shorter "lines" can be
     chance alignments of digits in stacked rows, and erasing them hid digits in
     testing;
   - leftover thin fragments of rules.
3. **Ignore print.** A glyph whose surroundings look like the reference at about
   the same spot is printed text (column headers) and is skipped. This test
   compares appearance, not just position: if the table is a row off, written
   digits land where the reference has print, and a position test would skip
   them.
4. **Group into lines.** Glyphs are grouped into written lines, and each line is
   segmented into character slots: connected components when they look like
   separate characters, otherwise `digits` equal slots.
5. **Hide characters.** About 60% of the slots are hidden at random
   (`SystemRandom`, different per cell and per run), never leaving more than two
   neighbouring slots visible. A line that can't be segmented is hidden whole.
6. **Assign rows.** Lines are assigned to rows only to tell empty rows from
   unreadable ones, and for the report. Masks follow the writing itself.

### 4. Gates (fail-closed)
A document is published only if every page passes:

| Gate | Catches |
|---|---|
| Page alignment checks | unknown form or layout, blank or garbage page |
| Form type has a template | 34B, 35A, 36A, … until templates exist |
| QR code vs file name | form uploaded under the wrong station |
| **Column edge check** | table matched one rule off. The straightened column must have its printed rules at its own edges, not a stronger rule somewhere else. This compares the mapping against the paper, because comparing the masks against the same mapping can't detect it |
| **Leak check** | the redacted column is re-read over a wider window; 5 or more written characters in a row with no mask between them means something was missed |
| VLM audit (optional) | anything still readable anywhere: free-text comments, signatures outside the boxes |

The edge check and the leak check were both added after reviewing real outputs
showed misses that geometry alone allowed through.

### 5. Output
The output is a new image-only PDF with no text layer, annotations or metadata
(`io.py`), plus the provenance sidecar. Documents that fail go to `review/`,
each with a JSON report of why. `review/report.jsonl` logs every document.

## VLM assistance (`vlm.py`, Ollama)

A local vision model, e.g. `qwen2.5vl:7b` via Ollama, is used where judgement
helps and speed doesn't matter:

- **`--vlm-audit`:** before publishing, ask whether a complete ID number, phone
  number or signature is readable anywhere. A "yes" sends the form to review.
  This covers things templates can't, such as officers signing again in the
  comments area or writing their own ID number at the bottom. It is a gate, not
  a guarantee: a model can miss things.
- **`--vlm-fallback`:** for documents no template handles, the model locates
  signatures, IDs and phones, and the same masking is applied. Results go to
  `review/vlm/` for a person to approve. They are never auto-published.
- **`rfpii propose`:** drafts a new template's zones from a reference image, for
  a person to check with `rfpii preview`.

The model is asked for strict JSON (Ollama `format` with a schema, temperature
0), and images are sent at sizes that are multiples of 28 so that box
coordinates map back exactly.

## Results on real data (Form 34A, 215 KIEMS PDFs, two county zips)

| Outcome | Count |
|---|---|
| Published (table snapped) | 173 |
| Published (widened fallback) | 15 |
| Review: page alignment failed | 4 |
| Review: column edge check | 22 |
| Review: leak check | 1 |

That is 188 of 215 (87%) published, at about 7 files/s on 4 cores. Several
random samples of the published forms were reviewed by eye. Every review that
found a problem led to a fix, and those cases are now caught by the gates
above. Withheld forms are a mix of genuinely misplaced tables and forms whose
faint rules can't be confirmed. The VLM fallback, a person, or a better
template can clear them.

## Templates

| Template | Status |
|---|---|
| `form34a_2022` (2022 Form 34A, KIEMS) | Built from a real form and validated |
| Form 34B (multi-page; agents and CRO details on the last page) | Not yet; routed to review |
| 2021 by-election Forms 35A / 36A / 38A | Not yet; routed to review |

A template's reference image must have its handwriting erased, keeping the
printed rules (see `README.md`). The form's PII must never end up in the repo.

## Known limits
- **PII outside template zones** is only caught by the VLM audit. This includes
  officers re-signing beside the stamp, IDs written in the comments, and an
  agent's ID written into the name column.
- **Faint or heavily curled tables** go to review rather than being published.
  That's by design.
- **Thresholds come from 215 forms in two counties.** Run a pilot on other
  counties and look at `report.jsonl` before trusting a full run.
