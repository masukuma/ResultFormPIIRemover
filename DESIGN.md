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
| National ID / passport numbers | **3 random characters hidden** |
| Phone numbers | **First digits kept, 3 of the last 6 hidden**: `0734471897` keeps `0734` and hides 3 of `471897` |
| Names | Left visible |
| QR code, stamp, station codes, votes | Left visible: these are what make the form verifiable |

The hidden characters are random, but never more than two neighbouring
characters are left visible together.

The 2022 process was marred by claims that results were altered between
transmission and publication. A redaction step adds another place where that
claim could be made. So the process must **demonstrate, to anyone who asks,
that no result was changed**. See [Verifiability](#verifiability) below, and
[VERIFYING.md](VERIFYING.md) for the procedure a scrutineer follows.

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

**Signature zones** (`action: signature`) get one black box **per signature**,
not one block per region:

1. **Find the ink.** The column is straightened and searched 24 px beyond each
   side, and 30 px further down, because signatures overrun their box and bent
   pages drift downward.
2. **Remove what isn't a signature:**
   - printed rules;
   - the rings of official stamps;
   - flat strokes 8 px or less in height, which are pieces of rules.
3. **Group strokes.** Strokes that touch, mostly sideways, form one signature.
   A group spanning several rows (neighbours' strokes touching) is cut at the
   row boundaries.
4. **Mask.** Each signature gets its own padded box. Empty rows are untouched.

A coverage check then finds signatures again on the original and requires
each to be at least 90% under black in the output. Blackout zones (plain
rectangles) remain available for areas that must be hidden entirely, and are
mapped edge by edge so they stay covered on curled pages.

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
   segmented into **one slot per character**: joined-up writing is split at
   the expected character pitch, and if that doesn't give a plausible count,
   the line is divided into `digits` equal slots.
5. **Hide characters**, following the zone's rule:
   - `id`: 3 random characters.
   - `phone`: 3 random digits among the last 6. The prefix (network code, e.g.
     `0734`) stays visible.

   Choices are random, with no more than two neighbouring characters left
   visible together. The seed is drawn per document and recorded in the
   manifest, so the redaction can be reproduced.

   A hidden character is covered from the midpoint with its left neighbour to
   the midpoint with its right, over the line's full height. Rule removal can
   shave a 5's or 7's top bar off the detected glyph box, so masking only the
   box could leave part of a "hidden" digit showing. A line that can't be
   segmented is hidden whole.
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
The output is a new PDF (`io.py`). Each page holds exactly one image: the
redacted JPEG, passed through byte for byte, or a lossless PNG for non-JPEG
sources. The page has the same size and layout as the source page. There is
no text layer, no annotations and no metadata. JPEG markers other than
JFIF/ICC/Adobe (for example EXIF, which can hold camera serials or GPS) are
dropped.

Next to each PDF goes its signed manifest (`.manifest.json`), plus the ledger
and batch roots described below. Documents that fail go to `review/`, each
with a JSON report of why. `review/report.jsonl` logs every document.

## Verifiability

### The edit is restricted to black blocks
The pipeline never edits the image freely. Masks are drawn on a separate
canvas, rounded **out** to whole 8×8 blocks (`pipeline.py`), and only then
applied to the image as **stored**:

- **JPEG sources (all KIEMS transmissions; `proof.py`).** The edit is made on
  the JPEG's quantised DCT coefficients:
  - a masked luma block gets the DC value that decodes to black, and zero AC;
  - a chroma block whose whole area is masked is set to neutral grey;
  - every other block's coefficients, and the quantisation tables, are copied
    unchanged.

  The new JPEG is embedded in the PDF byte for byte, so the published file
  contains the original's image data apart from the black blocks.
- **Other sources.** The same blocks are set to 0 in the pixels, which are
  written losslessly as PNG.

A black block can hide a digit but cannot change it into another digit. If
every changed block is listed and solid black, no result can have been
altered, only obscured, and the manifest records where.

### Checks made before anything is published
- **Self-check.** The output is compared with the original, block by block,
  before it is written. Anything other than "identical, or listed and black"
  stops publication.
- **Protected regions.** Templates declare `protect` regions: the header with
  the form serial and QR code, station names and codes, the votes table, and
  the polling-station counts. A mask block touching one stops publication.

### The record
- **Manifest.** For each document (`process.py`) it records:
  - the source file's SHA-256 (the IEBC portal / KIEMS copy) and the hash of
    the embedded image;
  - the output's SHA-256;
  - the tool version and exact git commit;
  - each template's fingerprint (hash of its YAML and reference image);
  - the random seed;
  - the bitmap of blacked blocks, the protected regions as image polygons,
    and the page alignment;
  - per agent row, the rule applied and which character *positions* were
    hidden (never the characters themselves);
  - any VLM verdicts.

  It is signed with the publisher's Ed25519 key (`manifest.py`).
- **Ledger.** `ledger.jsonl` is append-only and hash-chained. Each entry commits
  to the previous one, so altering, dropping or back-dating any entry breaks
  the chain.
- **Batch root.** Each run ends with a signed Merkle root over its manifests.
  Publish this one hash widely when the batch is released. Inclusion of any
  manifest can then be proven, and nothing can be swapped in later.

### Independent verification (`rfpii verify`, see VERIFYING.md)
Given the published PDF, its manifest, the public key, the ledger and the
original, the verifier checks:

- the signature, the ledger chain, and that the manifest is in the ledger;
- both files' hashes;
- **every block's coefficients** (identical, or listed and black);
- that no listed block touches a protected region;
- that each results region decodes to identical pixels in both files.

It also writes an overlay showing every changed block. With `--rerun` it
repeats the redaction from the recorded seed and templates and must
reproduce the output bit for bit.

### Tested
On the 215 real Forms 34A, all 188 published forms verify. Re-running a
random sample of 12 reproduced each one bit for bit. Tampering is detected in
each of these cases (also in the test suite):

- changing one coefficient of one block in the votes table (caught by four
  separate checks);
- editing the manifest after signing;
- supplying a different original;
- editing an entry in the ledger.

### Threat model, briefly
| Concern | Answer |
|---|---|
| "The redaction step changed a result." | Proven not to, block by block, against the original, by anyone. |
| "The publisher swapped a file after release." | The ledger chain and the published batch root fail. |
| "The manifest was edited to hide a change." | The signature fails. The coefficient check doesn't trust the manifest anyway: an unlisted change fails it. |
| "The masking hid a result." | Protected regions are checked by the redactor and by the verifier, and the overlay shows every black block. |
| "The original itself is fake." | Out of scope for redaction. The manifest ties each form to its source hash, KIEMS kit serial, capture time and QR, for comparison with IEBC records. |
| "The signing key was stolen." | Keep it offline. Publish its fingerprint once. Rotate by announcing a new key in a signed statement from the old one. |

## VLM assistance (`vlm.py`, Ollama Cloud or local)

A vision model is used where judgement helps and speed doesn't matter. There
are three ways to run it:

- **Ollama Cloud directly:** set `OLLAMA_API_KEY`. Requests go to
  `https://ollama.com/api/chat` with a bearer token. The default cloud model is
  `qwen3-vl:235b`, the newest Qwen vision model with a confirmed Ollama Cloud
  tag.
- **A local Ollama signed in to the cloud:** use a `*-cloud` model name with
  the local host.
- **Fully local:** the default is `qwen3.8:27b`, Qwen's newest family on
  Ollama, which has image input built in.

Check ollama.com/library for current names. Both defaults report boxes on a
0–1000 grid; boxes are scaled back per axis.

Box coordinates are read as 0–1000 for Qwen3.x and Qwen3-VL, and as pixels
for Qwen2.5-VL, chosen automatically from the model name.

**Privacy.** The audit sends only the *redacted* page, so it is safe for the
cloud. The fallback and template proposals need the *unredacted* form. These
are refused for any non-local host unless `--allow-remote-unredacted` is
given, which should only be used if that data transfer has been approved.

**Scrutiny.** A VLM never edits pixels itself: its boxes go through the same
block masking and the same proof as everything else. Every audit verdict is
recorded in the manifest with the model, host, prompt hash and the hash of
the exact image sent.

What the model does:

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
| Published | 188 |
| Review: column edge check | 22 |
| Review: page alignment failed or other | 4 |
| Review: leak check | 1 |

That is 188 of 215 (87%) published, and all 188 verify. Throughput is about
3.7 files/s on 4 cores, including the coefficient-level edit and the
self-check. With per-signature masks, the blacked-out area is a median of
3.0% of each image (it was 5.4% with whole-column signature blocks), at most
7.3%. Several
random samples of the published forms were reviewed by eye. Every review that
found a problem led to a fix, and those cases are now caught by the gates
above. Withheld forms are a mix of genuinely misplaced tables and forms whose
faint rules can't be confirmed. The VLM fallback, a person, or a better
template can clear them.

## Templates

| Template | Status |
|---|---|
| `form34a_2022` (2022 Form 34A, KIEMS) | Built from a real form; 188 of 215 published, all verified |
| `form34b_2022_block`, `form34b_2022_results`, `form34b_2022_results_b` | Form 34B layout "A" (the system-printed form, the most common); see below |
| Form 34B layouts B and C, and heavily distorted copies | Not yet; routed to review |
| 2021 by-election Forms 35A / 36A / 38A | Not yet; routed to review |

### Form 34B (constituency tally)

Form 34B is multi-page. The results table runs over as many pages as the
constituency has polling stations. The declaration block follows the last
row, so it sits at a different place on a different page in each document,
and it often shares a page with the last results rows. The block holds:

- the agents table (4 rows): name, ID/passport, party, phone, signature, date;
- the Constituency Returning Officer: name, ID number, signature, date;
- the handover to the Commission Chairperson: names, ID numbers, signatures, times.

Results pages carry no personal data. So 34B uses three templates:

- **`form34b_2022_block`:** the reference is just the block, so it is found
  wherever it is on a page. It has priority, so a page that also shows
  results rows is redacted with it. The masks are:
  - agent IDs and phones: the ID and phone rules, placed by snapping to the
    agents table's printed rules;
  - each signature: its own box;
  - the RO and Chairperson ID numbers: the ID rule;
  - names, dates and Form 34A counts: left visible (the counts are protected).
- **`form34b_2022_results`, `form34b_2022_results_b`:** the reference is just
  the typed column-header row with the candidate names. There are two
  variants because constituencies used two generators. These templates have
  no zones, so results pages are published with every block identical to
  the original.
- **Document rule:** a 34B document is published only if the block was found
  on **exactly one** page. If it was found on none (a bad scan, or a layout
  without a template), the page holding the personal data could otherwise
  pass as a plain results page. If it was found on two, the block is split or
  documents were merged. Either way the document goes to review.

The references come from the blank, system-generated form, rendered from an
IEBC PDF with empty fields, so there is no handwriting to erase. The variant
results header is a typed row cut from a scan.

Features for a reference cut from a page are extracted at the page's scale
(`page_long_side`), so that text sizes match the scans.

**Single underlined fields** (the RO's and Chairperson's ID and signature)
are handled differently from table cells:

- **Zone height:** each zone spans from just below the previous printed line
  to just below its own, so handwriting of any height is inside and the name
  on the line above is not.
- **Anchors:** each zone is anchored to its section's printed "ID Number:"
  label. The tool finds that label within ±120 px and moves the zone with it,
  because layout variants put the section a line higher or lower. If the
  label isn't found, the page fails.
- **Stamps:** round official stamps are detected, and printed ink wholly
  inside a stamp is not treated as a signature, so the stamp stays readable.

**Sample results (12 constituencies, Dropbox `Forms/34B`):**
- **Published:** 6 (Bomet East, Butula, Funyula, Garissa Township, Matayos,
  Nambale). All 6 verify against their originals.
- **Review, other layouts:** 3 documents use layouts without a template:
  Budalangi (layout B: aggregate results and voter-turnout tables, and a
  5-row agents table), Daadab (layout C, spreadsheet-style) and Fafi (a
  heavily curved phone photo).
- **Review, layout A that wasn't recognised:** Konoin (upside-down, grey
  background), Runyenjes (speckled "COPY" watermark) and Manyatta (a results
  page with a weak header match).

**A safety bug found and fixed while building this.** The test that tells
printed text from handwriting compared raw pixels around each glyph. A digit
written on an underline matched the reference's underline, and was treated
as print: it was neither masked nor flagged by the leak check, which uses
the same test. One RO ID number came out fully readable in testing. The test
now compares ink with printed rules removed, and requires printed ink nearby
in the reference. Re-checking all 188 published Forms 34A with the corrected
leak check found no affected form (in the 34A table, the rules are removed
before the test).

A template's reference image must have its handwriting erased, keeping the
printed rules (see `README.md`). The form's PII must never end up in the repo.

## Known limits
- **Form 34B layouts.** Only layout A has templates. Each other layout (B, C,
  …) needs its own block template, built from a blank or cleaned copy. The
  document rule keeps every unknown layout out of publication.
- **Blobs that can't be split into characters are hidden whole.** When an ID
  or phone number can't be segmented (e.g. faint ink), the whole number is
  hidden. That is the safe failure, but it is more than the 3-character rule.
- **Signature boxes merge in dense tables.** Where agents sign across
  neighbouring rows, the boxes merge into one patch. Any writing inside a
  signature box (e.g. a phone number continued into the column, or part of
  the "Signature" label) is hidden with it.
- **Block granularity.** Blocks are 8 px wide and a handwritten digit is about
  15–20 px, so a black block can also clip the edge of a neighbouring visible
  character.
- **Strength of the masking.** Hiding 3 digits of an 8-digit ID, or 3 of a
  phone number, leaves 1000 possible values. That is enough against casual
  reading, not against someone able to test candidates.
- **PII outside template zones** is only caught by the VLM audit. This includes
  officers re-signing beside the stamp, IDs written in the comments, and an
  agent's ID written into the name column.
- **Faint or heavily curled tables** go to review rather than being published.
  That's by design.
- **Thresholds come from 215 forms in two counties.** Run a pilot on other
  counties and look at `report.jsonl` before trusting a full run.
