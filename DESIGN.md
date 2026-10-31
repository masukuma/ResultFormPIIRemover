# Design: PII removal for published election result forms

## Goal

Polling-station result forms (Form 34A and others) are photographed and
transmitted by KIEMS kits and then published. Their **results** (station,
candidates, votes) and the officials' and agents' **names** must stay readable,
and readers must still be able to see that a form is genuine. Each kind of
personal data gets a different treatment:

| Field | Treatment |
|---|---|
| Signatures (officers, agents) | **Masked through the middle**: a black band along each signature; the tops and bottoms of its strokes show that it was signed |
| National ID / passport numbers | **3 random characters hidden** (all of them with `--full-ids`) |
| Phone numbers | **First digits kept, 3 of the last 6 hidden**: `0734471897` keeps `0734` and hides 3 of `471897` (all of them with `--full-phones`) |
| Names | Left visible |
| QR code, station codes, votes | Left visible: these are what make the form verifiable |
| Stamps | **Kept and verified**, except a field's line across one: where a stamp lies over a signature, an ID or a phone number, that line is blacked and the rest of the stamp kept (`--keep-stamps`: never masked; see "Stamps") |

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
page (`GE2022040228FORM34B03`). `kiems.rs` parses both:

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

A form type with no template (older printings, by-election forms) is read
without one instead: see [Forms without a template](#forms-without-a-template-genericrs).
Either way, stamps are found before masking and no mask is ever applied over
one: see [Stamps are never masked](#stamps-are-never-masked-stampspy).

### 1. Page alignment (`align.rs`)
ORB features are matched to the template's reference image, and a homography is
fitted with USAC/MAGSAC. This handles rotation (including upside-down and
sideways photos), scale and perspective. The fit is checked by inlier count,
inlier ratio, plausible geometry, and normalised cross-correlation (NCC)
between the reference and the scan warped back. A page failing any check is
withheld.

### 2. Table snapping (`structure.rs`)
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

The widened search stops at the zones' **fences**: printed labels just
outside the table, located on the page the way anchor labels are. For Form
34A these are the column headers above the agents table ("ID/Passport No.",
"Tel. Contact", "Signature"), "Agents or Candidates (if present)" below the
officers' signatures, and "Reasons for Refusal to Sign" below the table.
Without them the 200 px reached the Presiding Officer's comments, and the
officers' signatures reached the table's headers: words of the comments and
the printed headers were blacked out. A fence may cut half a row into the
cells as placed; a label found deeper than that, or not found at all, leaves
that side of the search wide, so a bad match can only mask more, never less.
On the 215 Forms 34A this changed the 15 published pages that fall back, each
with 10-37% fewer blocks blacked, and nothing else.

A table's searches stop at its fences even when it snapped: a signature is
looked for a little above its row, and a header's print taken for its ink
blacked the header out on many pages. A snapped table's cells are never cut
by a fence, only what lies outside them.

**Each column on its own rows.** A sheet is folded and bent, and the
reference photo is no exception: its rows run 1.6% downhill to the right.
Each column's zones are drawn on the rows printed over that column (the
agents' Signature rows lie 12 px below their ID rows). The snapped grid
holds rows at their height at the table's centre, so a point drawn on the
reference is first carried along the reference's slope onto it (and along
its column rules' lean). A label to one side bounds a column at the label's
level carried across the same way: "Reasons for Refusal", at the far left,
is level with the table's bottom rule at the Signature column, where the
last agent's signature hangs below it. A widened search may also run on
past it by 5% of the distance across, for a page tilted against the
alignment. Finding each column's rows on the page from its own faint printed
rules was tried: rows of handwriting make the same faint dips, and the fits
locked onto them, so it isn't used.

**The officers' signature lines.** They sit above the agents table and are
placed by it, but where its rules can't be matched they are widened only
downward: the page alignment is sure of the declaration, and above each line
is the writing of the line before. Writing that reaches into the search from
its top and stops above the first line's middle is that line's (a
constituency written just over the Presiding Officer's signature line), not
a signature. A zone's `print` labels ("Date" at the end of each line, inside
its search) are found on the page and their print set aside, so they are
never taken for writing; only those labels are, not the whole reference,
whose signature cells still hold traces of the handwriting erased from it.

**Signatures in a table that didn't snap.** Its rows are then the
template's, and down a column of eleven the writing can drift most of a row
off them; cut at those rows, a signature is banded in the wrong place. On
one Form 34A the last agent's signature hung below where the search stopped
and only the top of it was hidden. In such a table all of each signature's
ink is hidden, in slices 8 px wide, each from the top to the bottom of the
ink in it (ink more than 10 px apart in a slice is covered piece by piece),
with pieces of a stroke either side of a printed rule joined. The paper
beside and between the strokes stays visible: the space after a phone
number, the gap to the next signature, the date beside it. A box round each
signature merged a column of touching signatures and the phone numbers
before them into one blot (72% of that form's Signature column black,
against 44% along the ink). Two signature lines a line apart (the
officers') keep their bands.

**Finding a table again by its headers.** The bottom of a curled page can
put the agents table 100–250 px from where the page alignment expects it,
at a slightly different scale; then the rules are searched for in the wrong
place and the snap gives up. When it does, the table's column headers ("ID/
Passport No.", "Tel. Contact", "Signature", the zones' upper fences) are
searched for over a wide area, and the offset and x-scale they agree on (at
least two, within 20 px vertically and 10 px of a straight-line fit) move
the table before it is snapped again. If it still doesn't snap, the moved
position is where the widened fallback starts from.

**Placing a column again by its header.** When the edge check below finds a
column off its printed rules (a snap matched the rules one off, or a
fallback placement drifted), the column is moved to where its own header is,
then each of its edges onto the printed rule nearest it within 25 px, which
corrects a table photographed narrower or wider than the page. It must then
pass the same check; on a table that didn't snap, a header matched clearly
(0.55 or better) counts as evidence where faint rules can't be seen. The
signature column of a table whose columns moved is moved by its own header
too, if that passes the check. The page notes record each move.

**Each column's header as a witness.** The edge check can't see a table
matched a whole column or a whole row off: every agents column has a rule at
each edge. Each column's own header ("ID/Passport No.", "Tel. Contact",
"Signature") is therefore looked for wherever its column was placed. Found
clearly (NCC 0.6 or better) more than 25 px across or half a row or more up
or down, the column is moved down to it and placed again by it as above,
signature columns included, or the page is held. Over the 215 Forms 34A the
headers of well-placed columns were found within 25 px across and 18 px up
or down.

On the 215 Forms 34A these took the forms held by the edge check from 22 to
5; together with the fences (which cleared a leak-check failure that was
the header read as writing) 205 publish instead of 188. Every page they
changed was compared with the previous output by eye: nothing PII that was
masked is uncovered, and headers that were blacked out are readable.

### 3. Masking (`masking.rs`)

**Signature zones** (`action: signature`) get one black band **per signature**,
through its middle, not one block per region. Readers see that a form was
signed (the tops and bottoms of the strokes show above and below the band),
not the signature itself:

1. **Find the ink.** The column is straightened and searched 24 px beyond each
   side, and 30 px further down, because signatures overrun their box and bent
   pages drift downward.
2. **Remove what isn't a signature:**
   - printed rules;
   - in a table column, what is left of its own rules. A faint rule scans
     as dashes the rule removal leaves, and where a row's line meets the
     column's edge they cross in small "+"s tall enough to pass for
     writing: on one photo every empty row of the agents' Signature column
     was masked. Vertical runs within 4 px of the column's edges are
     dropped; what is left of a "+" is flat, and goes with the flat strokes
     below. A stroke crossing an edge loses only what runs along it. Flat
     runs along the rows' lines are left alone: a signature written as a
     short dash on its row's line is as flat as they are;
   - the stamps found on the page (see [Stamps](#stamps-are-never-masked-stampspy)),
     ring and lettering. A stamp's ring once joined the writing around it
     into one "signature", whose mask then framed the stamp with black
     corners (writing under a stamp is never masked anyway);
   - strike lines drawn through unused rows (below);
   - flat strokes 8 px or less in height, which are pieces of rules.
3. **Group strokes.** Strokes that touch, mostly sideways, form one signature.
   A group spanning several rows (neighbours' strokes touching) is cut at the
   row boundaries. A group counts only if one of its strokes is unbroken
   for 40 ink pixels or more: crumbs of a security print's lettering, close
   together, were joined into "signatures" and masked in empty fields. And
   writing cut off by the window's bottom edge that starts well below the
   last row is the next line's (an officer's name written large under the
   agents table), not a signature running over its row.
4. **Mask.** Each signature gets its own band, along it through its middle.
   The band follows the writing in 24 px pieces, so a signature rising or
   falling along its line is covered all along. It is half the signature's
   height thick (the height of its taller pieces, so a long flat tail
   doesn't thin it; at least 10 px). Each piece is centred on the middle
   row of its own ink and on that of the writing around it (the median of
   the middles of the two pieces either side); where they differ (a loop,
   a tall letter) it covers both. The mask reaches 2 px beyond the band,
   whose edge rows, warped back from the page, blend with the paper.
   Empty rows are untouched.

**Underlined fields** (a signature on a line, not in a table) take only
writing on their own line. The date is written on the next line, often so
close that the strokes of both were joined into one "signature", and the
date was masked with it:
- a group of strokes counts only if at least 30% of its ink is on the
  field's line (tails and flourishes reach far above or below), or if it
  crosses the line with a signature's worth of ink (120 px) above it,
  however far the rest hangs below: writing from the next line doesn't
  cross this one. Some 2013 officers wrote a short scrawl on the line and
  loops well below it, most of the ink under the line, and their
  signatures were left visible;
- strokes lying mostly on the line below or above (the date, the ID) are
  left out even where they run close to the signature;
- the field's printed underline joins the strokes crossing it (ink just
  above and just below it), so a loop of the signature through the line
  stays part of it; lettering of the security print sitting on the line is
  not joined, which stretched one mask along the line to a stamp. The line
  sets no mask's width: on layout B the underline is shorter than the
  field, and was taken as part of the signature, stretching its mask to
  the line's end. The line is the one where the reference has it (within
  20 px: layout variants move it), or without a reference the lowest in
  the field: a signature's long, flat tail is a line too, and was taken out
  and left visible as one. Where
  the scan breaks the line into dashes, ink lying wholly within a few
  pixels of it is taken as the line too: joined through it, those pieces
  stretched the mask along the line and past a stamp;
- a field's line is at most two thin lines. A scan striped all over with
  long horizontal lines has no line taken out at all: every stripe would
  have been taken for it, and the signature written across them with it;
- with a reference, the printed line is found where the reference has it,
  however many long strokes an officer drew beside it.

The band hides the signature's middle, not the strokes running down into
the date line, so a date written under a signature stays readable.

A coverage check then finds signatures again on the original, works out each
one's band the same way, and requires the ink inside it to be at least 90%
under black in the output. Blackout zones (plain
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
   - leftover thin fragments of rules;
   - printed dotted leaders ("ID Number: ........");
   - an embossed or striped background. Some forms are embossed, and scan
     with a background of thin, dark horizontal lines every few pixels, its
     lettering kinks in them, as dark as print. In empty fields it was
     taken for writing and masked. Where many rows of a strip are mostly
     thin line (8% or more; 0-1% on plain scans, 16-31% on the embossed one
     in the samples), thin horizontal runs (12 px or longer, under 3 px
     tall) are dropped. Pen strokes are thicker, and a line crossing one
     leaves it whole. This applies to signatures as well;
   - strike lines found over the whole table (below);
   - the ink of stamps found on the page. Stamps are found before anything
     is masked: a ring read as a digit stretched one ID's masks up into the
     name above it.

**Lines drawn through unused rows.** Within one column a cross-out line is
only a short stroke, too short to tell from a pen stroke: on Form 34B layout
C it crossed a 218 px column at 9° and was masked as ID digits, and as a
signature in the signature column. So strike lines are looked for over the
whole printed table, in the template's frame, before any column is read
(`masking.find_strikes`). A strike is a straight stroke at least three rows
long, at 3–80° (a printed rule can't have that angle), unbroken along at
least 90% of its length and thin. A row of handwriting that a straight line
happens to pass through is broken up, so it never qualifies: erasing written
digits from the ink would leave them unmasked. The strikes found are erased
from the ink in every column of that table, in masking and in the leak
checks alike, and the page notes say how many there were.
3. **Ignore print.** A glyph whose surroundings look like the reference at about
   the same spot is printed text (column headers) and is skipped. This test
   compares appearance, not just position: if the table is a row off, written
   digits land where the reference has print, and a position test would skip
   them.
   A glyph must be at least 40% of the row height (a template can lower
   that: in Form 34B's tall rows and on its officers' lines people write
   small, and a whole ID written small was taken for unreadable ink and
   hidden). Outside the table's rows it must still be 40%, so a printed
   header's small lettering above the column is not taken for writing. On
   an officer's line, writing cut off by the field's top edge that ends in
   the field's upper half is the foot of the name above, written large and
   low; taken for a character, it took one of the three hidden slots.
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

   **`--full-ids` and `--full-phones`** hide every character instead: each
   ID or passport number, or each phone number with its prefix, is covered
   from its first character to its last. Each flag is for its own rule, so
   with only one of them the other numbers keep theirs. The line is still
   segmented into characters and every slot hidden, so the mask has the
   same height and reach as a hidden character's; a line that can't be
   segmented was hidden whole already. The leak check still looks for 5
   written characters in a row (8 for phones), and also for 3 off the
   protected stamps: the default rule leaves runs of up to two visible, a
   whole number none, so three showing are part of a number that should be
   hidden. Writing on a protected stamp stays visible by the stamp rule. On
   one 2022 Lamu form the stamp's edge lay over the first three digits of
   two phone numbers. Counted, they held the page, and so would every
   number a stamp touches with `--keep-stamps`; by default they are hidden
   along the field's line. On 225 pages
   (two 2022 Form 34A zips, the 2013 Diaspora scan), the same pages were
   published and held as with the default rule. The settings are recorded
   in the manifest (`settings.full_ids`, `settings.full_phones`) and
   `verify --rerun` repeats them; the VLM review is told that those numbers
   must be hidden entirely.

   A hidden character is covered from the midpoint with its left neighbour to
   the midpoint with its right, from the line's top to its bottom (the
   median over its characters) or the character's own extent where that is
   larger. Rule removal can shave a 5's or 7's top bar off the detected
   glyph box, so masking only the box could leave part of a "hidden" digit
   showing. A line that can't be segmented is hidden whole, and so is ink
   in a field that makes no characters at all: as wide as its ink, not the
   field. On an officer's line, the field's full width ran on behind the
   stamp beside the number.
6. **Assign rows.** Lines are assigned to rows only to tell empty rows from
   unreadable ones, and for the report. Masks follow the writing itself.

### 4. Gates (fail-closed)
A document is published only if every page passes:

| Gate | Catches |
|---|---|
| Page alignment checks | unknown form or layout, blank or garbage page |
| Form type has a template | 34B, 35A, 36A, … until templates exist (or read without one, below) |
| Reading without a template | a column of the agents table not found; officers named in the declaration but no signature line found; no PII fields at all |
| QR code vs file name | form uploaded under the wrong station |
| **Column edge check** | table matched one rule off. The straightened column must have its printed rules at its own edges, not a stronger rule somewhere else. This compares the mapping against the paper, because comparing the masks against the same mapping can't detect it |
| **Leak check** | the redacted column is re-read over a wider window; 5 or more written characters in a row with no mask between them means something was missed (8 for phone numbers; for numbers hidden whole, 3 off the stamps) |
| VLM audit (optional) | anything still readable anywhere: free-text comments, signatures outside the boxes |

The edge check and the leak check were both added after reviewing real outputs
showed misses that geometry alone allowed through.

### 5. Output
The output is a new PDF (`io.rs`). Each page holds exactly one image: the
redacted JPEG, passed through byte for byte, or a lossless PNG for non-JPEG
sources. The page has the same size and layout as the source page. There is
no text layer, no annotations and no metadata. JPEG markers other than
JFIF/ICC/Adobe (for example EXIF, which can hold camera serials or GPS) are
dropped.

**The page shows the image the way the transmitted file did.** Redaction
works on the image as stored, whichever way up it is: pages are matched to
their template, or read without one, at any quarter turn. The page then
draws the image turned or mirrored just as the source did:
- a JPEG file by its EXIF orientation tag. The tag is dropped with the rest
  of EXIF, so the turn goes into the page instead. A phone photo stored
  sideways and tagged upright was published sideways before;
- a PDF page by the matrix that places its image on the page, as well as its
  `/Rotate`. A scan placed turned was published unturned before.

`verify` checks this against the original: same page size, placement and
rotation. Mirrored orientations (EXIF 2, 4, 5 and 7; cameras don't write
them) are drawn correctly, but their pages can't be matched to a template,
and they go to review.

**A photo with no orientation tag (`--orient`).** Such a file is shown as
stored, and many are stored on their side: 32 of the 35 published 2017
Form 34A photos were. With `--orient`, Clef (a decision model on Ollama
Cloud) is asked which way up each such page stands. It is shown the page as
published, redacted and at most 1024 px long. Unredacted pages are never
sent. It is asked once: upright, on its right side, upside down or on its
left side. If it isn't judged upright (a probability of 0.5 or more), the
page is also asked about turned a quarter, a half and three quarters
clockwise. The four answers are taken together: the page is turned t
quarters if every answer fits that best (the sum of the log probabilities).
Asked alone, the two quarter turns are the ones Clef confuses.
The turn is made only if Clef judged the page upright when shown it turned
that way; otherwise the page is published as stored and marked undecided.
The turn goes into the page's `/Rotate`, so the image is untouched. The
manifest records it per page under `orientation`, with each image sent (its
SHA-256), the probabilities and the turn. `verify` accepts a recorded turn
only for an image file with no orientation tag, and `--rerun` applies it
without asking Clef.

On 13 redacted pages of six printings (2013 to 2022), each in its four
quarter turns, the four answers together got all 52 right, and one
question alone got 44. An upright page scored 0.94 or more for "upright",
and a turned one 0.09 or less, so one question is enough for a page that
stands upright. On the 2017 photos, Clef was asked 131 times for 35 pages,
none was undecided, and all 35 were published upright.

Next to each PDF goes its signed manifest (`.manifest.json`), plus the ledger
and batch roots described below. Documents that fail go to `review/`, each
with a JSON report of why. `review/report.jsonl` logs every document.

**Digital signature (`pdfsign.rs`).** With a certificate, each published PDF
is signed as PDF readers expect: an incremental update appends an invisible
signature field whose value is a detached PKCS#7 signature
(`adbe.pkcs7.detached`, SHA-256) over every byte of the file except the
signature itself. The redacted PDF's bytes stay an unchanged prefix of the
signed file. The manifest records the output's hash after signing and names
the certificate by its SHA-256, so the manifest's Ed25519 signature also
vouches for which certificate signed the PDF.

With no certificate given, `redact` uses the one `rfpii certgen` saved
(`~/.config/rfpii/signer.p12`). If there is none, it offers to create one:
RSA-3072, self-signed, for digital signatures only (key usage
digitalSignature and nonRepudiation; extended key usage emailProtection,
which Acrobat accepts, and documentSigning). The private key is kept in a
password-protected PKCS#12 file readable only by its owner, and the public
certificate is written beside it to be published. A self-signed signature
proves the file is intact and was signed by whoever holds that key. Who that
is, readers learn only by comparing the certificate's fingerprint with the
one the publisher announced. A certificate from a CA they already trust
makes that link for them.

## Stamps are kept (`stamps.rs`)

The presiding officer's stamp is what readers look for to tell a genuine
form. It is often pressed across a signature, an ID or a phone number. The
rule: **no blacked block may touch a stamp except along a field's line
across it**, and the rest of the stamp is checked unchanged. Until October
2026 the rule was absolute (no blacked block touched a stamp at all, even
when that left a signature or a number readable under it); `--keep-stamps`
still applies it.

- **Finding stamps.** Two ink maps are made: dark ink, and coloured ink (any
  colour that stands out from the paper's own, so faded violet or blue stamps
  count; the green security print doesn't). Circles found by a Hough transform
  in three radius bands are refined by least squares, each from three
  starting radii: a stamp's lettering forms circles of its own just inside
  the ring, and the fit must not settle on one of those. A circle is kept
  only if ink runs around it: continuously, densely, and with little ink
  just outside it. Those thresholds were measured on the sample forms. The
  radius range comes from the page: 60–135 px on a 2340 px KIEMS page (the
  stamps there measure 70–110 px; loops in signatures and dates come out at
  30–60 and must stay masked), or 1.6–4.2 × the printed text height on a form
  read without a template. A line OCR reads as stamp
  lettering ("PRESIDING OFFICER", "I.E.B.C.") starts a second, more permissive
  search around it. Where no ring is found round it, a circle is guessed
  from the line, as wide as it is; but not for a line reaching into a ring
  found. That line is the ring's own lettering read with the writing next
  to it: "LUJMOFFICER", the officer's name written through the stamp and
  then its "OFFICER", made a circle twice the stamp's size that took in
  the name, the ID and the signature, and left them visible.
- **A second look where masks are about to go.** A faint stamp under a
  signature is the most likely miss and the costliest. So, before masks are
  applied, the area around them is searched again with lower thresholds, and
  any ring found there that reaches into a mask is added. A ring there
  smaller than the stamp range must run all the way round: a handwritten
  "0" in a date, with a signature's strokes around it, fitted a small
  broken ring, and the signature inside it was left visible. And a ring
  there must run in long arcs: on the sample forms the inked arcs of real
  rings, faint ones included, average 14 or more of 180 steps round, while
  a column of IDs in blue pen (coloured, like stamp ink) put short strokes
  all round a circle (arcs of 7 on average) and took the IDs' masks off.
  Near masks a ring needs arcs of 10 or more.
- **Protection.** Each stamp's disc, plus a margin (6 px + 4% of the radius),
  is rounded out to whole 8×8 blocks and removed from the mask canvas. Writing
  under a stamp stays visible; the page notes say how much was lifted.
- **Gates.** Signature coverage ignores ink inside a stamp, so a stamp over a
  signature doesn't send the page to review. The VLM audit is told to ignore
  writing inside stamps and to report any stamp a mask covers
  (`stamp_covered`); either a covered stamp or readable PII fails the audit.
- **Record and verification.** Stamps are listed among the page's protected
  regions in the manifest (`kind: "stamp"`, the circle, and how it was
  found). `rfpii verify` checks that no blacked block touches one and that its
  pixels decode identically in the original and the redacted form, as for the
  results. The overlay outlines stamps in magenta.
- **A field's line across a stamp is hidden (the default since October
  2026).** Officers often stamp over their own details: in the 2013 Forms
  34, 560 of 6,608 published pages had half or more of an officer's ID line
  inside a stamp, and on some the whole ID was readable there. The Data
  Protection Act's minimisation duty (s.25(d), kept binding by s.51(1) even
  where publication is required) and the courts' use of the stamp as a mark
  of authenticity are both met by hiding only the line and keeping the
  rest (`legal/C-recommendations.md` §2.6). Each field's band across a stamp
  (its zone: an ID's or phone number's line, a signature's line, a table
  cell, a blacked-out zone as masked) is blacked whole, as far out as the
  stamp is kept clear; the near stamps found around masks are treated the
  same. What is written under a stamp can't be told from the stamp's own
  lettering: on Isiolo North-2 p17 the blue pen lies 10–20° of hue from the
  violet stamp, inside the stamp's own spread, and the lettering is as dark
  as the pen. So it isn't read there, and the three-characters rule gives
  way to hiding the part of the number under the stamp. Every other mask on
  the stamp is lifted, as before. The stamp stays listed as a protected
  region, with the 8×8 blocks of the bands across it (`except_blocks`,
  [x, y, w, h], one run of blocks per row): `verify` requires the rest of
  the stamp untouched and identical to the original, and the overlay still
  outlines it. The setting is recorded in the manifest (`settings.stamps`:
  `bands`, `protected` for `--keep-stamps`) and `verify --rerun` repeats it;
  manifests from before it re-run as recorded then (`settings.mask_stamps`:
  `false` protected, `true` the old `--mask-stamps`, which protected no
  stamp at all). The VLM review is told that writing inside a stamp follows
  the rules, and what it finds there is masked.

## Verifiability

### The edit is restricted to black blocks
The pipeline never edits the image freely. Masks are drawn on a separate
canvas, rounded **out** to whole 8×8 blocks (`pipeline.rs`), and only then
applied to the image as **stored**:

- **JPEG sources (all KIEMS transmissions; `proof.rs`).** The edit is made on
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
  the polling-station counts. Without a template, everything above the
  "Declaration" label is protected. Every stamp found on the page is protected
  too. A mask block touching any of them stops publication.

### The record
- **Manifest.** For each document (`process.rs`) it records:
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

  It is signed with the publisher's Ed25519 key (`manifest.rs`).
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
- that each results region decodes to identical pixels in both files;
- that each page shows its image as the original does (page size, placement
  and rotation, EXIF orientation included), or turned upright as the
  manifest records for a photo with no orientation tag.

It also writes an overlay showing every changed block. With `--rerun` it
repeats the redaction from the recorded seed and templates and must
reproduce the output bit for bit.

### Tested
On the 215 real Forms 34A, all 188 published forms verify. Re-running a
random sample reproduced each one bit for bit. Tampering is detected in
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
| "The masking hid the stamp, so the form can't be authenticated." | No mask block may touch a stamp; the verifier checks it and the overlay outlines each stamp. A stamp the finder missed would show on the overlay as an unoutlined stamp, and the VLM audit asks about it. |
| "The original itself is fake." | Out of scope for redaction. The manifest ties each form to its source hash, KIEMS kit serial, capture time and QR, for comparison with IEBC records. |
| "The signing key was stolen." | Keep it offline. Publish its fingerprint once. Rotate by announcing a new key in a signed statement from the old one. |

## Vision review (`vlm.rs`, Ollama Cloud or local)

The local rules miss things a person would see: a signature in the comments
made of short strokes, a table placed off its columns. Run at scale there is
no one to look, so with `--vlm-audit` vision models look at every redacted
page before it is published.

**Only redacted pages are sent.** The review sees exactly what would be
published, so it is safe to run on Ollama Cloud. (`--vlm-fallback` and
`rfpii propose`, which need the unredacted form, are refused for any
non-local host unless `--allow-remote-unredacted` is given.)

**What it does.** For each page, from the declaration down (the template's
`review_area`; the whole page without a template):
1. **Find.** The review model is asked for every spot that breaks the rules:
   an ID or phone number with all its digits readable, a signature with no
   band across it. It answers with boxes.
2. **Already masked?** A spot that is 8% or more pure black is something
   masked already (a banded signature, a number with digits hidden), and
   isn't asked about again.
3. **Check.** Each other spot is cropped 80 px wider all round, zoomed 2x,
   boxed in red, and the checking model is asked one narrow question: is
   there an unmasked signature or a complete number in the box? The answer
   must match what the finder named: a name the finder took for an ID, or a
   printed "Tel." header it took for a phone, is not kept because the
   checker sees a signature beside it. A reply that can't be read counts as
   yes.
4. **On a stamp?** With `--keep-stamps`, a spot at least half inside a
   protected stamp is left visible: the stamp rule would lift a mask there
   anyway. It is recorded (`on_stamps`) but not masked. By default every
   spot is masked, its band across a stamp included.
5. **Mask and repeat.** What is confirmed is masked through the same block
   masking and proof as everything else: signatures hidden whole, numbers
   with 3 digits hidden. The page is redacted again from its seed and
   reviewed again, up to `--vlm-rounds` (2) more times. A page is published
   only when a review finds nothing new; otherwise, or if a model can't be
   reached, it goes to review.

**Rescue (`--vlm-rescue`).** A page held only because a column's place or
masking couldn't be confirmed (the edge check or the leak check) is
redacted again with those columns, and the signatures of their table,
hidden whole: a quarter of a column wider either side and half a row above
and below. It is published if the review then finds nothing. This hides
more than the rules ask (whole numbers, parts of neighbouring columns), and
only the review makes it safe, so it requires `--vlm-audit`.

**Reproducible.** The manifest records, per page, what the review found
(`vlm_found`) and any columns hidden whole (`whole_columns`), and the whole
review (`vlm_review`: models, host, prompt hashes, the hash of each image
sent, every spot and its verdict). `rfpii verify --rerun` applies the
recorded findings instead of asking a model, so a re-run is bit-identical.

**Models.** Twelve vision models on Ollama Cloud were tried on redacted
Forms 34A (October 2026). The default for both steps is `gemma4:31b`: about
3 s a request. As a finder it found 19 of 22 signatures the rules had
missed on 17 pages, and 12 of 18 made-up IDs, phones and scribbles written
into empty rows; alone it also flagged something on every page (banded
signatures, dates, names by a stamp), which steps 2 and 3 remove: of 88
spots, the black filter dropped 40 false ones and no real one, and the
check kept 26 of 31 real misses and dropped 53 of 56 false alarms.
`minimax-m3.1-flash` found as much but took 80–200 s a page and did worse
as the checker; `kimi-k3` boxed accurately but took 4–8 minutes; others
answered in prose, timed out or boxed in an inconsistent scale. Most models
ignore the requested JSON format, so replies are read leniently (fenced or
wrapped JSON, `bbox`/`box`/`box_2d`, free-form labels), and Gemini-style
`box_2d` boxes ([y1, x1, y2, x2] on a 0–1000 grid) are recognised.

**Clef as the checker (`--vlm-checker clef`).** Clef is a decision model:
it doesn't write a reply but scores every option of a typed question in
one pass, through Ollama's `/v1/systemone`. It can't draw boxes, so it can
only check. Each spot is asked one multiple-choice question (an unmasked
or masked signature, a name, a date, a complete or partly hidden number,
printed text); a spot is masked when the options for what the finder named
reach `--vlm-check-threshold` (0.5). Both signature options count: spots
asked about are under 8% black, so a "masked" one is crossed by its
writer's own stroke. A plain yes/no question read names as signatures.
Measured on the same 215 Forms 34A (October 2026):
- The same crop gets the same score every time, to 4 decimals; the score
  and threshold are recorded per spot.
- On its own a check takes 0.5–1 s against about 3 s for `gemma4:31b`. But
  it shares the account's limit with the finder (12 of 24 requests to both
  at once were refused), and the account served at most about 2.6 Clef
  requests a second however many were sent at once.
- The full run took 404 s against 380 s, and published the same 210. It
  masked 59 spots on 46 pages against 45 on 38: by eye 40 real, 8 unclear
  and 11 not (printed "Signature" labels, empty boxes beside them, names,
  digits the rules had already masked), against 38, 5 and 2. The extra
  masks also sent more pages to a second round of review.
- All 210 verify and the 51 pages it changed replay bit for bit.

So it is not faster here and masks more that isn't personal data; the
default checker stays `gemma4:31b`. Its use is a checker whose verdicts are
repeatable and come with a probability to set a threshold on.

**At scale.** Ollama Cloud refused (HTTP 429) more than 12 requests at once
from the account this was tested with, so requests are capped across all
workers (`--vlm-concurrency`, 10) and retried with backoff on 429, 5xx and
dropped connections. The review adds 2.3 model calls a form on
average (one look per round, one check per spot not already black), each
about 3 s with `gemma4:31b`; a form spends most of its time waiting, so
run about three workers per core, as memory allows (each worker peaks at
about 1.3 GB). On 4 cores with 8 workers the 215 Forms 34A took 380 s
(0.57 forms/s; without the review 0.8 forms/s). For 50,000 forms that is
about 120,000 calls: at 12 at once on one account, about 8–9 hours of model
time, which more cores don't shorten. More accounts, a plan with a higher
limit, or a local GPU running the same model lift that limit; the local work
then runs at about 5 core-seconds a form (about 4 hours on 16 cores).

**Results on the 215 Forms 34A** (`--vlm-audit --vlm-rescue`): 210
published, against 205 without the review and 188 before any of this.
- The five pages whose columns couldn't be placed were rescued: their
  ID/phone columns hidden whole, and the review found nothing after.
- The review masked 45 more spots on 38 published pages. Checked by eye
  against the originals, 38 were real misses (signatures, mostly in the
  comments), 5 were unclear (a name written beside a signature, a box
  mostly beside one) and 2 were not: a date and the word "signature"
  written out. It left 8 spots on stamps visible. 172 pages needed nothing
  after the first look, 35 after a second and 3 after a third.
- An earlier run, before the label match in step 3 and step 4 were added,
  masked 50 spots, of which 9 were not PII (writing on stamps, a date, a name, the printed
  "Tel. Contact" header, a stray mark) and 4 unclear.
- All 210 verify, and `--rerun` reproduces each of the 42 pages the review
  changed bit for bit, from the manifest alone.
- Five go to review: four the page alignment can't match, and one whose
  table is placed so far off that hiding its columns would touch the
  results.
- With each column on its own rows and checked against its own header,
  209 publish. The sixth held is the form whose table the snap had put a
  row high and 70-160 px across: it had published with an agent's phone
  number in full. Its Tel. Contact column can't be placed where its header
  is, and hiding it whole would touch the results. The header check moved
  the columns of 30 published pages, mostly whole tables a row or more up
  or down; checked by eye on six of them, five now sit on their printed
  rows and one Signature column is still a row high (its header matched in
  the wrong place, close enough to pass). The review masked 46 spots on 37
  pages; all 209 verify and the 42 pages it changed replay bit for bit.
- The models don't answer the same way twice. The same spot was called a
  phone number in one run and a vote count in the next (it was masked by
  the rules either way), and a page the review still flagged after its
  last round in one run was published in the next. Each run's findings
  are recorded, so what was published can always be reproduced; what a
  new run would find can't.

What it doesn't do: it is a second pair of eyes, not a guarantee. Some
signatures it doesn't find or the check reads as words; writing on a stamp
stays visible by the stamp rule even when the review points at it.

## Results on real data (Form 34A, 215 KIEMS PDFs, two county zips)

| Outcome | Count |
|---|---|
| Published | 188 |
| Review: column edge check | 22 |
| Review: page alignment failed or other | 4 |
| Review: leak check | 1 |

That is 188 of 215 (87%) published, and all 188 verify. Throughput was about
3.7 files/s on 4 cores, including the coefficient-level edit and the
self-check; the stamp search added since cuts it to about 1–1.5 files/s.
Adding stamp protection changed none of these outcomes: the same 188 forms
publish, and 177 of them have a stamp below the header (262 in all), each
protected. Every stamp the search finds or stops finding between versions is
checked against the original by eye. With per-signature masks, the blacked-out area is a median of
3.0% of each image (it was 5.4% with whole-column signature blocks), at most
7.3%. Several
random samples of the published forms were reviewed by eye. Every review that
found a problem led to a fix, and those cases are now caught by the gates
above. Withheld forms are a mix of genuinely misplaced tables and forms whose
faint rules can't be confirmed. The VLM fallback, a person, or a better
template can clear them.

Since then the comments are read too (see "Comments on a templated form"):
signatures written there are hidden on 21 pages, and six pieces of comment
words on five pages are masked with them; the outcomes above are unchanged.

## Templates

| Template | Status |
|---|---|
| `form34a_2022` (2022 Form 34A, KIEMS) | Built from a real form; 188 of 215 published, all verified |
| `form34b_2022_block`, `form34b_2022_results`, `form34b_2022_results_b` | Form 34B layout A (the system-printed form, the most common); see below |
| `form34b_2022_block_b`, `form34b_2022_results_layout_b` | Form 34B layout B (aggregate results, turnout, agents table; no handing over) |
| `form34b_2022_block_c`, `form34b_2022_results_layout_c` | Form 34B layout C (Garissa county's printing, dotted lines) |
| Other Form 34B printings, and heavily distorted copies | Not yet; routed to review |
| 2017 Forms 34A, 2021 by-election Forms 35A / 36A / 38A, other printings | Read without a template; see below |

### Form 34B (constituency tally)

Form 34B is multi-page. The results table runs over as many pages as the
constituency has polling stations. The declaration block follows the last
row, so it sits at a different place on a different page in each document,
and it often shares a page with the last results rows. The block holds:

- the agents table (4 rows): name, ID/passport, party, phone, signature, date;
- the Constituency Returning Officer: name, ID number, signature, date;
- the handover to the Commission Chairperson: names, ID numbers, signatures, times.

Results pages carry no personal data. So each 34B layout uses a block
template and one or two results-header templates (layout A's are described
here; B and C follow the same pattern, below):

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
- **Document rule:** a 34B document is published only if a block, of any
  layout, was found on **exactly one** page. If it was found on none (a bad scan, or a layout
  without a template), the page holding the personal data could otherwise
  pass as a plain results page. If it was found on two, the block is split or
  documents were merged. Either way the document goes to review.

The references come from the blank, system-generated form, rendered from an
IEBC PDF with empty fields, so there is no handwriting to erase. The variant
results header is a typed row cut from a scan.

Features for a reference cut from a page are extracted at the page's scale
(`page_long_side`), so that text sizes match the scans. Form 34B templates
take features at 2400 px with a 20000-feature budget (`feature_dim`,
`n_features`), where Form 34A uses 1200 px and 5000: 34B scans are up to
4960 px with a dense security print, and at the default settings the
background's corners crowded out the printed text's. Raising them recovered
eight layout A pages that weren't recognised.

**Layouts B and C.** Some constituencies printed a different form:

- **Layout B** (Budalangi): after the results, an "Aggregate Results" table
  (votes per candidate in figures and words), a "Voter Turn Out" table, a
  5-row agents table, and the Returning Officer's name, ID, signature and
  date on underlines. There is no handing over section. The aggregate and
  turnout tables are protected regions.
- **Layout C** (Garissa county's printing; Daadab): "Form 34 B /
  Constituency / County" above the results, a 3-row agents table with a
  green header, the Returning Officer on dotted lines, and a handing over /
  taking over box, also on dotted lines.

There is no blank of either, so their references are cut from scans and
cleaned: every piece of handwriting, the stamps, the typed results and the
turnout figures are erased, the security-print background is removed, and
the printed rules and dotted leaders that ran under the writing are
redrawn at their measured positions. Each reference was checked at full
resolution for anything left behind.

Two things these layouts needed:

- **Dotted leaders are print.** A partial-mask zone on a dotted line saw the
  dots as ink it couldn't read and hid the whole field, even when empty.
  `ink_mask` now removes runs of six or more tiny, evenly spaced blobs on
  one line, which handwriting doesn't produce.
- **Tight windows for the agents table.** In every layout the Returning
  Officer's name starts right below the table, and the default search
  window of an empty last row reached it: the name was read as an ID and
  three of its letters masked. Those columns search 4 px above and below
  (layouts B and C), or 20 px below (layout A, whose name line is further
  down).

- **Rules that seem to run on up.** On some scans a dark band across the
  top of the page makes the agents table's column rules seem to run on up
  past it. Where their lower ends are where the table ends (within 20 px),
  they place the table, as for a table hung under another (below);
  otherwise the zones fell back to windows widened 200 px down, which took
  in the officer's name.
- **A table hung under another.** In layout C the agents table sits right
  under the results table, with the same columns, so its column rules run
  on up into the results table and their top end doesn't mark the agents
  table. Snapping first failed there, and the ID and phone columns fell back
  to zones widened 200 px downward, which swept up the Returning Officer's
  name. That region is marked `open_top`: only the rules' lower end, and
  the row rules, place the table.

On layout C the lines are 51 px apart and handwriting is often taller than
that. A signature that runs down into the date line still covers the part
of the date it overlaps; the rest of the date stays visible (see
underlined fields under [Masking](#3-masking-maskingpy)).

**Single underlined fields** (the RO's and Chairperson's ID and signature)
are handled differently from table cells:

- **Zone height:** each zone spans from just below the previous printed line
  to just below its own, so handwriting of any height is inside and the name
  on the line above is not.
- **Anchors:** each zone is anchored to its section's printed "ID Number:"
  label. The tool finds that label within ±120 px and moves the zone with it,
  because layout variants put the section a line higher or lower. If the
  label isn't found, the page fails. Every zone tied to a label is placed
  through the mapping the label was found in: the zones lie within a line
  or two of it. Placed through their own local mappings, which on a curled
  page can disagree with the label's by a line, a signature's window landed
  on the ID's line, and one mask hid the whole ID and both labels.
- **Stamps:** round official stamps are detected, and printed ink wholly
  inside a stamp is not treated as a signature, so the stamp stays readable.

**Sample results (all 42 constituencies in Dropbox `Work/Kenya/2022/Forms/34B`):**

| Outcome | Documents |
|---|---|
| Published, layout A | 24 |
| Published, layout B | 1 (Budalangi) |
| Published, layout C | 1 (Daadab) |
| Review: other printings (Kabuchai's aggregate-results form; Bumula's "Chief Agents Affirmation"; Balambala, with handing over at the county; Sirisiya, with the block split over two pages) | 4 |
| Review: layout A or C pages not recognised (landscape scans of the rotated form, "COPY" watermarks, small or curved scans: Baringo North and South, Bomet Central, Chepalungu, Ijara, Kanduyi, Konoin, Manyatta, Mt Elgon, Runyenjes, Teso North) | 11 |
| Review: heavily curved phone photo (Fafi) | 1 |

Every published block page was checked by eye against its original.

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

## Forms without a template (`generic.rs`)

Older forms (2017 Forms 34A, the 2021 by-election Forms 35A, 36A and 38A) came
in many printings and were scanned or photographed in every orientation.
A template per printing doesn't scale, so a form with no template is
**read**, using the same printed labels and rules a person would look for.
It is on by default (`--no-generic` turns it off) and needs the OCR models
(`setup-deps.sh` installs them).

1. **OCR, locally** (`ocr.rs`). RapidOCR's PP-OCR models on ONNX Runtime, on the CPU;
   nothing leaves the machine. The recogniser's output is decoded here, so
   every character has a position and a confidence. Labels are matched
   allowing for misreads ("Shature", "bate 17-10-2019").
2. **Upright and to scale.** Orientation comes from the shape of the text
   boxes and the 180° classifier, then the page is deskewed and scaled so
   that printed text is 34 px high. A homography maps every field back to the
   original image.
3. **Results are protected.** Everything above the declaration (its
   "Declaration" label, or its "We, the undersigned…" sentence when OCR
   misses the label) is a protected region. A page with neither goes to
   review. So does a page whose form type has templates (Forms 34A and 34B):
   it is only ever redacted by a template. An earlier fallback, protecting
   down to the last line naming a count, was removed after a Form 34B
   results page, which has no declaration, took its title for that line;
   the rest of the page was left open to masking.
4. **Agents table.** The column headers ("ID/Passport No.", "Tel. Contact",
   "Signature") are found by OCR. Each column runs between the printed rules
   either side of its header, and the rows come from the horizontal rules.
   A column rule hidden by handwriting in every row is recovered from the
   ink profile; if it can't be, the page goes to review.
5. **Declaration.** Each officer's line is found by its "Signature" and "Date"
   labels; the signature field lies between them. The band is OCR'd again at
   2.5× because, at page scale, a label running into handwriting is often
   missed. Both reads are kept, so a label is often read twice: it makes
   one field, from the read that reads more of it, and the other read's
   words are no next label ("ID No. / Rassport:" ends at "No.", and the
   good read's "Passport" was taken for the label after it, leaving no
   room for the field). Nothing is written under a 2013 officer's last line
   but the stamp's lettering, over the security stripe: there a signature
   hanging below the line is the line's, as far as a printed line lower.
   Taken for the next line's, such signatures were left showing. The field
   itself keeps its height: a taller one moved its middle down, and a
   signature written high, reaching in from the top of the search and
   ending above the new middle, was taken for the line above's. The officers' written names are kept for step 7. A page naming
   more officers than it has signature lines goes to review; a stamp's
   "PRESIDING OFFICER", read in capitals inside the stamp's circle and often
   run into the name written under it ("PRESiDINGNSOZOGZ"), names no
   officer. The printed label, in small letters, still does under a stamp.
6. **Numbers written anywhere else.** An ID (7–9 digits that aren't a date,
   or digits after "ID") or a phone number written below the results gets the
   same partial masking as in the table. A label ("ID No. / Passport:",
   "Tel.", "Mobile No.") names the field after it. OCR often misreads the
   slash in "ID No. / Passport:" as a letter or digit ("IDNo.1Passport:"),
   or the word itself ("Passoort:"): a word two letters or fewer off
   "passport", before a colon, is still the label. "Mobile" heads a label
   only at the start of a line or before "No." or a colon: it also ends the
   names of mobile polling stations ("LOMUKE MOBILE (039)"), and the station
   code after it was masked as a phone number on 37 pages of the 2013 run.
   A field is no higher than a printed line, whatever box OCR drew: OCR boxes
   a label with the ID written after it, and a field that high reached the
   "Date:" on the line below and masked it.
7. **Signatures outside any field**, typically an officer signing off under
   the comments ("name, ID, signature, date"). OCR tells writing from
   signatures: names, numbers and words are read, either confidently or the
   same way twice (on the page and in a re-read at 2.5×); a signature is read
   as nothing, as noise, or differently each time. Names on these forms are
   written in capitals, so joined-up lower case on its own line next to a
   sign-off is a signature even when it is read ("Kahenya"). Writing is
   taken for a signature:
   - next to an officer's own name, or on a sign-off line (one carrying a
     number, a date, an ID or "sign"), when it isn't read;
   - on the line above or below a sign-off, when it is cursive on a line of
     its own, or joined strokes OCR reads nothing in;
   - on a line with a word in capitals, or beside the stamp (officers sign
     by it), when it is a few strokes OCR reads nothing in.

   Names and numbers read well are taken out first, so a signature written
   on from a name is judged by itself. The mask is the box of the writing
   found, so it leaves the name, and a printed label the signature touches,
   readable. A straight line struck across a box is not a signature. Comments are left alone even where OCR can't read them: a
   remark such as "NO DISPUTED VOTES" or "6 REJECTED" is part of the
   record a scrutineer needs. Printed labels ("Presiding Officer's
   Comments") are never masked.
8. **Masking and proof are shared.** The fields found go through the same
   signature and digit masking as template zones, with glyph sizes scaled to
   the text height, and then through the same block proof, gates, manifest
   and verification. Typed "XXXX" fill and printed labels inside a field are
   left alone: a label as far as the characters OCR read in it, not the box
   OCR drew round it. OCR often boxes "Signature:" together with the
   signature written after it and reads the label alone; taking the whole
   box for print hid those signatures from the finder, which judged the line
   empty and left them visible: many of the 276 officers' lines judged
   empty in the first run over the 2013 Forms 34 were such signatures.
   And nothing is printed on an officer's line between "Signature" and
   "Date": a word OCR reads there was written, even one that reads as a
   label (a signature read as "No", the agents' table's heading). 2013's
   "ID No. / Passport:" is print as well: a signature's search reaches up
   into it, and "assport" was masked as part of the signature.

The manifest records what was read (the page frame, each field, the notes)
and the OCR engine with its models' hashes, so `rfpii verify --rerun` can
reproduce the page bit for bit with the same models.

### Results on the older forms (`Work/Kenya/Extracted`)

The folder holds 2017 Forms 34A from the repeat presidential election and
2018–2021 by-election Forms 35A, 36A and 38A, one folder per station or
ward; five of its 46 folders are empty. One page was taken from each of the
other 41 (three from the repeat election), 42 in all, covering every printing
in the folder:

| Outcome | Pages |
|---|---|
| Published | 36 |
| Review: a column header of the agents table not read (handwriting or a stamp across it) | 3 |
| Review: dim photo, nowhere to tell where the results end | 1 |
| Review: source PDF truncated, no page in it | 2 |

Each published page was checked by eye, original against redacted, with
the protected stamps outlined and the blacked blocks shaded:

- Every stamp is found and untouched (the IEBC logo is often taken for one
  as well, which does no harm).
- Signatures in the declaration and the agents table are masked through
  the middle, and so are
  the officers' sign-offs under the comments.
- IDs and phone numbers, in the table or written elsewhere, keep all but 3
  digits.
- Comments stay readable, including remarks about rejected and disputed
  votes.

Some sign-off lines lose more than they should: part of a name OCR can't
read, or a date, next to the signature. That errs on the side of privacy;
the officers' names stay readable in the declaration.

Building this found real misses, each now covered by the checks above and
by a test where one can be written:
- signatures written across or beside a stamp;
- a cursive signature that OCR reads as a word;
- a signature merged with a printed label OCR missed;
- a faint stamp the circle search fitted off its ring.

### Comments on a templated form

A template can name areas of free writing (`free_text`). Form 34A names its
comments area, "Reasons for Refusal to Sign" and the Presiding Officer's
comments, where officers often sign off and sometimes write an ID or phone
number. The area is straightened to the working text height (`text_h`) and
read the way steps 6 and 7 read a page without a template: numbers written
there are partially masked and go through the leak check, and signatures are
hidden whole (the box of the writing). The template's own printed letters
are taken out first, and stamps found anywhere on the page are protected,
including one the area cuts through.

The rule is stricter here, because the comments are where most of the
handwriting is: block capitals, often across the security print, which OCR
misses much of. A word OCR never finds as a line reads as nothing, the same
as a signature; read by the rule above, the first run took 145 pieces of
writing on 68 pages for signatures, most of them comment words, names and
dates. In the comments:
- the area is read a third time, on its ink alone (security print and rules
  removed), which finds lines the other two readings miss;
- writing read as capitals or digits with fair confidence is text;
- only a date or a number makes a sign-off line ("all agents signed" doesn't);
- a signature needs a joined stroke at least 1.4 times the text height wide
  that isn't a straight line, and fewer than 8 letter-sized pieces (a name
  with a flourish is mostly letters); being next to a word in capitals is
  not a reason (in the comments, that is usually another word);
- unread writing within a stroke's gap of a signature is part of it.

On the 215 Forms 34A the same 188 publish and all of them verify; `--rerun`
reproduces each of the 27 with writing found in the comments. Every piece of
writing masked there (45, on 26 pages) was checked by eye against the
original:
- 36 are signature strokes: 22 signatures on 21 pages, now hidden whole;
- one is too faint to tell;
- two don't matter to a reader: a line struck across the table, a mark by
  the stamp;
- six are parts of comment words, on five pages: "THE" (twice, on one page),
  "fair", and the "Y", "L" and "WA" of words written just above a signature
  whose strokes touch them.

One ID written in the comments is partially masked. What stays visible:
signatures made of short separate strokes or small initials, which these
rules can't tell from block capitals (at least 13 of them among what the
first run found), and any signature OCR reads as text. Forms 34B and the
older forms are unchanged: no free-text areas there, and the rule for forms
without a template is the same as before.

## Implementation

The tool is written in Rust (`rust/`). It was first written in Python, which
stayed the reference until the Rust version made the same decisions on all
the samples: on the 215 Forms 34A, the 42 Forms 34B and the 42 older forms
(the same seeds for both), every document was published or withheld alike,
with the same blocks blacked out, the same masks, notes, protected regions
and page alignment. Each version's verifier accepts the other's published
forms. With four workers, the Rust version took 112 s for the 215 Forms 34A
(Python: about 260 s); on forms read without a template OCR dominates, and
both run the same models (42 older forms: 300 s against 450 s).

To get the same output, and not just similar output, it calls the native
libraries the Python version ran on, built the same way. A masking decision
can turn on the last digit of a float, so a similar library is not enough:

| What | Library | How it is linked |
|---|---|---|
| Image processing, features, homographies | OpenCV 5.0 | built from the opencv-python source distribution with the wheel's CPU dispatch, Intel IPP and LAPACK options |
| PDF rendering and image extraction | MuPDF 1.28.5 | PyMuPDF's `libmupdf`, through a small C shim (`csrc/render.c`) that makes the same calls PyMuPDF does |
| JPEG coefficients | libjpeg | a private, renamed static copy (`csrc/jdct.c`), so it can't clash with MuPDF's or OpenCV's |
| OCR | ONNX Runtime 1.29 with RapidOCR's models | loaded at run time; RapidOCR's pre- and post-processing ported line by line |
| Least squares (`np.polyfit`, `lstsq`, `inv`) | numpy's OpenBLAS (LAPACK `dgelsd`, `dgesv`) | loaded at run time |

numpy's semantics are reproduced where results depend on them: float32
arithmetic, pairwise summation, how `linspace`, `arange` and `percentile`
compute, and Python's random number generator for the seeded choice of
hidden characters. `rust/scripts/setup-deps.sh` installs the libraries from
the packages they ship in.

Documents are processed in parallel (`--workers`), each on one thread, so
results don't depend on how the work is split.

## Known limits
- **Form 34B layouts.** Layouts A, B and C have templates. Four other
  printings in the sample don't, and 12 documents of known layouts weren't
  recognised (rotated landscape scans, watermarks, small or curved scans).
  Each new printing needs its own block template, built from a blank or
  cleaned copy. The document rule keeps every unknown layout out of
  publication.
- **Layouts B and C each come from one document.** Their references are cut
  from the only sample of each; other constituencies using the same
  printing may lay it out slightly differently (a 6-row agents table, typed
  names). Those go to review until a template covers them.
- **Blobs that can't be split into characters are hidden whole.** When an ID
  or phone number can't be segmented (e.g. faint ink), the whole number is
  hidden. That is the safe failure, but it is more than the 3-character rule.
- **Faint strokes of a signature can show.** A stroke too faint to read
  as ink (a thin flourish between a signature's initial and its final
  tick) is not masked; the parts either side of it are, each on its own.
- **Writing joined to a signature is banded with it.** Writing that touches
  a signature (a phone number continued into the column, part of the
  "Signature" label) is taken as part of it and loses its middle too.
- **How much of a signature shows is a judgement.** The band hides the
  middle half of each signature's height; the tops and bottoms of the
  strokes show. A signature made mostly of tall loops shows more of them
  than a compact one. Masks are applied in whole 8×8 blocks, which adds up
  to 7 px to the band at each edge.
- **Thin pen strokes on embossed forms.** On a form whose background
  scans as thin lines (below), pen strokes as thin as that background (a
  fine, flat flourish) are dropped with it and stay visible; the rest of
  the signature is masked.
- **Block granularity.** Blocks are 8 px wide and a handwritten digit is about
  15–20 px, so a black block can also clip the edge of a neighbouring visible
  character.
- **Strength of the masking.** Hiding 3 digits of an 8-digit ID, or 3 of a
  phone number, leaves 1000 possible values. That is enough against casual
  reading, not against someone able to test candidates.
- **The vision review is a second pair of eyes, not a guarantee.** On
  forms whose comment signatures the rules had missed, it masked about two
  in three; a signature that looks like a word, or one it boxes beside
  rather than on, stays visible. Its answers also vary from run to run (a
  form held once published the next time); each published form is
  reproducible from its manifest regardless.
- **A rescued page hides more than the rules ask.** Its ID and phone
  columns are hidden whole, with a quarter of a column either side, which
  can take part of a name or party next to them.
- **Pages that can't be aligned still go to review** (4 of 215 Forms 34A:
  photos the page alignment can't match), and so does a page whose table
  is placed so far off that hiding a column whole would touch the results.
- **PII outside template zones** is only caught by the vision review on
  templated forms, apart from the areas a template reads as free text (the
  Form 34A comments). This includes officers re-signing beside the stamp above the
  comments, and an agent's ID written into the name column. Forms read
  without a template, and the Form 34A comments, are searched for these
  (steps 6 and 7 above), but by rule, not by understanding: in the comments
  some signatures are missed and a few words are masked (see "Comments on a
  templated form"). Run `--vlm-audit` on them.
- **Writing under a stamp off a field's line stays visible.** Only each
  field's band across a stamp is hidden; a signature's flourish or a number
  written high above its line, inside a stamp, shows. With `--keep-stamps`
  everything under a stamp shows. The page notes record both.
- **A stamp the finder misses is not protected.** Rings that are very faint,
  mostly hidden, or far from circular can be missed. The second search near
  masks and the VLM audit's `stamp_covered` check reduce this; the overlay
  shows every stamp that was protected.
- **Reading without a template depends on OCR.** Dim photos, labels the OCR
  can't find, and source PDFs that are truncated send pages to review. OCR
  adds about 8–15 s per page on one core.
- **Re-runs are bit-identical on the same software and hardware.** ONNX
  Runtime can round differently on another CPU, so a `--rerun` of a form read
  without a template elsewhere may differ in a few blocks; the coefficient
  check, which doesn't depend on re-running, is unaffected.
- **Faint or heavily curled tables** go to review rather than being published.
  That's by design.
- **Thresholds come from 215 forms in two counties.** Run a pilot on other
  counties and look at `report.jsonl` before trusting a full run.
