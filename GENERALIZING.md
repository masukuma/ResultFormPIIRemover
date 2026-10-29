# Reading forms no one has made a template for

The next general election produces 1,835 different polling-station result
forms: one Form 34A (President), 290 Forms 35A (one per constituency), 1,450
Forms 36A (one per ward), 47 Forms 37A and 47 Forms 38A (one per county).
Each lists its own candidates, so each is printed differently, and none is
seen before election night. Every one of them also has B-series (tally)
and C-series (final tally) forms. A template per printing, drawn by hand
before the election, is impossible.

So the app doesn't depend on templates. It knows what every A-, B- and
C-series form has in common, and learns each printing's exact layout from
the forms themselves as they arrive.

## 1. What every results form has in common

The forms are laid down by regulation. Whatever the election or the area,
a form is the same sequence of parts. Only the size of the results part
changes with the number of candidates (or stations, on a tally form).

**A-series (polling station: 34A to 38A).** One page, sometimes two:

| Part | Printed anchors | Personal details | Action |
|---|---|---|---|
| Header | "FORM 3xA", election title, QR (2022), station name and codes | none | protected |
| Results | candidates' names, "No. of valid votes", totals | none | protected |
| Counts | "Polling Station Counts", "Registered voters", "Rejected", "Disputed" | none | protected |
| Declaration | "Declaration", "We, the undersigned…", one line per officer: "Presiding Officer", "Deputy Presiding Officer", each with "Signature" and "Date" | officers' signatures | band through each signature |
| Agents | "Agents or Candidates (if present)"; a ruled table headed "Name", "ID/Passport No.", "Party", "Tel. Contact", "Signature", "Date" | agents' IDs, phones, signatures | 3 ID characters, 3 of the last 6 phone digits, band through each signature |
| Free text | "Reasons for Refusal to Sign", "Presiding Officer's Comments" | signatures, sometimes an ID or phone | the same, comments stay readable |
| Stamps | the round IEBC stamp, anywhere | — | never masked |

The 2013 Form 34 (President, polling station) is the same parts over two
pages, laid out differently:
- **Front:** the results, then the declaration, the presiding officer's
  name, ID or passport number, signature and date one under another.
- **Back:** the deputy's lines (the same four), then a table of the
  candidates' agents with only "Name", "Signature" and "Reasons for refusal
  to sign", then the presiding officer's comments. No results at all.

**B-series (tally: 34B to 38B).** Several pages:
- **Results pages:** a table of polling stations (rows) by candidates (columns). They carry no one's details.
- **Declaration page:** the totals (protected), then the same declaration, officer lines and agents table as an A form. On this page the officers are the Returning Officer and Deputy, and the agents are the candidates or their agents.

**C-series (final tally: 34C, 37C, 38C …).** The same shape as a B form, one
level up: results pages (constituencies or counties by candidates), then a
declaration page signed by the county or national returning officer, with
the candidates' agents.

The personal details are always the same three things: signatures, ID or
passport numbers and phone numbers. They are always in the same three kinds
of place:
- an officer's line, found by a role label and "Signature" or "ID";
- a signatories table, found by its column headings;
- free text, under "Comments" or "Reasons".

Names, parties, dates, station names and every number in the results stay
readable.

## 2. Three layers

### 2.1 The structural reader (no prior knowledge)

`generic.rs` reads any page from the parts above:
1. It turns the page upright, deskews it and scales it.
2. OCR reads the printed labels.
3. Printed rules give the tables.
4. Each label's meaning gives the role of the field next to it or below it.
5. Everything above the declaration is protected. A page that opens with
   an officer's lines and prints no counts goes on with the declaration of
   the page before (the 2013 back page): it has no results to protect.
6. A signatories table missing the ID or phone column is taken as it is
   only when every other column's heading is read, and none of them is
   for an ID or phone number: otherwise the missing column may simply not
   have been read. An agents' section with no table read under it sends
   the page to review.

This works on any form. But on one photo by itself, a faint rule, a curled
page or a misread label is enough to lose a column.

Measured without templates on the 2022 Forms 34A, which do have a template
to compare with:
- It publishes 54 of 215, against 204 with the template.
- On the forms both publish, it covers 76% of the masks the template makes.
- 35 of those 54 forms miss at least one mask.
- On the 42 Forms 34B it publishes none: the results pages have no
  declaration, so the reader can't tell where their results end.

One photo is not enough evidence.

### 2.2 Learned layouts (the app writes its own templates)

Thousands of photos of one printing arrive together: every polling station
in a ward sends the same Form 36A. Together they show what no single photo
does.

1. **Group by printing.** KIEMS file names give the form and the station
   code, and so the printing:
   - 34A: national, one printing;
   - 35A: per constituency;
   - 36A: per ward;
   - 37A and 38A: per county.

   Older files are grouped by their codes, or by alignment alone (2.4).
2. **Pick a seed** among the group's pages: sharp, flat, upright, and read
   well by OCR.
3. **Align** every page to the seed: a homography from their printed
   features (ORB, taken tile by tile: a photo's strongest corners are often
   in the table or floor around the form), then dense optical flow (DIS),
   smoothed. Photos of a curled or folded page are off any one homography
   by up to 60 px in places, and a table's repeating rows give few
   distinct features to correct that locally. Paper bends smoothly, so a
   flow field smoothed over 25 px follows the bend while one page's
   handwriting averages out.
4. **The median is the blank form.** Printing is identical on every page.
   Handwriting differs, so a pixel-wise median of the aligned pages keeps
   the print and drops the writing. The frame is cut to the form (what most
   pages print, and a text line around it). The median, sharpened, is the
   template's reference image.
5. **Where people write.** A page has writing where it is much darker than
   85% of the pages are there: print is as dark on nearly every page, and
   even where most people write (the first agents' rows) their strokes
   differ from page to page. Accumulated over the group, it marks exactly
   where people write:
   - each officer's line;
   - every agent row;
   - the comments.
6. **What each place is.** The structural reader reads the median (in a
   few versions) for its labels: the officers' lines, the comments. The
   agents' table is read from its rules instead: on the 2022 Form 34A they
   are printed light grey, and where most people write the ghosts of their
   writing are as dark, so a reader's thresholds lose them. Averaged along
   its length, a rule is still a clear ridge: down the table between each
   two headings for the columns, across it for the rows; the table ends
   where its outer rules do. Those rules are then drawn into the reference,
   where they are printed, for pages to be snapped to. Each field gets its
   role from its label, under the A/B/C rules above.
7. **Write the template.** The result is a template in the same format as a
   hand-made one:
   - protected results;
   - officers' signature lines;
   - agents' columns with their rows;
   - the labels that bound them;
   - the comments area.

   Every published form names it in its manifest, so `verify --rerun`
   repeats the redaction exactly.
8. **Redact the group** with the learned template, through the same checks
   as any template: alignment, column edges, headers, leak check and
   protected regions.

A learned layout is checked before it is used:
- It must publish most of the pages it was learned from.
- It must agree with the structural reader's own reading of those pages.

Pages that don't align to it (another printing in the same group) are
learned from separately.

### 2.3 Checks

These are unchanged and apply to everything:
- per-page geometric and leak checks;
- no mask ever touching a protected region or a stamp;
- the vision review of the *redacted* page;
- review for anything unsure.

A cloud model never sees an unredacted form.

### 2.4 B and C documents

A tally document is read page by page:
- **Results pages:** a page with no declaration or signatories vocabulary, and no
  writing outside its numeric cells, is published unchanged. It is checked,
  not assumed.
- **Declaration page:** read as an A form's lower part. Its layout is
  learned across all the documents of the same form type, for example the
  290 Forms 35B, aligned on the declaration page alone.

## 3. Election night

`rfpii redact --learn learned/ …` learns every printing in the batch that
has no template in `learned/` yet, writes it there, and redacts the batch
with it. Later batches reuse what is already learned. `rfpii learn` does
the learning alone.

- **Learning a printing:** the first forms of a printing are held until its
  layout is learned, then everything is published. Learning needs a handful
  of pages (5 to 20). The smallest printings, a ward's Form 36A, have 20 to
  60 polling stations.
- **Small printings:** a printing too small to learn from is read by the
  structural reader under stricter checks, or goes to review.
- **Lower blocks shared across printings:** the declaration-and-agents block
  is printed the same on every Form 36A. A small ward can therefore borrow
  that block from the other wards: align on the block alone.
- **Speed:** learning costs little. Pages then run at template speed, not at
  OCR speed.

## 4. Where it stands (milestone 1)

Measured without any hand-made template (`rfpii learn`, then `redact`):

| Printing | Pages | Learned from | Published | Hand template |
|---|---|---|---|---|
| 2022 Form 34A (national) | 215 | 40, in about 2 minutes on 4 cores | 194 | 204 |
| 2017 Form 34A, first round (sideways photos on tables) | 46 | 32 | 35 | none |
| 2017 Form 34A, repeat election | 25 | 19 | 21 | none |

On the 2022 forms the learned layout masks what the hand template masks,
but by design (partial digits, signature bands, stamps never masked) and
where the hand template itself is off: it places the agents' table a row
or two out on some pages, masking headers and leaving signatures. Most
forms it doesn't publish fail the alignment checks, and go to review.

The weak point is the officer's sign-off under the comments (name,
signature, date, written on a comment line). It is found by the free-text
reader, which flips with small changes of scale and position: on 4 of the
2022 forms the learned layout leaves a sign-off signature that the hand
template, tuned on these forms, masks. Making
that reader robust belongs to milestone 3; until then the vision review
(`--vlm-audit`) is what catches it.

The 2017 photos show two more weak points, which the leak check misses
because it reads a column where the column was placed. A column moved
across by a false match of its heading ("Tel. Contact" matched the
"Candidate" of the heading beside it) is masked in the wrong column. And
an ID written above its row's line falls outside the first row. One of
each turned up among the 56 forms published from the two 2017 batches,
checking by eye those whose masks changed with the layouts. Until the
structural reader checks a moved column against the words of its
heading, the vision review is the backstop here too.

The 2013 Forms 34 are flatbed scans, clean and upright, read by the
structural reader alone (no template, nothing learned):

| Scans | Pages | Published | Sent to review |
|---|---|---|---|
| Diaspora (5 forms, front and back) | 10 | 10 | 0 |
| Baringo North, first file, page by page | 153 | 146 | 7 |

Of the seven, four have a stamp near the deputy's line that OCR read as a
second officer; of the others, a back page wasn't recognised as one, an
agents' heading row wasn't read whole, and an ID label was misread. On the pages checked by eye the masks are
right; an officer's ID or signature under the stamp stays visible, as
everywhere. The scans come a constituency to a PDF (up to 160 pages of
many forms), and a PDF is redacted as one document: one page that fails
holds back the whole file. They were measured page by page.

The by-election samples have one page per printing: too few to learn
from. They are the structural reader's (milestone 3).

## 5. Hand-made templates

These become test fixtures: the 2022 Form 34A and 34B templates are the
reference that learned layouts are measured against. The app no longer
needs them.

## 6. Milestones

| | Goal | Measured on |
|---|---|---|
| 1 | Learn A-series layouts from the batch: `rfpii learn`, `redact --learn` | 2022 34A (learned vs hand template: same masks, same publish rate); 2017 34A; by-election 35A/36A/38A printings |
| 2 | B- and C-series documents without templates | 2022 34B (42 documents, 3 printings), 34C pages |
| 3 | Structural reader for printings too small to learn; templates optional | by-election forms one per printing; 2017 forms |
