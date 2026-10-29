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
5. Everything above the declaration is protected.

This works on any form. But on one photo by itself, a faint rule, a curled
page or a misread label is enough to lose a column.

Measured without templates on the 2022 Forms 34A, which do have a template
to compare with:
- It publishes 72 of 215, against 204 with the template.
- On the forms both publish, it covers 72% of the masks the template makes.
- 49 of those 72 forms miss at least one mask.
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
3. **Align** every page to the seed by its printed features (ORB and a
   homography, as for templates).
4. **The median is the blank form.** Printing is identical on every page.
   Handwriting differs, so a pixel-wise median of the aligned pages keeps
   the print and drops the writing. On 15 photos of Forms 34A the median is
   already a legible blank form.
5. **Where people write.** Ink on a page that the median doesn't have is
   writing. Accumulated over the group, it marks exactly where people write:
   - each officer's line;
   - every agent row;
   - the comments.
6. **What each place is.** The structural reader reads the clean median:
   labels, rules, rows and columns. On a blank, flat, sharp form it doesn't
   misread. Each writing area gets its role from its label, under the A/B/C
   rules above. A writing area no rule explains stops the learning: the
   printing goes to review.
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

## 4. Hand-made templates

These become test fixtures: the 2022 Form 34A and 34B templates are the
reference that learned layouts are measured against. The app no longer
needs them.

## 5. Milestones

| | Goal | Measured on |
|---|---|---|
| 1 | Learn A-series layouts from the batch: `rfpii learn`, `redact --learn` | 2022 34A (learned vs hand template: same masks, same publish rate); 2017 34A; by-election 35A/36A/38A printings |
| 2 | B- and C-series documents without templates | 2022 34B (42 documents, 3 printings), 34C pages |
| 3 | Structural reader for printings too small to learn; templates optional | by-election forms one per printing; 2017 forms |
