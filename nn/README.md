# nn/: data for rfpii's own networks

The data side of NEURAL.md, milestone 1: datasets of labelled pages for
training and measuring the local networks, a gold set they are measured
on and never trained on, and a read-only viewer to look at both. And
milestone 2: M3's rule stem, the weight-space programming tools (printer,
probe, try, diff, a sandbox and a session driver for a local language
model), and the stem trained by gradient descent as a baseline
([below](#milestone-2-the-rule-stem-and-weight-space-programming)).

What is here:

| | |
|---|---|
| `rfpii dataset synth` (Rust, `rust/src/dataset/`) | synthetic forms with exact labels and masks; with `--redacted`, the same pages redacted by the pipeline, labelled with what still shows |
| `rfpii dataset build` | the teacher's labels (the current pipeline's) from published manifests and their originals, and the pages it held |
| `rfpii dataset check`, `stats`, `compare` | verify a dataset; count it; score one dataset's boxes against another's on the same pages |
| `rfpii_nn/dataset.py` | read a dataset: index, labels, images, masks, splits |
| `rfpii_nn/gold.py` | plan the gold set; the exclusion every training reader must use |
| `rfpii_nn/view.py` | the local, read-only viewer |
| `rfpii dataset ink` | the pipeline's own ink masks (dark ink, rules, stamp ink) of any pages: an ink set |
| `rfpii_nn/stem.py` | M3's rule stem: the architecture, the numpy reference, PyTorch, ONNX |
| `rfpii_nn/inkset.py` | read an ink set; score a stem against it (agreement and IoU) |
| `rfpii_nn/wsp.py` | the weight-space tools: print, probe, try (with its budget), diff; `score`, `export` |
| `rfpii_nn/sandbox.py`, `sandbox_kernel.py` | the isolated Python kernel the language model's code runs in |
| `rfpii_nn/session.py` | a weight-space session with the local language model (or a stub) |
| `rfpii_nn/stem_hand.py`, `stem_train.py` | the stem written by hand from the rules; trained by gradient descent |
| `stems/` | the stems' weights, under 10 kB each: `hand.npz` (and, once milestone 2's runs are done, `trained.npz`, `written.npz` and their scores) |
| `tests/` | `python3 -m pytest -q` (in `nn/`); or `python3 -m unittest discover -s nn/tests -t nn` from the root |

Python, milestone 1: the standard library, numpy and Pillow; nothing else,
and nothing fetched from the network. Milestone 2 adds PyTorch (CPU),
`onnx` and `onnxruntime` (for the stem; the tests that need them skip
without them) and `llama-cpp-python` (for a session with the local model;
the tests use a stub).

## Quick start

```sh
cd rust && cargo build --release && cd ..
B=rust/target/release/rfpii

# 2,000 synthetic pages, 1,280 px on the long side, for training
$B dataset synth --templates templates --out nn/data/synth --count 2000 --max-side 1280
# redacted ones, for the local reviewer
$B dataset synth --templates templates --out nn/data/redacted --count 500 --seed 100000 --redacted --max-side 1280
$B dataset check nn/data/synth
$B dataset stats nn/data/synth

# the teacher's labels from a run (personal data: see below)
$B dataset build public/ --originals Forms/ --templates templates --review review/ --out nn/data/teacher

# the gold set: its synthetic part, and a reference sample of the teacher's pages
cd nn
python3 -m rfpii_nn.gold plan --gold data/gold --reference data/teacher --templates ../templates
python3 -m rfpii_nn.gold make-synthetic --gold data/gold --rfpii ../rust/target/release/rfpii --templates ../templates
python3 -m rfpii_nn.view --gold data/gold        # http://127.0.0.1:8765/
python3 -m rfpii_nn.view --dataset data/synth
```

`nn/data/` is in `.gitignore`: datasets are never committed.

## The dataset format

A dataset is a folder. Schema names carry a version; a reader refuses a
schema it doesn't know.

```text
dataset.json        rfpii/nn-dataset/1: what it is and how it was made
pages/<id>.<ext>    each page image
labels/<id>.json    its labels, rfpii/nn-labels/1
masks/<id>.png      its ink mask (optional)
index.jsonl         one line per page
splits.json         rfpii/nn-splits/1: the pages of train, val and test
SHA256SUMS          the SHA-256 of every other file
```

**`dataset.json`**: `schema`, `kind` (`teacher` or `synthetic`), `created`
and `updated` (UTC; `SOURCE_DATE_EPOCH` fixes them, so a dataset made twice
from one seed is the same file for file), `rfpii` (version and commit),
`parameters` (the last run's: seeds, count, templates, profile, sizes;
for a teacher dataset the manifests, originals and review folder),
`personal_data` (true for teacher datasets, false for synthetic ones),
`ink` (`exact`, `rules` or `none`), `mask_bits`, `classes`, `pages`,
`splits` (counts), `runs` (every run that wrote to it), `notes`.

**`pages/<id>.<ext>`**: the page as the pipeline sees it. A teacher page
from a JPEG (a KIEMS PDF's photo, a phone photo) is the original JPEG's
bytes, not recompressed; any other page is a PNG of its pixels as stored
(no EXIF turn applied). Synthetic pages are JPEGs.

**`labels/<id>.json`** (`rfpii/nn-labels/1`):

- `page`: `width`, `height`, `kind` (`form`, `blank`, `not_form`, or
  `unknown` for a held page), `template`, `form`, `printing` (the
  template), `year`, `capture` (`kiems`, `scan`, `photo`;
  `synthetic_scan`, `synthetic_photo`), `turn` (degrees clockwise the
  content stands turned: turn the image that much counter-clockwise to
  stand it up), `source` (teacher: the original's SHA-256, page number,
  path, the stored image's SHA-256), `kiems` (county, constituency,
  station, when known), `held` and `reason`.
- `split_key`, and `teacher` (the manifest's SHA-256 and tool version,
  the settings, seed, template fingerprint) or `synthetic` (seed,
  generator version, how the page was photographed, the homography from
  the drawing canvas to the image).
- `exhaustive`: the classes labelled completely on this page. Synthetic
  pages: all. Teacher pages read with a template: all but `name` (the
  pipeline never looks for names). Held pages and pages read without a
  template: none. Training must not count a missing box of a class that
  isn't listed as a miss.
- `boxes`: each has `class`, `box` [x0, y0, x1, y1] (image px, inside the
  image), `source` (`template_zone`, `blocks`, `vlm_found`, `spot_mask`,
  `stamp`, `protect`, `synthetic`), and where known `zone` (the template
  zone, rows as `agent_id[3]`), `role` (`officer`, `agent`, `unknown`),
  `polygon`, `cell` (the field's cell), `empty: true` (a field cell with
  nothing written in it; its box is the cell), `struck: true` (an empty
  cell crossed by a line drawn through unused rows), `slots` (each
  character's [x0, x1] for ID and phone numbers, on pages that are not
  turned), `char_boxes` (synthetic: each character's box), `rule` (`id`
  or `phone`), `circle` [cx, cy, r] (stamps), `under_stamp: true` (a field
  a stamp lies over), `approximate: true` (teacher: from blacked areas
  outside template zones), `redaction` (redacted pages, below).
- Classes: `signature`, `id_number`, `phone_number`, `name`, `stamp`,
  `results_region`, `agents_table`, `declaration`, `comments`.

**`masks/<id>.png`**: 8-bit, one value per pixel, bits: 1 print and rules,
2 pen, 4 stamp ink, 8 blacked out by redaction. Pen under a stamp sets 2
and 4. Synthetic masks are exact (`ink: "exact"`): pen is every pixel
drawn with the pen, stamp ink every pixel of a stamp at least a quarter
opaque, print what the pipeline's dark-ink rule finds on the template's
reference (so the reference's photographed shadows are not print). On a
smaller page a bit is set where it covers at least a quarter of the pixel.
Teacher masks are the pipeline's own ink rules (`ink: "rules"`): rules
can't tell pen from print, so all dark ink is bit 1 and pen (bit 2) is
never set; coloured ink is bit 4.

**`index.jsonl`**: per page `id`, `image`, `image_sha256`, `labels`,
`labels_sha256`, `mask`, `mask_sha256`, `split_key`, `split`, `kind`,
`held`, `template`, `form`, `year`, `capture`, `classes` (boxes per class,
empty cells left out), `empty` (empty cells), and `leaks` on redacted
pages.

**Splits** are fixed by each page's split key alone: the first 8 bytes of
SHA-256(split key), big-endian, as v: train if v·10 < 8·2⁶⁴, val if
v·10 < 9·2⁶⁴, else test (80/10/10). Adding pages never moves a page. A
teacher page's key is its county and constituency (KIEMS file name or
QR code), so no polling station or officer is in two splits; without
them, the folder the original is in. A synthetic page's key is its seed.

**`SHA256SUMS`**: every file but itself, `sha256  path`, sorted.
`rfpii dataset check` verifies it, the schemas, that every page's
labels and mask match its image's size, every box lies inside its image,
every slot inside its box, and that each page's split is its key's.

## Synthetic datasets

`rfpii dataset synth --templates DIR --out DATASET --count N [--seed S]`.
Sample i has seed S + i and depends on nothing else: the same seed gives
the same files, byte for byte. Each sample picks a template with fields
(Form 34A, the Form 34B blocks of layouts A, B and C) and starts from its
reference image, a real blank printing; a block is put on a whole page,
under a typed results table. On it, in template coordinates:

- ID numbers (7 or 8 digits) and phone numbers (10 digits, 07… or 01…,
  sometimes spaced) in the zones' cells, written character by character in
  a stroke font with its own height, width, slant, spacing, jitter and pen
  width, in blue or black; each character's span is recorded;
- signatures: splines with loops and peaks, initials with a trailing line,
  or tight scrawls; some with a flourish leaving the cell;
- names in the name columns (capitals, or a cursive script), parties and
  dates (digits that are not IDs), counts in results cells;
- empty rows, sometimes struck through;
- round stamps in violet, blue or red, with ring text and a date, patchy
  and see-through; usually by the officers' lines, a quarter of the time
  over an agent's number (pen under a stamp);
- in the Form 34A comments: words, and sometimes a signature, a number or
  a name.

Where the templates don't say (names, parties, dates, declarations, the
reference's own stamp), `rust/src/dataset/layout.rs` does, read off the
references. On layout C the synthetic forms write phone numbers and
signatures in the printed "Tel. Contact" and "Signature" columns, which
the template's `agent_phone` and `agent_signatures` zones are not on (see
NEURAL.md, "Where it stands").

Then the page is photographed or scanned (`--profile mixed`, the default:
40% photos with perspective and a background; scans otherwise; 12% turned
90, 180 or 270 degrees; folds; light; blur; noise; JPEG quality 60–95;
`clean`: light scans only; `photo`: photos only). `--negatives 0.12` (the
default share) adds results pages, forms nobody filled in, blank pages and
pages that are not forms. `--max-side 1280` makes the pages smaller, with
their labels and masks.

**Redacted pages** (`--redacted`), for the local reviewer (stage C): each
page is redacted by the pipeline as `rfpii redact` would (templates, rules,
leak check; the blocks blacked in the JPEG's coefficients). On 60% of
pages, one to three fields' masks are then undone or cut back on purpose:
a signature left whole, cut to one row of blocks or to its left half; a
number shown whole or with one hidden character too few. Each field with
writing gets `redaction`: `status` (`masked_ok`, `leak_signature`,
`leak_number`), `ink_hidden` (share of its ink under black),
`chars`/`chars_hidden` (numbers) or `middle_hidden` (signatures), and
`perturbed` (how its mask was cut back, or null: the pipeline's own
result). A number is `masked_ok` when at least 3 characters are hidden (a
phone number: 3 of its last 6); a character is hidden when half its ink is
under black. A signature is `masked_ok` when along its body, in 32 px
pieces, the ink between 40% and 60% of its height is 80% under black in
pieces holding 75% of the body's ink, and a quarter of all its ink is under
black (a flourish is not judged: the pipeline's band follows the body).
Pages the pipeline holds are left out (`parameters.held_by_pipeline`).
OCR must be available for Form 34A pages (their comments are read with
it); without it they are held.

**Size and speed** on a 4-core machine (NEURAL.md, "Where it stands"): a
full-size synthetic page takes about 1.1 MB (JPEG 1 MB, mask 0.1 MB,
labels 11 kB); at `--max-side 1280`, 212 kB (JPEG 160 kB, mask 36 kB,
labels 11 kB), so 10,000 pages take about 2.1 GB. `synth` writes 5 to 6
pages a second either way; `--redacted` about 0.3, as it runs the
pipeline on every page.

## Teacher datasets

`rfpii dataset build MANIFESTS... --originals DIR --templates DIR [--review DIR] --out DATASET`.

Manifests are found under the paths given (a run's `--out`); each
original is found by its SHA-256 among `--originals` (files, folders and
zip members), decoded exactly as the pipeline stored it, and checked
against the manifest's `source_image_sha256`. Labels:

| From the manifest | Gives |
|---|---|
| template zones placed by `alignment.H` | each field cell, rows expanded: class from the zone (signature, ID, phone), role from its mask record; `empty: true` where the masking found the cell empty |
| `blocks` (the blacked 8×8 blocks) in a cell's search area | the field's box: a number hidden whole is its blocks' box; a number with some characters hidden is fitted from where those characters are (`lines[].hidden`, which of how many), which also gives its slots (`slots_from: "hidden_blocks"`); a signature's band is the middle half of its height, so its box is the band grown to twice its height |
| `protected` | stamps (circle, polygon, how found) and results regions |
| the template and `layout.rs` | agents table, comments, declaration |
| `vlm_found`, `spot_masks` | boxes with that source |
| blacked areas in the comments and on pages read without a template | boxes, `approximate: true`, classed from the mask records or the nearest field label |
| `--review DIR/report.jsonl` | the pages held, with their reason and no field labels: the pool to learn from next |

What can't be recovered: names (never looked for); where a table column
was snapped to its printed rules or a field moved with its printed label
(only the page's alignment is recorded, so a cell can sit a few pixels off
where no blocks correct it); the extent of writing in a cell nothing was
masked in; the exact field boxes of free text and of pages read without a
template (only the blacked areas). The ink mask is the rules' (above).

`rfpii dataset compare TRUTH TEACHER` scores the teacher against synthetic
truth on the same pages; NEURAL.md, "Where it stands", has the numbers.

## The gold set

No person labels pages. The gold set has two parts, both planned in
`gold/plan.json` (`rfpii/nn-gold-plan/1`), both never trained on:

- **the synthetic gold set**: a synthetic dataset from its own seed range
  (by default 1,000,000,000 to 1,000,000,300; train with seeds below it),
  exact labels: what is measured against truth. `make-synthetic` makes it
  with the binary.
- **the reference set**: about 300 real pages sampled from teacher
  datasets, stratified by template (or form) and year × capture (KIEMS
  JPEG, scanned PDF, photo) × published or held; each stratum gets at
  least one page and the rest in proportion, picked by a fixed hash of the
  image. Its labels are the teacher's: it measures agreement with the
  pipeline, not truth.

A plan only grows: running `plan` again with more datasets adds pages and
raises the version (`gold/VERSION`, `history`); no page is taken out and
the synthetic seeds never move.

Every training reader goes through `Dataset.pages(split, exclude=...)` or
`training_pages(dataset, exclusion)`; `exclude` has no default, and
training passes `gold.Exclusion.load(gold_dir)`, which leaves out every
reference page (by image SHA-256, in any dataset) and every synthetic page
in the gold seed range.

## The viewer

`python3 -m rfpii_nn.view --gold DIR` or `--dataset DIR [--port 8765]`:
each page with its boxes (colour by class; leaks thick, in pink; empty
cells dashed; slots under numbers) and its mask, zoom and pan, next and
previous, filters (pages with a class, with a leak, held, with flags) and
the list of boxes. Read-only: nothing is drawn, edited or saved. A
marked place shows milestone 3's disagreement flags (`labels.flags`), read
only.

It listens on 127.0.0.1 only, serves only the pages it was given (asked for
by number, never by path; a page whose files would leave its dataset is
refused), answers only requests addressed to it (Host, and Origin when
present), only GET and HEAD, sends `Cache-Control: no-store` and a
Content Security Policy that allows nothing but its own files, and loads
nothing from anywhere else.

## Milestone 2: the rule stem and weight-space programming

NEURAL.md §5 and §11.2. M3's **rule stem** (`rfpii_nn/stem.py`): 684
weights in 20 small convolutions that approximate, from the colour page,
the pipeline's three ink rules: stamp ink (`stamps::color_ink`), dark ink
(`stamps::dark_ink`) and long thin rules (`masking::rule_lines` over the
dark ink). The architecture is fixed; its weights are written three ways
and scored the same way: by a local language model in a weight-space
session, by hand from the rules, and by gradient descent.

### The pipeline's masks: ink sets

```sh
B=rust/target/release/rfpii
# every page of some sources, or a list of chosen pages (JSON lines: source, page, key, group)
$B dataset ink --out nn/data/ink-synth nn/data/synth/pages --max-side 1280
$B dataset ink --out nn/data/ink-real --list pages.jsonl --max-side 1280 --personal-data
```

Each page is made 1,280 px on its long side (`INTER_AREA`, as the stamp
finder makes its working copy) and stored as a PNG, so a network reads
exactly the pixels the masks come from. `masks/<id>.png` has bit 1 for
dark ink, 2 for rules (an opening of the dark ink with a line a fortieth
of the long side, 32 px), 4 for stamp ink: each the pipeline's own
function. `index.jsonl` records the source, page, split key and split (the
datasets' rule: by constituency for real forms, by seed for synthetic
ones); `train` is for training and test runs, `val` and `test` together
are the held-out pages. Real pages are personal data (§10): keep their
ink sets out of the repository (`nn/data/` or a scratch folder).

### The stem and its scores

```sh
cd nn
python3 -m rfpii_nn.stem_hand --out stems/hand.npz                 # written by hand from the rules
python3 -m rfpii_nn.stem_train --ink A --ink B --out stems/trained.npz --minutes 45   # gradient descent
python3 -m rfpii_nn.wsp score stems/hand.npz --ink A --ink B --json scores.json       # training and held-out
python3 -m rfpii_nn.wsp export stems/hand.npz --out hand.onnx --ink A                # ONNX, checked
python3 -m rfpii_nn.wsp print stems/hand.npz --layer dark_raw                       # the printer
python3 -m rfpii_nn.wsp diff stems/hand.npz SESSION_DIR/written.npz
```

`score` reports, per output and per group of pages, the **agreement**
(the share of pixels where the stem and the pipeline agree; the test is
99% on every output) and the **IoU** of the ink class (pixels both mark /
pixels either marks), with recall and precision. Ink is a small share of a
page (stamp ink about 0.5%), so agreement alone says little: a stem that
marks no stamp ink at all agrees on 99.5% of pixels.

The ONNX file takes `image` (N, 3, H, W) float32, R, G, B 0-255, any size,
and gives `ink` (N, 3, H, W): stamp, dark and rule probabilities (a pixel
is marked above 0.5). Every convolution pads by repeating the edge pixels
(ONNX `Pad`, mode `edge`). The numpy reference (`stem.forward_numpy`) is
the definition; the tests check PyTorch and ONNX Runtime against it.

### The tools

In `rfpii_nn/wsp.py`, over the stem (any small part can be described the
same way):

- **print** (`wsp.text`): each layer's name, kind, kernel, activation and
  shapes, what it computes, the meaning of each input channel, and its
  weights to 4 significant figures (kernels as rows, 1x1 layers as a
  matrix with the bias); a layer of zeros in one line. In the sandbox the
  same weights are numpy arrays (`W`), and `SPEC` the layers as data.
- **probe** (`wsp.probe`): a candidate's activations on chosen training
  pages, as numbers: each channel's quantiles, mean and share above 0.5;
  the outputs' counts against the pipeline (tp, fp, fn, tn); and, with
  `where="fn:stamp"` (or tp, fp, tn, pipe, nopipe, stem, nostem) and
  `samples` (at most 256), the channels' values at that many such pixels,
  picked at random, without their positions. Never an image.
- **try** (`wsp.Budget`): one test run, on a copy of the candidate, scored
  on every training page: agreement, IoU, recall and precision per output,
  IoU by group; 50 a session, then refused.
- **diff** (`wsp.diff`): how many weights two versions differ in, per
  layer, and the largest changes, old and new.

### A session

```sh
python3 -m rfpii_nn.session --ink A --ink B --model /home/user/nn-models/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf \
    --out SESSION_DIR --tries 50 --hours 4.5
python3 -m rfpii_nn.wsp score SESSION_DIR/written.npz --ink A --ink B
```

The model is given the goal, the pipeline's source (the Rust functions,
read from this repository when the session starts, with their thresholds),
OpenCV's documented definitions of the calls they make, the stem as the
printer shows it, and the sandbox's tools. Each reply's last `python`
block runs in the sandbox, and what it printed comes back as the next
message, with the test runs left and the best so far. It ends when the
model calls `submit(W, reasons)` and writes DONE, when the test runs are
used up after a submission, or at the time or turn limit; the answer is the
stem submitted last, or else the best tried.

The model runs here, with llama.cpp (`llama-cpp-python`, built for the
CPU), driven token by token so its context only grows (Qwen3.5 has
recurrent layers, whose state can't be rolled back), with a budget of
thinking tokens per reply (`--think`, then it is told to write its code)
and of answer tokens (`--answer`); when the context nears its end it is
renewed with a summary of the test runs and the current weights. Nothing
is sent anywhere.

**The sandbox** (`sandbox.py`) is a persistent Python kernel in a
separate process: its own new folder; a network namespace with no
interfaces (`unshare -n`); the user `nobody` (`setpriv`), who can't read
the scratch folders the pages are in; a clean environment (no proxy, no
keys); limits on memory (4 GB), file size and open files; a time limit per
cell (the kernel is restarted with the weights last kept); and an audit
hook that refuses sockets, new processes, native libraries and any file
outside its folder and Python's own. It has numpy, `W`, `SPEC` and the
tools; test runs and probes are answered by the parent, which holds the
pages. Where a layer isn't available (not root, no `unshare`), the session
records which ones were used.

**The log** (in SESSION_DIR): `session.jsonl` (every prompt, reply, cell
and its output, tool call, test run and its scores, with token counts,
times and speeds), `transcript.md` (the same, to read), `tries/try_NN.npz`
(each candidate tried), `submitted.npz`, `written.npz` (the answer) and
`summary.json`. Logs hold numbers, never page pixels; their scores are of
pages that may be personal data, so keep them with the ink sets.

`--stub replies.json` drives the same loop with canned replies; the tests
do so (`tests/test_session.py`), so no language model is needed to test.

## Personal data (NEURAL.md §10)

Teacher datasets and the reference set are personal data: names, ID and
phone numbers and signatures of officers and agents, processed for a new
purpose, training, which the DPIA must cover.

- Keep them on this machine's local disk, on an encrypted volume, with
  access for named people only. Never commit them (`nn/data/` is ignored),
  copy them to a shared drive, or send them to a cloud model.
- Put them under the retention schedule as the review folder is: a class
  for each folder in `retention.yaml` (for example
  `{name: teacher datasets, path: nn/data/teacher, keep_days: 365}` and one
  for `nn/data/gold`), then `rfpii retain --policy retention.yaml`
  reports what is due and `--apply` deletes it.
- An erasure request reaches them too: remove the form's pages (by its
  SHA-256) from every teacher dataset and the reference set, and train the
  next round without them.
- Weights trained on them stay internal (§10). Synthetic datasets hold no
  personal data: the names and numbers are random, the stations made up.
- The viewer serves them to this machine only and tells the browser not to
  keep them.
