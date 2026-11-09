# A local neural network, trained to the same results

How rfpii comes to run on a neural network of its own, on the machine that
does the redaction, and how that network is trained round by round until
it gives the same results the current pipeline gives, then better ones.
This is a design: nothing in it is built yet.

## 0. Why, and what "the same results" means

Today a page goes through hand-made templates, alignment, table snapping
and rules (DESIGN.md); pages without a template go through OCR and the
generic reader (GENERALIZING.md); and the cloud vision models review crops
and locate fields on held pages (HARNESS.md). Three things push towards a
network of our own:

- **Nothing leaves the machine.** The cloud review sends crops, and held
  pages, to a host abroad: transfers that need recording (reg. 41(2)) and
  that regulation 26 bears on for election data (`legal/`). A local model
  makes both questions go away.
- **The known limits are perception problems** (DESIGN.md, "Known limits"):
  pen against stamp ink, faint strokes, rules that look like strokes,
  numbers that can't be split into characters, writing outside template
  zones, pages that can't be aligned, new printings that need a template.
  These are what convolutional networks are good at.
- **Speed and cost.** OCR takes 8–15 s a page on one core; small networks
  take well under a second. Model calls are about two thirds of the
  measured cost of a form (PRICING.md).

**The same results** is defined as a test, not a feeling. On a page the
current pipeline publishes, the network-driven pipeline must:

1. hide every character and signature the current output hides (after both
   are rounded to 8×8 blocks: recall 1.0 on the teacher's masks);
2. hide no more than the teacher plus a set margin (5% more blacked blocks
   on the page; every extra block must be on a field the rules hide);
3. touch no protected region, keep every stamp's ring as now, and reach the
   same release decision (published, or held for the same reason class).

Better results, beyond that: fewer pages held, and the misses the cloud
review catches today caught locally (§6, stage C).

## 1. What does not change

The network only **proposes**: where the fields, the stamps and the ink
are. Everything that makes a published form trustworthy stays as it is:

- the rules (3 characters of an ID, 3 of a phone number's last 6, a band
  through each signature, stamp bands, `--full-ids`, `--full-phones`);
- the masking itself (deterministic code, whole 8×8 blocks, coefficient
  edits of the JPEG), the pre-publish self-check, the leak check and the
  gates, which fail closed;
- the manifest, the ledger, the signatures and `rfpii verify`;
- the release gate and the 0.5% leak-rate bound of a job (HARNESS.md §8–9).

A network's answer is recorded in the manifest, the way the vision
review's finds are (`vlm_found`), and `verify --rerun` replays it (§4).

## 2. The four models

Small convolutional networks, run on the CPU with ONNX Runtime, which
rfpii already loads for OCR (`ocr.rs`, the `ort` crate). Trained in
PyTorch, exported to ONNX.

| | Model | Input | Output | Replaces or helps |
|---|---|---|---|---|
| M1 | **Page classifier**: a MobileNetV3-Small or EfficientNet-Lite0 backbone with three heads | the page at 384 px | form type and printing (34A 2022, 34B layout B, 35A 2013, …), turn (0/90/180/270°), kind (form, tally sheet, blank, not a form, too faint) | the template choice, Clef's orientation check, held-page triage |
| M2 | **Field detector**: a single-stage, anchor-free detector (FCOS or CenterNet style) on a small CNN backbone with a feature pyramid, 3–8 M parameters | the page at 1,280 px long side | boxes with a class and a score: signature, ID number, phone number, name, stamp, results region, agents table, declaration, comments | finding fields outside template zones, locating fields on held pages, the cloud review's *find* step; placing tables on pages the alignment can't match |
| M3 | **Ink segmenter**: a U-Net with a light encoder | 512×512 tiles of the page at full resolution, overlapping | per pixel: paper, print and rules, pen, stamp ink | the ink extraction under the masking: rules against strokes, faint strokes, embossed backgrounds, and (to be measured) pen under a stamp |
| M4 | **Character slots**: a 1-D CNN over a field's crop, read left to right, with a boundary head | the crop of one ID or phone number, 48 px high | how many characters, and where each starts and ends | splitting a number into characters (no more numbers hidden whole because they couldn't be split) |

M4 finds **where** the characters are, never **what** they are. It is
trained on slot boundaries, not transcriptions, so the system never reads
an ID or phone number: the masking needs the positions only, and data
minimisation (s.25(d)) asks for no more.

## 3. Where the labels come from

| Source | What it gives | Quality |
|---|---|---|
| **The teacher**: the current pipeline's published pages | from each manifest: the masks (with zone, mode and role), the stamps (circle and polygon), the protected results regions, the vision review's finds and spot masks, the template, the turn | boxes good; ink masks only approximate (the pipeline's own ink extraction) |
| **Synthetic forms** | `synthetic.rs` extended: handwriting-like strokes, digits and signatures drawn on blank forms of each printing, stamps in several inks, folds, photos at an angle, blur, JPEG; every pixel's class known exactly | exact, but not real |
| **Adjudicated pages** | the disagreements of each round, settled (§5) | the best; few |
| **Gold set** | about 300 pages, labelled by people, never trained on | the measure (§5.1) |

The corpus to start from: the 215 Forms 34A of 2022, the 6,608 published
pages of the 2013 Forms 34, the 2017 forms, and the pages each of them
held. The IEBC portal serves tens of thousands more forms per election;
downloading them for training is a new processing activity the DPIA must
cover (§9).

Splits are by constituency, so no polling station or officer is in both
training and test. A page's labels carry the version of the teacher that
made them, so a fixed pipeline bug can relabel exactly what it touched.

## 4. Running the network inside rfpii

- **`rust/src/nn.rs`** loads the four ONNX files with ONNX Runtime, one
  session each, on the CPU, single-threaded per page (pages run in
  parallel), with graph optimisations fixed, so a run is repeatable on the
  same hardware.
- **Pre-processing is defined once and tested twice.** Python and Rust
  resize, pad and normalise the same way; a test feeds both a fixed image
  and compares the input tensors byte for byte (golden vectors). This is
  where ported models usually go wrong.
- **Each manifest records** the models (name, version, SHA-256 of the ONNX
  file, runtime version) and their outputs per page: M1's classes, M2's
  boxes, M4's slots, and a hash of M3's mask per tile (with the masks
  themselves stored beside the manifest, as the review keeps its crops).
- **`verify --rerun` replays the recorded outputs.** ONNX Runtime can round
  differently on another CPU (DESIGN.md already notes this for OCR), so the
  proof of a published form never depends on re-running a network. A new
  **`verify --reinfer`** runs the recorded models again and reports how
  close they come (box IoU, mask agreement): a check on the model, not on
  the form.
- **Which stage is on** (§6) is a setting, recorded in the manifest's
  `settings`, so a form says how it was read.

## 5. The iterative process

```
round 0   freeze the teacher T0 (this pipeline) and score it on the gold set
round k   1. harvest   labels: the teacher's outputs + adjudicated pages + synthetic
          2. train     student S_k (recipe, seed and data recorded)
          3. score     S_k and T on the gold set
          4. shadow    run S_k beside the pipeline on the whole corpus
          5. compare   list every disagreement, ranked by risk
          6. settle    rules first, then people, for the top of the list
          7. fix       relabel; fix the teacher where it was wrong
          8. gate      promote S_k one stage if its scores pass (§6)
          until S_k matches T on the gold set two rounds running and
          disagreements fall below 0.5% of fields
```

Both sides get better: a disagreement the teacher loses is a pipeline bug
to fix, as every review round has been so far.

### 5.1 The gold set

About 300 pages, drawn across years, forms, printings, scans and photos,
published and held. Each is labelled by a person, with a second person
checking a quarter of them: every signature, ID, phone number, name,
stamp and results region, as boxes, and pixel masks for the pen on 60 of
them. It is never used for training, and it changes only by adding pages,
each change versioned, so scores stay comparable across rounds.

### 5.2 What a round measures

| Measure | On | Target to pass a stage |
|---|---|---|
| Field recall, by class (signature, ID, phone) | gold set | ≥ the teacher's, and ≥ 0.99 for IDs and phones |
| Teacher-mask recall (the same-results test, §0) | the teacher's published pages | 1.0 after block rounding, on every page |
| Extra blacked area | the same | ≤ 5% more blocks; none on a protected region |
| Collateral | gold set | masks on names, parties, results: no more than the teacher |
| Character slots | gold IDs and phones | exact count on ≥ 98%; each boundary within 4 px |
| Page class and turn | gold set | ≥ 99.5% |
| Pages held | corpus | fewer than the teacher, with every page it publishes passing all the gates |
| Leak rate | a job's random sample (HARNESS.md §8) | bounded below 0.5%, as now |
| Time per page | 4-core CPU | under 2 s for all four models |
| Repeatability | `verify --reinfer` on the same machine | identical outputs |

### 5.3 Settling a disagreement

Kinds, in order of risk:

1. **The student finds PII the teacher left visible** (a box with nothing
   masked under it): a possible leak in today's output. First.
2. **The teacher masks what the student doesn't see**: a student miss.
3. **They differ on a protected region, a stamp or a page's class or
   turn.**
4. **Masks that overlap but differ in extent.**

Rules settle what they can: whether a box overlaps a results region, what
the leak check says, whether a stamp's ring is intact. People settle the
rest in a local review page (a browser on the same machine): the page, both
proposals, three buttons (student, teacher, neither) and a pen to draw the
box. Nothing goes to a cloud model; the vision review is not a judge here,
since the network is meant to replace it.

What it costs per round: the first rounds will raise thousands of
disagreements, mostly kind 4; only kinds 1 to 3 and a random tenth of
kind 4 go to people. Expect 2–4 hours of a reviewer per round at first,
falling as the student converges.

### 5.4 Training, locally

- A `nn/` folder in the repository: the dataset builder (reads manifests,
  sources and synthetic forms), the four training recipes, ONNX export,
  the gold-set scorer and the review page.
- A recipe is code plus a config with its seed. A round records the data
  list (file SHA-256s), the recipe's commit and the library versions; the
  same inputs give the same weights where the libraries allow.
- **Hardware.** One local GPU with 16–24 GB trains each model in hours;
  M1 and M4 train on a CPU in a day, M2 and M3 are impractical without a
  GPU. Inference needs only the CPU.
- Each model gets a **model card**: data, scores on the gold set, known
  failures, the round it came from. A registry (`nn/registry.json`) lists
  every version and its stage.

## 6. Promotion: what the network is trusted to do, stage by stage

Each stage is a setting, can be turned off at once, and needs the
targets of §5.2 met on the gold set and a sample of live output checked by
a person.

| Stage | The network… | Can it hide less than today? |
|---|---|---|
| **A. Shadow** | runs beside the pipeline; its answers are recorded, compared, and used for nothing | no |
| **B. Second opinion** | adds masks where it finds a field the pipeline left visible: the union of both, never less | no: it can only add |
| **C. Local reviewer** | replaces the cloud review in `--vlm-audit` and the job's QA (find, then check, on crops): no model calls, no transfers abroad | no: the review only adds masks; the job's leak-rate sample stays |
| **D. Held pages** | classifies held pages (M1) and locates their fields (M2), locally; such pages publish only through every gate and are always reviewed | no: they were held |
| **E. Primary reader** | M1 picks the form, M2 places the fields, M3 gives the ink, M4 the characters; templates and OCR become checks against it | yes: from here a student miss is a leak, so this stage needs the strictest gate (teacher-mask recall 1.0 over at least 5,000 pages, and a full round of live sampling) |

Stages B to D are safe by construction: they can only hide more, or touch
pages that were held anyway. Stage C alone removes the transfers abroad
from ordinary jobs, and D removes them from held pages. Stage E is where
templates stop being needed for new printings.

## 7. What it would change in the product

- **Jobs** (HARNESS.md): a job with stage C needs no model host; model
  calls and their transfers disappear from the journal and the report.
- **The API** (API.md): the same endpoints; the price is set by the owner
  and doesn't change. The operator's cost per form falls (no tokens; about
  1–2 s more CPU per page, against 8–15 s less where OCR is no longer run).
- **verify**: unchanged for published forms (recorded outputs);
  `--reinfer` added.

## 8. Risks

- **Too few real labels of the hard cases.** Pen under a stamp and new
  printings are rare in 215 forms. Synthetic data, the 2013 and 2017
  corpora and portal downloads (once the DPIA covers them) are the answer;
  the gold set says whether it worked.
- **The student learns the teacher's mistakes.** That is why the gold set
  is labelled by people, and why kind-1 disagreements go first.
- **Shift between years, counties and devices.** Splits by constituency,
  a gold set across years, and a pilot (HARNESS.md) before each new set.
- **A confident miss.** Networks fail without warning. Until stage E, a
  miss can't hide less than today; at stage E, the gates, the leak check
  and the job's random sample stay in force.
- **Floating point across machines.** Handled by recording outputs (§4).

## 9. Personal data in training

- Training data is personal data (names, IDs, phones, signatures of
  officers and agents), processed for a new purpose: training. The DPIA
  annex (`rfpii dpia-annex`) gains a section for it; the controller record
  names the purpose.
- It stays on the local machine: an encrypted disk, access for named
  people, and a retention schedule (`rfpii retain`) for the label store.
- **Weights trained on real forms stay internal.** Detectors and
  segmenters don't reproduce their inputs the way generative models can,
  but this is not proven for every input, so such weights are not
  published or shared without an assessment. Weights trained on synthetic
  forms only can be published.
- M4 never reads characters (§2), and no model outputs text.
- An erasure request reaches the training set too: the form's labels are
  removed from the label store, and the next round trains without them.

## 10. Milestones

1. **Data:** the dataset builder from manifests, the synthetic generator
   extended, the gold set labelled (the review page doubles as the
   labelling tool).
2. **M1 and M2, stage A:** shadow runs on the corpus, the first
   disagreement report.
3. **Rounds to stage B and C:** the local reviewer replaces the cloud
   review.
4. **M3 and M4, stage D:** held pages read locally.
5. **Stage E:** primary reader, templates as checks.

Each milestone ends with its round's scores on the gold set, written up
in this document as DESIGN.md does for the pipeline.

## 11. Open decisions

- **Hardware for training:** a local GPU (16–24 GB), or CPU only (M1 and
  M4 only, at first)?
- **More data:** may forms be downloaded from the IEBC portal for training,
  under an updated DPIA?
- **Reviewers:** who settles disagreements and labels the gold set, and
  how many hours a round can they give?
- **The target stage:** C (no cloud models in ordinary jobs), D (none at
  all) or E (no templates)?
- **Weights:** may weights trained on real forms ever leave the machine?
