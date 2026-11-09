# A local neural network, programmed and trained to the same results

How rfpii comes to run on a neural network of its own, on the machine that
does the redaction, and how that network is brought, round by round, to
give the same results the current pipeline gives, then better ones: by
weight-space programming (a local language model reading and writing the
network's weights directly, §5) where it works, and by training where it
doesn't. This is a design; its first milestone, the data, is built
(§12, "Where it stands").

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
review catches today caught locally (§7, stage C).

**How the network gets there.** Each model has a large part learned by
training and small parts kept legible enough to program by hand: by a
language model on the same machine, reading their weights and writing new
ones (weight-space programming, §5). The small parts start as the
pipeline's own rules written into weights, so the first network already
gives the pipeline's results where those rules apply; between training
rounds, faults that sit in one place (a threshold, a silenced input) are
repaired in the weights directly; and before each promotion the language
model reads the small parts and says what they respond to, as a pointer
for tests, not a verdict. Training does
the rest, and every change, programmed or trained, passes the same gate.

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

**Each model in two parts.** A **backbone**, trained by gradient descent,
turns pixels into features: millions of weights, too many to read. On top
of it, a few **programmable parts** of about 20,000 weights or fewer each,
small enough to show a language model whole (§5):

| Model | Programmable parts |
|---|---|
| M1 | the three heads (a linear layer each over a 128-feature bottleneck the backbone ends in: about 2,600 weights for 20 form types) and their thresholds |
| M2 | each class's final 1×1 convolution and score threshold; the box-fusion rule |
| M3 | a **rule stem** before the backbone: 1×1 and small convolutions that approximate, from the colour image, what the pipeline's ink rules compute today (stamp ink as a colour difference from the paper in Lab, dark ink by adaptive threshold, long thin rules by oriented kernels; the median filter and local thresholds are approximated, not reproduced), fed to the U-Net as extra channels; and the per-class output layer |
| M4 | the whole model (a 1-D CNN of about 8,000 weights over a 48 px strip) |

M4 finds **where** the characters are, never **what** they are. It is
trained on slot boundaries, not transcriptions, so the system never reads
an ID or phone number: the masking needs the positions only, and data
minimisation (s.25(d)) asks for no more.

## 3. Where the labels come from

| Source | What it gives | Quality |
|---|---|---|
| **The teacher**: the current pipeline's published pages | from each manifest: the masks (with zone, mode and role), the stamps (circle and polygon), the protected results regions, the vision review's finds and spot masks, the template, the turn | boxes good; ink masks only approximate (the pipeline's own ink extraction) |
| **Synthetic forms** | `synthetic.rs` extended: handwriting-like strokes, digits and signatures drawn on blank forms of each printing, stamps in several inks, folds, photos at an angle, blur, JPEG; every pixel's class known exactly | exact, but not real |
| **Adjudicated pages** | the disagreements of each round, settled (§6) | the best; few |
| **Gold set** | a synthetic gold set from its own seeds, and a reference sample of the teacher's pages; never trained on | the measure: of truth, and of agreement with the pipeline (§6.1) |

The corpus to start from: the 215 Forms 34A of 2022, the 6,608 published
pages of the 2013 Forms 34, the 2017 forms, and the pages each of them
held. The IEBC portal serves tens of thousands more forms per election;
downloading them for training is a new processing activity the DPIA must
cover (§10). Decided (9 November 2026): forms are downloaded for training
from the IEBC portal (forms.iebc.or.ke) and from the owner's Dropbox
archive. They are kept outside the repository, and a provenance file beside
them records each file's source, address, election, county, size, SHA-256
and time of retrieval, so a file can be traced, removed on an erasure
request, or excluded from a round.

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
- **Which stage is on** (§7) is a setting, recorded in the manifest's
  `settings`, so a form says how it was read.

## 5. Weight-space programming

**What it is.** A language model is given a network's raw weights, as
text or in a Python session, with a goal in plain words, and answers with
new weights: it works out what the weights compute and edits them, or
writes a network from a specification, without a training loop. The term
comes from Takehi, Teoh, Zhang and Khattab (MIT CSAIL, 2 November 2026), a
report the authors call a blog post, not peer-reviewed.

**What has been shown**, all on MLPs and transformers (none on a
convolutional network yet):

| | Network | Setting | Result |
|---|---|---|---|
| Read | 4 game policies, MLP, 17,253 weights each | GPT-6 Astra, all weights as text, no tools | named the game of all 4 |
| Read | middle blocks of 3 character-level transformers, 7.2 M weights | Python over the weights, no running them | named each one's language; found every copying head and no other |
| Read | 2 fine-tunes of Gemma 3 270M, one sycophantic | weights and Python | **could not tell** which, in 4 attempts, until shown attention patterns |
| Fix | a car-control policy, 171,664 weights, stuck in a local minimum | weights, the scene, one episode, no tools, one reply | 1,024 of 1,024 episodes succeed, up from 0, by changing 3.5% of the weights |
| Write | 3 game policies, 600→32→32→6 | the game's rules and source code, no data, no tools | beat PPO trained for 100 M steps in all 3 |
| Write | a Go network, 1.48 M weights | a Python sandbox, up to 50 test runs | beat an equal-sized PPO network trained on 20 M games |
| Fix | Qwen3.5-0.8B, reasoning loops | a Python kernel, 10, 25 or 50 test runs | 131,072 weights (0.016%) changed; MATH-500 7.9% to 12.3%, 14.1%, **48.1%** with 50 runs |
| Fix | Gemma 3 270M, which doesn't loop | a Python kernel, 5 test runs | **no gain** (6.8% to 7.0%): it fixes a defect, it doesn't add ability |
| Audit | 18 small clinical networks with planted flaws, 433 to 4,801 weights (Nusrat and Nusrat, 3 November 2026) | Claude Opus 5.5, blind, no tools | named the broken rule in 16 of 18 and repaired each by changing a median of **3 weights** (fine-tuning changed nearly all of them); but **cleared only 2 of 5 healthy networks** and **found 1 of 3 hidden triggers**; testing against the protocol found every flaw, and plain black-box testing found the confounds and the shift faster |
| Local | the 4 game policies above | Qwen 3.8 27B (17 GB) on a laptop, as text, no tools (Willison, 3 November 2026) | named 3 of 4, missed Freeway; **46 to 65 minutes** each (43,506 reasoning tokens for one); needed a long context |

Critics point out that the report mixes two things: reading weights with
no tools, and editing them with code, where the language model takes the
steps an ML researcher would (Honnibal, Goedecke, as cited by Willison).

So: reading and repairing small networks works; writing a network from a
specification works; a defect that sits in one place can be fixed by
changing very few weights, even in a large model, given a sandbox and
enough test runs; it adds no ability a network lacks; its audits raise
false alarms and miss hidden triggers; and **testing finds what it finds,
faster**. The design takes each of these as given.

**Two modes.**

- **Whole, as text, no tools**: the part's weights in the prompt. For
  reading and small repairs, on parts of up to about 20,000 weights (what
  the no-tools experiments read).
- **Sandbox**: a Python session over the weights (numpy, no GPU), plus a
  budget of test runs, each a forward pass of the candidate on a set of
  pages, scored. For writing a part, and for repairs that need checking.
  The 0.8B fix needed 50 runs, and 10 or 25 barely helped, so a session gets
  50 runs.

**What it does here.** Three jobs, on the programmable parts (§2):

| Job | When | Goal given | Done by |
|---|---|---|---|
| **Write** | round 0 | "compute, as M3's rule stem, the stamp-ink, dark-ink and rule masks `stamps.rs` and `masking.rs` compute today", with that source code and its thresholds, as the game experiments gave rules and source code | sandbox mode; scored pixel by pixel against the pipeline's masks |
| **Repair** | between training rounds | a fault from the disagreement report, in words and with its pages ("M2 scores signatures on stamped pages too low: 41 misses, all with a ring under them") | the fewest weights that fix those pages and change no others; at most 1% of the part's weights, or it is treated as needing training |
| **Read** | before each promotion | "say what each output of this part responds to, and name any input it ignores or any feature it shouldn't use (the QR code, the page border, the KIEMS header)" | no tools; the reading goes into the model card |

Repair is for **defects**: a threshold off, an input silenced, a
class swapped. A new printing of a form, or a kind of page never seen, is
a gap in the data, and only training fills it.

**The reading is a pointer, never a verdict.** In the clinical study it
flagged 3 of 5 healthy networks and found 1 of 3 hidden triggers, and tests
found everything it found. So:

- each claim it makes is turned into a test before anyone acts on it (a
  claim that M2 relies on the QR code becomes an occlusion test: cover the
  QR code on the gold set and measure the change);
- the same checks run automatically every round, as plain statistics, which
  the clinical study found catch the same faults: an input channel whose
  weights have shrunk to near zero (silenced), a unit that never fires, a
  bias or threshold far from its neighbours;
- shortcut tests run every round whatever the reading says: occlusion of
  the QR code, the header, the page border and the stamps, and pages from a
  county the round never trained on;
- a reading can't clear a model for promotion, and its flags block one only
  once a test confirms them.

**How a session runs.**

1. A tool prints the part's weights: layer names, shapes, values to 4
   significant figures, the meaning of each input channel and output
   written beside them; in sandbox mode, the same as arrays.
2. The language model gets the goal, the weights and, in sandbox mode,
   **probe** (the part's activations on chosen pages, as numbers), **try**
   (one test run: apply a candidate edit to a copy and score it on the
   round's regression set; counted against the 50) and **diff**.
3. It answers with an edit: the weights changed, their new values and its
   reasons.
4. The edit becomes a new model version and goes through the whole gate
   (§6.2) like a trained one. Nothing it writes reaches a published form
   any other way. Its diff, usually a few weights, is kept beside the model
   card, where a person can read it.

**The language model runs on this machine** (the owner's decision, 9
November 2026, for every part, those trained on synthetic forms included).
Weights trained on real forms are derived from personal data (§10). The reference is an open model of the Qwen 3.8 27B class:
17 GB, so it fits a 24 GB GPU (the training GPU), with a context of at least
128,000 tokens. A reading of a 17,000-weight part takes about an hour, so a
round affords a handful of sessions. The published results used frontier
cloud models (GPT-6 Astra, Claude Opus 5.5), and the local model did less
well (3 of 4). No part is shown to a cloud model. Training runs on this
instance's CPU with 15 GB of memory (§6.4), where a 27 B model doesn't fit:
milestone 2 picks the largest local model that does, and measures it.

**Backbones.** Not by default. The 0.8B fix shows that a defect traced to
one direction inside a large network can be removed in the sandbox; if a
reading and a test point to such a defect in a backbone, a sandbox repair
may be tried as an experiment, behind the same gate. Otherwise backbones
are trained.

**Keeping a repair.** A repair fixes the weights now; the next training
round would undo it. So every repair also adds its pages to the training
set and to a **regression set** that every later version must pass, and the
repaired part is the starting point of the next round's training of that
part. A repair that keeps being undone means the fault is in the data, not
the part.

**A repair is promoted only after training confirms it** (the owner's
decision, 9 November 2026). A repaired version is scored and shadowed like
any other, but the gate (§6, step 11) promotes it only once the next
training round, started from the repaired part, has kept the fix and passes
the gate itself; what is promoted is that trained version.

**Whether it earns its place.** Each round records the faults repaired,
the repairs that passed the gate, those that survived the next training
round, the test runs and hours spent, against a training round's. If,
after three rounds, repairs pass the gate less often than retraining
fixes the same faults, repair stops and the design is training alone; if
the written rule stem misses its test (§11), the stem is trained. Nothing
else depends on weight-space programming.

## 6. The iterative process

```
round 0   freeze the teacher T0 (this pipeline) and score it on the gold set;
          write the rule stems from the pipeline's rules (§5, write);
          train the backbones on the teacher's labels: student S_0
round k   1. harvest   labels: the teacher's outputs + settled pages + synthetic
          2. train     student S_k from S_(k-1) (recipe, seed and data recorded)
          3. score     S_k and T on the gold set and the regression set
          4. shadow    run S_k beside the pipeline on the whole corpus
          5. compare   list every disagreement, ranked by risk
          6. settle    rules settle what they can; the rest is flagged
          7. diagnose  sort the settled faults: a gap in the data, or a fault
                       in one place
          8. repair    defects in one place: weight-space programming on the
                       part they sit in (§5, repair) -> S_k'
          9. fix       relabel; fix the teacher where it was wrong; add the
                       repaired pages to the training and regression sets
         10. read      read every programmable part (§5, read); turn each
                       claim into a test; run the statistics and shortcut
                       tests
         11. gate      promote S_k one stage if its scores pass (§7); a
                       repaired S_k' is promoted only once training has
                       confirmed it (§5, keeping a repair)
          until S_k matches T on the gold set two rounds running and
          disagreements fall below 0.5% of fields
```

Repairs are fast (hours) and narrow; training is slow (a day or more) and
broad. A round can end after step 8 with a repaired model and go straight to
step 3 again, without training, when only faults in one place were found;
the repaired model is then measured, not promoted, until a training round
confirms it.

Both sides get better: a disagreement the teacher loses is a pipeline bug
to fix, as every review round has been so far.

### 6.1 The gold set

No person labels pages (the owner's decision), so the gold set has two
parts, planned in `gold/plan.json` (nn/README.md):

- **the synthetic gold set**: synthetic forms made from a seed range of
  their own (by default 1,000,000,000 to 1,000,000,300), across every
  printing with a template, photos and scans, with exact labels and pen
  masks. It measures truth, on synthetic forms.
- **the reference set**: about 300 real pages sampled from the teacher's
  datasets, across templates and years, KIEMS photos, scans and photos,
  published and held. Its labels are the teacher's, so it measures
  agreement with the pipeline, not truth.

Neither is ever used for training (every training reader goes through the
gold set's exclusion), and the plan changes only by adding pages, each
change versioned, so scores stay comparable across rounds.

### 6.2 What a round measures

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

### 6.3 Settling a disagreement

Kinds, in order of risk:

1. **The student finds PII the teacher left visible** (a box with nothing
   masked under it): a possible leak in today's output. First.
2. **The teacher masks what the student doesn't see**: a student miss.
3. **They differ on a protected region, a stamp or a page's class or
   turn.**
4. **Masks that overlap but differ in extent.**

Rules settle what they can: whether a box overlaps a results region, what
the leak check says, whether a stamp's ring is intact. No person settles
the rest (the owner's decision): it is flagged, listed in the round's
report by kind and page, and shown read-only in the local viewer
(nn/README.md). Nothing goes to a cloud model; the vision review is not a
judge here, since the network is meant to replace it. How flagged
disagreements count against a stage's gate is for milestone 3 to set.

### 6.4 Training, locally

- A `nn/` folder in the repository: the dataset builder (reads manifests,
  sources and synthetic forms), the four training recipes, ONNX export,
  the gold-set scorer and the read-only viewer.
- A recipe is code plus a config with its seed. A round records the data
  list (file SHA-256s), the recipe's commit and the library versions; the
  same inputs give the same weights where the libraries allow.
- **Hardware.** One local GPU with 16–24 GB trains each model in hours;
  M1 and M4 train on a CPU in a day, M2 and M3 are impractical without a
  GPU at full size. Inference needs only the CPU. Decided: training runs
  on the project's cloud instance, CPU only (4 cores, 15 GB of memory,
  about 5 GB of disk, not kept between sessions), so datasets are kept
  compact (JPEG pages, PNG masks, pages made 1,280 px long where that is
  enough) and M2 and M3 are trained small and at reduced size.
- Each model gets a **model card**: data, scores on the gold set, known
  failures, the round it came from. A registry (`nn/registry.json`) lists
  every version and its stage.

## 7. Promotion: what the network is trusted to do, stage by stage

Each stage is a setting, can be turned off at once, and needs the
targets of §6.2 met on the gold set and a sample of live output checked by
a person. A version made by weight-space programming is promoted by the
same gate as a trained one, never on the language model's word: it can
find and repair faults but cannot certify a network (§5). Stage E also
needs every shortcut test (§5) passed, and every flag of the reading
either tested and dismissed or fixed.

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

## 8. What it would change in the product

- **Jobs** (HARNESS.md): a job with stage C needs no model host; model
  calls and their transfers disappear from the journal and the report.
- **The API** (API.md): the same endpoints; the price is set by the owner
  and doesn't change. The operator's cost per form falls (no tokens; about
  1–2 s more CPU per page, against 8–15 s less where OCR is no longer run).
- **verify**: unchanged for published forms (recorded outputs);
  `--reinfer` added.

## 9. Risks

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
- **Weight-space programming is a week old.** One unreviewed report, one
  preprint and one blog test; nothing yet on convolutional networks; the
  0.8B result fixed a defect (reasoning loops) and added no ability; the
  clinical audits raised false alarms and missed hidden triggers; and the
  local model was slower and weaker than the frontier ones (3 of 4, an hour
  each). So it works on small parts and defects, every edit passes the full
  gate, its readings are tested before they count, and the design works
  without it (§5).
- **A repair that fits its pages and breaks others.** The regression set
  and the whole gold set are scored for every repair, not only the pages it
  was made for.

## 10. Personal data in training

- Training data is personal data (names, IDs, phones, signatures of
  officers and agents), processed for a new purpose: training. The DPIA
  annex (`rfpii dpia-annex`) gains a section for it; the controller record
  names the purpose.
- It stays on the local machine: an encrypted disk, access for named
  people, and a retention schedule (`rfpii retain`) for the label store.
- **Weights may leave the machine** (the owner's decision, 9 November
  2026), those trained on real forms included: they can be stored and
  shared elsewhere. Detectors and segmenters don't reproduce their inputs
  the way generative models can, but this is not proven for every input,
  so the DPIA annex records the decision and its reasoning, and each model
  card says whether real forms were in its training data.
- M4 never reads characters (§2), and no model outputs text.
- The language model that reads and writes weights runs on this machine
  (§5); no cloud model reads or writes them. A
  probe shows it activations of chosen pages, numbers, not images; it never
  sees a page.
- An erasure request reaches the training set too: the form's labels are
  removed from the label store, and the next round trains without them.

## 11. Milestones

1. **Data:** the dataset builder from manifests, the synthetic generator
   extended (also to redacted pages that say what still shows, for the
   local reviewer), the gold set planned (synthetic and reference), and a
   read-only viewer of labelled pages.
2. **Weight-space tools and the first test of them:** the weight printer,
   probe, try and diff tools, the local language model, and M3's rule stem
   written from the pipeline's rules (§5, write). The test: do the written
   stem's masks agree with the pipeline's on at least 99% of the pixels of
   the training pages? If not, the stem is trained instead, and repair
   is tried later on smaller parts first.
3. **M1 and M2, stage A:** shadow runs on the corpus, the first
   disagreement report.
4. **Rounds to stage B and C:** the local reviewer replaces the cloud
   review, with repair measured each round (§5).
5. **M3 and M4, stage D:** held pages read locally.
6. **Stage E:** primary reader, templates as checks.

Each milestone ends with its round's scores on the gold set, written up
in this document as DESIGN.md does for the pipeline.

## 12. Where it stands

### Milestone 1: the data

Built (9 November 2026), as nn/README.md describes:

- **`rfpii dataset synth`** (`rust/src/dataset/`): synthetic forms on the
  four printings with a template (Form 34A; the Form 34B blocks of layouts
  A, B and C), drawn on the templates' reference images, with exact boxes,
  character slots and ink masks (print, pen, stamp ink drawn on separate
  layers), photographed or scanned, some turned, folded, blurred; and
  negatives (results pages, unfilled forms, blank pages, pages that are
  not forms). A seed gives the same files, byte for byte.
- **`synth --redacted`**: the same pages redacted by the pipeline itself,
  some masks then undone or cut back on purpose, every field labelled with
  what it still shows (`masked_ok`, `leak_signature`, `leak_number`): the
  training data for stage C, the local reviewer.
- **`rfpii dataset build`**: the teacher's labels from published manifests
  and their originals, and the held pages as an unlabelled pool;
  **`check`**, **`stats`**, **`compare`**.
- **`nn/`**: the dataset reader, the gold set's plan (synthetic gold set and
  reference set, §6.1) with the exclusion every training reader goes
  through, and a read-only viewer on 127.0.0.1.

**How good the teacher's labels are, where the truth is known.** 100
synthetic pages of each kind (about 25 per printing), redacted with
`rfpii redact`, labelled with `rfpii dataset build` and compared with the
truth they were drawn with (`rfpii dataset compare`), over all four
printings, layout C included. Light scans ("clean"), and photos and scans
as they come ("mixed": 41 photos, 12 pages turned). "Found" is recall at
the IoU given; "mean best IoU" averages, over the true boxes, the best
overlap any teacher box of the class has with each:

| | clean | mixed |
|---|---|---|
| pages published by the pipeline | 92 of 100 | 90 of 100 |
| agents table, comments, declaration: mean IoU | 0.99 | 0.99 |
| results regions: found; median best IoU | 0.70; 0.98 | 0.73; 0.99 |
| ID numbers: found at IoU 0.5 / 0.3; mean best IoU | 0.71 / 0.80; 0.57 | 0.69 / 0.83; 0.56 |
| phone numbers | 0.75 / 0.86; 0.58 | 0.72 / 0.82; 0.55 |
| signatures | 0.69 / 0.88; 0.54 | 0.66 / 0.86; 0.54 |
| stamps: found at IoU 0.5; mean best IoU (precision) | 0.88; 0.84 (0.69) | 0.83; 0.77 (0.76) |
| numbers with slots on both: same count; mean boundary error | 90%; 16.5 px | 91%; 17.9 px |

So the teacher places fields where they are, but its boxes are coarse:
they are drawn from whole 8×8 blocks, a signature's from its band (the
middle half, grown back to full height), a partly hidden number's from its
hidden characters, and a cell the masking found empty is the whole cell.
A detector trained on them should be scored at IoU 0.3 for fields, and
character slots should come from synthetic forms (M4), not the teacher.
The teacher's stamps include the IEBC logo in the 34A header and other
rings its search favours (it favours recall, by design); the synthetic
truth labels the stamp printed on the 34A reference, which the stamp
finder also finds. The results regions the truth has and the teacher
doesn't are the typed results tables above a Form 34B block, which no
template protects.

**Two things the measurement found in the pipeline itself.**

- **Layout C's phone column is not masked.** The `form34b_2022_block_c`
  template's `agent_phone` zone (x 1844–2158) lies on the printed
  "Signature" column and its `agent_signatures` zone (2164–2483) on the
  "Date" column; the printed "Tel. Contact" column is 1524–1839, and no
  zone covers it. On synthetic pages written in the printed columns, none
  of 54 phone numbers was found (recall 0.00, both runs), and on redacted
  pages all 18 were published whole; 27 of 63 signatures were only partly
  hidden. The other three templates' zones match their printed columns.
  The one published layout C document (Daadab) leaked nothing: its agents
  table is struck through and empty. **Fixed** (9 November 2026): the two
  zones now lie between the printed rules (phone x 1525–1839, signature
  1844–2159). On 30 synthetic layout C pages redacted with the same seeds,
  phone numbers published whole fell from 37 of 37 to 2 of 37, both cut
  back on purpose by the generator, and agents' signatures left showing
  from 32 of 33 to 8, 3 of them cut back on purpose; the other 5 have
  64–74% of their middle hidden, against the 75% the rule asks.
- **Faint numbers can be published whole.** Outside layout C, on 89
  redacted synthetic pages the pipeline's own masks (none cut back) left
  37 of 589 fields showing by the rules of nn/README.md: 6 of 204 IDs (3
  of them with no character hidden at all), 3 of 148 phone numbers and 28
  of 237 signatures (their middle not hidden along enough of them). In the
  teacher comparison, the written numbers the masking read as empty were,
  looked at one by one, mostly thin, light writing, published whole. The leak check rereads with the same ink rule, so a
  stroke too faint for one is too faint for the other (DESIGN.md, "Known
  limits"). How often real pens are that faint is not known; these are the
  misses the local reviewer is for.

**Size and speed** (4 cores): `synth` writes 5.2–5.7 pages a second at full
size or at 1,280 px; a page takes 1.1 MB at full size and 212 kB at
1,280 px (10,000 pages: about 2.1 GB). `synth --redacted` runs the pipeline
on every page: 0.27 pages a second, and the pipeline holds about one page
in ten. `build` labels 5.6 pages a second, the rules' ink mask included;
`rfpii redact` itself ran at 0.34 pages a second.

**What remains for milestone 1**: the teacher datasets and the reference
set, from the real corpus. It is on this machine (this section said
otherwise until milestone 2), outside the repository: the 2013 originals
and their latest published run with its manifests, the 2022 Forms 34B as
published with their originals, a run of 2022 Forms 34A, and the forms
downloaded from the owner's Dropbox archive (the Forms 34A of 2022, a zip
per county) and the IEBC portal, each file in the provenance file beside
them (§3). Run `rfpii dataset build` over each published run, then
`gold.py plan` and `make-synthetic` (nn/README.md). No person labels the gold set (§6.1).
Layout C pages published before the template's fix (above) are relabelled
by running them again.

## 13. Open decisions

Decided by the owner on 9 November 2026:

- **Training hardware:** this cloud instance, CPU only (§6.4).
- **Reviewers:** none. No person labels pages or settles disagreements;
  disagreements are flagged (§6.1, §6.3).
- **Target:** stage C, the local reviewer on published, redacted pages
  (§7).
- **More data:** yes, from the IEBC portal and the owner's Dropbox archive,
  with each file's provenance recorded (§3).
- **Weights:** they may leave the machine, those trained on real forms
  included (§10).
- **The weight-programming model:** it must run locally, for every part
  (§5).
- **Repairs:** a repaired version is not promoted before a training round
  confirms it (§5).

Still open:

- **A person's check before promotion:** §7 asks for a sample of live
  output checked by a person at each stage; with no reviewers, what
  replaces it?

## Sources on weight-space programming

Read in full on 9 November 2026; the figures in §5 are from these.

- R. Takehi, J. Teoh, A. Zhang, O. Khattab, *Sparks of Weight-Space
  Programming: LLMs can natively read & write neural network weights*,
  MIT CSAIL, 2 November 2026, https://rikiyat.github.io/blog/weight-space-programming/
  (logs and prompts: GitHub `RikiyaT/lm_understand-logs`).
- H. Nusrat, O. Nusrat, *Weight-space programming for auditing clinical
  decision networks*, arXiv 2611.05341 [physics.med-ph], 3 November 2026,
  https://arxiv.org/abs/2611.05341 (code, networks and every model answer
  in its ancillary files).
- S. Willison, *Qwen 3.8 27B on my laptop can interpret model weights, too*,
  3 November 2026, https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/
  (which also cites the criticism by Matthew Honnibal and Sean Goedecke).
- Not to be confused with *weight-space learning*, the older field of
  networks that take other networks' weights as input.
