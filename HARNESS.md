# A harness that runs redaction jobs

`rfpii redact` masks a batch of forms. A real job asks for much more than
one command. When the 2013 Forms 34 were redacted (6,984 pages, 51 files),
an operator also had to:

- take stock of the sources: zips, a truncated PDF, three copies of the same
  scan;
- choose the reading mode and the number of workers, and watch memory and
  disk;
- pilot one file;
- run the files one at a time;
- count the pages published and held, and why each was held;
- rank the published pages by risk and look at the riskiest;
- look at random samples from each county;
- `verify --rerun` a few pages;
- work out what the held pages were;
- report the results.

Seven signature leaks were found that way, on pages that had passed every
check: five, then two signatures hanging below an officer's last line. Each
was fixed in the code, and the set was run again.

This is the design of a harness that does that work unattended, driven by
models on Ollama Cloud. It is given a form or a set of forms, and
instructions on what to mask. It returns:

- the published forms;
- the forms held back, with the reason for each;
- a report of what was checked and how.

**The principle:** the models decide and look; `rfpii` makes every edit;
plain code enforces privacy and decides what is released.

## 1. A job

A job is a folder, and a job file:

```yaml
inputs:                       # files, folders, zips; fetched by code, never by a model
  - /data/2013/IEBC2/
instructions: |               # free text from the requester
  Hide ID numbers completely. Officers stamp over their details:
  hide what is under the stamps too. Keep the phone prefixes.
outputs: /jobs/2013-34          # staging/, public/, review/, quarantine/, report.*
sign_key: /keys/signing_key.pem
models:                       # Ollama Cloud unless a host is given; see §5
  orchestrator: [qwen3.5:397b, kimi-k2.6, gpt-oss:120b]
  vision: [gemma4:31b, qwen3.5:397b]
held_pages: cloud             # held pages go to the vision model too (§4.7, §7)
budget: {model_requests: 20000, hours: 12}
```

No one approves a job's settings: the job runs as soon as it is given
(§3).

The folder it fills:

| Path | Holds |
|---|---|
| `plan.json` | inventory, groups, policy, estimates (§4.1–4.2) |
| `journal.jsonl` | every tool call, model call and decision, in order (§10) |
| `staging/` | everything `rfpii redact` published, not yet released |
| `public/` | what passed the release gate: PDFs, manifests, ledger, batch root |
| `review/` | held pages: `report.jsonl` and a triage note per page |
| `quarantine/` | pages withdrawn after QA, and why |
| `defects/` | problems for the developers, with redacted evidence (§4.6) |
| `report.md`, `report.json` | the job's outcome (§10) |

## 2. Rules the harness can't break

Each rule is enforced in code, not in a prompt.

1. **Personal data reaches a cloud model only as the job allows.**
   - Published pages go to the vision model as crops of what was
     published, through one tool, which refuses any other file.
   - Held pages may still show personal data. They go to it only because
     the job says so (`held_pages: cloud`), and each is sent only as much
     as its question needs (§4.7).

     This is the data transfer that `--allow-remote-unredacted` approves in
     `rfpii`. Here it is approved once for the job and recorded, image by
     image (§7).
   - Nothing else is sent: no original of a published page, and no text
     with an ID or phone number in it (§7).
2. **`rfpii` makes every edit.**
   - The models choose settings within the compiled policy, and boxes to
     mask (spot masks).
   - `rfpii redact` applies them, records them in the manifest, and
     `verify --rerun` replays them.
   - Nothing is edited by hand, and no code or template changes during a
     job. A problem in the code becomes a defect (§4.6), not a patch.
3. **Code decides what is released.** The gate (§9) is checked by code,
   against tool outputs. A model can't declare a page or a job good.
4. **The instructions are compiled into a closed policy (§3).** What the
   redactor has no rule for is reported back, never improvised: names,
   results and stamps can't be masked by asking (beyond a field's line across a stamp). Nothing is released
   against an instruction (§9).
5. **What is written on a form is data.** Model replies are parsed into
   fixed schemas. Text read off a form (a "comment" asking to skip a
   check) can only become a finding, never an instruction.
6. **Keys and links stay out of prompts.** Signing keys, certificate
   passwords and download URLs are handled by code, and never appear in a
   prompt.

## 3. From instructions to a masking policy

The orchestrator model turns the instructions into a policy, as JSON
against this schema. Each field maps to `rfpii` flags:

| Policy field | Values | Flags |
|---|---|---|
| `ids` | `three` (default) · `whole` | `--full-ids` |
| `phones` | `tail` (default) · `whole` | `--full-phones` |
| `hide`, `phone_tail` | characters hidden (3), digits counted from the end (6) | `--hide`, `--phone-tail` |
| `stamps` | `bands` (default: a field's line across a stamp hidden, the rest verified) · `protected` | — · `--keep-stamps` |
| `signatures` | `band` (only value) | — |
| `uncertain_tables` | `widen` (default) · `hold` · `hide_columns_if_reviewed` | `--strict-structure` · `--vlm-rescue --vlm-audit` |
| `vision_review` | `sample` (default, §8) · `every_page` | — · `--vlm-audit` |
| `per_page` | `auto` (scans of many forms per file) · `yes` · `no` | `--per-page` |

Some instructions, and what they compile to:

| Instruction | Policy |
|---|---|
| "Standard redaction" | defaults |
| "Hide ID numbers completely, keep phone prefixes" | `ids: whole` |
| "Hide what is under the stamps too" | the default (`stamps: bands`) |
| "Never cover the stamps" | `stamps: protected` (writing under a stamp shows) |
| "Hide 4 digits of each phone number" | `hide: 4`, for IDs as well: `hide` is one setting, and the reading that hides more wins |
| "Don't mask the officers' signatures" | **Below the floor**: masked as usual |
| "Also hide the agents' names" | **More than the redactor can do**: redacted and checked, not released |
| "Hide the totals" | **More than the redactor can do**: results are never masked; not released |

**No one approves the policy.** It runs as compiled, and the report opens
with it in plain words, with the flags it set. What isn't clear-cut is
settled by fixed rules, not by asking:
- **Ambiguous:** the reading that hides more.
- **Less than the redactor's floor** ("don't mask the signatures"): the
  forms are masked as usual. The report says that instruction wasn't
  followed, and why.
- **More than the redactor can do** ("hide the names", "hide the totals"):
  the forms are redacted and checked, but none is released (§9).
  Publishing them would show what the instruction asked to hide. The
  report names the rule `rfpii` would need, or, for results, why they are
  never masked.

## 4. The process

The phases are fixed, in code. Within a phase, the orchestrator works in a
loop with the phase's tools until it returns the phase's structured
verdict. The loop has a step limit and a share of the budget.

| Phase | What it does | Done by hand in 2013 |
|---|---|---|
| 1 Intake | expand, hash and count every file; find duplicates and broken files; sort into groups | listed the zips; found Sabatia truncated and KIAMBU CONSTITTUENCY three times |
| 2 Plan | reading mode per group, per-page, workers, order; time, memory and disk estimates | chose `--per-page`, 4 workers; freed disk mid-run |
| 3 Pilot | redact a sample from each group and check every page of it | piloted Baringo North; fixed the reader before the full run |
| 4 Run | redact group by group, resumable, with a heartbeat | a script per file; a waiter re-armed every 2 hours |
| 5 QA | analytics over every manifest; risk-ranked and random visual review; verify | `summary.py`, `empty_sig.py`, `ink_check.py`, montages, `verify --rerun` |
| 6 Remediate | spot masks for single leaks; quarantine and a defect for a pattern | the Moyale and Tiaty hanging signatures: fixed in code, run again |
| 7 Triage | what the held pages are; what, if anything, can be retried | Kiambaa's tally sheets, Emuhaya's turned scans, Lari's faint ones |
| 8 Release | the gate (§9); copy to `public/`; batch root | — |
| 9 Report | numbers, reasons, checks, follow-ups | the final report, the GENERALIZING.md table |

### 4.1 Intake

Code does all of intake:

- expands zips, keeping each member's name;
- hashes every file and finds duplicates;
- opens each PDF and finds broken ones ("cannot find startxref");
- counts pages;
- tells the kinds of file apart:
  - KIEMS JPEG-in-PDF, by its file name and QR;
  - photos, by their EXIF;
  - multi-page scans.

It then sorts the pages into **groups**, by kind of page and by printing
where it can tell: for example, 2022 Forms 34A, or 2013 Form 34 fronts.
Groups are what the leak-rate bound and the release gate work on (§8,
§9). Within a group, pages are also counted by folder (county,
constituency), for the statistics, to spread samples, and for the
report. The orchestrator reads
the inventory as scrubbed JSON: names of folders and files, counts and
kinds, no page content. It flags what needs a decision, such as
duplicates, broken files, or a group much smaller or larger than its
neighbours.

### 4.2 Plan

For each group, the orchestrator picks:

- the reader: a template, `--learn`, or reading without one;
- per-page or whole-document;
- the order of the groups.

Code checks the plan against the machine and refuses one that can't fit:

- **Memory.** Four workers on the 2013 scans used 6–8.5 GB, and a process
  once reached about 10 GB on a 15 GB machine. Workers are sized from what
  each kind of page measured in the pilot, with a margin.
- **Disk.** The 2013 output was 3.7 GB for 6,984 pages. The estimate comes
  from the pilot's bytes per page. The run stops before the disk is full,
  not after.
- **Time.** About 0.4 pages a second, or 5 hours for the 2013 set. The plan
  must fit `budget.hours`.

### 4.3 Pilot

- **The sample.** 2% of each group, at least 20 pages and at most 500 in
  all, chosen at random.
- **The checks.** Every pilot page gets the full QA of §4.5, including a
  visual review of every page.
- **Going on.** The full run goes ahead only if:
  - the pilot found no confirmed leak that remediation (§4.6) couldn't
    clear;
  - each group's publish rate is within what the plan expected.

  A group that fails is set to "review only": it is run, but nothing from
  it is released until its problem is understood.
- **The note.** The orchestrator writes a pilot note: what was seen in each
  group, and why the plan goes on as it is or changes.

### 4.4 Run

- **One file at a time.** One `rfpii redact` process per source file, as in
  the 2013 run, so that one file's memory doesn't add to another's and a
  crash costs one file.
- **Resumable.** With `--resume` (new, §6), pages already in the ledger with
  the same settings are skipped.
- **A heartbeat every minute**, recorded in the journal: pages done, rate,
  memory, free disk. Code acts on it, not a model:
  - no progress for 10 minutes: kill the process and resume;
  - an out-of-memory kill: one worker fewer, and resume;
  - free disk below the estimate for the rest of the run: stop and report.

### 4.5 QA

QA runs on everything in `staging/`, before anything is released. It has
three parts.

**Analytics** (code, over every manifest, no images):
- published and held counts per file and per group; held reasons,
  normalised (numbers and quoted text taken out);
- per page, the risk signals of §8, combined into a score;
- outliers against the page's own group. For example, a page with one
  officer's field where its group has two, or ID fields with fewer
  characters than its group's.

**Visual review** (vision model, published crops only):
- every page above a risk threshold, and a random sample from each group
  (§8);
- each crop is a page's officer block or agents' table, as published;
- **find, then check**, as `--vlm-audit` does: the finder names spots that
  break the policy's rules, and the checker confirms each on a zoomed crop.
  Crops can be sent in tiles of eight to save requests, with each flagged
  tile checked alone. Tiling is used only if the evaluation (§11) shows it
  finds as much as single crops.

**Verification** (code):
- `rfpii verify` on every page: proof, manifest signature, ledger;
- `verify --rerun` on a random 1% of each group (at least 5 pages), and on
  every page that got spot masks.

### 4.6 Remediate

The orchestrator handles each confirmed finding.

- **A single leak.** The page is redacted again with the confirmed boxes as
  spot masks (`--spot-masks`, new, §6). They take the same treatment as
  `--vlm-audit` finds:
  - a signature hidden whole;
  - a number hidden by the policy's rule.

  They are recorded in the manifest (`spot_masks`: box, label, the model,
  the prompt's hash, the hash of the image it saw). `verify --rerun`
  replays them, and the page is checked again.
- **A pattern.** Three or more confirmed leaks of one kind in a group, or a
  risk signal that every confirmed leak shares, make a pattern. Then:
  - every page with that signal is reviewed, or all of them go to
    `quarantine/`;
  - a **defect** is written: what leaks, the signal that finds it, the pages
    affected, and redacted crops showing it.

  The hanging signatures of the 2013 run are the kind of pattern this is
  for. Their signal, a signature line judged empty with writing just below
  it, finds every page like them.
- **A defect is fixed in the developers' loop, not by the harness.** Once a
  new version is released, the harness runs the affected pages again and
  compares their masks with the previous run's, as was done by hand with
  fixed seeds. What the change hides or shows is reviewed before release.

### 4.7 Triage held pages

A held page failed a check, so it may still show personal data. The job
lets the vision model on Ollama Cloud look at held pages
(`held_pages: cloud`). Each question gets only as much of the page as it
needs.

- **What is this page?** Every held page is sent as a **triage view**:
  - redacted with every mask the redactor did place;
  - every field it found but couldn't confirm hidden whole;
  - at a resolution too low to read handwriting: 512 px on the long side,
    to be confirmed by the test in §11.

  The model answers in a fixed schema:
  - the kind of page: this form, another form, a tally sheet, blank, not a
    form;
  - turned or cut off, too faint;
  - where the declaration and the agents' table are.
- **Can it be published?** A page of the right kind whose fields couldn't
  be placed is sent at full resolution. The model is asked where its
  signatures, ID numbers and phone numbers are: `--vlm-fallback`, which
  runs with `--allow-remote-unredacted` because the job allows it.
  - The page is redacted again with those boxes, which are recorded in its
    manifest.
  - It then goes through every gate and through QA like any other page.
  - A page saved this way is always reviewed visually, never left to the
    sample.
- **Text.** The orchestrator also sees, for each held page:
  - its normalised held reason;
  - text diagnostics computed by code, each a number, a measurement or a
    yes/no:
    - the turn it was read at;
    - skew;
    - contrast and ink coverage;
    - OCR confidence;
    - which words of the form's printed vocabulary were found (an
      allow-list: "Declaration", "Presiding Officer", "TALLY"…);
    - the fields that were found.
- **The 2013 held pages show what triage has to tell apart:**
  - Kiambaa's 70 are tally sheets, not Forms 34;
  - Emuhaya's 67 are turned and cut off;
  - Lari's 20 are too faint to read.

  Each kind shows in the triage view, and in the diagnostics. Lari's
  faint pages are the ones a full-resolution look might still save.
- **Actions, within the policy:**
  - retry a page with another reader;
  - locate its fields, as above;
  - `--vlm-rescue`, if the policy allows hiding columns whole;
  - mark pages "not a form for this job", listed apart in the report;
  - send the rest to people, with a category and a suggested action.

### 4.8 Release and report

The gate (§9) is checked per group. A group that passes is copied to
`public/`, with its ledger entries and a batch root over what was
released. A group that fails stays in `staging/`, and the report says why.
The report is described in §10.

## 5. Models

| Role | Needs | Candidates (Ollama Cloud, 2026) |
|---|---|---|
| Orchestrator: policy, plans, phase decisions, notes | tool calling, reasoning, long context, JSON | `qwen3.5:397b`, `kimi-k2.6`, `glm-5.1`, `deepseek-v4-flash`, `gpt-oss:120b` |
| Vision finder and checker (published crops only) | vision, JSON | `gemma4:31b` (the current `--vlm-audit` default: on 88 spots flagged on redacted Forms 34A, as checker it kept 26 of 31 real misses and dropped 53 of 56 false alarms), `qwen3.5:397b` |
| Scored checks | the same answer every time | Clef (`--vlm-checker clef`), as now |
| Held pages: triage, and locating fields (§4.7) | vision, JSON | the vision models above |

- **Configurable, with fallbacks.** The cloud catalogue changes: several
  tags were retired in June 2026. So the roles are lists, in order of
  preference.
- **Checked at start.** At start the harness reads `/api/tags` and each
  model's `/api/show` capabilities (`tools`, `vision`), as `vlm.rs` already
  does for `decision`. It drops what is missing, and refuses to start with
  an empty role.
- **Model calls:**
  - temperature 0;
  - a JSON schema in `format`;
  - backoff on 429;
  - a shared limit on concurrent requests (`--vlm-concurrency`).
- **An unreadable reply** counts as a finding in QA, and as "no
  decision" in a phase. Either way it is the safe side.
- **Every model is chosen by measurement (§11), not by name.** That is how
  `gemma4:31b` was picked for `--vlm-audit`.

## 6. Tools

The orchestrator acts only through these. Each takes and returns JSON, and
is an `rfpii` subcommand, so it can be tested without a model.

| Tool | Kind | Exists? |
|---|---|---|
| `inventory(inputs)` → files, pages, kinds, hashes, duplicates, broken files, groups | read | new: `rfpii inventory` |
| `redact(group or pages, policy, plan)` → run id | write (staging) | `rfpii redact`, plus `--resume`, `--pages`, `--spot-masks` (new) |
| `status(run)` → heartbeat | read | new (from the ledger and the process) |
| `stats(run, group?)` → counts, normalised held reasons, mask modes | read | new: `rfpii stats` (what `summary.py` did) |
| `risk(run, group?)` → pages ranked by the signals of §8 | read | new: `rfpii stats --risk` (`empty_sig.py`, `ink_check.py`, `id_in_stamp.py`) |
| `crop(manifest, region)` → image, for a model | read; the only image path | new: `rfpii crop`. Refuses anything not in `staging/` with a verified manifest |
| `review(crops, policy)` → confirmed findings | model | the find and check of `vlm.rs`, made a subcommand |
| `verify(pages, rerun?)` → pass or fail per check | read | `rfpii verify` |
| `diagnose(held page)` → text diagnostics (§4.7) | read | new: `rfpii diagnose` |
| `held_view(held page, triage or locate)` → image, for a model | read; held pages' only image path | new: `rfpii held-view`. Only with `held_pages: cloud`; full resolution only to locate fields on a page triage found to be of the right kind |
| `spot_mask(page, findings)` → the page redacted again | write (staging) | new: `redact --spot-masks` |
| `quarantine(pages, reason)` | write | new (moves files; journal entry) |
| `defect(title, signal, pages, crops)` | write | new (a file in `defects/`) |
| `release(group)` | write | new; runs only if the gate passes |

Code checks each write tool's preconditions:
- a spot mask needs a confirmed check, by its id in the journal;
- a policy change needs the compiled policy to allow it;
- a release needs the gate.

## 7. Privacy, in detail

- **Images.** `crop` is the only tool that returns pixels of published
  pages (held pages have `held_view`, below). It takes a
  manifest and a region, never a path. It renders from the published PDF
  only after that page has passed `verify`, and logs the SHA-256 of what it
  returns. Code checks every image a cloud request carries against the
  hashes `crop` and `held_view` logged before sending it. Nothing else can
  attach an image.
- **Held pages.** They have no published PDF, so `crop` can't reach them.
  - **Images.** They go through `held_view`, and only in a job with
    `held_pages: cloud`. It sends the triage view, and full resolution
    only to locate fields (§4.7).
  - **The record.** Every image it sends is hashed in the journal and
    listed in the page's triage record. So it can always be said which
    held pages left the machine, and at what resolution.
  - **Text.** Their diagnostics are numbers and allow-listed words.
- **Text.** Everything sent to a cloud model goes through the scrubber
  first:
  - quoted OCR text is replaced by `'…'`, as `summary.py` did;
  - every run of 5 or more digits is replaced by `#` (an ID has 7–8 digits,
    a phone number 10; coordinates and counts have 4 or fewer);
  - only allow-listed manifest fields are passed on.

  Names are public on the published forms, but the scrubber doesn't rely on
  that: a name read beside a field is quoted OCR text, and goes too.
- **A local model instead.** If the held-page role names a local host,
  nothing about held pages leaves the machine, as with `--vlm-fallback` on
  a local model now.
- **Tested.** Canary tests ask `crop` for an original and for a held
  page, and `held_view` for a page of a job without `held_pages: cloud`.
  All must fail. A test over a job's journal finds no image hash that
  isn't a logged crop or held view, and no run of 5 digits in any prompt.

## 8. Risk signals and sampling

**Signals,** from what the 2013 checks found:

| Signal | Why |
|---|---|
| a signature line judged empty, with writing on or just below it outside stamps | the 2013 signature leaks, among them the hanging signatures of Moyale (2) p117 and Tiaty p083 |
| a field hidden whole ("full") | a line that couldn't be segmented; a wide mask can cover print, or miss part of the writing |
| writing inside a stamp on an ID or signature line (default policy) | 560 of the 6,608 pages published in 2013 had half or more of an officer's ID line under a stamp |
| a field near the stamp-ring or the page edge; the page read at a quarter turn | where reading went wrong before |
| a layout unlike its group's (fields, officers, characters per ID) | twice-read labels made two ID fields on 139 Baringo North pages |
| low OCR confidence on the labels | faint scans |
| a leak-check run one short of its threshold | the check passed, just |

**Sampling.** Pages above the risk threshold are all reviewed. Below it,
a random sample from each group bounds the leak rate that could remain,
at **0.5%**:
- **The sample size.** Finding no leak in 600 random pages bounds the rate
  below 0.5% at 95% confidence (the rule of three, 3/600).
- **Small groups.** A group of 600 pages or fewer is reviewed whole.
- **A leak found in the sample** is fixed (§4.6), but it counts. The
  sample grows to 950 pages with one leak, and 1,260 with two (the Poisson
  95% bounds, 4.74 and 6.30, over 0.5%). Three are a pattern (§4.6).
- **Spread.** The sample is spread over a group's folders in proportion to
  their size, so no county or scanner is missed.

The 2013 set, as fronts and backs, needs 1,200 of its 6,608 published
pages. 50,000 Forms 34A in one or two groups need 600 to 1,200, plus the
risky pages.

## 9. The release gate

A group is released only if all of these hold, checked by code:

1. Every staged page passes `verify`: proof, signature, ledger.
2. Every `verify --rerun` in the group passes.
3. Every page above the risk threshold was reviewed.
4. The random sample bounds the group's leak rate below 0.5% (§8), and
   every leak it found was fixed.
5. No pattern (§4.6) touching the group is still open.
6. The policy that was run is the policy compiled from the instructions:
   the job's journal shows its flags on every redact call.
7. Every instruction was met. If the instructions asked to hide something
   the redactor can't (§3), nothing from the job is released.

A failed condition holds the group; the report names it.

## 10. Journal and report

**The journal** is the audit trail, with one JSON line per event:
- tool calls with their arguments and the hashes of their results;
- model calls: model, host, prompt hash, hashes of the images sent, the
  parsed reply;
- decisions: phase, choice, reasons;
- heartbeats.

From the manifests and the journal, someone else can see what every model
saw and decided. They can't change any output: the outputs are
reproducible from their manifests without any model.

**The report** gives:
- the policy and its flags, and how each unclear instruction was settled
  (§3);
- the plan;
- per file and per group: pages published and held;
- held reasons and triage categories, with counts;
- QA: pages reviewed (risk-ranked and random), findings confirmed, spot
  masks, quarantines;
- the held pages sent to the cloud model, and at what resolution;
- verify results;
- the leak-rate bound met per group;
- defects;
- batch roots;
- time and model requests used;
- what is left for people.

`report.json` has the same in fixed fields.

## 11. Evaluation

Before the harness runs on anything that will be published, it is measured
on jobs whose answers are known.

- **QA recall.** The 2013 set redacted by the binary from before the fixes
  of `claude/2013-reader-fixes`. Its known leaks are the seven signature
  leaks, including the hanging signatures of Moyale (2) p117 and Tiaty
  p083. The double ID fields of Baringo North are a known layout anomaly.
  The harness must flag every one, through its signals or its sample. False alarms are counted per 1,000 pages reviewed.
- **Policy compiler.** About 50 instructions written to cover §3's table,
  each with the policy or refusal expected.
- **Triage view.** The vision model is asked to read the IDs and phone
  numbers on triage views of pages whose numbers are known. The resolution
  is set where it reads none: no digit right more often than chance.
- **Triage.** The 376 pages held in the final 2013 run, labelled by hand
  (tally sheet, turned, faint, truncated, table not read…). The harness's
  categories, from triage views at that resolution, are scored against the
  labels. So are the pages located at full resolution and then published,
  each reviewed for leaks.
- **Cost.** Model requests and hours per 1,000 pages, for 2013 scans and
  for 2022 Forms 34A.
- **Privacy.** The canary tests of §7, on every build.

## 12. Build plan

| | Builds | Done when |
|---|---|---|
| 1 | The tools of §6 that need no model: `inventory`, `stats`, `stats --risk`, `crop`, `held-view`, `diagnose`, `redact --resume --pages --spot-masks` | the 2013 run's numbers and risk lists come out of `rfpii` alone |
| 2 | `rfpii job` without models: fixed phases, default policy, analytics and verification, gate, report | a job runs from start to report with no model; nothing is released without the gate |
| 3 | The models: policy compiler, orchestrator decisions, visual review, held-page triage and locating; journal; the scrubber and the canary tests | §11's privacy tests pass; a 2013 file runs end to end |
| 4 | Evaluation (§11) and tuning: thresholds, sample sizes, tiling, model choice | QA recall on the known leaks is complete; cost per 1,000 pages is known |

It is built inside `rfpii` (`rust/src/job/`), reusing the Ollama client and
guard of `vlm.rs`: one binary, one privacy guard, and the same tests.

## 13. Decisions and open questions

Decided:
- **No one approves a job's settings.** The policy runs as compiled, and
  unclear points are settled by the rules of §3.
- **The leak-rate bound is 0.5%:** 600 random pages per group with no
  leak (§8).
- **Held pages go to a vision model on Ollama Cloud** (§4.7): a triage view
  for every held page, and full resolution to locate fields where that
  could get a page published.

Open:
- At 50,000 forms, is `--vlm-audit` on every page affordable? If so, the
  harness's own sampling adds a second look rather than the only one.
- Is a group released as soon as it passes, or is the whole job released
  at once?
