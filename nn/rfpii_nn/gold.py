"""The gold set (NEURAL.md §6.1): what every model is measured on and never
trained on. No person labels pages, so it has two parts:

- the **synthetic gold set**: a synthetic dataset made from its own range of
  seeds (by default 1,000,000,000 on), never used for training, with exact
  labels: the measure of truth;
- the **reference set**: a fixed, stratified sample of real pages from
  teacher datasets (strata: template or form and year x how the page was
  captured x published or held), whose labels are the teacher's: the
  measure of agreement with the pipeline, not of truth.

`gold/plan.json` records both. A plan only grows: pages are added, never
taken out, and each change raises its version, so scores stay comparable.
Every training reader excludes both parts through `Exclusion`.

    python3 -m rfpii_nn.gold plan --gold nn/data/gold --reference nn/data/teacher
    python3 -m rfpii_nn.gold make-synthetic --gold nn/data/gold --rfpii rust/target/release/rfpii
    python3 -m rfpii_nn.gold show --gold nn/data/gold
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

from .dataset import Dataset, seed_of

PLAN_SCHEMA = "rfpii/nn-gold-plan/1"
GOLD_SEED = 1_000_000_000
DEFAULT_SIZE = 300
SAMPLE_KEY = "rfpii-gold-1"


def _now():
    t = os.environ.get("SOURCE_DATE_EPOCH")
    when = datetime.datetime.fromtimestamp(int(t), datetime.timezone.utc) if t else datetime.datetime.now(datetime.timezone.utc)
    return when.replace(microsecond=0).isoformat()


def stratum(entry):
    """A page's stratum: template (else form) and year, capture, and
    whether it was published or held."""
    what = entry.get("template") or ("form " + entry["form"] if entry.get("form") else "unknown")
    year = entry.get("year") or "-"
    capture = entry.get("capture") or "-"
    state = "held" if entry.get("held") else "published"
    return "%s|%s|%s|%s" % (what, year, capture, state)


def _rank(entry):
    return hashlib.sha256((SAMPLE_KEY + ":" + entry["image_sha256"]).encode()).hexdigest()


def allocate(sizes, total):
    """Pages per stratum: in proportion to its size, at least one each
    (while `total` allows), by largest remainder; never more than it has."""
    keys = sorted(sizes)
    if not keys or total <= 0:
        return {k: 0 for k in keys}
    if total >= sum(sizes.values()):
        return dict(sizes)
    out = {k: 0 for k in keys}
    # one each first, the largest strata first
    left = total
    for k in sorted(keys, key=lambda k: (-sizes[k], k)):
        if left > 0 and sizes[k] > 0:
            out[k] = 1
            left -= 1
    grand = sum(sizes.values())
    shares = {k: total * sizes[k] / grand for k in keys}
    extra = {k: max(0, int(shares[k]) - out[k]) for k in keys}
    for k in keys:
        take = min(extra[k], sizes[k] - out[k], left)
        out[k] += take
        left -= take
    order = sorted(keys, key=lambda k: (-(shares[k] - int(shares[k])), k))
    while left > 0:
        moved = False
        for k in order:
            if left > 0 and out[k] < sizes[k]:
                out[k] += 1
                left -= 1
                moved = True
        if not moved:
            break
    return out


def sample_reference(datasets, size, keep=()):
    """A deterministic stratified sample of `size` pages from teacher
    datasets (one list of index entries each, with their dataset path),
    keeping every page of `keep` (a plan's pages, by image SHA-256)."""
    pool = {}
    for path, entries in datasets:
        for e in entries:
            pool.setdefault(e["image_sha256"], (path, e))
    kept = [pool[s] for s in keep if s in pool]
    strata = {}
    for path, e in pool.values():
        strata.setdefault(stratum(e), []).append((path, e))
    sizes = {k: len(v) for k, v in strata.items()}
    quota = allocate(sizes, max(size, len(keep)))
    chosen = {e["image_sha256"] for _, e in kept}
    for k in sorted(strata):
        have = sum(1 for _, e in kept if stratum(e) == k)
        for path, e in sorted(strata[k], key=lambda pe: _rank(pe[1])):
            if have >= quota[k]:
                break
            if e["image_sha256"] not in chosen:
                kept.append((path, e))
                chosen.add(e["image_sha256"])
                have += 1
    return [{"dataset": path, "id": e["id"], "image_sha256": e["image_sha256"], "stratum": stratum(e)} for path, e in kept]


def plan_path(gold):
    return os.path.join(gold, "plan.json")


def load_plan(gold):
    with open(plan_path(gold), encoding="utf-8") as f:
        p = json.load(f)
    if p.get("schema") != PLAN_SCHEMA:
        raise ValueError("%s: schema %r" % (plan_path(gold), p.get("schema")))
    return p


def plan(gold, references=(), size=DEFAULT_SIZE, synthetic_seed=GOLD_SEED, synthetic_count=DEFAULT_SIZE,
         templates="templates", profile="mixed"):
    """Write or extend `gold/plan.json`. An existing plan keeps its pages and
    its synthetic seeds; pages are only added."""
    os.makedirs(gold, exist_ok=True)
    old = load_plan(gold) if os.path.exists(plan_path(gold)) else None
    datasets = []
    for r in references:
        ds = Dataset(r)
        if ds.kind != "teacher":
            raise ValueError("%s is a %s dataset: the reference set is drawn from teacher datasets" % (r, ds.kind))
        datasets.append((ds.path, ds.index))
    keep = [p["image_sha256"] for p in old["reference"]["pages"]] if old else []
    pages = sample_reference(datasets, size, keep)
    # pages of the old plan whose datasets weren't given this time stay
    if old:
        have = {p["image_sha256"] for p in pages}
        pages = [p for p in old["reference"]["pages"] if p["image_sha256"] not in have] + pages
    synthetic = old["synthetic"] if old else {
        "seeds": [synthetic_seed, synthetic_seed + synthetic_count],
        "count": synthetic_count,
        "profile": profile,
        "dataset": "synthetic",
        "command": "rfpii dataset synth --templates %s --out %s --seed %d --count %d --profile %s"
                   % (templates, os.path.join(gold, "synthetic"), synthetic_seed, synthetic_count, profile),
    }
    if old and (synthetic_seed, synthetic_count) != (old["synthetic"]["seeds"][0], old["synthetic"]["count"]):
        print("note: the plan keeps its synthetic seeds %s" % old["synthetic"]["seeds"], file=sys.stderr)
    pages.sort(key=lambda p: (p["stratum"], p["image_sha256"]))
    strata = {}
    for p in pages:
        strata[p["stratum"]] = strata.get(p["stratum"], 0) + 1
    added = len(pages) - (len(old["reference"]["pages"]) if old else 0)
    version = (old["version"] + (1 if added else 0)) if old else 1
    history = (old.get("history") or []) if old else []
    if not old or added:
        history.append({"version": version, "at": _now(), "reference_pages_added": added,
                        "reference_datasets": [d for d, _ in datasets]})
    p = {
        "schema": PLAN_SCHEMA,
        "version": version,
        "created": old["created"] if old else _now(),
        "rule": "never trained on; pages are only added; each change raises the version",
        "synthetic": synthetic,
        "reference": {"size": len(pages), "target": size, "labels": "the teacher's (agreement, not truth)",
                      "strata": dict(sorted(strata.items())), "pages": pages},
        "history": history,
    }
    with open(plan_path(gold), "w", encoding="utf-8") as f:
        json.dump(p, f, indent=1)
        f.write("\n")
    with open(os.path.join(gold, "VERSION"), "w", encoding="utf-8") as f:
        f.write("%d\n" % version)
    return p


def make_synthetic(gold, rfpii, templates="templates", workers=None):
    """Make the synthetic gold set the plan names, with the rfpii binary."""
    p = load_plan(gold)
    s = p["synthetic"]
    out = os.path.join(gold, s["dataset"])
    cmd = [rfpii, "dataset", "synth", "--templates", templates, "--out", out,
           "--seed", str(s["seeds"][0]), "--count", str(s["count"]), "--profile", s.get("profile", "mixed")]
    if os.path.exists(os.path.join(out, "dataset.json")):
        cmd.append("--append")
    if workers:
        cmd += ["--workers", str(workers)]
    subprocess.run(cmd, check=True)
    return out


class Exclusion:
    """What the gold set takes out of training: the reference pages (by
    image SHA-256, in whatever dataset they turn up) and every synthetic
    page whose seed is in the gold range."""

    def __init__(self, image_sha256=(), seed_ranges=()):
        self.shas = set(image_sha256)
        self.ranges = [tuple(r) for r in seed_ranges]

    @classmethod
    def load(cls, gold):
        p = load_plan(gold)
        return cls((x["image_sha256"] for x in p["reference"]["pages"]), [p["synthetic"]["seeds"]])

    def excludes(self, entry):
        if entry.get("image_sha256") in self.shas:
            return True
        seed = seed_of(entry)
        return seed is not None and any(lo <= seed < hi for lo, hi in self.ranges)


class _Nothing(Exclusion):
    def excludes(self, entry):
        return False


# for reading data outside training (tests, looking at a dataset)
NOTHING = _Nothing()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m rfpii_nn.gold", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan", help="write or extend gold/plan.json")
    a.add_argument("--gold", required=True)
    a.add_argument("--reference", action="append", default=[], help="a teacher dataset (repeatable)")
    a.add_argument("--size", type=int, default=DEFAULT_SIZE)
    a.add_argument("--synthetic-seed", type=int, default=GOLD_SEED)
    a.add_argument("--synthetic-count", type=int, default=DEFAULT_SIZE)
    a.add_argument("--templates", default="templates")
    m = sub.add_parser("make-synthetic", help="make the synthetic gold set with the rfpii binary")
    m.add_argument("--gold", required=True)
    m.add_argument("--rfpii", default="rust/target/release/rfpii")
    m.add_argument("--templates", default="templates")
    m.add_argument("--workers", type=int)
    s = sub.add_parser("show", help="print the plan's summary")
    s.add_argument("--gold", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        p = plan(args.gold, args.reference, args.size, args.synthetic_seed, args.synthetic_count, args.templates)
    elif args.cmd == "make-synthetic":
        print(make_synthetic(args.gold, args.rfpii, args.templates, args.workers))
        return 0
    else:
        p = load_plan(args.gold)
    print("gold plan v%d: synthetic seeds %s (%d pages); reference set %d pages in %d strata"
          % (p["version"], p["synthetic"]["seeds"], p["synthetic"]["count"], p["reference"]["size"], len(p["reference"]["strata"])))
    for k, n in p["reference"]["strata"].items():
        print("  %4d  %s" % (n, k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
