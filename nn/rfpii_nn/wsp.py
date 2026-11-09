"""Weight-space programming tools (NEURAL.md §5, "How a session runs"):
the printer, probe, try and diff, over a programmable part (here M3's rule
stem, ``stem.py``).

- **print**: the part's weights as text, values to 4 significant figures,
  each layer with its name, shape, what it computes, and the meaning of
  each input channel and output beside them (``text``); in sandbox mode
  the same as arrays (``arrays``).
- **probe**: a part's activations on chosen pages, as numbers: each
  channel's distribution, the outputs' confusion with the pipeline's masks,
  and optionally values sampled at chosen kinds of pixel, without their
  positions. Never an image (§10).
- **try**: one scored test run of a candidate, on a copy, on the training
  pages; counted against a budget (50 a session).
- **diff**: which weights two versions differ in, and by how much.

Command line (``python3 -m rfpii_nn.wsp``): ``print``, ``diff``, ``score``
(any stem on any ink sets, training and held-out, outside a session's
budget) and ``export`` (ONNX, checked against the numpy reference).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import inkset, stem

INPUT_DOC = {"R": "red, 0-255", "G": "green, 0-255", "B": "blue, 0-255"}


def g4(v):
    """A number to 4 significant figures."""
    v = float(v)
    if v == 0:
        return "0"
    return f"{v:.4g}"


def channel_doc(name):
    if name in INPUT_DOC:
        return INPUT_DOC[name] + " (input)"
    for l in stem.LAYERS:
        if name in l.outputs:
            if len(l.outputs) > 1 and name not in ("L", "a", "b", "pL", "pa", "pb", "pL_h", "pa_h", "pb_h"):
                return f"unit {l.outputs.index(name)} of {l.name}: {l.what}"
            if len(l.outputs) > 1:
                return f"{name} of {l.name}: {l.what}"
            return l.what
    raise KeyError(name)


def _row(vals):
    return "[" + " ".join(g4(v) for v in vals) + "]"


def layer_text(l, w, b):
    kh, kw = l.kernel
    n = int(np.prod(l.w_shape)) + int(np.prod(l.b_shape))
    kind = "depthwise conv" if l.depthwise else "conv"
    out = [f"## layer {l.name}: {kind} {kh}x{kw}, activation {l.act}, {len(l.inputs)} in -> {len(l.outputs)} out, "
           f"w {tuple(l.w_shape)}, b {tuple(l.b_shape)} ({n} weights)",
           f"computes: {l.what}"]
    if len(l.inputs) <= 6:
        for i, c in enumerate(l.inputs):
            out.append(f"  in[{i}] {c}: {channel_doc(c)}")
    else:
        out.append(f"  in[0..{len(l.inputs) - 1}] {l.inputs[0]}..{l.inputs[-1]}: {channel_doc(l.inputs[0]).split(':', 1)[-1].strip()}")
    if not np.any(w) and not np.any(b):
        out.append("  every weight 0" + ("" if kh == 1 and kw == 1 and not l.depthwise else f" (kernels {kh}x{kw}; row i is offset i-{kh // 2}, column j offset j-{kw // 2})"))
        return "\n".join(out)
    if kh == 1 and kw == 1 and not l.depthwise:
        # a matrix: one row per output, one column per input, then the bias
        out.append("  w[out, in] (rows: outputs; columns: " + ", ".join(l.inputs) + ") | bias")
        for o, name in enumerate(l.outputs):
            out.append(f"  out[{o}] {name}: {_row(w[o, :, 0, 0])} | {g4(b[o])}")
        return "\n".join(out)
    for o, name in enumerate(l.outputs):
        out.append(f"  out[{o}] {name}: bias {g4(b[o])}")
        ins = [l.inputs[o]] if l.depthwise else list(l.inputs)
        for i, c in enumerate(ins):
            k = w[o, 0 if l.depthwise else i]
            label = f"    kernel {name}<-{c} ({kh}x{kw}; row i is offset i-{kh // 2}, column j offset j-{kw // 2})"
            if not np.any(k):
                out.append(label + ": all 0")
            elif kh == 1 or kw == 1:
                out.append(label + ": " + _row(k.ravel()))
            else:
                out.append(label + ":")
                out.extend("      " + _row(r) for r in k)
    return "\n".join(out)


def text(w, layers=None):
    """The printer: every layer (or those named) as text."""
    w = stem.clean(w)
    names = layers or [l.name for l in stem.LAYERS]
    return "\n\n".join(layer_text(stem.BY_NAME[n], w[n + ".w"], w[n + ".b"]) for n in names)


def spec():
    """The architecture as plain data, for the sandbox."""
    return [{"name": l.name, "inputs": list(l.inputs), "outputs": list(l.outputs), "kernel": list(l.kernel),
             "act": l.act, "depthwise": l.depthwise, "what": l.what,
             "w_shape": list(l.w_shape), "b_shape": list(l.b_shape)} for l in stem.LAYERS]


def arrays(w):
    """Sandbox mode: the weights as a dict of numpy arrays (copies)."""
    return {k: v.copy() for k, v in stem.clean(w).items()}


def diff(a, b, top=12):
    """Text: per layer, how many weights differ between a and b, the
    largest change, and the largest changes themselves (old -> new)."""
    a, b = stem.clean(a), stem.clean(b)
    lines, total = [], 0
    for l in stem.LAYERS:
        for s in (".w", ".b"):
            k = l.name + s
            d = b[k] - a[k]
            idx = np.argwhere(d != 0)
            if len(idx) == 0:
                continue
            total += len(idx)
            order = np.argsort(-np.abs(d[tuple(idx.T)]))[:top]
            ch = ", ".join(f"{k}{list(map(int, idx[i]))} {g4(a[k][tuple(idx[i])])} -> {g4(b[k][tuple(idx[i])])}" for i in order)
            lines.append(f"{k}: {len(idx)} of {d.size} changed, largest |change| {g4(np.abs(d).max())}: {ch}")
    head = f"{total} of {stem.n_weights()} weights differ"
    return head + ("\n" + "\n".join(lines) if lines else "")


def changed(a, b):
    a, b = stem.clean(a), stem.clean(b)
    return int(sum(np.count_nonzero(a[k] != b[k]) for k in a))


# --- pages, probe and try ------------------------------------------------------------------


class Pages:
    """The pages a session scores on, held in memory: images and masks."""

    def __init__(self, sets, side="train"):
        self.lines = sorted(inkset.pages_of(sets, side), key=lambda sl: (sl[1].get("group", ""), sl[1]["id"]))
        self.images = [s.image(l) for s, l in self.lines]
        self.masks = [s.mask(l) for s, l in self.lines]

    def __len__(self):
        return len(self.lines)

    def listing(self):
        """What the model may know of each page: its number, group and size."""
        return [{"page": i, "group": l.get("group", ""), "size": [l["width"], l["height"]]} for i, (_, l) in enumerate(self.lines)]


def run_torch(w, rgb, keep=()):
    return stem.forward_torch(w, rgb, keep=keep)


WHERE = ("tp", "fp", "fn", "tn", "pipe", "nopipe", "stem", "nostem")


def _select(cond, pred, bits):
    kind, _, out = cond.partition(":")
    if kind not in WHERE or out not in stem.OUTPUTS:
        raise ValueError(f"where: {cond!r}; use one of {WHERE} then ':' and an output {stem.OUTPUTS}, e.g. 'fn:stamp'")
    p = pred[out] > 0.5
    t = (bits & stem.OUTPUT_BITS[out]) != 0
    return {"tp": p & t, "fp": p & ~t, "fn": ~p & t, "tn": ~p & ~t, "pipe": t, "nopipe": ~t, "stem": p, "nostem": ~p}[kind]


def stats(a):
    a = np.asarray(a, np.float64).ravel()
    q = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    return {"min": g4(a.min()), "p1": g4(q[0]), "p5": g4(q[1]), "p25": g4(q[2]), "p50": g4(q[3]), "p75": g4(q[4]),
            "p95": g4(q[5]), "p99": g4(q[6]), "max": g4(a.max()), "mean": g4(a.mean()), "above_0.5": g4((a > 0.5).mean())}


def probe(w, pages, which, channels, where=None, samples=0, seed=0):
    """Activations of chosen channels on chosen pages, as numbers.

    For each page: each channel's distribution (quantiles, mean, share above
    0.5); the outputs' counts against the pipeline's masks; and, with
    ``where`` (e.g. "fn:stamp": pixels the pipeline marks as stamp ink and
    the stem does not) and ``samples`` (at most 256), the channels' values
    at that many such pixels picked at random, in no order and without
    their positions.
    """
    channels = list(channels)
    known = set(stem.channels())
    bad = [c for c in channels if c not in known]
    if bad:
        raise ValueError(f"unknown channels {bad}; channels are {stem.channels()}")
    samples = max(0, min(int(samples), 256))
    rng = np.random.default_rng(seed)
    out = []
    for i in which:
        if not 0 <= int(i) < len(pages):
            raise ValueError(f"page {i}: pages are 0..{len(pages) - 1}")
        rgb, bits = pages.images[i], pages.masks[i]
        acts = run_torch(w, rgb, keep=[c for c in channels if c not in stem.INPUTS])
        for k, c in enumerate(stem.INPUTS):
            if c in channels:
                acts[c] = rgb[:, :, k].astype(np.float32)
        r = {"page": int(i), "group": pages.lines[i][1].get("group", ""),
             "channels": {c: stats(acts[c]) for c in channels},
             "outputs": inkset.counts(acts, bits)}
        if where:
            sel = _select(where, acts, bits)
            idx = np.flatnonzero(sel.ravel())
            r["where"] = where
            r["count"] = int(len(idx))
            if samples and len(idx):
                pick = rng.choice(idx, size=min(samples, len(idx)), replace=False)
                r["samples"] = {c: [float(g4(v)) for v in acts[c].ravel()[pick]] for c in channels}
        out.append(r)
    return out


class Budget:
    """Test runs: each scores a copy of a candidate on the training pages."""

    def __init__(self, pages, total=50, log=None):
        self.pages = pages
        self.total = total
        self.used = 0
        self.log = log  # callable(event dict)
        self.history = []

    @property
    def left(self):
        return self.total - self.used

    def run(self, w, note=""):
        if self.used >= self.total:
            raise RuntimeError(f"no test runs left (budget {self.total})")
        w = stem.clean(w)  # a copy: the candidate itself is never touched
        t0 = time.time()
        total, groups = {}, {}
        for rgb, bits, (_, line) in zip(self.pages.images, self.pages.masks, self.pages.lines):
            c = inkset.counts(run_torch(w, rgb), bits)
            inkset.add(total, c)
            inkset.add(groups.setdefault(line.get("group", ""), {}), c)
        self.used += 1
        s = inkset.summary(total)
        res = {"try": self.used, "left": self.left, "note": str(note)[:300], "seconds": round(time.time() - t0, 1),
               "passes_99": inkset.passes(s),
               **{n: {k: round(s[n][k], 5) for k in ("agreement", "iou", "recall", "precision")} for n in stem.OUTPUTS},
               "iou_by_group": {g: {n: round(inkset.summary(v)[n]["iou"], 4) for n in stem.OUTPUTS} for g, v in sorted(groups.items())}}
        self.history.append({**res, "weights": w})
        if self.log:
            self.log({"event": "try", **res})
        return res

    def best(self):
        """The try with the highest lowest-agreement, then mean IoU."""
        if not self.history:
            return None
        return max(self.history, key=lambda h: (min(h[n]["agreement"] for n in stem.OUTPUTS) >= 0.99,
                                                sum(h[n]["iou"] for n in stem.OUTPUTS)))


# --- command line --------------------------------------------------------------------------


def score_stem(w, sets, side):
    """Scores of a stem on one side (train or held) of the ink sets: over
    all its pages, and by group (synthetic, 2013-34, 2022-34A, ...)."""
    total, groups = {}, {}
    pages = inkset.pages_of(sets, side)
    for st, line in pages:
        c = inkset.counts(run_torch(w, st.image(line)), st.mask(line))
        inkset.add(total, c)
        inkset.add(groups.setdefault(line.get("group", ""), {}), c)
    s = inkset.summary(total)
    s["pages"] = len(pages)
    s["passes_99"] = inkset.passes(s)
    s["by_group"] = {g: {**inkset.summary(v), "pages": sum(1 for _, l in pages if l.get("group", "") == g)} for g, v in sorted(groups.items())}
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m rfpii_nn.wsp", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("print", help="a stem's weights as text")
    p.add_argument("stem")
    p.add_argument("--layer", action="append")
    p = sub.add_parser("diff", help="which weights two stems differ in")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("score", help="a stem against the pipeline's masks, training and held-out pages")
    p.add_argument("stem")
    p.add_argument("--ink", action="append", required=True)
    p.add_argument("--json", help="write the scores here")
    p = sub.add_parser("export", help="ONNX, checked against the numpy reference on a page")
    p.add_argument("stem")
    p.add_argument("--out", required=True)
    p.add_argument("--ink", help="an ink set: check on its first page")
    a = ap.parse_args(argv)
    if a.cmd == "print":
        print(text(stem.load(a.stem), a.layer))
    elif a.cmd == "diff":
        print(diff(stem.load(a.a), stem.load(a.b)))
    elif a.cmd == "score":
        w = stem.load(a.stem)
        sets = inkset.open_sets(a.ink)
        res = {}
        for side in ("train", "held"):
            s = score_stem(w, sets, side)
            res[side] = s
            print(f"{side}: {s['pages']} pages, every mask at 99% agreement or more: {s['passes_99']}")
            print(inkset.fmt(s))
            for g, v in s["by_group"].items():
                print(f"  {g:10s} ({v['pages']} pages) " + "  ".join(f"{n} {v[n]['agreement']*100:.2f}% / {v[n]['iou']:.3f}" for n in stem.OUTPUTS))
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    elif a.cmd == "export":
        w = stem.load(a.stem)
        stem.export_onnx(w, a.out)
        print(f"{a.out}: written")
        if a.ink:
            st = inkset.InkSet(a.ink)
            rgb = st.image(st.index[0])
            ref = stem.forward_numpy(w, rgb)
            got = stem.forward_onnx(a.out, rgb)
            for n in stem.OUTPUTS:
                d = np.abs(ref[n] - got[n])
                print(f"{n}: largest |numpy - onnx| {d.max():.2e}; masks differ at {int(np.count_nonzero((ref[n] > 0.5) != (got[n] > 0.5)))} of {d.size} pixels")


if __name__ == "__main__":
    main()
