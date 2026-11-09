"""A weight-space programming session (NEURAL.md §5, "How a session runs"):
the local language model writes M3's rule stem in sandbox mode.

The model is given the goal, the pipeline's source and thresholds (the Rust
functions themselves, read from the repository, with OpenCV's documented
definitions of the calls they make), the stem's architecture as the
printer shows it, and a Python sandbox over the weights with ``try_run``
(50 test runs), ``probe``, ``diff``, ``show``, ``forward`` and ``submit``.
Each reply's last ``python`` block runs in the sandbox; what it prints comes
back as the next message. The session ends when the model submits a stem
and says DONE, or at the end of its time or turns.

Everything is logged in the session folder: ``session.jsonl`` (every
prompt, reply, cell, tool call and test-run score, with token counts and
speeds), ``transcript.md``, ``tries/try_NN.npz`` (each candidate tried),
``written.npz`` (the answer: the stem submitted last, or else the best
tried) and ``summary.json``.

The language model runs on this machine (llama.cpp, through
llama-cpp-python), never a cloud model. Tests drive the same loop with a
stub (``--stub replies.json``).

``python3 -m rfpii_nn.session --ink A --ink B --model M.gguf --out DIR``
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from . import inkset, sandbox, stem, wsp

REPO = Path(__file__).resolve().parents[2]

# --- what the model is told ---------------------------------------------------------------

SOURCES = [
    ("rust/src/stamps.rs", ["dark_ink", "color_ink"]),
    ("rust/src/util.rs", ["gray", "gaussian", "adaptive_inv", "morph", "ones_kernel", "rect_kernel"]),
    ("rust/src/masking.rs", ["rule_lines"]),
    ("rust/src/dataset/ink.rs", ["rule_len", "ink_bits"]),
]

OPENCV = """\
OpenCV's documented behaviour for the calls above (8-bit images; every result is rounded to a whole number 0-255):
- cvtColor(BGR2GRAY): Y = 0.299 R + 0.587 G + 0.114 B.
- GaussianBlur, 3x3, sigma 0: the kernel is [0.25, 0.5, 0.25] along each axis.
- adaptiveThreshold(maxValue 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, blockSize 25, C 12): T(x, y) is the mean of
  the 25x25 window centred on the pixel (edge pixels repeated beyond the image), rounded; dst = 255 where
  src(x, y) <= T(x, y) - 12, else 0.
- cvtColor(BGR2Lab), 8-bit: R, G, B scaled to 0-1 and made linear (sRGB: c <= 0.04045 ? c/12.92 : ((c+0.055)/1.055)^2.4);
  X = 0.412453 R + 0.357580 G + 0.180423 B, Y = 0.212671 R + 0.715160 G + 0.072169 B, Z = 0.019334 R + 0.119193 G + 0.950227 B;
  X /= 0.950456, Z /= 1.088754; f(t) = t^(1/3) if t > 0.008856 else 7.787 t + 16/116;
  L = 116 f(Y) - 16 (903.3 Y if Y <= 0.008856), a = 500 (f(X) - f(Y)), b = 200 (f(Y) - f(Z));
  stored as L * 255/100, a + 128, b + 128.
- medianBlur(ksize 31): each channel on its own, the median of the 31x31 window centred on the pixel (edges repeated).
- morphologyEx(MORPH_OPEN, kernel): erode, then dilate, with the same kernel and the same anchor, the kernel's centre
  (column kw/2, row kh/2, integer division). erode: dst(x, y) = min over the kernel's cells (i, j) of
  src(x + j - kw/2, y + i - kh/2); dilate: the same with max. Beyond the image, erosion sees ink and dilation none.
  So for a 2x2 kernel the window is the pixel and those above, to the left and above-left, in both steps; for a
  horizontal line 32 wide, columns x-16 .. x+15, in both steps.
- getStructuringElement(MORPH_RECT, (w, h)) and Mat ones (h rows, w columns): every cell 1.
"""

GOAL = """\
You are programming a neural network's weights directly (weight-space programming), in a Python sandbox.

THE PART: M3's rule stem, {n} weights in {nl} small convolution layers (below). From a colour page (R, G, B, 0-255)
it computes three outputs, each a value per pixel; a pixel is marked where the output is above 0.5:
- stamp: the pipeline's stamp-ink mask, stamps::color_ink;
- dark: the pipeline's dark-ink mask, stamps::dark_ink;
- rule: the pipeline's rule mask, masking::rule_lines(dark ink, 32, 32): pages are 1,280 px on the long side, so
  rule_len is 32.
The median filter and the local thresholds can't be reproduced exactly by convolutions: approximate them.

THE GOAL: write weights so that each output agrees with the pipeline's own mask, pixel by pixel, on the training
pages. The test passes when every output agrees on at least 99% of the pixels. Ink is a small share of a page
(dark about 14%, rule 7%, stamp 0.5%), so 99% agreement alone says little: also make the IoU of each output's
ink (marked pixels shared with the pipeline / marked by either) as high as you can.

THE SANDBOX (numpy only; no files outside your folder, no network, no other programs). Defined for you:
- W: dict of the current weights, numpy float32 arrays: W["<layer>.w"], W["<layer>.b"]. It starts {start}.
  W is kept from one reply to the next; it is your candidate.
- SPEC: the layers as a list of dicts (name, inputs, outputs, kernel, act, depthwise, what, w_shape, b_shape);
  CHANNELS, OUTPUTS: channel names.
- show(W, layers=None) -> str: the printer (values to 4 significant figures, with each channel's meaning).
- forward(W, rgb, keep=None) -> {{channel: (H, W) array}}: run the stem (the numpy reference) on an image you
  make yourself, rgb (H, W, 3) 0-255, e.g. a 64x64 test card; keep="all" returns every channel.
- try_run(W, note="") -> dict: ONE TEST RUN, scored on a copy of W against the pipeline's masks on the {pages}
  training pages: each output's agreement, iou, recall, precision, IoU by group of pages, and runs left. You have
  {tries} test runs in all; each takes about {try_s} s. Use them to measure, not to search blindly.
- probe(W, pages, channels, where=None, samples=0) -> list: numbers on chosen training pages (not counted):
  each channel's quantiles, mean and share above 0.5; the outputs' counts (tp, fp, fn, tn) against the pipeline;
  with where="fn:stamp" (tp, fp, fn, tn, pipe, nopipe, stem or nostem, ':', an output) and samples (at most
  256), the channels' values at that many such pixels, picked at random, without their positions. You never see a
  page: only numbers. Pages: {listing}.
- diff(A, B) -> str: which weights differ; blank() -> a stem of zeros; budget() -> test runs left.
- submit(W, reasons): hand in your stem with your reasons. The last one submitted is your answer.

HOW TO REPLY: think briefly (don't restate the architecture or the source), then write ONE ```python block of at
most about 60 lines. It runs in the sandbox; you get back what it printed (print what you need). A reply is cut off
after {answer} tokens. Each reply should make progress: write or fix a few layers, check them with forward() on small
test images, measure with try_run() when a whole output is ready. Arrays of the right size are reshaped to their
layer's shape. When you are done, or when your test runs are used up, call submit(W, "your reasons") and write DONE.

The convolution, in every layer: out[o, y, x] = b[o] + sum over inputs c and kernel cells (i, j) of
w[o, c, i, j] * in[c, y + i - kh//2, x + j - kw//2], the image padded by repeating its edge pixels; a depthwise
layer's output o reads only input o (w has shape (C, 1, kh, kw)). Activations: none, relu, sigmoid
(1 / (1 + exp(-z))).
"""


def rust_fn(path, name):
    """The text of a Rust function (its doc comment included), by name."""
    src = (REPO / path).read_text()
    m = re.search(rf"^(?:///[^\n]*\n)*pub fn {name}\b", src, re.M)
    if not m:
        raise KeyError(f"{path}: fn {name}")
    i = src.index("{", m.end())
    depth = 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[m.start():j + 1]
    raise ValueError(f"{path}: fn {name} not closed")


def source_text():
    parts = []
    for path, names in SOURCES:
        parts.append(f"// {path}\n" + "\n\n".join(rust_fn(path, n) for n in names))
    return "\n\n".join(parts)


def listing_text(pages):
    groups, cur = [], None
    for p in pages.listing():
        if cur and cur[0] == p["group"] and cur[2] == p["page"] - 1:
            cur[2] = p["page"]
        else:
            cur = [p["group"], p["page"], p["page"]]
            groups.append(cur)
    return "; ".join(f"{a}-{b} {g}" if a != b else f"{a} {g}" for g, a, b in groups)


def system_prompt(pages, tries, start_name, try_s, answer=1800):
    goal = GOAL.format(n=stem.n_weights(), nl=len(stem.LAYERS), start=start_name, pages=len(pages), tries=tries,
                       try_s=try_s, listing=listing_text(pages), answer=answer)
    arch = wsp.text(stem.zeros()) if start_name == "with every weight 0" else None
    return (goal + "\nTHE PIPELINE'S SOURCE (Rust; the masks you are scored against are ink_bits of each page):\n```rust\n"
            + source_text() + "\n```\n\n" + OPENCV + "\nTHE ARCHITECTURE, as the printer shows it"
            + (" (every weight 0 now)" if arch else "") + ":\n" + (arch or "") + "\n")


# --- the language model --------------------------------------------------------------------


class LlamaBackend:
    """The local model through llama-cpp-python, driven token by token so
    that the context only ever grows (the model's recurrent state can't be
    rolled back), with a budget for thinking."""

    def __init__(self, model, n_ctx=65536, threads=4, seed=0, temp=0.6, top_p=0.95, top_k=20):
        from llama_cpp import Llama

        self.llm = Llama(model_path=str(model), n_ctx=n_ctx, n_threads=threads, n_threads_batch=threads, n_batch=512,
                         seed=seed, verbose=False)
        self.n_ctx = n_ctx
        self.rng = np.random.default_rng(seed)
        self.temp, self.top_p, self.top_k = temp, top_p, top_k
        self.stop = {self.llm.token_eos(), self.tok("<|im_end|>")[0]}
        self.think_end = self.tok("</think>")[0]
        self.name = Path(model).name

    def tok(self, text):
        return self.llm.tokenize(text.encode(), add_bos=False, special=True)

    @property
    def used(self):
        return self.llm.n_tokens

    def reset(self):
        self.llm.reset()  # clears the KV cache and the recurrent state

    def append(self, text):
        t0 = time.time()
        toks = self.tok(text)
        if self.llm.n_tokens + len(toks) >= self.n_ctx - 8:
            raise OverflowError("context full")
        self.llm.eval(toks)
        return {"tokens": len(toks), "seconds": round(time.time() - t0, 2)}

    def _sample(self):
        import ctypes

        n_vocab = self.llm.n_vocab()
        ptr = self.llm._ctx.get_logits_ith(-1)
        logits = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(n_vocab,)).astype(np.float64)
        top = np.argpartition(-logits, self.top_k)[: self.top_k]
        z = logits[top] / self.temp
        p = np.exp(z - z.max())
        p /= p.sum()
        order = np.argsort(-p)
        keep = order[: int(np.searchsorted(np.cumsum(p[order]), self.top_p) + 1)]
        q = p[keep] / p[keep].sum()
        return int(top[keep[self.rng.choice(len(keep), p=q)]])

    def generate(self, max_think, max_answer, thinking=True):
        t0 = time.time()
        out, think_n, answer_n, forced, truncated = [], 0, 0, False, False
        in_think = thinking
        while True:
            if self.llm.n_tokens >= self.n_ctx - 8:
                truncated = True
                break
            t = self._sample()
            if t in self.stop:
                break
            self.llm.eval([t])
            out.append(t)
            if in_think:
                think_n += 1
                if t == self.think_end:
                    in_think = False
                elif think_n >= max_think:
                    extra = self.tok("\n\nI must stop thinking and write the code now.\n</think>\n\n")
                    self.llm.eval(extra)
                    out += extra
                    in_think, forced = False, True
            else:
                answer_n += 1
                if answer_n >= max_answer:
                    truncated = True
                    break
        text = self.llm.detokenize(out, special=True).decode("utf-8", "replace")
        dt = time.time() - t0
        return text, {"tokens": len(out), "think_tokens": think_n, "answer_tokens": answer_n, "seconds": round(dt, 1),
                      "tok_per_s": round(len(out) / dt, 2) if dt else None, "think_forced": forced, "truncated": truncated}


class StubBackend:
    """A stand-in for tests: returns the replies it is given, in turn
    (a list of strings, or a function of the turn number and the last
    message)."""

    def __init__(self, replies, n_ctx=1 << 20):
        self.replies = replies
        self.n_ctx = n_ctx
        self.seen = []
        self.turn = 0
        self.used = 0
        self.name = "stub"

    def reset(self):
        self.used = 0

    def append(self, text):
        self.seen.append(text)
        self.used += len(text) // 4
        return {"tokens": len(text) // 4, "seconds": 0.0}

    def generate(self, max_think, max_answer, thinking=True):
        r = self.replies(self.turn, self.seen[-1]) if callable(self.replies) else (self.replies[self.turn] if self.turn < len(self.replies) else "DONE")
        self.turn += 1
        self.used += len(r) // 4
        return r, {"tokens": len(r) // 4, "think_tokens": 0, "answer_tokens": len(r) // 4, "seconds": 0.0, "tok_per_s": None,
                   "think_forced": False, "truncated": False}


# --- the session ---------------------------------------------------------------------------

CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def code_of(reply):
    """The last python block of the reply's answer (after its thinking)."""
    answer = reply.split("</think>")[-1]
    blocks = CODE.findall(answer)
    return blocks[-1] if blocks else None


def cut_code(reply):
    """A python block a reply was cut off in: its lines up to the last that
    still make a whole program, and how many lines were dropped; or None."""
    answer = reply.split("</think>")[-1]
    m = list(re.finditer(r"```(?:python|py)[ \t]*\n", answer))
    if not m or CODE.search(answer[m[-1].start():]):
        return None
    lines = answer[m[-1].end():].split("\n")
    for drop in range(0, min(len(lines), 40)):
        code = "\n".join(lines[: len(lines) - drop])
        try:
            compile(code, "<cell>", "exec")
        except SyntaxError:
            continue
        if code.strip():
            return code, drop
    return None


def done_of(reply):
    return bool(re.search(r"\bDONE\b", reply.split("</think>")[-1].replace("```", "\n")))


class Session:
    def __init__(self, backend, sets, out, tries=50, hours=4.0, max_turns=120, start=None, think=1200, answer=1800,
                 cell_timeout=900.0, isolation="auto", compact_at=0.8):
        self.backend = backend
        self.out = Path(out)
        (self.out / "tries").mkdir(parents=True, exist_ok=True)
        self.log_f = open(self.out / "session.jsonl", "a")
        self.md = open(self.out / "transcript.md", "a")
        self.pages = wsp.Pages(sets, "train")
        self.budget = wsp.Budget(self.pages, tries, log=self.log)
        self.hours, self.max_turns, self.think, self.answer = hours, max_turns, think, answer
        self.compact_at = compact_at
        self.start_w = stem.clean(start) if start is not None else stem.zeros()
        self.start_name = "with every weight 0" if start is None else "from the weights given"
        self.submitted = None
        self.probes = 0
        self.sandbox = sandbox.Sandbox(self.start_w, {"try": self._try, "probe": self._probe, "submit": self._submit,
                                                      "budget": lambda a, w: self.budget.left},
                                       timeout=cell_timeout, isolation=isolation)

    def log(self, ev):
        ev = {"t": round(time.time() - self.t0, 1) if hasattr(self, "t0") else 0.0, **ev}
        self.log_f.write(json.dumps(ev, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        self.log_f.flush()

    def note(self, title, text):
        self.md.write(f"\n## {title}\n\n{text}\n")
        self.md.flush()

    # tool handlers (called from the sandbox's requests)
    def _try(self, args, w):
        if w is None:
            raise ValueError("try_run needs weights")
        r = self.budget.run(w, args.get("note", ""))
        stem.save(w, self.out / "tries" / f"try_{r['try']:02d}.npz")
        self.note(f"test run {r['try']}", "```json\n" + json.dumps(r, indent=1) + "\n```")
        return r

    def _probe(self, args, w):
        pages = list(args.get("pages") or [])[:8]
        self.probes += 1
        if self.probes > 400:
            raise RuntimeError("no more probes this session")
        r = wsp.probe(w, self.pages, pages, args.get("channels") or [], args.get("where"), args.get("samples", 0), seed=self.probes)
        self.log({"event": "probe", "args": args, "pages": pages})
        return r

    def _submit(self, args, w):
        self.submitted = (w, str(args.get("reasons", ""))[:4000])
        stem.save(w, self.out / "submitted.npz")
        self.log({"event": "submit", "reasons": self.submitted[1], "tries_used": self.budget.used})
        return {"ok": True, "tries_used": self.budget.used}

    def _feedback(self, r, turn):
        parts = []
        if r.get("out"):
            parts.append("Printed:\n" + r["out"].rstrip())
        else:
            parts.append("(nothing printed)")
        if r.get("error"):
            parts.append("Error:\n" + r["error"].rstrip())
        if r.get("w_ok") not in (True, None):
            parts.append(str(r["w_ok"]))
        b = self.budget.best()
        best = ""
        if b:
            best = " Best test run so far: #{} ({}).".format(
                b["try"], ", ".join(f"{n} {b[n]['agreement']*100:.2f}% / IoU {b[n]['iou']:.3f}" for n in stem.OUTPUTS))
        left_h = self.hours - (time.time() - self.t0) / 3600
        parts.append(f"[Turn {turn}. Test runs left: {self.budget.left} of {self.budget.total}.{best} "
                     f"Time left: {max(left_h, 0) * 60:.0f} min.{' Submitted: yes.' if self.submitted else ''}]")
        if self.budget.left == 0 and not self.submitted:
            parts.append("No test runs are left: call submit(W, reasons) with your best stem now, then write DONE.")
        elif left_h < 0.15 and not self.submitted:
            parts.append("Time is nearly up: call submit(W, reasons) with your best stem now, then write DONE.")
        return "\n\n".join(parts)

    def _summary_message(self):
        """After the context is renewed: where the session stands."""
        rows = [f"#{h['try']}: " + ", ".join(f"{n} {h[n]['agreement']*100:.2f}%/{h[n]['iou']:.3f}" for n in stem.OUTPUTS)
                + (f"  ({h['note']})" if h["note"] else "") for h in self.budget.history]
        return ("The conversation so far was cleared to free memory; the sandbox kept its state (W and your variables). "
                "Test runs so far:\n" + ("\n".join(rows) or "none") + "\n\nW now:\n" + wsp.text(self.sandbox.weights)
                + f"\n\n[Test runs left: {self.budget.left}. Submitted: {'yes' if self.submitted else 'no'}.]")

    def run(self):
        self.t0 = time.time()
        t_try = max(1, round(0.45 * len(self.pages)))
        sysmsg = system_prompt(self.pages, self.budget.total, self.start_name, t_try, self.answer)
        self.log({"event": "start", "model": self.backend.name, "tries": self.budget.total, "hours": self.hours,
                  "pages": len(self.pages), "weights": stem.n_weights(), "isolation": self.sandbox.isolation,
                  "think": self.think, "answer": self.answer})
        self.log({"event": "prompt", "role": "system", "text": sysmsg})
        self.note("system", sysmsg)
        first = "Begin. Write the stem's weights."
        ps = self.backend.append(f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n{first}<|im_end|>\n<|im_start|>assistant\n<think>\n")
        self.log({"event": "prompt_eval", **ps})
        turn, reason = 0, "max turns"
        stats = {"prompt_tokens": ps["tokens"], "prompt_seconds": ps["seconds"], "gen_tokens": 0, "gen_seconds": 0.0, "renewals": 0}
        try:
            while turn < self.max_turns:
                turn += 1
                if (time.time() - self.t0) / 3600 > self.hours:
                    reason = "time"
                    break
                reply, g = self.backend.generate(self.think, self.answer)
                stats["gen_tokens"] += g["tokens"]
                stats["gen_seconds"] += g["seconds"]
                self.log({"event": "reply", "turn": turn, "text": reply, **g})
                self.note(f"turn {turn}: reply ({g['tokens']} tokens, {g['seconds']} s)", reply)
                code, cut = code_of(reply), None
                if code is None and g.get("truncated"):
                    cut = cut_code(reply)
                    if cut:
                        code = cut[0]
                if code:
                    r = self.sandbox.run(code)
                    self.log({"event": "cell", "turn": turn, "code": code, "cut": cut[1] if cut else None, **r})
                    fb = self._feedback(r, turn)
                    if cut:
                        fb = (f"Your reply was cut off at {self.answer} tokens, inside its code: the first "
                              f"{len(code.splitlines())} lines ran (the {cut[1]} after them did not). Write less per reply.\n\n" + fb)
                elif g.get("truncated"):
                    fb = self._feedback({"out": "", "error": f"Your reply was cut off at {self.answer} tokens before any code "
                                                              "ran. Write shorter replies: a few layers at a time."}, turn)
                else:
                    fb = self._feedback({"out": "", "error": "No ```python block in your reply: write one."}, turn)
                if done_of(reply) and self.submitted:
                    reason = "done"
                    break
                if self.budget.left == 0 and self.submitted:
                    reason = "test runs used up"
                    break
                self.note(f"turn {turn}: result", fb)
                msg = f"<|im_end|>\n<|im_start|>user\n{fb}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                if self.backend.used + len(msg) // 2 + self.think + self.answer > self.compact_at * self.backend.n_ctx:
                    self.backend.reset()
                    stats["renewals"] += 1
                    msg = (f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n{self._summary_message()}\n\n{fb}"
                           "<|im_end|>\n<|im_start|>assistant\n<think>\n")
                    self.log({"event": "renew_context", "turn": turn})
                p = self.backend.append(msg)
                stats["prompt_tokens"] += p["tokens"]
                stats["prompt_seconds"] += p["seconds"]
        finally:
            self.sandbox.cleanup()
        return self._finish(turn, reason, stats)

    def _finish(self, turns, reason, stats):
        if self.submitted:
            w, how = self.submitted[0], "submitted"
        elif self.budget.history:
            w, how = self.budget.best()["weights"], "best test run (nothing was submitted)"
        else:
            w, how = self.sandbox.weights, "the sandbox's last W (no test run, nothing submitted)"
        stem.save(w, self.out / "written.npz")
        secs = time.time() - self.t0
        summary = {
            "model": self.backend.name, "end": reason, "turns": turns, "seconds": round(secs, 1),
            "tries_used": self.budget.used, "tries_total": self.budget.total, "probes": self.probes,
            "answer": how, "reasons": self.submitted[1] if self.submitted else None,
            "weights_changed_from_start": wsp.changed(self.start_w, w),
            "prompt_tokens": stats["prompt_tokens"], "gen_tokens": stats["gen_tokens"],
            "prompt_tok_per_s": round(stats["prompt_tokens"] / stats["prompt_seconds"], 1) if stats["prompt_seconds"] else None,
            "gen_tok_per_s": round(stats["gen_tokens"] / stats["gen_seconds"], 2) if stats["gen_seconds"] else None,
            "context_renewals": stats["renewals"], "isolation": self.sandbox.isolation, "kernel_restarts": self.sandbox.restarts,
            "tries": [{k: v for k, v in h.items() if k != "weights"} for h in self.budget.history],
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        self.log({"event": "end", **{k: v for k, v in summary.items() if k != "tries"}})
        self.note("end", "```json\n" + json.dumps({k: v for k, v in summary.items() if k != "tries"}, indent=1) + "\n```")
        self.log_f.close()
        self.md.close()
        return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m rfpii_nn.session", description=__doc__.splitlines()[0])
    ap.add_argument("--ink", action="append", required=True, help="ink set (rfpii dataset ink); repeat")
    ap.add_argument("--out", required=True, help="the session's folder (log, tries, answer)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="GGUF model file (run locally with llama.cpp)")
    g.add_argument("--stub", help="JSON list of canned replies (tests)")
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--turns", type=int, default=120)
    ap.add_argument("--think", type=int, default=1200, help="thinking tokens a reply may use")
    ap.add_argument("--answer", type=int, default=1800, help="tokens a reply's answer may use")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", help="weights to start from (default: every weight 0)")
    ap.add_argument("--cell-timeout", type=float, default=900.0)
    a = ap.parse_args(argv)
    sets = inkset.open_sets(a.ink)
    backend = LlamaBackend(a.model, n_ctx=a.ctx, threads=a.threads, seed=a.seed) if a.model else StubBackend(json.loads(Path(a.stub).read_text()))
    s = Session(backend, sets, a.out, tries=a.tries, hours=a.hours, max_turns=a.turns, think=a.think, answer=a.answer,
                start=stem.load(a.start) if a.start else None, cell_timeout=a.cell_timeout)
    summary = s.run()
    print(json.dumps({k: v for k, v in summary.items() if k != "tries"}, indent=1))


if __name__ == "__main__":
    main()
