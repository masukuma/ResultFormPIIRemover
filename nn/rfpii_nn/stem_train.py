"""The rule stem trained by gradient descent: the baseline that says what
training alone reaches in the same architecture (NEURAL.md §11.2: if the
written stem misses its test, the stem is trained).

Adam on random crops of the training pages of the ink sets, binary
cross-entropy plus a soft IoU term per output, against the pipeline's
masks. Crops are padded by their edges as the stem pads a page, so the loss
is taken only on each crop's middle (the stem sees about 60 px around a
pixel).

The weights are trained through a fixed rescaling, so that every weight
moves on its own scale (Adam's steps are about the same size for every
parameter, and this stem's weights range from 1/255 to hundreds): a
layer's weight is ``theta * out_scale / (in_scale * taps)`` and its bias
``theta_b * out_scale``, with 255 for channels of grey or colour levels, 1
for the others, and an output scale of 10 logits for a sigmoid layer. The
architecture and what it can compute are unchanged. ``theta`` starts from
PyTorch's default convolution initialisation; biases start at zero.

``python3 -m rfpii_nn.stem_train --ink A --ink B --out nn/stems/trained.npz``
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import inkset, stem

LEVELS = {"R", "G", "B", "gray", "blur", "mean_h", "mean", "L", "a", "b", "pL_h", "pa_h", "pb_h", "pL", "pa", "pb"} | {f"h{i}" for i in range(stem.HIDDEN)}
LOGITS = 10.0


def scales(l):
    """(weight scale (out, in, 1, 1) or (out, 1, 1, 1), bias scale (out,)) of a layer."""
    s_in = np.array([255.0 if c in LEVELS else 1.0 for c in l.inputs], np.float32)
    s_out = np.array([255.0 if c in LEVELS else (LOGITS if l.act == "sigmoid" else 1.0) for c in l.outputs], np.float32)
    taps = l.kernel[0] * l.kernel[1]
    if l.depthwise:
        ws = (s_out / (s_in * taps))[:, None, None, None]
    else:
        ws = (s_out[:, None] / (s_in[None, :] * taps))[:, :, None, None]
    return ws.astype(np.float32), s_out


def init_theta(seed):
    """PyTorch's default convolution initialisation, for theta."""
    import torch

    g = torch.Generator().manual_seed(seed)
    th = {}
    for l in stem.LAYERS:
        fan_in = (1 if l.depthwise else len(l.inputs)) * l.kernel[0] * l.kernel[1]
        bound = 1.0 / np.sqrt(fan_in)
        th[l.name + ".w"] = ((torch.rand(l.w_shape, generator=g) * 2 - 1) * bound).numpy() * fan_in
        th[l.name + ".b"] = np.zeros(l.b_shape, np.float32)
    return th


def scaled_module(theta):
    import torch

    class Scaled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.ParameterDict()
            self.ws, self.bs = {}, {}
            for l in stem.LAYERS:
                ws, bs = scales(l)
                self.ws[l.name], self.bs[l.name] = torch.from_numpy(ws), torch.from_numpy(bs)
                self.theta[l.name + "__w"] = torch.nn.Parameter(torch.from_numpy(np.asarray(theta[l.name + ".w"], np.float32)))
                self.theta[l.name + "__b"] = torch.nn.Parameter(torch.from_numpy(np.asarray(theta[l.name + ".b"], np.float32)))

        def params(self):
            p = {}
            for l in stem.LAYERS:
                p[l.name + ".w"] = self.theta[l.name + "__w"] * self.ws[l.name]
                p[l.name + ".b"] = self.theta[l.name + "__b"] * self.bs[l.name]
            return p

        def run(self, x, keep=(), logits=False):
            return stem.torch_run(self.params(), x, keep, logits)

        def weights(self):
            return {k: v.detach().cpu().numpy().copy() for k, v in self.params().items()}

    return Scaled()


def load_pages(sets, side):
    imgs, masks = [], []
    for st, line in inkset.pages_of(sets, side):
        imgs.append(st.image(line))
        masks.append(st.mask(line))
    return imgs, masks


def batch(imgs, masks, rng, n, size):
    xs, ys = [], []
    for _ in range(n):
        k = int(rng.integers(len(imgs)))
        h, w = masks[k].shape
        y0, x0 = int(rng.integers(0, max(1, h - size))), int(rng.integers(0, max(1, w - size)))
        xs.append(imgs[k][y0:y0 + size, x0:x0 + size])
        m = masks[k][y0:y0 + size, x0:x0 + size]
        ys.append(np.stack([(m & stem.OUTPUT_BITS[o]) != 0 for o in stem.OUTPUTS]))
    return np.stack(xs).transpose(0, 3, 1, 2).astype(np.float32), np.stack(ys).astype(np.float32)


def train(sets, steps, seed=0, size=224, margin=48, bsz=8, lr=0.03, log=None, minutes=None, threads=4):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    imgs, masks = load_pages(sets, "train")
    size = min(size, *(m.shape[0] for m in masks), *(m.shape[1] for m in masks))
    margin = min(margin, size // 4)
    model = scaled_module(init_theta(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    hist = []
    step = 0
    for step in range(1, steps + 1):
        x, y = batch(imgs, masks, rng, bsz, size)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        o = model.run(x, logits=True)
        loss, parts = 0.0, {}
        sl = (slice(None), slice(margin, size - margin), slice(margin, size - margin))
        for k, n in enumerate(stem.OUTPUTS):
            z, t = o[n + ".logit"][sl], y[:, k][sl]
            p = torch.sigmoid(z)
            bce = F.binary_cross_entropy_with_logits(z, t)
            inter = (p * t).sum()
            soft_iou = 1 - (inter + 1) / (p.sum() + t.sum() - inter + 1)
            loss = loss + bce + 0.5 * soft_iou
            parts[n] = round(float(soft_iou.detach()), 4)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            row = {"step": step, "loss": round(loss.item(), 5), "soft_iou_loss": parts, "seconds": round(time.time() - t0, 1)}
            hist.append(row)
            if log:
                print(json.dumps(row), file=log, flush=True)
        if minutes and time.time() - t0 > 60 * minutes:
            break
    return model.weights(), hist, step, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ink", action="append", required=True, help="ink set (rfpii dataset ink); repeat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--minutes", type=float, default=None, help="stop after this long")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    sets = inkset.open_sets(a.ink)
    w, hist, steps, secs = train(sets, a.steps, a.seed, lr=a.lr, log=sys.stdout, minutes=a.minutes, threads=a.threads)
    stem.save(w, a.out)
    print(json.dumps({"out": a.out, "steps": steps, "seconds": round(secs, 1)}))


if __name__ == "__main__":
    main()
