"""M3's rule stem (NEURAL.md §2): a few 1x1 and small convolutions that
approximate, from the colour page, the pipeline's ink rules.

Three outputs, each a probability per pixel (a pixel is marked where it is
above 0.5), scored against the pipeline's own masks (`rfpii dataset ink`):

- ``stamp``: stamp ink, as ``stamps::color_ink`` computes it (Lab, each
  channel's 31x31 median as the paper, da^2 + db^2 > 81 and da > -2 and
  dL < 5, a 2x2 opening);
- ``dark``: dark ink, as ``stamps::dark_ink`` (grey, 3x3 Gaussian, the
  mean over 25x25 minus 12 as the threshold, inverted, a 2x2 opening);
- ``rule``: long thin rules, as ``masking::rule_lines`` over the dark ink
  (an opening with a horizontal or a vertical line of 32 px).

The median and the local thresholds are approximated, not reproduced: a
weighted local average stands in for the median, and steep sigmoids for
the comparisons.

The architecture is fixed (``LAYERS``); its weights are what weight-space
programming writes, what a person writes by hand and what training finds.
Weights are a dict ``{"<layer>.w": array, "<layer>.b": array}``; a full
convolution's ``w`` is (out, in, kh, kw), a depthwise one's (channels, 1,
kh, kw), ``b`` is (out,). Every convolution is a cross-correlation over
the image padded by repeating its edge pixels, with odd kernels centred on
the pixel: ``out[o, y, x] = b[o] + sum_{c, i, j} w[o, c, i, j] *
in[c, y + i - kh//2, x + j - kw//2]``.

Three implementations: ``forward_numpy`` (the reference), ``StemTorch``
(training, quick test runs) and the ONNX export, run by ONNX Runtime; the
tests check they agree.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import numpy as np

SCHEMA = "rfpii/nn-stem/1"
INPUTS = ["R", "G", "B"]
OUTPUTS = ["stamp", "dark", "rule"]
#: mask bits of an ink set (rfpii dataset ink) for each output
OUTPUT_BITS = {"stamp": 4, "dark": 1, "rule": 2}
#: pages are made 1,280 px on the long side, so rules are 32 px (a fortieth)
RULE_LEN = 32
HIDDEN = 16  # hidden units of the colour transform
CHROMA = 16  # half-plane tests on the colour difference


@dataclasses.dataclass(frozen=True)
class Layer:
    name: str
    inputs: tuple
    outputs: tuple
    kernel: tuple  # (kh, kw), odd
    act: str  # "none", "relu", "sigmoid"
    depthwise: bool
    what: str

    @property
    def w_shape(self):
        kh, kw = self.kernel
        if self.depthwise:
            return (len(self.outputs), 1, kh, kw)
        return (len(self.outputs), len(self.inputs), kh, kw)

    @property
    def b_shape(self):
        return (len(self.outputs),)


def _L(name, inputs, outputs, kernel, act, what, depthwise=False):
    return Layer(name, tuple(inputs), tuple(outputs), tuple(kernel), act, depthwise, what)


_H = [f"h{i}" for i in range(HIDDEN)]
_C = [f"c{i}" for i in range(CHROMA)]

LAYERS = [
    # dark ink: stamps::dark_ink
    _L("gray", INPUTS, ["gray"], (1, 1), "none",
       "grey level 0-255 from R, G, B (the pipeline converts the page to grey first)"),
    _L("blur", ["gray"], ["blur"], (3, 3), "none",
       "grey smoothed by the pipeline's 3x3 Gaussian blur"),
    _L("mean_h", ["blur"], ["mean_h"], (1, 25), "none",
       "the 25x25 local mean of blur, first pass: along each row"),
    _L("mean", ["mean_h"], ["mean"], (25, 1), "none",
       "the 25x25 local mean of blur, second pass: down each column"),
    _L("dark_raw", ["blur", "mean"], ["dark_raw"], (1, 1), "sigmoid",
       "1 where blur <= mean - 12: the adaptive threshold, inverted"),
    _L("dark_erode", ["dark_raw"], ["dark_erode"], (3, 3), "sigmoid",
       "first half of the 2x2 opening of dark_raw: erosion"),
    _L("dark", ["dark_erode"], ["dark"], (3, 3), "sigmoid",
       "OUTPUT dark ink: second half of the 2x2 opening, dilation"),
    # rules: masking::rule_lines over the dark ink
    _L("rule_h_erode", ["dark"], ["rule_h_erode"], (1, 33), "sigmoid",
       "first half of the opening of dark by a horizontal line of 32 px: erosion"),
    _L("rule_h", ["rule_h_erode"], ["rule_h"], (1, 33), "sigmoid",
       "horizontal rules: second half of that opening, dilation"),
    _L("rule_v_erode", ["dark"], ["rule_v_erode"], (33, 1), "sigmoid",
       "first half of the opening of dark by a vertical line of 32 px: erosion"),
    _L("rule_v", ["rule_v_erode"], ["rule_v"], (33, 1), "sigmoid",
       "vertical rules: second half of that opening, dilation"),
    _L("rule", ["rule_h", "rule_v"], ["rule"], (1, 1), "sigmoid",
       "OUTPUT rules: rule_h or rule_v"),
    # stamp ink: stamps::color_ink
    _L("lab_hidden", INPUTS, _H, (1, 1), "relu",
       "hidden units of the colour transform from R, G, B to Lab"),
    _L("lab", _H, ["L", "a", "b"], (1, 1), "none",
       "OpenCV's 8-bit Lab of the pixel: L 0-255, a and b 0-255 with 128 for grey"),
    _L("paper_h", ["L", "a", "b"], ["pL_h", "pa_h", "pb_h"], (1, 31), "none",
       "paper colour around the pixel, first pass: along each row (each channel on its own)", depthwise=True),
    _L("paper", ["pL_h", "pa_h", "pb_h"], ["pL", "pa", "pb"], (31, 1), "none",
       "paper colour around the pixel, second pass: down each column; the pipeline takes each channel's 31x31 median",
       depthwise=True),
    _L("chroma", ["L", "a", "b", "pL", "pa", "pb"], _C, (1, 1), "sigmoid",
       "tests on the pixel's difference from the paper (dL = L - pL, da = a - pa, db = b - pb)"),
    _L("stamp_raw", _C, ["stamp_raw"], (1, 1), "sigmoid",
       "1 where da^2 + db^2 > 81 and da > -2 and dL < 5"),
    _L("stamp_erode", ["stamp_raw"], ["stamp_erode"], (3, 3), "sigmoid",
       "first half of the 2x2 opening of stamp_raw: erosion"),
    _L("stamp", ["stamp_erode"], ["stamp"], (3, 3), "sigmoid",
       "OUTPUT stamp ink: second half of the 2x2 opening, dilation"),
]
BY_NAME = {l.name: l for l in LAYERS}


def channels():
    """Every channel name, in the order it is made: inputs first."""
    out = list(INPUTS)
    for l in LAYERS:
        out.extend(l.outputs)
    return out


def n_weights():
    return sum(int(np.prod(l.w_shape)) + int(np.prod(l.b_shape)) for l in LAYERS)


def zeros():
    """A blank stem: every weight 0."""
    w = {}
    for l in LAYERS:
        w[l.name + ".w"] = np.zeros(l.w_shape, np.float32)
        w[l.name + ".b"] = np.zeros(l.b_shape, np.float32)
    return w


def check_weights(w):
    """Problems with a weights dict: missing, unknown, misshapen or not finite."""
    problems = []
    for l in LAYERS:
        for k, shape in ((l.name + ".w", l.w_shape), (l.name + ".b", l.b_shape)):
            if k not in w:
                problems.append(f"{k}: missing")
                continue
            a = np.asarray(w[k])
            if a.size != int(np.prod(shape)):  # any shape of the right size is reshaped
                problems.append(f"{k}: shape {a.shape}, expected {shape}")
            elif not np.all(np.isfinite(a)):
                problems.append(f"{k}: not finite")
    for k in w:
        if k.rsplit(".", 1)[0] not in BY_NAME or not k.endswith((".w", ".b")):
            problems.append(f"{k}: not a weight of the stem")
    return problems


def clean(w):
    """The weights as new float32 arrays of the right shapes (copies, never
    the caller's arrays; an array of the right size in another shape is
    reshaped, in C order); raises on problems."""
    p = check_weights(w)
    if p:
        raise ValueError("; ".join(p))
    return {k: np.array(w[k], np.float32).reshape(s) for k, s in shapes().items()}


def shapes():
    s = {}
    for l in LAYERS:
        s[l.name + ".w"] = l.w_shape
        s[l.name + ".b"] = l.b_shape
    return s


def save(w, path):
    w = clean(w)
    path = Path(path)
    if path.suffix == ".json":
        path.write_text(json.dumps({"schema": SCHEMA, "weights": {k: v.tolist() for k, v in w.items()}}) + "\n")
    else:
        np.savez_compressed(path, **w)


def load(path):
    path = Path(path)
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        if d.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} file")
        return clean({k: np.array(v, np.float32) for k, v in d["weights"].items()})
    with np.load(path, allow_pickle=False) as z:
        return clean({k: z[k] for k in z.files})


def to_bytes(w):
    b = io.BytesIO()
    np.savez_compressed(b, **clean(w))
    return b.getvalue()


# --- the numpy reference ------------------------------------------------------------------


def _sigmoid(z):
    return (1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))).astype(np.float32)


def _act(z, act):
    if act == "relu":
        return np.maximum(z, 0.0)
    if act == "sigmoid":
        return _sigmoid(z)
    return z


def conv_numpy(x, w, b, depthwise=False):
    """x (C, H, W) float32; edge-padded cross-correlation with odd kernels."""
    c, h, wd = x.shape
    o, _, kh, kw = w.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)), mode="edge")
    out = np.zeros((o, h, wd), np.float32)
    for i in range(kh):
        for j in range(kw):
            patch = xp[:, i:i + h, j:j + wd]
            if depthwise:
                k = w[:, 0, i, j]
                if np.any(k):
                    out += k[:, None, None] * patch
            else:
                k = w[:, :, i, j]
                if np.any(k):
                    out += np.tensordot(k, patch, axes=(1, 0))
    out += np.asarray(b, np.float32)[:, None, None]
    return out


def forward_numpy(w, rgb, keep=None):
    """The stem on one page: rgb (H, W, 3) uint8 or float, 0-255.

    Returns {channel: (H, W) float32} for the outputs, and for every channel
    named in ``keep`` (``keep="all"`` keeps every one).
    """
    w = clean(w)
    x = np.asarray(rgb, np.float32)
    acts = {n: x[:, :, i] for i, n in enumerate(INPUTS)}
    for l in LAYERS:
        inp = np.stack([acts[c] for c in l.inputs])
        z = conv_numpy(inp, w[l.name + ".w"], w[l.name + ".b"], l.depthwise)
        y = _act(z, l.act)
        for k, n in enumerate(l.outputs):
            acts[n] = y[k]
    want = set(OUTPUTS)
    if keep == "all":
        want = set(acts)
    elif keep:
        want |= set(keep)
    return {n: acts[n] for n in acts if n in want}


# --- PyTorch and ONNX ---------------------------------------------------------------------


def torch_run(params, x, keep=(), logits=False):
    """The stem in PyTorch, functional: params {"<layer>.w"/".b": tensor},
    x (N, 3, H, W) float, 0-255. Returns {channel: (N, H, W)} for the
    outputs and the channels in ``keep``; with logits, also
    "<output>.logit" before the outputs' sigmoid."""
    import torch
    import torch.nn.functional as F

    acts = {n: x[:, i] for i, n in enumerate(INPUTS)}
    out = {}
    for l in LAYERS:
        inp = torch.stack([acts[c] for c in l.inputs], 1)
        kh, kw = l.kernel
        if kh > 1 or kw > 1:
            inp = F.pad(inp, (kw // 2, kw // 2, kh // 2, kh // 2), mode="replicate")
        z = F.conv2d(inp, params[l.name + ".w"], params[l.name + ".b"], groups=len(l.inputs) if l.depthwise else 1)
        if logits and l.name in OUTPUTS:
            out[l.name + ".logit"] = z[:, 0]
        y = torch.relu(z) if l.act == "relu" else torch.sigmoid(z) if l.act == "sigmoid" else z
        for k, n in enumerate(l.outputs):
            acts[n] = y[:, k]
    for n in list(OUTPUTS) + list(keep):
        out[n] = acts[n]
    return out


def torch_module(w=None):
    """The stem as a torch.nn.Module (imported lazily: numpy alone runs the rest)."""
    import torch

    class StemTorch(torch.nn.Module):
        def __init__(self, weights):
            super().__init__()
            weights = clean(weights if weights is not None else zeros())
            self.params = torch.nn.ParameterDict()
            for l in LAYERS:
                self.params[l.name + "__w"] = torch.nn.Parameter(torch.from_numpy(weights[l.name + ".w"].copy()))
                self.params[l.name + "__b"] = torch.nn.Parameter(torch.from_numpy(weights[l.name + ".b"].copy()))

        def weights(self):
            return {l.name + s: self.params[l.name + "__" + s[1:]].detach().cpu().numpy().copy() for l in LAYERS for s in (".w", ".b")}

        def run(self, x, keep=(), logits=False):
            p = {l.name + s: self.params[l.name + "__" + s[1:]] for l in LAYERS for s in (".w", ".b")}
            return torch_run(p, x, keep, logits)

        def forward(self, x):
            o = self.run(x)
            return torch.stack([o[n] for n in OUTPUTS], 1)

    return StemTorch(w)


def forward_torch(w, rgb, keep=(), threads=None):
    """forward_numpy's answer from the torch module (float32, quicker)."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    m = torch_module(w)
    x = torch.from_numpy(np.ascontiguousarray(np.asarray(rgb, np.float32).transpose(2, 0, 1)))[None]
    with torch.no_grad():
        o = m.run(x, keep=keep)
    return {k: v[0].numpy() for k, v in o.items()}


def export_onnx(w, path, opset=17):
    """Write the stem as ONNX: input "image" (N, 3, H, W) float32 R, G, B
    0-255; output "ink" (N, 3, H, W): stamp, dark and rule probabilities."""
    import torch

    m = torch_module(w).eval()
    x = torch.zeros(1, 3, 64, 48)
    torch.onnx.export(m, (x,), str(path), input_names=["image"], output_names=["ink"],
                      dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "ink": {0: "n", 2: "h", 3: "w"}},
                      opset_version=opset, dynamo=False)
    return Path(path)


def forward_onnx(path, rgb, threads=1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.ascontiguousarray(np.asarray(rgb, np.float32).transpose(2, 0, 1))[None]
    y = s.run(["ink"], {"image": x})[0][0]
    return {n: y[i] for i, n in enumerate(OUTPUTS)}
