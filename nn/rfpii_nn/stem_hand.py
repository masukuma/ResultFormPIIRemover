"""The rule stem's weights written by hand, from the pipeline's rules: the
baseline the language model's stem is compared with.

Every weight comes from a rule in ``stamps.rs`` / ``masking.rs`` or from
OpenCV's documented definitions of the operations they call, never from a
page:

- grey: OpenCV's 0.299 R + 0.587 G + 0.114 B;
- the 3x3 Gaussian: OpenCV's kernel for size 3 (1/4, 1/2, 1/4 each way);
- the 25x25 mean: 1/25 along a row, then down a column;
- the threshold: steep sigmoid of mean - blur - 11.5 (integers: >= 12);
- OpenCV's openings with an even kernel use the same anchor for the
  erosion and the dilation (the window of a 2x2 square is the pixel and
  the ones above and to its left; of a 32 px line, 16 before and 15
  after), so the opened mask sits a pixel lower and to the right of the
  ink; each erosion is "all of the window", each dilation "any of it";
- Lab: a small fit of the 16-unit colour layer to OpenCV's 8-bit Lab
  formula (``lab_opencv``), on colours, not pages;
- the paper: the mean over 31 px each way in place of the median;
- the colour test: da^2 + db^2 > 81 as 14 half-planes around the half of
  the disc where da > -2 (a polygon within 0.1 of the circle), and the
  gates da > -2 and dL < 5.

``python3 -m rfpii_nn.stem_hand --out nn/stems/hand.npz``
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from . import stem

# steepness of each comparison (logit units per unit of its input)
G_DARK = 8.0  # per grey level
G_MORPH_E = 12.0  # per pixel counted, erosion
G_MORPH_D = 20.0  # per pixel counted, dilation
G_CHROMA = 3.0  # per Lab level
G_AND = 6.0


def lab_opencv(rgb):
    """OpenCV's COLOR_RGB2Lab for 8-bit images, as documented (sRGB
    linearised, D65 white, L scaled to 0-255, a and b offset by 128),
    before rounding. rgb (..., 3) 0-255."""
    c = np.asarray(rgb, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
    xyz = lin @ m.T
    xyz = xyz / np.array([0.950456, 1.0, 1.088754])

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    L = np.where(xyz[..., 1] > 0.008856, 116.0 * fy - 16.0, 903.3 * xyz[..., 1])
    return np.stack([L * 255.0 / 100.0, 500.0 * (fx - fy) + 128.0, 200.0 * (fy - fz) + 128.0], -1)


def colour_samples(n=60000, seed=0):
    """Colours to fit the Lab layer on: the whole cube, and more of what
    pages hold (paper, faint and strong inks, near-greys)."""
    r = np.random.default_rng(seed)
    cube = r.uniform(0, 255, (n // 2, 3))
    base = r.uniform(150, 255, (n // 2, 1)) + r.normal(0, 4, (n // 2, 3))
    tint = r.normal(0, 1, (n // 2, 3)) * r.uniform(0, 60, (n // 2, 1))
    near = np.clip(base + tint, 0, 255)
    return np.concatenate([cube, near])


def fit_lab(hidden=stem.HIDDEN, seed=0, steps=4000):
    """Weights of lab_hidden (relu) and lab, fitted to ``lab_opencv``.

    A least-squares fit of the output layer over hidden units placed by a
    fixed rule (directions through the cube, hinges spread along them),
    then refined by plain gradient steps on colours. Returns
    (w1 (H, 3), b1 (H,), w2 (3, H), b2 (3,)) and the mean and largest
    absolute error on held-out colours.
    """
    x = colour_samples(seed=seed)
    y = lab_opencv(x)
    r = np.random.default_rng(seed + 1)
    dirs = r.normal(0, 1, (hidden, 3))
    dirs[:4] = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    proj = x @ dirs.T
    b1 = -np.quantile(proj, r.uniform(0.02, 0.6, hidden), axis=0).diagonal()
    w1 = dirs.copy()
    xs, ys = x / 255.0, y
    w1s, b1s = w1 * 255.0, b1.copy()

    def hid(xx):
        return np.maximum(xx @ w1s.T + b1s, 0.0)

    hmat = np.concatenate([hid(xs), np.ones((len(xs), 1))], 1)
    sol = np.linalg.lstsq(hmat, ys, rcond=None)[0]
    w2, b2 = sol[:-1].T.copy(), sol[-1].copy()
    # gradient refinement of both layers (Adam), on the same colours
    params = [w1s, b1s, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    lr, bsz = 3e-3, 4096
    for t in range(1, steps + 1):
        idx = r.integers(0, len(xs), bsz)
        xb, yb = xs[idx], ys[idx]
        z = xb @ w1s.T + b1s
        h = np.maximum(z, 0.0)
        out = h @ w2.T + b2
        g = 2.0 * (out - yb) / bsz
        gw2, gb2 = g.T @ h, g.sum(0)
        gh = (g @ w2) * (z > 0)
        gw1, gb1 = gh.T @ xb, gh.sum(0)
        for k, (p, gp) in enumerate(zip(params, [gw1, gb1, gw2, gb2])):
            m[k] = 0.9 * m[k] + 0.1 * gp
            v[k] = 0.999 * v[k] + 0.001 * gp * gp
            p -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    xt = colour_samples(20000, seed + 7)
    err = np.abs(np.maximum((xt / 255.0) @ w1s.T + b1s, 0) @ w2.T + b2 - lab_opencv(xt))
    # in 0-255 input units: w1 / 255
    return (w1s / 255.0, b1s, w2, b2), (float(err.mean()), float(err.max()))


def polygon(k=14, span_deg=105.0, radius=9.0):
    """Directions and offset of k half-planes whose union, with da > -2,
    approximates da^2 + db^2 > radius^2 on the side da > -2."""
    th = np.radians(np.linspace(-span_deg, span_deg, k))
    step = 2 * span_deg / (k - 1)
    r = radius * (1 + math.cos(math.radians(step / 2))) / 2
    return np.stack([np.cos(th), np.sin(th)], 1), r


def hand_weights(seed=0):
    w = stem.zeros()
    w["gray.w"][0, :, 0, 0] = [0.299, 0.587, 0.114]
    g3 = np.array([0.25, 0.5, 0.25], np.float32)
    w["blur.w"][0, 0] = np.outer(g3, g3)
    w["mean_h.w"][0, 0, 0, :] = 1.0 / 25
    w["mean.w"][0, 0, :, 0] = 1.0 / 25
    # blur <= mean - 12  <=>  mean - blur - 11.5 > 0 (grey levels are whole numbers)
    w["dark_raw.w"][0, :, 0, 0] = [-G_DARK, G_DARK]
    w["dark_raw.b"][0] = -11.5 * G_DARK
    for er, di in (("dark_erode", "dark"), ("stamp_erode", "stamp")):
        # OpenCV's 2x2 window: rows y-1..y, columns x-1..x (kernel rows/cols 0..1 of 3)
        w[er + ".w"][0, 0, 0:2, 0:2] = G_MORPH_E
        w[er + ".b"][0] = -3.5 * G_MORPH_E
        w[di + ".w"][0, 0, 0:2, 0:2] = G_MORPH_D
        w[di + ".b"][0] = -0.5 * G_MORPH_D
    n = stem.RULE_LEN  # 32: window offsets -16..+15, kernel taps 0..31 of 33
    w["rule_h_erode.w"][0, 0, 0, 0:n] = G_MORPH_E
    w["rule_h_erode.b"][0] = -(n - 0.5) * G_MORPH_E
    w["rule_h.w"][0, 0, 0, 0:n] = G_MORPH_D
    w["rule_h.b"][0] = -0.5 * G_MORPH_D
    w["rule_v_erode.w"][0, 0, 0:n, 0] = G_MORPH_E
    w["rule_v_erode.b"][0] = -(n - 0.5) * G_MORPH_E
    w["rule_v.w"][0, 0, 0:n, 0] = G_MORPH_D
    w["rule_v.b"][0] = -0.5 * G_MORPH_D
    w["rule.w"][0, :, 0, 0] = [G_MORPH_D, G_MORPH_D]
    w["rule.b"][0] = -0.5 * G_MORPH_D
    (w1, b1, w2, b2), _ = fit_lab(seed=seed)
    w["lab_hidden.w"][:, :, 0, 0] = w1
    w["lab_hidden.b"][:] = b1
    w["lab.w"][:, :, 0, 0] = w2
    w["lab.b"][:] = b2
    for c in range(3):
        w["paper_h.w"][c, 0, 0, :] = 1.0 / 31
        w["paper.w"][c, 0, :, 0] = 1.0 / 31
    dirs, r = polygon(stem.CHROMA - 2)
    g = G_CHROMA
    for k, (ca, sb) in enumerate(dirs):
        # inputs L, a, b, pL, pa, pb: test ca*da + sb*db > r
        w["chroma.w"][k, :, 0, 0] = [0, g * ca, g * sb, 0, -g * ca, -g * sb]
        w["chroma.b"][k] = -g * r
    kd, kl = stem.CHROMA - 2, stem.CHROMA - 1
    w["chroma.w"][kd, :, 0, 0] = [0, g, 0, 0, -g, 0]  # da > -2  <=>  da >= -1
    w["chroma.b"][kd] = 1.5 * g
    w["chroma.w"][kl, :, 0, 0] = [-g, 0, 0, g, 0, 0]  # dL < 5  <=>  dL <= 4
    w["chroma.b"][kl] = 4.5 * g
    # any direction, and both gates: sum(dirs) + 20 gate_a + 20 gate_L > 40.5
    w["stamp_raw.w"][0, :kd, 0, 0] = G_AND
    w["stamp_raw.w"][0, kd, 0, 0] = 20 * G_AND
    w["stamp_raw.w"][0, kl, 0, 0] = 20 * G_AND
    w["stamp_raw.b"][0] = -40.5 * G_AND
    return stem.clean(w)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    w = hand_weights(a.seed)
    stem.save(w, a.out)
    _, (mean_err, max_err) = fit_lab(seed=a.seed)
    print(f"{a.out}: {stem.n_weights()} weights; Lab fit: mean error {mean_err:.2f}, largest {max_err:.2f} levels")


if __name__ == "__main__":
    main()
