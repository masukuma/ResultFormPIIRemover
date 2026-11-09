"""M3's rule stem: weights, the numpy reference, PyTorch and ONNX parity,
and the hand-written weights against OpenCV's own operations."""

import importlib.util
import os
import unittest

import numpy as np

from rfpii_nn import stem, stem_hand

from .helpers import TempDir

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_ONNX = HAVE_TORCH and importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None
HAVE_CV2 = importlib.util.find_spec("cv2") is not None

_HAND = None


def hand():
    global _HAND
    if _HAND is None:
        _HAND = stem_hand.hand_weights()
    return _HAND


def card(h=96, w=128, seed=0):
    """A test card: paper with noise, a long rule, a vertical rule, a short
    stroke, a violet patch and a blue scribble."""
    r = np.random.default_rng(seed)
    img = np.clip(np.full((h, w, 3), (232, 230, 222), np.float64) + r.normal(0, 2, (h, w, 3)), 0, 255)
    img[30:33, 5:w - 5] = 35
    img[5:h - 5, 100:102] = 40
    img[60:67, 20:24] = 50
    img[70:82, 50:66] = (125, 75, 170)
    img[45:47, 40:70] = (40, 60, 160)
    return img.astype(np.uint8)


class WeightsTest(unittest.TestCase):
    def test_shapes_and_round_trip(self):
        w = stem.zeros()
        self.assertEqual(stem.n_weights(), sum(v.size for v in w.values()))
        self.assertLess(stem.n_weights(), 20000)  # small enough to show whole (§2)
        w["gray.w"][0, :, 0, 0] = [0.299, 0.587, 0.114]
        with TempDir() as d:
            for ext in ("npz", "json"):
                p = os.path.join(d, "s." + ext)
                stem.save(w, p)
                back = stem.load(p)
                self.assertTrue(all(np.array_equal(back[k], w[k]) for k in w))

    def test_bad_weights_are_named(self):
        w = stem.zeros()
        w["gray.w"] = np.zeros((2, 2))
        w["blur.b"] = np.array([np.nan], np.float32)
        del w["mean.w"]
        w["extra.w"] = np.zeros(1)
        p = " ".join(stem.check_weights(w))
        for s in ("gray.w: shape", "blur.b: not finite", "mean.w: missing", "extra.w: not a weight"):
            self.assertIn(s, p)
        self.assertRaises(ValueError, stem.clean, w)
        ok = stem.zeros()
        ok["gray.w"] = [[0.299, 0.587, 0.114]]  # the right size in another shape is reshaped
        self.assertEqual(stem.clean(ok)["gray.w"].shape, (1, 3, 1, 1))

    def test_a_blank_stem_marks_nothing(self):
        out = stem.forward_numpy(stem.zeros(), card())
        self.assertEqual(set(out), set(stem.OUTPUTS))
        for n in stem.OUTPUTS:
            self.assertTrue(np.all(out[n] == 0.5))  # sigmoid(0): not above 0.5


class HandTest(unittest.TestCase):
    def test_lab_formula_is_opencvs(self):
        if not HAVE_CV2:
            self.skipTest("OpenCV for Python not installed")
        import cv2
        x = np.random.default_rng(1).integers(0, 256, (4000, 3)).astype(np.uint8)
        ours = np.round(stem_hand.lab_opencv(x))
        theirs = cv2.cvtColor(x[None], cv2.COLOR_RGB2Lab)[0]
        self.assertLessEqual(np.abs(ours - theirs).max(), 2)

    def test_the_hand_stem_marks_the_card(self):
        out = stem.forward_numpy(hand(), card())
        dark, rule, stamp = out["dark"] > 0.5, out["rule"] > 0.5, out["stamp"] > 0.5
        self.assertTrue(dark[31, 60] and rule[31, 60])  # the long rule
        self.assertTrue(dark[50, 101] and rule[50, 101])  # the vertical rule
        self.assertTrue(dark[63, 22] and not rule[63, 22])  # a short stroke is not a rule
        self.assertTrue(stamp[76, 58] and not stamp[31, 60])  # violet is stamp ink, black is not
        self.assertFalse(dark[10, 10] or stamp[10, 10] or rule[10, 10])  # paper

    def test_the_hand_stem_follows_opencv(self):
        """Dark ink and rules as OpenCV computes them on the card, shift included."""
        if not HAVE_CV2:
            self.skipTest("OpenCV for Python not installed")
        import cv2
        img = card(160, 200, seed=3)
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dark = cv2.morphologyEx(cv2.adaptiveThreshold(cv2.GaussianBlur(g, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY_INV, 25, 12), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        rule = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (32, 1))) | \
            cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 32)))
        out = stem.forward_numpy(hand(), img)
        for name, ref in (("dark", dark > 0), ("rule", rule > 0)):
            p = out[name] > 0.5
            core = (slice(4, -4), slice(4, -4))  # borders are padded differently
            iou = np.count_nonzero(p[core] & ref[core]) / max(1, np.count_nonzero(p[core] | ref[core]))
            self.assertGreater(iou, 0.97, name)


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TorchParityTest(unittest.TestCase):
    def test_torch_matches_numpy(self):
        rng = np.random.default_rng(4)
        img = card(80, 112, seed=4)
        for w in (hand(), {k: rng.normal(0, 0.05, v.shape).astype(np.float32) for k, v in stem.zeros().items()}):
            a = stem.forward_numpy(w, img, keep="all")
            b = stem.forward_torch(w, img, keep=[c for c in stem.channels() if c not in stem.INPUTS])
            for n in a:
                if n in stem.INPUTS:
                    continue
                # float32 sums in another order: 1e-4 of a channel's range, or 1e-3 of a probability
                scale = max(1.0, float(np.abs(a[n]).max()))
                self.assertLess(float(np.abs(a[n] - b[n]).max()) / scale, 1e-3 if scale == 1.0 else 1e-4, n)


@unittest.skipUnless(HAVE_ONNX, "PyTorch, onnx or onnxruntime not installed")
class OnnxParityTest(unittest.TestCase):
    def test_onnx_matches_numpy(self):
        import onnx
        with TempDir() as d:
            p = os.path.join(d, "stem.onnx")
            stem.export_onnx(hand(), p)
            onnx.checker.check_model(onnx.load(p))
            for h, w in ((96, 128), (61, 77)):  # any page size
                img = card(h, w, seed=h)
                ref = stem.forward_numpy(hand(), img)
                got = stem.forward_onnx(p, img)
                for n in stem.OUTPUTS:
                    self.assertEqual(got[n].shape, (h, w))
                    self.assertLess(float(np.abs(ref[n] - got[n]).max()), 1e-3, n)
                    self.assertEqual(int(np.count_nonzero((ref[n] > 0.5) != (got[n] > 0.5))), 0, n)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TrainTest(unittest.TestCase):
    def test_a_few_steps_lower_the_loss_and_save_a_stem(self):
        from rfpii_nn import inkset, stem_train

        from .helpers import fake_inkset
        with TempDir() as d:
            sets = inkset.open_sets([fake_inkset(os.path.join(d, "ink"), n=3)])
            w, hist, steps, _ = stem_train.train(sets, 60, seed=1, size=64, margin=8, bsz=2, threads=1)
            self.assertEqual(steps, 60)
            self.assertLess(hist[-1]["loss"], hist[0]["loss"])
            stem.save(w, os.path.join(d, "t.npz"))
            self.assertEqual(stem.check_weights(stem.load(os.path.join(d, "t.npz"))), [])

    def test_the_rescaling_keeps_the_function(self):
        import torch

        from rfpii_nn import stem_train
        theta = {k: v / np.broadcast_to(s, v.shape) for l in stem.LAYERS
                 for k, v, s in ((l.name + ".w", hand()[l.name + ".w"], stem_train.scales(l)[0]),
                                 (l.name + ".b", hand()[l.name + ".b"], stem_train.scales(l)[1]))}
        m = stem_train.scaled_module(theta)
        img = card(48, 64)
        x = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1).copy())[None]
        with torch.no_grad():
            got = m.run(x)
        ref = stem.forward_numpy(hand(), img)
        for n in stem.OUTPUTS:
            self.assertLess(float(np.abs(got[n][0].numpy() - ref[n]).max()), 1e-3, n)
