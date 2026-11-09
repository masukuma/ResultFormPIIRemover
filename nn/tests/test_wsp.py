"""The weight-space tools: printer, diff, probe and try (with its budget)."""

import importlib.util
import os
import unittest

import numpy as np

from rfpii_nn import inkset, stem, wsp

from .helpers import TempDir, fake_inkset

HAVE_TORCH = importlib.util.find_spec("torch") is not None


class PrinterTest(unittest.TestCase):
    def test_weights_as_text_with_meanings(self):
        w = stem.zeros()
        w["dark_raw.w"][0, :, 0, 0] = [-8.123456, 8.0]
        w["dark_raw.b"][0] = -92.0
        t = wsp.text(w)
        for l in stem.LAYERS:
            self.assertIn(f"## layer {l.name}:", t)
        self.assertIn("in[1] mean: the 25x25 local mean", t)
        self.assertIn("out[0] dark_raw: [-8.123 8] | -92", t)  # 4 significant figures
        self.assertIn("## layer blur: conv 3x3", t)
        self.assertIn("every weight 0 (kernels 3x3; row i is offset i-1, column j offset j-1)", t)
        w["blur.w"][0, 0, 1, 1] = 1.0
        self.assertIn("    kernel blur<-gray (3x3; row i is offset i-1, column j offset j-1):\n      [0 0 0]\n      [0 1 0]", wsp.text(w, ["blur"]))
        one = wsp.text(w, ["rule_h"])
        self.assertIn("1x33", one)
        self.assertNotIn("## layer gray", one)

    def test_arrays_are_copies(self):
        w = stem.zeros()
        a = wsp.arrays(w)
        a["gray.w"][:] = 1
        self.assertEqual(float(w["gray.w"].sum()), 0.0)
        self.assertEqual(wsp.spec()[0]["name"], "gray")

    def test_diff(self):
        a = stem.zeros()
        b = stem.zeros()
        b["gray.w"][0, 1, 0, 0] = 0.587
        b["rule.b"][0] = -10
        d = wsp.diff(a, b)
        self.assertTrue(d.startswith("2 of %d weights differ" % stem.n_weights()), d)
        self.assertIn("gray.w: 1 of 3 changed", d)
        self.assertIn("gray.w[0, 1, 0, 0] 0 -> 0.587", d)
        self.assertEqual(wsp.changed(a, b), 2)
        self.assertTrue(wsp.diff(a, a).startswith("0 of"))


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TryProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TempDir()
        d = self.tmp.__enter__()
        self.sets = inkset.open_sets([fake_inkset(os.path.join(d, "ink"), n=4, groups=("synthetic", "2013-34"))])
        self.pages = wsp.Pages(self.sets, "train")

    def tearDown(self):
        self.tmp.__exit__()

    def test_try_scores_a_copy_and_counts(self):
        events = []
        b = wsp.Budget(self.pages, total=2, log=events.append)
        w = stem.zeros()
        w["dark.b"][0] = 50.0  # everything dark
        r = b.run(w, "all dark")
        w["dark.b"][0] = -50.0  # changing it afterwards changes nothing tried
        self.assertEqual((r["try"], r["left"]), (1, 1))
        self.assertEqual(r["dark"]["recall"], 1.0)
        self.assertLess(r["dark"]["precision"], 0.2)
        self.assertEqual(set(r["iou_by_group"]), {"synthetic", "2013-34"})
        self.assertEqual(float(b.history[0]["weights"]["dark.b"][0]), 50.0)
        b.run(w)
        with self.assertRaises(RuntimeError):
            b.run(w)
        self.assertEqual(len(events), 2)
        # neither passes 99% on every mask, so the higher IoU wins: all dark finds the dark ink
        self.assertEqual(b.best()["try"], 1)

    def test_probe_gives_numbers_not_images(self):
        w = stem.zeros()
        w["gray.w"][0, :, 0, 0] = [0.299, 0.587, 0.114]
        r = wsp.probe(w, self.pages, [0, 1], ["gray", "R"], where="pipe:stamp", samples=500)
        self.assertEqual(len(r), 2)
        s = r[0]
        self.assertEqual(s["channels"]["R"]["max"], "236")
        self.assertEqual(s["count"], 120)  # the violet patch
        self.assertEqual(len(s["samples"]["gray"]), 120)  # at most 256, and no more than there are
        self.assertTrue(all(isinstance(v, float) for v in s["samples"]["R"]))
        self.assertNotIn("positions", s)
        self.assertEqual(set(s["outputs"]), set(stem.OUTPUTS))
        with self.assertRaises(ValueError):
            wsp.probe(w, self.pages, [0], ["nope"])
        with self.assertRaises(ValueError):
            wsp.probe(w, self.pages, [99], ["gray"])
        with self.assertRaises(ValueError):
            wsp.probe(w, self.pages, [0], ["gray"], where="fn:ink", samples=3)

    def test_inkset_scores(self):
        st = self.sets[0]
        self.assertEqual(len(st.pages("train")) + len(st.pages("held")), len(st))
        line = st.index[0]
        c = inkset.counts({"stamp": np.zeros(st.mask(line).shape), "dark": (st.mask(line) & 1) * 1.0,
                           "rule": np.ones(st.mask(line).shape)}, st.mask(line))
        s = inkset.summary(c)
        self.assertEqual(s["dark"]["iou"], 1.0)
        self.assertEqual(s["stamp"]["iou"], 0.0)
        self.assertGreater(s["stamp"]["agreement"], 0.98)  # why IoU is reported beside agreement
        self.assertFalse(inkset.passes(s))  # the rule mask marks everything


if __name__ == "__main__":
    unittest.main()
