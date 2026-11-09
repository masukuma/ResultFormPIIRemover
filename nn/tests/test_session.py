"""A weight-space session driven by a stub model: the loop, the budget,
the answer and the log, with no language model needed."""

import importlib.util
import json
import os
import unittest

import numpy as np

from rfpii_nn import inkset, session, stem

from .helpers import TempDir, fake_inkset

HAVE_TORCH = importlib.util.find_spec("torch") is not None


class PromptTest(unittest.TestCase):
    def test_the_source_is_read_from_the_repository(self):
        src = session.source_text()
        for fn in ("pub fn dark_ink", "pub fn color_ink", "pub fn rule_lines", "pub fn ink_bits", "pub fn adaptive_inv",
                   "pub fn rule_len"):
            self.assertIn(fn, src)
        self.assertIn("da * da + db * db > 81", src)  # the thresholds come with the code
        self.assertIn("adaptive_inv(&util::gaussian(gray, 3)?, 25, 12.0)", src)

    def test_replies_are_parsed(self):
        r = "<think>\nmaybe ```python\nprint('no')\n```\n</think>\n\nHere:\n```python\nx = 1\n```\nthen\n```python\nprint(x)\n```\nDONE"
        self.assertEqual(session.code_of(r), "print(x)\n")
        self.assertTrue(session.done_of(r))
        self.assertIsNone(session.code_of("<think>```python\nprint(1)\n```</think> nothing"))
        self.assertFalse(session.done_of("<think>DONE</think> not yet"))
        cut = "</think>\n```python\nx = 1\ny = [1,\n  2,\n  3"
        self.assertIsNone(session.code_of(cut))
        self.assertEqual(session.cut_code(cut), ("x = 1", 3))
        self.assertIsNone(session.cut_code("</think>\n```python\nx = 1\n```"))


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class StubSessionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TempDir()
        d = self.tmp.__enter__()
        self.sets = inkset.open_sets([fake_inkset(os.path.join(d, "ink"), n=3)])
        self.out = os.path.join(d, "session")

    def tearDown(self):
        self.tmp.__exit__()

    def test_a_session_writes_tries_and_submits(self):
        replies = [
            "<think>\nGrey first.\n</think>\n\n```python\nW['gray.w'][0, :, 0, 0] = [0.299, 0.587, 0.114]\n"
            "r = try_run(W, 'grey only')\nprint('agreement', r['dark']['agreement'], 'left', r['left'])\n```",
            "<think>\nhm\n</think>\n\nI will think about it.",
            "```python\nprint(probe(W, [0], ['gray'])[0]['channels']['gray']['max'])\nW['dark.b'][0] = -3.0\n"
            "submit(W, 'grey, and nothing marked')\n```\nDONE",
        ]
        backend = session.StubBackend(replies)
        s = session.Session(backend, self.sets, self.out, tries=5, hours=0.5)
        summary = s.run()
        self.assertEqual(summary["end"], "done")
        self.assertEqual((summary["turns"], summary["tries_used"]), (3, 1))
        self.assertEqual(summary["answer"], "submitted")
        self.assertEqual(summary["reasons"], "grey, and nothing marked")
        self.assertEqual(summary["weights_changed_from_start"], 4)
        w = stem.load(os.path.join(self.out, "written.npz"))
        self.assertEqual(float(w["dark.b"][0]), -3.0)
        self.assertTrue(os.path.exists(os.path.join(self.out, "tries", "try_01.npz")))
        # what the model was told: the goal and source first, then each result
        self.assertIn("pub fn color_ink", backend.seen[0])
        self.assertIn("Test runs left: 4 of 5", backend.seen[1])
        self.assertIn("agreement", backend.seen[1])
        self.assertIn("No ```python block", backend.seen[2])
        events = [json.loads(l)["event"] for l in open(os.path.join(self.out, "session.jsonl"))]
        for e in ("start", "prompt", "reply", "cell", "try", "probe", "submit", "end"):
            self.assertIn(e, events)
        self.assertIn("test run 1", open(os.path.join(self.out, "transcript.md")).read())

    def test_without_a_submission_the_best_try_is_the_answer(self):
        def replies(turn, last):
            b = [-50.0, 50.0, -50.0][turn] if turn < 3 else 0.0
            return f"```python\nW['dark.b'][0] = {b}\nprint(try_run(W)['try'])\n```"

        s = session.Session(session.StubBackend(replies), self.sets, self.out, tries=3, hours=0.5, max_turns=5)
        summary = s.run()
        self.assertEqual(summary["tries_used"], 3)
        self.assertEqual(summary["answer"], "best test run (nothing was submitted)")
        self.assertEqual(summary["turns"], 5)
        best = max(summary["tries"], key=lambda t: sum(t[n]["iou"] for n in stem.OUTPUTS))
        w = stem.load(os.path.join(self.out, "written.npz"))
        self.assertEqual(float(w["dark.b"][0]), 50.0 if best["try"] == 2 else -50.0)
        # after the last run, the model is told to submit
        self.assertTrue(np.isfinite(summary["seconds"]))


if __name__ == "__main__":
    unittest.main()
