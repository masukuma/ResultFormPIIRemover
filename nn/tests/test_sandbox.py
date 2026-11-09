"""The sandbox: a persistent kernel over the weights, isolated from the
network, the pages and other programs."""

import os
import unittest

import numpy as np

from rfpii_nn import sandbox, stem


class SandboxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = []
        cls.secret = os.path.abspath(os.path.join(os.path.dirname(__file__), "helpers.py"))

        def fake_try(args, w):
            cls.calls.append(w)
            return {"agreement": 0.5, "note": args.get("note")}

        cls.sb = sandbox.Sandbox(stem.zeros(), {"try": fake_try, "budget": lambda a, w: 7}, timeout=15)

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()
        assert not os.path.exists(cls.sb.dir)

    def run_cell(self, code):
        return self.sb.run(code)

    def test_state_persists_and_tools_answer(self):
        r = self.run_cell("W['gray.w'][0, :, 0, 0] = [0.299, 0.587, 0.114]\nx = 41\nprint(try_run(W, 'first'), budget())")
        self.assertIsNone(r["error"])
        self.assertIn("'note': 'first'", r["out"])
        self.assertIn(" 7", r["out"])
        self.assertAlmostEqual(float(self.calls[-1]["gray.w"][0, 1, 0, 0]), 0.587, places=6)
        r = self.run_cell("print(x + 1); print(show(W, ['gray']).splitlines()[-1])")
        self.assertIn("42", r["out"])
        self.assertIn("[0.299 0.587 0.114]", r["out"])
        self.assertAlmostEqual(float(self.sb.weights["gray.w"][0, 2, 0, 0]), 0.114, places=6)

    def test_no_network_no_programs_no_pages(self):
        for code, word in (("import socket; socket.socket()", "socket"),
                           ("import subprocess; subprocess.run(['true'])", "subprocess"),
                           ("import os; os.system('true')", "os.system"),
                           ("import ctypes; ctypes.CDLL('libc.so.6')", "ctypes"),
                           (f"open({self.secret!r}).read()", "outside the work folder"),
                           ("open('/etc/hostname').read()", "outside the work folder"),
                           ("import os; os.listdir('/home')", "outside the work folder")):
            r = self.run_cell(code)
            self.assertIn("PermissionError", r["error"] or "", code)
            self.assertIn(word, r["error"], code)
        r = self.run_cell("open('mine.txt', 'w').write('ok'); print(open('mine.txt').read())")
        self.assertEqual(r["out"].strip(), "ok")

    def test_a_bad_w_is_not_kept_and_a_stuck_cell_is_stopped(self):
        self.run_cell("W['blur.w'][0, 0, 1, 1] = 1.0")
        r = self.run_cell("W['blur.w'] = np.zeros(3)")  # the wrong size
        self.assertIn("not a valid stem", str(r["w_ok"]))
        self.assertEqual(float(self.sb.weights["blur.w"][0, 0, 1, 1]), 1.0)
        self.run_cell("W['blur.w'] = np.zeros((1, 1, 3, 3), np.float32); W['blur.w'][0, 0, 1, 1] = 2.0")
        r = self.run_cell("while True:\n    pass")
        self.assertTrue(r["timed_out"])
        r = self.run_cell("print(float(W['blur.w'][0, 0, 1, 1]))")
        self.assertEqual(r["out"].strip(), "2.0")  # the kernel came back with W as kept

    def test_isolation_is_reported(self):
        self.assertEqual(set(self.sb.isolation), {"netns", "nobody"})
        if os.geteuid() == 0 and self.sb.isolation["nobody"]:
            r = self.run_cell("import os; print(os.getuid())")
            self.assertEqual(r["out"].strip(), str(sandbox.NOBODY))
        if self.sb.isolation["netns"]:
            # with no network at all, even the audit hook aside, nothing is reachable
            r = self.run_cell("import os; print(sorted(os.listdir('.')) != [])")
            self.assertIsNone(r["error"])


if __name__ == "__main__":
    unittest.main()
