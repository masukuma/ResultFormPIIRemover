"""The parent's side of the sandbox (``sandbox_kernel.py``): starts the
kernel in its own working folder, with no network and as an unprivileged
user where the machine allows it, under resource limits and a time limit
per cell, and answers its tool calls (test runs, probes, a submitted stem)
with the handlers it is given. The pages stay in this process.

Isolation, in layers, each used where available and recorded:
``unshare -n`` (a network namespace with no interfaces), ``setpriv`` to
the user ``nobody`` (who can't read the scratch folders the pages are in),
a clean environment (no proxy, no keys), resource limits (memory, file
size, open files), a fresh folder of its own, and the kernel's audit hook.
"""

from __future__ import annotations

import json
import os
import resource
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import stem

NN_DIR = Path(__file__).resolve().parent.parent
KERNEL = Path(__file__).resolve().parent / "sandbox_kernel.py"
NOBODY = 65534


def isolation_available():
    """Which isolation layers this machine offers to the sandbox."""
    root = os.geteuid() == 0
    have = {"unshare": bool(shutil.which("unshare")), "setpriv": bool(shutil.which("setpriv"))}
    ok = {"netns": False, "nobody": False}
    if root and have["unshare"]:
        ok["netns"] = subprocess.run(["unshare", "-n", "true"], capture_output=True).returncode == 0
    if root and have["setpriv"]:
        ok["nobody"] = subprocess.run(["setpriv", f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", "true"],
                                      capture_output=True).returncode == 0
    return ok


def _limits(mem_bytes):
    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 20, 256 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


class SandboxError(RuntimeError):
    pass


class Sandbox:
    """A persistent kernel. ``handlers``: {"try"|"probe"|"submit"|"budget":
    f(args, weights or None) -> JSON-able result}."""

    def __init__(self, start, handlers, timeout=600.0, mem_gb=4.0, isolation="auto", workdir=None, rpc_per_cell=60):
        self.handlers = handlers
        self.timeout = timeout
        self.mem = int(mem_gb * (1 << 30))
        self.rpc_per_cell = rpc_per_cell
        avail = isolation_available() if isolation == "auto" else {"netns": False, "nobody": False}
        self.isolation = avail
        self.own_dir = workdir is None
        self.dir = Path(workdir or tempfile.mkdtemp(prefix="rfpii-wsp-", dir="/tmp"))
        self.dir.mkdir(parents=True, exist_ok=True)
        if avail["nobody"]:
            os.chown(self.dir, NOBODY, NOBODY)
        os.chmod(self.dir, 0o700)
        self.weights = stem.clean(start)
        self.proc = None
        self.restarts = 0

    # --- the process --------------------------------------------------------------------

    def _cmd(self):
        cmd = []
        if self.isolation["netns"]:
            cmd += ["unshare", "-n"]
        if self.isolation["nobody"]:
            cmd += ["setpriv", f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", "--inh-caps=-all"]
        return cmd + [sys.executable, "-I", str(KERNEL), str(self.dir), str(NN_DIR)]

    def start(self):
        np.savez(self.dir / "start.npz", **self.weights)
        if self.isolation["nobody"]:
            os.chown(self.dir / "start.npz", NOBODY, NOBODY)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.dir), "LANG": "C.UTF-8", "OMP_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
        self.proc = subprocess.Popen(self._cmd(), cwd=self.dir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, preexec_fn=_limits(self.mem), text=True, bufsize=1)

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        self.proc = None

    def cleanup(self):
        self.close()
        if self.own_dir:
            shutil.rmtree(self.dir, ignore_errors=True)

    def _restart(self):
        self.close()
        self.restarts += 1
        self.start()

    # --- one cell -------------------------------------------------------------------------

    def _readline(self, deadline):
        buf = ""
        while True:
            left = deadline - time.time()
            if left <= 0:
                return None
            r, _, _ = select.select([self.proc.stdout], [], [], min(left, 1.0))
            if r:
                line = self.proc.stdout.readline()
                if not line:
                    raise SandboxError("the kernel stopped")
                buf += line
                if buf.endswith("\n"):
                    return buf
            elif self.proc.poll() is not None:
                raise SandboxError(f"the kernel stopped (exit {self.proc.returncode})")

    def _load(self, name):
        p = (self.dir / name).resolve()
        if p.parent != self.dir.resolve() or not p.is_file() or p.stat().st_size > 8 << 20:
            raise SandboxError(f"bad weights file {name!r}")
        with np.load(p, allow_pickle=False) as z:
            return stem.clean({k: z[k] for k in z.files})

    def run(self, code):
        """Run a cell. Returns {"out", "error", "seconds", "timed_out", "rpc"}."""
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        t0 = time.time()
        deadline = t0 + self.timeout
        self.proc.stdin.write(json.dumps({"op": "exec", "code": code}) + "\n")
        self.proc.stdin.flush()
        calls = []
        while True:
            try:
                line = self._readline(deadline)
            except SandboxError as e:
                self._restart()
                return {"out": "", "error": f"{e}; the kernel was restarted with W as last kept (other variables are lost)",
                        "seconds": round(time.time() - t0, 1), "timed_out": False, "rpc": calls}
            if line is None:
                self._restart()
                return {"out": "", "error": f"the cell ran past {self.timeout:.0f} s and was stopped; the kernel was restarted with W as last kept (other variables are lost)",
                        "seconds": round(time.time() - t0, 1), "timed_out": True, "rpc": calls}
            msg = json.loads(line)
            if msg.get("op") == "done":
                kept = msg.get("w_ok")
                if kept is True:
                    try:
                        self.weights = self._load("current.npz")
                    except SandboxError as e:
                        kept = str(e)
                return {"out": msg.get("out", ""), "error": msg.get("error"), "w_ok": kept,
                        "seconds": round(time.time() - t0, 1), "timed_out": False, "rpc": calls}
            if msg.get("op") != "rpc":
                continue
            fn = msg.get("fn")
            t1 = time.time()
            try:
                if len(calls) >= self.rpc_per_cell:
                    raise SandboxError(f"at most {self.rpc_per_cell} tool calls a cell")
                if fn not in self.handlers:
                    raise SandboxError(f"no tool {fn!r}")
                w = self._load(msg["file"]) if msg.get("file") else None
                res = {"ok": True, "result": self.handlers[fn](msg.get("args") or {}, w)}
            except Exception as e:  # the tool's refusal goes back to the model's code
                res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            calls.append({"fn": fn, "args": msg.get("args"), "ok": res["ok"], "seconds": round(time.time() - t1, 1)})
            deadline += time.time() - t1  # tool time doesn't count against the cell
            self.proc.stdin.write(json.dumps(res) + "\n")
            self.proc.stdin.flush()
