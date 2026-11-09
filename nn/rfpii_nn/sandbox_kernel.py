"""The sandbox's side of a weight-space session: a persistent Python kernel
that runs the language model's code over the weights.

Started by ``sandbox.Sandbox`` as ``python3 -I sandbox_kernel.py WORKDIR
NN_DIR``, in WORKDIR, with no network, as an unprivileged user where the
machine allows it. It sees the weights (``W``, numpy arrays), the
architecture (``SPEC``) and the tools; test runs and probes are asked of the
parent, which holds the pages, and come back as numbers. An audit hook,
installed once numpy is loaded, refuses sockets, new processes, native
libraries and any file outside WORKDIR and the Python library folders.

Protocol (JSON lines): the parent sends {"op": "exec", "code": ...}; the
kernel answers {"op": "rpc", "fn": ..., "args": ..., "file": ...} for each
tool call (the parent replies {"ok": ..., "result" | "error": ...}), then
{"op": "done", "out": ..., "error": ..., "w_ok": ...}.
"""

import contextlib
import io
import json
import os
import sys
import traceback

WORK = os.path.realpath(sys.argv[1])
NN = os.path.realpath(sys.argv[2])
sys.path.insert(0, NN)
os.chdir(WORK)

import numpy as np  # noqa: E402

from rfpii_nn import stem, wsp  # noqa: E402

sys.path.remove(NN)  # the rest of nn/ (datasets among it) is not the kernel's

PROTO_OUT = os.fdopen(os.dup(1), "w", buffering=1)
PROTO_IN = os.fdopen(os.dup(0), "r", buffering=1)
# anything written to the real stdout/stderr goes to a file, not the protocol
_sink = os.open(os.path.join(WORK, "kernel.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.dup2(_sink, 1)
os.dup2(_sink, 2)
_null = os.open(os.devnull, os.O_RDONLY)
os.dup2(_null, 0)

OUT_LIMIT = 6000


def _rpc(fn, args=None, weights=None):
    msg = {"op": "rpc", "fn": fn, "args": args or {}}
    if weights is not None:
        _rpc.n += 1
        f = f"rpc_{_rpc.n:04d}.npz"
        np.savez(os.path.join(WORK, f), **stem.clean(weights))
        msg["file"] = f
    PROTO_OUT.write(json.dumps(msg) + "\n")
    PROTO_OUT.flush()
    line = PROTO_IN.readline()
    if not line:
        raise SystemExit("the session ended")
    r = json.loads(line)
    if not r.get("ok"):
        raise RuntimeError(r.get("error", "refused"))
    return r.get("result")


_rpc.n = 0


def try_run(weights, note=""):
    """One test run (counted against the budget): the candidate is scored, on a
    copy, against the pipeline's masks on the training pages. Returns each
    output's agreement, IoU, recall and precision, IoU by group of pages, and
    the runs left."""
    return _rpc("try", {"note": str(note)}, weights)


def probe(weights, pages, channels, where=None, samples=0):
    """Activations on chosen training pages, as numbers (not counted): per page
    and channel, quantiles, mean and the share above 0.5; the outputs' counts
    (tp, fp, fn, tn) against the pipeline; with where="fn:stamp" (tp, fp, fn,
    tn, pipe, nopipe, stem, nostem, then an output) and samples=N (at most
    256), the channels' values at N such pixels picked at random."""
    return _rpc("probe", {"pages": list(map(int, pages)), "channels": list(channels), "where": where, "samples": int(samples)}, weights)


def submit(weights, reasons):
    """Hand in a stem as the session's answer, with your reasons. The last one
    submitted is the answer."""
    return _rpc("submit", {"reasons": str(reasons)}, weights)


def budget():
    """Test runs left."""
    return _rpc("budget")


def forward(weights, rgb, keep=None):
    """Run a stem on an image you make yourself (H, W, 3), 0-255: returns
    {channel: (H, W)} for the outputs and the channels named in keep ("all"
    for every one). The numpy reference: slow on big images."""
    return stem.forward_numpy(weights, rgb, keep)


def show(weights, layers=None):
    """The printer: the weights as text (layers: a list of names, or all)."""
    return wsp.text(weights, layers)


def diff(a, b):
    """Which weights two stems differ in, and by how much."""
    return wsp.diff(a, b)


def blank():
    """A stem with every weight 0."""
    return stem.zeros()


G = {"__name__": "__session__", "np": np, "SPEC": wsp.spec(), "CHANNELS": stem.channels(), "OUTPUTS": list(stem.OUTPUTS),
     "try_run": try_run, "probe": probe, "submit": submit, "budget": budget, "forward": forward, "show": show,
     "diff": diff, "blank": blank}
with np.load(os.path.join(WORK, "start.npz"), allow_pickle=False) as z:
    G["W"] = {k: z[k].astype(np.float32) for k in z.files}


def _under(p, roots):
    return any(p == r or p.startswith(r.rstrip("/") + "/") for r in roots)


READ_OK = sorted({os.path.realpath(p) for p in sys.path if p and os.path.isdir(p) and not _under(os.path.realpath(p), [NN])}
                 | {os.path.join(NN, "rfpii_nn"), WORK, "/dev/null", os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)})
READ_OK = [r for r in READ_OK if r not in ("/", "/usr", "/usr/local")]
BLOCKED = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
           "ctypes.", "os.kill", "os.killpg", "pty.", "winreg", "webbrowser", "urllib.Request", "http.client", "ftplib",
           "smtplib", "imaplib", "poplib", "nntplib", "telnetlib")
PATH_EVENTS = ("os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.link", "os.symlink",
               "os.truncate", "os.utime", "shutil.", "os.listdir", "os.scandir", "glob.glob", "os.chdir")


def _hook(event, args):
    if event == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if isinstance(path, int):
            return
        p = os.path.realpath(os.fsdecode(path))
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if write and not _under(p, [WORK]):
            raise PermissionError(f"sandbox: no writing outside the work folder ({p})")
        if not write and not _under(p, READ_OK):
            raise PermissionError(f"sandbox: no reading outside the work folder ({p})")
    elif event.startswith(BLOCKED):
        raise PermissionError(f"sandbox: {event} is not allowed")
    elif event.startswith(PATH_EVENTS):
        for a in args:
            if isinstance(a, (str, bytes, os.PathLike)) and not _under(os.path.realpath(os.fsdecode(a)), [WORK] + (READ_OK if event in ("os.listdir", "os.scandir", "glob.glob") else [])):
                raise PermissionError(f"sandbox: {event} outside the work folder")


sys.addaudithook(_hook)


def _clip(s):
    return s if len(s) <= OUT_LIMIT else s[: OUT_LIMIT // 3] + f"\n...[{len(s) - OUT_LIMIT} characters cut]...\n" + s[-2 * OUT_LIMIT // 3:]


while True:
    line = PROTO_IN.readline()
    if not line:
        break
    msg = json.loads(line)
    if msg.get("op") != "exec":
        continue
    buf, err = io.StringIO(), None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(msg["code"], "<cell>", "exec"), G)
        except SystemExit as e:
            if str(e) == "the session ended":
                break
            err = "SystemExit is not allowed in a cell"
        except BaseException:
            err = "".join(traceback.format_exc().splitlines(True)[-12:])
    w_ok = None
    try:
        kept = stem.clean(G.get("W", {}))
        np.savez(os.path.join(WORK, "current.npz"), **kept)
        if isinstance(G.get("W"), dict):
            G["W"].update(kept)  # every array back in its layer's shape, in place
        w_ok = True
    except Exception as e:  # W is not a valid stem: keep the last one
        w_ok = f"W is not a valid stem and was not kept: {e}"[:600]
    PROTO_OUT.write(json.dumps({"op": "done", "out": _clip(buf.getvalue()), "error": err, "w_ok": w_ok}) + "\n")
    PROTO_OUT.flush()
