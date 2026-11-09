"""A local, read-only viewer of labelled pages: each page with its labels
(and, from milestone 3, its disagreement flags) drawn over it, zoom and
pan, next and previous, filters by class and flag.

    python3 -m rfpii_nn.view --gold nn/data/gold          # the gold set's pages
    python3 -m rfpii_nn.view --dataset nn/data/synth      # a whole dataset

It serves on 127.0.0.1 only, and only the pages it was given: the gold
plan's (its reference pages and its synthetic gold set) or one dataset's.
A page is asked for by its number in that list, never by a path. It
answers only requests addressed to itself (the Host header, and the Origin
header when there is one, must be this server), only GET and HEAD, and
fetches nothing from anywhere: the page's script and style are served from
this package. Teacher pages are personal data: the viewer keeps them on
this machine and tells the browser not to cache them.
"""

import argparse
import json
import mimetypes
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .dataset import Dataset

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATIC_FILES = {"/": "view.html", "/static/view.js": "view.js", "/static/view.css": "view.css"}
CSP = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


class Pages:
    """The pages the viewer may show: (dataset, index entry) pairs."""

    def __init__(self, items, title):
        self.items = items
        self.title = title

    @classmethod
    def from_dataset(cls, path):
        ds = Dataset(path)
        return cls([(ds, e) for e in ds.index], "%s (%s dataset)" % (os.path.basename(ds.path.rstrip("/")), ds.kind))

    @classmethod
    def from_gold(cls, gold):
        from .gold import load_plan
        p = load_plan(gold)
        items = []
        opened = {}
        for r in p["reference"]["pages"]:
            path = r["dataset"]
            if path not in opened and os.path.exists(os.path.join(path, "dataset.json")):
                opened[path] = Dataset(path)
            ds = opened.get(path)
            e = ds.by_id.get(r["id"]) if ds else None
            if e is not None and e.get("image_sha256") == r["image_sha256"]:
                items.append((ds, e))
        syn = os.path.join(gold, p["synthetic"]["dataset"])
        if os.path.exists(os.path.join(syn, "dataset.json")):
            ds = Dataset(syn)
            lo, hi = p["synthetic"]["seeds"]
            from .dataset import seed_of
            items.extend((ds, e) for e in ds.index if seed_of(e) is not None and lo <= seed_of(e) < hi)
        return cls(items, "gold set v%d" % p["version"])

    def summary(self):
        out = []
        for i, (ds, e) in enumerate(self.items):
            out.append({"n": i, "id": e["id"], "dataset": ds.kind, "kind": e.get("kind"), "template": e.get("template"),
                        "split": e.get("split"), "held": bool(e.get("held")), "classes": e.get("classes") or {},
                        "leaks": e.get("leaks", 0), "flags": e.get("flags", 0), "has_mask": bool(e.get("mask"))})
        return out


def make_handler(pages, port_holder):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rfpii-view"
        sys_version = ""

        def log_message(self, fmt, *args):  # quiet: page ids are not logged
            pass

        def _ours(self):
            port = port_holder[0]
            allowed = {"127.0.0.1:%d" % port, "localhost:%d" % port}
            if self.headers.get("Host", "") not in allowed:
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {"http://" + a for a in allowed}:
                return False
            return True

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _refuse(self, code, why):
            self._send(code, json.dumps({"error": why}))

        def do_GET(self):
            if not self._ours():
                return self._refuse(403, "not addressed to this viewer")
            path = urlsplit(self.path).path
            if path in STATIC_FILES:
                name = STATIC_FILES[path]
                with open(os.path.join(STATIC, name), "rb") as f:
                    return self._send(200, f.read(), mimetypes.guess_type(name)[0] or "text/plain")
            if path == "/api/pages":
                return self._send(200, json.dumps({"title": pages.title, "pages": pages.summary()}))
            m = re.fullmatch(r"/api/page/(\d{1,7})/(labels|image|mask)", path)
            if not m:
                return self._refuse(404, "no such thing")
            n, what = int(m.group(1)), m.group(2)
            if n >= len(pages.items):
                return self._refuse(404, "no such page")
            ds, e = pages.items[n]
            try:
                if what == "labels":
                    return self._send(200, json.dumps(ds.labels(e)))
                f = ds.image_path(e) if what == "image" else ds.mask_path(e)
                if f is None:
                    return self._refuse(404, "no mask")
                with open(f, "rb") as fh:
                    return self._send(200, fh.read(), mimetypes.guess_type(f)[0] or "application/octet-stream")
            except (OSError, ValueError) as ex:
                return self._refuse(404, str(ex))

        def do_HEAD(self):
            return self.do_GET()

        def _no(self):
            self._refuse(405, "read-only")

        do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _no

    return Handler


def serve(pages, port=8765):
    """A server for `pages` on 127.0.0.1 (port 0: any free port); returns
    it, not started."""
    holder = [port]
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(pages, holder))
    holder[0] = srv.server_address[1]
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m rfpii_nn.view", description="A read-only local viewer of labelled pages.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gold", help="a gold folder (its plan.json)")
    g.add_argument("--dataset", help="a dataset folder")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    pages = Pages.from_gold(args.gold) if args.gold else Pages.from_dataset(args.dataset)
    srv = serve(pages, args.port)
    print("%s: %d pages at http://127.0.0.1:%d/ (Ctrl-C to stop)" % (pages.title, len(pages.items), srv.server_address[1]))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
