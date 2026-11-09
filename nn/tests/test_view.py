import http.client
import json
import os
import threading
import unittest

from rfpii_nn import gold, view

from .helpers import TempDir, fake_dataset


class ViewerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TempDir()
        d = cls.tmp.__enter__()
        boxes = [{"class": "signature", "box": [1, 1, 20, 10], "source": "blocks", "zone": "z"}]
        cls.ds = fake_dataset(os.path.join(d, "t"), "teacher", [("p1", {"held": False}, boxes), ("p2", {"held": True}, []),
                                                                ("p3", {"image": "../../outside.png"}, [])])
        with open(os.path.join(d, "outside.png"), "wb") as f:
            f.write(b"secret")
        cls.gold = os.path.join(d, "g")
        gold.plan(cls.gold, [cls.ds], size=2)
        cls.srv = view.serve(view.Pages.from_gold(cls.gold), port=0)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.__exit__()

    def req(self, method, path, host=None, origin=None, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        c.putheader("Host", host or "127.0.0.1:%d" % self.port)
        if origin:
            c.putheader("Origin", origin)
        if body is not None:
            c.putheader("Content-Length", str(len(body)))
        c.endheaders(body)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r, data

    def test_it_serves_the_page_and_its_own_script_only(self):
        r, data = self.req("GET", "/")
        self.assertEqual(r.status, 200)
        self.assertIn(b"/static/view.js", data)
        self.assertNotIn(b"http", data.replace(b"http-equiv", b""))
        self.assertIn("default-src 'none'", r.getheader("Content-Security-Policy"))
        self.assertEqual(r.getheader("Cache-Control"), "no-store")
        for p in ("/static/view.js", "/static/view.css"):
            r, data = self.req("GET", p)
            self.assertEqual(r.status, 200)
            self.assertNotIn(b"https://", data)

    def test_it_lists_and_serves_the_plans_pages(self):
        r, data = self.req("GET", "/api/pages")
        self.assertEqual(r.status, 200)
        pages = json.loads(data)["pages"]
        self.assertEqual(len(pages), 2, "only the plan's pages")
        ids = [p["id"] for p in pages]
        n = ids.index("p1") if "p1" in ids else 0
        r, data = self.req("GET", "/api/page/%d/labels" % n)
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(data)["schema"], "rfpii/nn-labels/1")
        r, data = self.req("GET", "/api/page/%d/image" % n)
        self.assertEqual((r.status, r.getheader("Content-Type")), (200, "image/png"))
        self.assertTrue(data.startswith(b"\x89PNG"))
        r, data = self.req("GET", "/api/page/%d/mask" % n)
        self.assertEqual(r.status, 200)
        self.assertEqual(self.req("GET", "/api/page/9/image")[0].status, 404)

    def test_paths_are_refused(self):
        for p in ("/api/page/../../outside.png", "/static/../view.py", "/static/view.py", "/../outside.png",
                  "/api/page/0/../../../etc/passwd", "/api/page/-1/image", "/api/page/0/labels/x", "/%2e%2e/outside.png"):
            r, data = self.req("GET", p)
            self.assertEqual(r.status, 404, p)
            self.assertNotIn(b"secret", data)

    def test_foreign_hosts_and_origins_and_writes_are_refused(self):
        self.assertEqual(self.req("GET", "/api/pages", host="evil.example:%d" % self.port)[0].status, 403)
        self.assertEqual(self.req("GET", "/api/pages", origin="http://evil.example")[0].status, 403)
        self.assertEqual(self.req("GET", "/api/pages", origin="http://localhost:%d" % self.port)[0].status, 200)
        self.assertEqual(self.req("GET", "/api/pages", host="localhost:%d" % self.port)[0].status, 200)
        self.assertEqual(self.req("POST", "/api/pages", body=b"{}")[0].status, 405)
        self.assertEqual(self.req("PUT", "/api/page/0/labels", body=b"{}")[0].status, 405)
        self.assertEqual(self.req("POST", "/api/pages", origin="http://evil.example", body=b"{}")[0].status, 405)

    def test_it_listens_on_localhost_only(self):
        self.assertEqual(self.srv.server_address[0], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
