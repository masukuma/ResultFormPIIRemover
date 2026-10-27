"""VLM plumbing against a stub Ollama server (no model needed)."""

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pytest

from resultform_pii import vlm
from resultform_pii.pipeline import redact_boxes


class _Stub(BaseHTTPRequestHandler):
    replies: list = []
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.requests.append((self.path, body))
        reply = {"message": {"role": "assistant", "content": json.dumps(_Stub.replies.pop(0))}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def ollama():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Stub.replies, _Stub.requests = [], []
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_audit_request_and_verdict(ollama):
    client = vlm.Ollama(vlm.VlmSettings(host=ollama, model="m"))
    page = np.full((2340, 1654, 3), 255, np.uint8)
    _Stub.replies = [{"id_number_visible": False, "phone_number_visible": True, "signature_visible": False,
                      "details": "phone in comments"}]
    a = client.audit(page)
    assert not a.ok and a.findings["details"] == "phone in comments"
    path, body = _Stub.requests[0]
    assert path == "/api/chat" and body["model"] == "m" and body["stream"] is False
    assert body["format"]["required"] and body["options"]["temperature"] == 0
    img = cv2.imdecode(np.frombuffer(base64.b64decode(body["messages"][0]["images"][0]), np.uint8), 0)
    assert max(img.shape) <= 1288 and img.shape[0] % 28 == 0 and img.shape[1] % 28 == 0


def test_locate_scales_boxes_back(ollama):
    client = vlm.Ollama(vlm.VlmSettings(host=ollama))
    page = np.full((2340, 1654, 3), 255, np.uint8)
    _Stub.replies = [{"items": [{"label": "signature", "bbox": [100, 200, 50, 220]},
                                {"label": "name", "bbox": [0, 0, 10, 10]}]}]
    boxes = client.locate(page)
    assert len(boxes) == 1 and boxes[0].label == "signature"  # unknown label dropped
    # sent at 896x1288 (multiples of 28 within 1288); x1 < x0 in the reply gets sorted
    scale = (1654 * 1288 / 2340) // 28 * 28 / 1654
    b = boxes[0]
    assert np.allclose([b.x0, b.y0, b.x1, b.y1], np.array([50, 200, 100, 220]) / scale, atol=1e-6)


def test_redact_boxes_masks_signature():
    page = np.full((400, 600, 3), 230, np.uint8)
    cv2.putText(page, "Sig", (100, 200), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 0), 3)
    r = redact_boxes(page, [vlm.Box("signature", 90, 140, 260, 220)])
    assert (r.image[150:210, 100:250] == 0).all()
