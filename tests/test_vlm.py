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
    client = vlm.Ollama(vlm.VlmSettings(host=ollama, model="qwen2.5vl:7b"))  # pixel boxes
    page = np.full((2340, 1654, 3), 255, np.uint8)
    _Stub.replies = [{"items": [{"label": "signature", "bbox": [100, 200, 50, 220]},
                                {"label": "name", "bbox": [0, 0, 10, 10]}]}]
    boxes = client.locate(page)
    assert len(boxes) == 1 and boxes[0].label == "signature"  # unknown label dropped
    # sent at 896x1288 (multiples of 28 within 1288); x1 < x0 in the reply gets sorted
    sx, sy = (1654 * 1288 / 2340) // 28 * 28 / 1654, 1288 / 2340
    b = boxes[0]
    assert np.allclose([b.x0, b.y0, b.x1, b.y1], np.array([50, 200, 100, 220]) / [sx, sy, sx, sy], atol=1e-6)


def test_redact_boxes_masks_signature():
    page = np.full((400, 600, 3), 230, np.uint8)
    cv2.putText(page, "Sig", (100, 200), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 0), 3)
    r = redact_boxes(page, [vlm.Box("signature", 90, 140, 260, 220)])
    assert (r.image[150:210, 100:250] == 0).all()


def test_qwen3_coords_and_audit_record(ollama):
    client = vlm.Ollama(vlm.VlmSettings(host=ollama, model="qwen3-vl:235b"))
    assert client.s.coords == "norm1000"  # Qwen3-VL boxes are 0..1000
    _Stub.replies = [{"id_number_visible": False, "phone_number_visible": False, "signature_visible": False,
                      "details": ""}]
    a = client.audit(np.full((500, 400, 3), 255, np.uint8))
    assert a.ok and a.record["model"] == "qwen3-vl:235b" and len(a.record["image_sent_sha256"]) == 64


def test_remote_host_refuses_unredacted_and_sends_key(monkeypatch):
    s = vlm.VlmSettings(host="https://ollama.com", api_key="k123")
    client = vlm.Ollama(s)
    assert not client.local and client.s.model == vlm.DEFAULT_CLOUD_MODEL
    with pytest.raises(PermissionError):
        client.locate(np.full((500, 400, 3), 255, np.uint8))
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self, *a):
            return json.dumps({"message": {"content": json.dumps(
                {"id_number_visible": False, "phone_number_visible": False, "signature_visible": False,
                 "details": ""})}}).encode()

    def fake_urlopen(req, timeout=0):
        sent["auth"] = req.headers.get("Authorization")
        sent["url"] = req.full_url
        return Resp()

    monkeypatch.setattr(vlm.urllib.request, "urlopen", fake_urlopen)
    client.audit(np.full((500, 400, 3), 255, np.uint8))  # redacted pages may go to the cloud
    assert sent == {"auth": "Bearer k123", "url": "https://ollama.com/api/chat"}


def test_cloud_without_key_is_rejected(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        vlm.Ollama(vlm.VlmSettings(host="https://ollama.com", api_key=None))


def test_qwen3_boxes_are_on_a_1000_grid(ollama):
    client = vlm.Ollama(vlm.VlmSettings(host=ollama))  # default local: latest Qwen
    assert client.s.model == vlm.DEFAULT_LOCAL_MODEL and client.s.coords == "norm1000"
    page = np.full((2340, 1654, 3), 255, np.uint8)
    _Stub.replies = [{"items": [{"label": "signature", "bbox": [500, 500, 600, 550]}]}]
    b = client.locate(page)[0]
    assert np.allclose([b.x0, b.y0, b.x1, b.y1], [827, 1170, 992.4, 1287], atol=2)
