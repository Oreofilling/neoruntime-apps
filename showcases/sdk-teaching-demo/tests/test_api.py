"""HTTP surface: envelope shape, routes, settings POST, 404s, MJPEG
first chunk, and the page's appUrl wiring (static check)."""

import http.client
import json
import threading

import pytest


@pytest.fixture
def server(app, main_mod, monkeypatch):
    monkeypatch.setattr(main_mod, "HTTP_PORT", 0)  # ephemeral port
    httpd = main_mod.make_server(app)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def request(server, method, path, body=None, content_type=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                      timeout=5)
    headers = {}
    if body is not None:
        headers["Content-Type"] = content_type or "application/json"
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, dict(response.getheaders()), data


def test_root_serves_page_with_appurl(server):
    status, headers, body = request(server, "GET", "/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert b"function appUrl(path)" in body
    assert b"/api/status" in body


def test_health_envelope(server):
    status, _, body = request(server, "GET", "/api/health")
    envelope = json.loads(body)
    assert status == 200
    assert envelope["success"] is True
    assert envelope["data"]["ok"] is True
    assert envelope["data"]["app"] == "sdk-teaching-demo"
    assert envelope["data"]["bform_running"] is False


def test_status_sections(server):
    status, _, body = request(server, "GET", "/api/status")
    envelope = json.loads(body)
    assert status == 200 and envelope["success"] is True
    data = envelope["data"]
    for section in ("settings", "bform", "overlay", "routing", "aform",
                    "events", "model"):
        assert section in data
    assert data["model"]["id"] == "yolov8n_384_640"
    assert data["events"]["topic"] == "app/sdk-teaching-demo/detection"
    assert data["settings"]["policy"] == "prefer_hardware"


def test_settings_post_valid_and_invalid(server):
    status, _, body = request(server, "POST", "/api/settings",
                              json.dumps({"min_score": 0.7}))
    envelope = json.loads(body)
    assert status == 200
    assert envelope["data"]["min_score"] == 0.7

    status, _, body = request(server, "POST", "/api/settings",
                              json.dumps({"policy": "warp9"}))
    assert status == 400
    assert json.loads(body)["success"] is False


def test_settings_post_bad_json(server):
    status, _, body = request(server, "POST", "/api/settings", "not json")
    assert status == 400
    assert "invalid JSON" in json.loads(body)["error"]


def test_unknown_paths_404_envelope(server):
    for method, path in (("GET", "/nope"), ("POST", "/api/nope")):
        status, _, body = request(server, method, path)
        envelope = json.loads(body)
        assert status == 404
        assert envelope == {"success": False, "error": "not found"}


def test_refusal_endpoint_returns_message(server):
    status, _, body = request(server, "POST", "/api/routing/refusal", "{}")
    envelope = json.loads(body)
    assert status == 200 and envelope["success"] is True
    assert "HardwareUnavailable" in envelope["data"]["message"]


def test_snippets_endpoint_serves_teach_payload(server):
    status, _, body = request(server, "GET", "/api/snippets")
    envelope = json.loads(body)
    assert status == 200 and envelope["success"] is True
    data = envelope["data"]
    assert [s["id"] for s in data["stations"]] == [
        "s1", "s2", "s3", "s4", "s5"]
    assert data["errors"], "error lessons must ride the payload"
    assert data["next"]["steps"]


def test_stream_mjpeg_first_chunk(server, app):
    app.buffer.update(b"frame-one-bytes")  # 15-byte payload
    conn = http.client.HTTPConnection("127.0.0.1",
                                      server.server_address[1], timeout=5)
    conn.request("GET", "/stream.mjpg")
    response = conn.getresponse()
    assert response.status == 200
    assert "multipart/x-mixed-replace" in response.getheader("Content-Type")
    # the latest-wins buffer holds ONE frame: read exactly the 57-byte
    # multipart header block so the read never starves
    chunk = response.read(57)
    conn.close()
    assert chunk.startswith(b"--frame\r\n")
    assert b"Content-Type: image/jpeg\r\n" in chunk
    assert chunk.endswith(b"Content-Length: 15\r\n\r\n")
