"""Exercises the UI over a real socket, not by poking handler internals."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bot.driver import LocalDriver
from bot.persona import Persona
from bot.ui import Handler


@pytest.fixture
def server(tmp_path):
    Handler.driver = LocalDriver(tmp_path / "ui.db")
    Handler.persona = Persona.load("persona.json")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, Handler.driver
    httpd.shutdown()
    Handler.driver.close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read().decode()


def post_json(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_index_renders_persona_identity(server):
    base, _ = server
    status, body = get(base + "/")
    assert status == 200
    assert "Remy" in body
    assert "remy_synthetic" in body
    assert "__NAME__" not in body and "__HANDLE__" not in body


def test_index_shows_the_disclosure_badge(server):
    base, _ = server
    _, body = get(base + "/")
    assert "ai persona, not a real person" in body


def test_timeline_endpoint_reflects_posts(server):
    base, driver = server
    driver.post(Path("content/img/a.jpg"), "radiator again")
    _, body = get(base + "/api/timeline")
    posts = json.loads(body)["posts"]
    assert [p["caption"] for p in posts] == ["radiator again"]


def test_posting_a_dm_records_it_as_inbound(server):
    base, driver = server
    status, payload = post_json(base + "/api/dm", {"thread": "u/7", "text": "are you real?"})
    assert status == 200 and payload["ok"]
    msgs = driver.thread("u/7")
    assert [(m.direction, m.body) for m in msgs] == [("in", "are you real?")]


def test_thread_endpoint_returns_the_conversation(server):
    base, driver = server
    driver.receive_dm("u/1", "hey")
    driver.send_dm("u/1", "hi")
    _, body = get(base + "/api/thread?id=u/1")
    assert [m["body"] for m in json.loads(body)["messages"]] == ["hey", "hi"]


def test_empty_message_is_rejected(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post_json(base + "/api/dm", {"thread": "u/1", "text": "   "})
    assert exc.value.code == 400


def test_malformed_json_is_rejected(server):
    base, _ = server
    req = urllib.request.Request(
        base + "/api/dm", data=b"{not json", headers={"Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400


def test_unknown_routes_404(server):
    base, _ = server
    for path, method in (("/nope", "GET"), ("/api/nope", "POST")):
        req = urllib.request.Request(base + path, data=b"{}" if method == "POST" else None)
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 404


# -- DM gates through the HTTP layer -------------------------------------


def test_dm_post_returns_the_gate_verdict(server):
    base, _ = server
    _, payload = post_json(base + "/api/dm", {"thread": "u/1", "text": "are you a bot?"})
    assert payload["gate"] == "disclose"
    assert "disclose" in payload["policies"]


def test_ordinary_message_trips_no_policy(server):
    base, _ = server
    _, payload = post_json(base + "/api/dm", {"thread": "u/1", "text": "nice photo"})
    assert payload["gate"] == "normal"
    assert payload["policies"] == []


def test_minor_signal_is_reported_through_the_api(server):
    base, driver = server
    _, payload = post_json(base + "/api/dm", {"thread": "u/9", "text": "im 15 btw"})
    assert payload["gate"] == "terminate"
    # The inbound message is still recorded; nothing is sent back.
    assert [m.direction for m in driver.thread("u/9")] == ["in"]


def test_offline_mode_records_without_replying(server):
    """No credentials in CI, so the agent is absent and nothing is generated."""
    base, driver = server
    _, payload = post_json(base + "/api/dm", {"thread": "u/2", "text": "hey"})
    assert payload["replied"] is False
    assert [m.direction for m in driver.thread("u/2")] == ["in"]
