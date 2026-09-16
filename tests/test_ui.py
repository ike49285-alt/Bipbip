"""Exercises the tuning UI over a real socket."""

import json
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bot.driver import LocalDriver
from bot.persona import Persona
from bot.ui import THREAD, Handler


@pytest.fixture
def server(tmp_path):
    persona_path = tmp_path / "persona.json"
    shutil.copy("persona.json", persona_path)

    Handler.persona_path = persona_path
    Handler.persona = Persona.load(persona_path)
    Handler.driver = LocalDriver(tmp_path / "ui.db")
    Handler.agent = None          # offline: gates only, no network in tests
    Handler.annotations = {}

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, Handler.driver, persona_path
    httpd.shutdown()
    Handler.driver.close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read().decode()


def post(url, payload=None):
    body = json.dumps(payload or {}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def state(base):
    return json.loads(get(base + "/api/state")[1])


# -- page ----------------------------------------------------------------


def test_index_renders_persona_identity(server):
    base, _, _ = server
    status, body = get(base + "/")
    assert status == 200
    assert "Remy" in body and "remy_synthetic" in body
    assert "{{" not in body, "an unsubstituted template token leaked"


def test_index_reports_offline_mode(server):
    base, _, _ = server
    assert "gates only" in get(base + "/")[1]


# -- state ---------------------------------------------------------------


def test_state_exposes_the_live_voice_card(server):
    base, _, _ = server
    s = state(base)
    assert s["prompt"] == Handler.persona.voice_card()
    assert s["rules"] == list(Handler.persona.voice.rules)
    assert s["facts"] == list(Handler.persona.facts)
    assert s["live"] is False


def test_state_starts_empty(server):
    base, _, _ = server
    assert state(base)["messages"] == []


# -- chat ----------------------------------------------------------------


def test_chat_records_the_message_and_its_gate(server):
    base, driver, _ = server
    _, payload = post(base + "/api/chat", {"text": "are you a bot?"})
    assert payload["gate"] == "disclose"
    assert [m.body for m in driver.thread(THREAD)] == ["are you a bot?"]


def test_chat_annotates_the_turn_with_the_gate(server):
    base, _, _ = server
    post(base + "/api/chat", {"text": "are you a bot?"})
    ann = state(base)["messages"][0]["ann"]
    assert any(a["text"] == "disclose" for a in ann)


def test_ordinary_message_trips_nothing(server):
    base, _, _ = server
    _, payload = post(base + "/api/chat", {"text": "nice photo"})
    assert payload["gate"] == "normal" and payload["policies"] == []


def test_minor_signal_terminates_without_replying(server):
    base, driver, _ = server
    _, payload = post(base + "/api/chat", {"text": "im 15 btw"})
    assert payload["gate"] == "terminate"
    assert [m.direction for m in driver.thread(THREAD)] == ["in"]


def test_offline_mode_generates_nothing(server):
    base, _, _ = server
    _, payload = post(base + "/api/chat", {"text": "hey"})
    assert payload["replied"] is False
    assert any("no credentials" in a["text"] for a in state(base)["messages"][0]["ann"])


def test_empty_message_is_rejected(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(base + "/api/chat", {"text": "   "})
    assert exc.value.code == 400


# -- tuning --------------------------------------------------------------


def test_editing_rules_changes_the_prompt_immediately(server):
    base, _, _ = server
    _, payload = post(base + "/api/persona", {"rules": ["speak only in questions"]})
    assert payload["ok"] and payload["written"] is False
    s = state(base)
    assert s["rules"] == ["speak only in questions"]
    assert "speak only in questions" in s["prompt"]


def test_editing_facts_changes_the_prompt(server):
    base, _, _ = server
    post(base + "/api/persona", {"facts": ["owns a very loud kettle"]})
    assert "owns a very loud kettle" in state(base)["prompt"]


def test_edits_are_not_written_unless_asked(server):
    base, _, path = server
    before = path.read_text()
    post(base + "/api/persona", {"rules": ["temporary rule"]})
    assert path.read_text() == before


def test_saving_persists_to_disk(server):
    base, _, path = server
    _, payload = post(base + "/api/persona", {"rules": ["a durable rule"], "save": True})
    assert payload["written"] is True
    assert json.loads(path.read_text())["voice"]["rules"] == ["a durable rule"]


def test_an_invalid_edit_is_refused_and_changes_nothing(server):
    """Persona validation still guards the tuning endpoint."""
    base, _, _ = server
    before = state(base)["rules"]
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(base + "/api/persona", {"rules": [""]})
    assert exc.value.code == 400
    assert state(base)["rules"] == before


def test_tuning_cannot_reach_the_disclosure_boundary(server):
    """Only voice and facts are editable here; the bounds are not."""
    base, _, _ = server
    post(base + "/api/persona", {"rules": ["anything"], "facts": ["anything"]})
    assert Handler.persona.bounds.always_answer_truthfully
    assert Handler.persona.identity.disclosure


# -- reset ---------------------------------------------------------------


def test_reset_clears_the_conversation(server):
    base, driver, _ = server
    post(base + "/api/chat", {"text": "hey"})
    post(base + "/api/chat", {"text": "you there"})
    assert len(driver.thread(THREAD)) == 2
    post(base + "/api/reset")
    assert driver.thread(THREAD) == []
    assert state(base)["messages"] == []


# -- routing -------------------------------------------------------------


def test_unknown_routes_404(server):
    base, _, _ = server
    for path, method in (("/nope", "GET"), ("/api/nope", "POST")):
        req = urllib.request.Request(base + path, data=b"{}" if method == "POST" else None)
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 404


def test_malformed_json_is_rejected(server):
    base, _, _ = server
    req = urllib.request.Request(base + "/api/chat", data=b"{not json",
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400
