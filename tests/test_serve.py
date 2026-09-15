"""The local web UI, driven over a real socket."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from thirsttrap.chat import Repl, Session
from thirsttrap.llm import GrammarBackend
from thirsttrap.profile import Profile
from thirsttrap.serve import Handler


@pytest.fixture
def base():
    Handler.repl = Repl(
        session=Session(profile=Profile(path=None), backend=GrammarBackend(), pool=150, seed=3),
        top=3,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _open(url, data=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if data is None:
        return opener.open(url, timeout=10)
    request = urllib.request.Request(
        url, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return opener.open(request, timeout=10)


def say(base, line):
    return json.loads(_open(base + "/api/say", {"line": line}).read())


class TestPage:
    def test_the_page_is_served(self, base):
        response = _open(base + "/")
        assert response.status == 200
        body = response.read().decode()
        assert "<title>thirsttrap</title>" in body

    def test_a_content_security_policy_is_set(self, base):
        assert "default-src 'self'" in _open(base + "/").headers["Content-Security-Policy"]

    def test_an_unknown_path_is_404(self, base):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _open(base + "/nope")
        assert excinfo.value.code == 404


class TestApi:
    def test_state_reports_the_backend_and_profile(self, base):
        data = json.loads(_open(base + "/api/state").read())
        assert "grammar" in data["backend"]
        assert data["confidence"] == "untuned"
        assert data["posts"] == []

    def test_saying_something_returns_a_reply_and_scored_drafts(self, base):
        data = say(base, "finally quitting the job")
        assert data["reply"]
        assert len(data["posts"]) == 3
        post = data["posts"][0]
        assert post["text"] and isinstance(post["score"], float)
        assert set(post["components"]) == set(__import__("thirsttrap").score_post("x").components)

    def test_posts_come_back_ranked(self, base):
        scores = [p["score"] for p in say(base, "finally quitting the job")["posts"]]
        assert scores == sorted(scores, reverse=True)

    def test_an_adjustment_is_applied_and_reported(self, base):
        say(base, "finally quitting the job")
        data = say(base, "much shorter")
        assert "at most 60 characters" in data["constraints"]
        assert all(len(p["text"]) <= 60 for p in data["posts"])

    def test_a_standing_rule_is_adopted(self, base):
        say(base, "finally quitting the job")
        data = say(base, "never use exclamation marks")
        assert "never say '!'" in data["rules"]
        assert data["adopted"] == ["never say '!'"]

    def test_keeping_teaches_and_is_reported(self, base):
        say(base, "finally quitting the job")
        data = say(base, "/keep 1")
        assert len(data["kept"]) == 1
        assert "kept (1 total" in data["note"]

    def test_an_empty_line_is_a_noop(self, base):
        assert say(base, "   ")["posts"] == []

    def test_a_standing_rule_is_enforced_on_what_comes_back(self, base):
        say(base, "finally quitting the job")
        say(base, "never use exclamation marks")
        data = say(base, "tell me more")
        assert data["posts"]
        assert all("!" not in p["text"] for p in data["posts"])

    def test_malformed_json_is_rejected_cleanly(self, base):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            base + "/api/say", data=b"{not json",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            opener.open(request, timeout=10)
        assert excinfo.value.code == 400
