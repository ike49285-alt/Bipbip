"""Backend tests against a real loopback HTTP server.

Not a mock: these stand up an actual socket speaking Ollama's and the
OpenAI-compatible protocol, so the urllib code, timeouts, proxy bypass and
error handling are all genuinely exercised.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from thirsttrap.llm import (
    ENV_BACKEND,
    ENV_GGUF,
    ENV_HOST,
    ENV_MODEL,
    BackendError,
    GrammarBackend,
    LlamaCppBackend,
    OllamaBackend,
    OpenAICompatBackend,
    build,
    detect,
    parse_posts,
)

STATE = {"content": "one post\ntwo post", "status": 200, "shape": "normal"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "test-model"}, {"name": "other"}]})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "test-model"}]})
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.request_body = json.loads(self.rfile.read(length) or b"{}")
        STATE["last_request"] = self.request_body

        if STATE["status"] != 200:
            self._send({"error": "upstream failure"}, status=STATE["status"])
            return
        if STATE["shape"] == "garbage":
            self._send({"unexpected": "shape"})
            return

        if self.path == "/api/chat":
            self._send({"message": {"content": STATE["content"]}})
        elif self.path == "/v1/chat/completions":
            self._send({"choices": [{"message": {"content": STATE["content"]}}]})
        else:
            self._send({"error": "not found"}, status=404)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def reset_state():
    STATE.update(content="one post\ntwo post", status=200, shape="normal")


class TestOllamaBackend:
    def test_it_sees_a_running_server(self, server):
        assert OllamaBackend(host=server).available()

    def test_it_reports_an_absent_server(self):
        # Port 1 is reserved and nothing listens there.
        assert not OllamaBackend(host="http://127.0.0.1:1").available()

    def test_it_lists_pulled_models(self, server):
        assert OllamaBackend(host=server).models() == ["test-model", "other"]

    def test_listing_an_absent_server_returns_empty_rather_than_raising(self):
        assert OllamaBackend(host="http://127.0.0.1:1").models() == []

    def test_it_completes(self, server):
        text = OllamaBackend(host=server, model="test-model").complete("sys", "prompt")
        assert text == "one post\ntwo post"

    def test_it_sends_the_system_and_user_messages(self, server):
        OllamaBackend(host=server, model="test-model").complete("SYSTEM HERE", "PROMPT HERE")
        sent = STATE["last_request"]
        assert sent["model"] == "test-model"
        assert sent["stream"] is False
        assert sent["messages"][0] == {"role": "system", "content": "SYSTEM HERE"}
        assert sent["messages"][1] == {"role": "user", "content": "PROMPT HERE"}

    def test_a_server_error_raises_backend_error(self, server):
        STATE["status"] = 500
        with pytest.raises(BackendError):
            OllamaBackend(host=server).complete("sys", "prompt")

    def test_an_empty_completion_raises_rather_than_returning_nothing(self, server):
        STATE["content"] = ""
        with pytest.raises(BackendError, match="no content"):
            OllamaBackend(host=server).complete("sys", "prompt")

    def test_an_unreachable_host_raises_backend_error(self):
        with pytest.raises(BackendError):
            OllamaBackend(host="http://127.0.0.1:1").complete("sys", "prompt")

    def test_describe_names_model_and_host(self, server):
        assert "test-model" in OllamaBackend(host=server, model="test-model").describe()


class TestOpenAICompatBackend:
    def test_it_sees_a_running_server(self, server):
        assert OpenAICompatBackend(host=server).available()

    def test_it_completes(self, server):
        assert OpenAICompatBackend(host=server).complete("sys", "p") == "one post\ntwo post"

    def test_an_unexpected_shape_raises_backend_error(self, server):
        STATE["shape"] = "garbage"
        with pytest.raises(BackendError, match="unexpected response shape"):
            OpenAICompatBackend(host=server).complete("sys", "p")

    def test_a_server_error_raises_backend_error(self, server):
        STATE["status"] = 503
        with pytest.raises(BackendError):
            OpenAICompatBackend(host=server).complete("sys", "p")


class TestLlamaCppBackend:
    def test_a_missing_file_is_unavailable(self, tmp_path):
        assert not LlamaCppBackend(model_path=str(tmp_path / "absent.gguf")).available()

    def test_an_empty_path_is_unavailable(self):
        assert not LlamaCppBackend().available()

    def test_describe_names_the_file(self):
        assert "tiny.gguf" in LlamaCppBackend(model_path="/models/tiny.gguf").describe()


class TestGrammarBackend:
    def test_it_is_always_available(self):
        assert GrammarBackend().available()

    def test_it_refuses_prompts(self):
        with pytest.raises(BackendError):
            GrammarBackend().complete("sys", "p")


class TestBuild:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("ollama", OllamaBackend),
            ("openai-compat", OpenAICompatBackend),
            ("lmstudio", OpenAICompatBackend),
            ("llama-cpp", LlamaCppBackend),
            ("grammar", GrammarBackend),
        ],
    )
    def test_names_map_to_backends(self, name, cls):
        assert isinstance(build(name), cls)

    def test_an_unknown_name_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="ollama"):
            build("gpt4all")

    def test_explicit_arguments_win(self):
        backend = build("ollama", host="http://example:1234", model="mine")
        assert backend.host == "http://example:1234" and backend.model == "mine"

    def test_the_environment_fills_the_gaps(self, monkeypatch):
        monkeypatch.setenv(ENV_HOST, "http://envhost:9999")
        monkeypatch.setenv(ENV_MODEL, "env-model")
        backend = build("ollama")
        assert backend.host == "http://envhost:9999" and backend.model == "env-model"


class TestDetect:
    def test_an_explicit_preference_wins(self):
        assert isinstance(detect(prefer="grammar"), GrammarBackend)

    def test_the_environment_selects_a_backend(self, monkeypatch):
        monkeypatch.setenv(ENV_BACKEND, "openai-compat")
        assert isinstance(detect(), OpenAICompatBackend)

    def test_it_prefers_ollama_when_one_is_running(self, monkeypatch):
        monkeypatch.delenv(ENV_BACKEND, raising=False)
        monkeypatch.setattr("thirsttrap.llm.OllamaBackend.available", lambda self: True)
        assert isinstance(detect(), OllamaBackend)

    def test_it_falls_through_to_an_openai_server(self, monkeypatch):
        monkeypatch.delenv(ENV_BACKEND, raising=False)
        monkeypatch.setattr("thirsttrap.llm.OllamaBackend.available", lambda self: False)
        monkeypatch.setattr("thirsttrap.llm.OpenAICompatBackend.available", lambda self: True)
        assert isinstance(detect(), OpenAICompatBackend)

    def test_it_falls_back_to_the_grammar_when_nothing_is_running(self, monkeypatch):
        monkeypatch.delenv(ENV_BACKEND, raising=False)
        monkeypatch.delenv(ENV_GGUF, raising=False)
        monkeypatch.setattr("thirsttrap.llm.OllamaBackend.available", lambda self: False)
        monkeypatch.setattr("thirsttrap.llm.OpenAICompatBackend.available", lambda self: False)
        assert isinstance(detect(), GrammarBackend)

    def test_detect_never_raises(self, monkeypatch):
        monkeypatch.delenv(ENV_BACKEND, raising=False)
        assert detect() is not None


class TestParsePosts:
    def test_a_json_array(self):
        assert parse_posts('["one", "two"]') == ["one", "two"]

    def test_a_fenced_json_array(self):
        assert parse_posts('```json\n["one"]\n```') == ["one"]

    def test_a_numbered_list(self):
        assert parse_posts("1. one\n2) two\n(3) three") == ["one", "two", "three"]

    def test_bullets(self):
        assert parse_posts("- one\n* two\n• three") == ["one", "two", "three"]

    def test_preamble_is_dropped(self):
        assert parse_posts("Here are 2 posts:\none\ntwo") == ["one", "two"]

    def test_wrapping_quotes_are_stripped(self):
        assert parse_posts('"one"\n“two”') == ["one", "two"]

    def test_blank_lines_are_ignored(self):
        assert parse_posts("one\n\n\ntwo") == ["one", "two"]

    def test_the_limit_is_respected(self):
        assert parse_posts("a\nb\nc", limit=2) == ["a", "b"]

    @pytest.mark.parametrize("text", ["", "   ", "\n\n", None])
    def test_empty_input_yields_nothing(self, text):
        assert parse_posts(text) == []

    def test_malformed_json_falls_back_to_lines(self):
        # Mangled slightly by quote-stripping, but never silently dropped.
        assert len(parse_posts('["one", "two"')) == 1

    def test_a_post_containing_a_colon_survives(self):
        line = "The thing about quitting: nobody tells you it gets quieter."
        assert parse_posts(line) == [line]

    def test_preamble_is_only_stripped_before_the_first_post(self):
        # A later line ending in a colon is a post, not scaffolding.
        parsed = parse_posts("Here are 2 posts:\nreal post\nthe part nobody says:")
        assert parsed == ["real post", "the part nobody says:"]
