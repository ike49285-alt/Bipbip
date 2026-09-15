"""Local model backends.

Everything here talks to a model running on your own machine. There is no
hosted API and no API key. Three ways in, probed in order by `detect()`:

- `OllamaBackend`      -- Ollama's native API, usually 127.0.0.1:11434
- `OpenAICompatBackend`-- llama.cpp's llama-server, LM Studio, vLLM, anything
                          exposing /v1/chat/completions on localhost
- `LlamaCppBackend`    -- llama-cpp-python in-process, no server at all

HTTP here is `urllib` from the standard library, aimed at loopback. That keeps
the package dependency-free: `pip install thirsttrap` pulls in nothing, and
llama-cpp-python is an optional extra only the in-process backend needs.

`GrammarBackend` is the fallback when no model is reachable. It is genuinely
worse -- see the README -- but it means the tool still runs on a machine with
nothing installed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OPENAI_HOST = "http://127.0.0.1:8080"

# Loopback should answer immediately; a long hang means nothing is there.
PROBE_TIMEOUT = 1.5
GENERATE_TIMEOUT = 180.0

ENV_BACKEND = "THIRSTTRAP_BACKEND"
ENV_MODEL = "THIRSTTRAP_MODEL"
ENV_HOST = "THIRSTTRAP_HOST"
ENV_GGUF = "THIRSTTRAP_GGUF"


class BackendError(RuntimeError):
    """The backend was reachable but could not produce a completion."""


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Loopback must not go through any proxy the environment has set.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _reachable(url: str) -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=PROBE_TIMEOUT):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


@dataclass
class Backend:
    """Common surface. `complete` returns raw model text; parsing is the caller's."""

    name: str = "backend"

    def available(self) -> bool:
        raise NotImplementedError

    def complete(self, system: str, prompt: str) -> str:
        """One-shot: a single user message, no history."""
        return self.chat(system, [{"role": "user", "content": prompt}])

    def chat(self, system: str, messages: list[dict], json_mode: bool = False) -> str:
        """Continue a conversation. `messages` alternates user/assistant, ending on user."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


@dataclass
class OllamaBackend(Backend):
    name: str = "ollama"
    host: str = DEFAULT_OLLAMA_HOST
    model: str = ""  # empty: use whatever is pulled

    def available(self) -> bool:
        return _reachable(f"{self.host.rstrip('/')}/api/tags")

    def models(self) -> list[str]:
        """What this Ollama has pulled. Empty if it cannot be asked."""
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"{self.host.rstrip('/')}/api/tags", timeout=PROBE_TIMEOUT) as r:
                data = json.loads(r.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", []) if "name" in m]
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return []

    def resolve_model(self) -> str:
        """Use the configured model, else the first one this Ollama has pulled.

        Guessing a name the user has not pulled is the most common way this
        fails, and asking the server is free.
        """
        if self.model:
            return self.model
        pulled = self.models()
        if not pulled:
            raise BackendError(
                f"ollama at {self.host} has no models pulled -- run `ollama pull llama3.2`"
            )
        self.model = pulled[0]
        return self.model

    def chat(self, system: str, messages: list[dict], json_mode: bool = False) -> str:
        payload = {
            "model": self.resolve_model(),
            "stream": False,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if json_mode:
            payload["format"] = "json"

        try:
            data = _post_json(
                f"{self.host.rstrip('/')}/api/chat", payload, timeout=GENERATE_TIMEOUT
            )
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error", "")
            except Exception:
                pass
            raise BackendError(f"ollama: {detail or exc}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise BackendError(f"ollama at {self.host}: {exc}") from exc

        content = (data.get("message") or {}).get("content")
        if not content:
            raise BackendError(f"ollama returned no content (is {self.model!r} pulled?)")
        return content

    def describe(self) -> str:
        return f"ollama {self.model or '(first pulled model)'} @ {self.host}"


@dataclass
class OpenAICompatBackend(Backend):
    """llama-server, LM Studio, vLLM -- anything speaking /v1/chat/completions."""

    name: str = "openai-compat"
    host: str = DEFAULT_OPENAI_HOST
    model: str = "local-model"

    def available(self) -> bool:
        return _reachable(f"{self.host.rstrip('/')}/v1/models")

    def chat(self, system: str, messages: list[dict], json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            data = _post_json(
                f"{self.host.rstrip('/')}/v1/chat/completions", payload,
                timeout=GENERATE_TIMEOUT,
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise BackendError(f"openai-compatible server at {self.host}: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected response shape from {self.host}") from exc

    def describe(self) -> str:
        return f"openai-compat {self.model} @ {self.host}"


@dataclass
class LlamaCppBackend(Backend):
    """In-process GGUF via llama-cpp-python. No server, no socket."""

    name: str = "llama-cpp"
    model_path: str = ""
    n_ctx: int = 4096
    _llama: object = field(default=None, repr=False)

    def available(self) -> bool:
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._llama is None:
            from llama_cpp import Llama

            # verbose=False or every call prints a wall of load diagnostics.
            self._llama = Llama(
                model_path=self.model_path, n_ctx=self.n_ctx, verbose=False
            )
        return self._llama

    def chat(self, system: str, messages: list[dict], json_mode: bool = False) -> str:
        try:
            llama = self._load()
            result = llama.create_chat_completion(
                messages=[{"role": "system", "content": system}, *messages],
                max_tokens=1024,
                response_format={"type": "json_object"} if json_mode else None,
            )
        except Exception as exc:  # the binding raises its own types
            raise BackendError(f"llama-cpp: {exc}") from exc

        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("unexpected response shape from llama-cpp") from exc

    def describe(self) -> str:
        return f"llama-cpp {os.path.basename(self.model_path)}"


@dataclass
class GrammarBackend(Backend):
    """The dependency-free fallback. Ignores the prompt; the caller uses `propose`."""

    name: str = "grammar"

    def available(self) -> bool:
        return True

    def chat(self, system: str, messages: list[dict], json_mode: bool = False) -> str:
        raise BackendError("the grammar backend does not take prompts")

    def describe(self) -> str:
        return "grammar (no model found)"


def from_env() -> Backend | None:
    """Build the backend named by THIRSTTRAP_BACKEND, if one is."""
    choice = os.environ.get(ENV_BACKEND, "").strip().lower()
    if not choice:
        return None
    return build(choice)


def build(
    choice: str,
    host: str | None = None,
    model: str | None = None,
    gguf: str | None = None,
) -> Backend:
    """Construct a named backend, filling gaps from the environment."""
    host = host or os.environ.get(ENV_HOST)
    model = model or os.environ.get(ENV_MODEL)
    gguf = gguf or os.environ.get(ENV_GGUF, "")

    if choice == "ollama":
        backend = OllamaBackend()
        if host:
            backend.host = host
        if model:
            backend.model = model
        return backend
    if choice in {"openai", "openai-compat", "llama-server", "lmstudio"}:
        backend = OpenAICompatBackend()
        if host:
            backend.host = host
        if model:
            backend.model = model
        return backend
    if choice in {"llama-cpp", "llamacpp", "gguf"}:
        return LlamaCppBackend(model_path=gguf)
    if choice == "grammar":
        return GrammarBackend()
    raise ValueError(
        f"unknown backend {choice!r}; try ollama, openai-compat, llama-cpp or grammar"
    )


def detect(prefer: str | None = None) -> Backend:
    """Find something to generate with. Never fails -- the grammar always answers."""
    if prefer:
        return build(prefer)

    configured = from_env()
    if configured is not None:
        return configured

    for backend in (OllamaBackend(), OpenAICompatBackend()):
        if backend.available():
            return backend

    gguf = os.environ.get(ENV_GGUF)
    if gguf:
        candidate = LlamaCppBackend(model_path=gguf)
        if candidate.available():
            return candidate

    return GrammarBackend()


# -- parsing ------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:\w+)?\s*|\s*```\s*$", re.MULTILINE)
_LEADER_RE = re.compile(r"^\s*(?:[-*•]|\(?\d{1,2}[.):])\s*")
_WRAPPING_QUOTES = "\"'“”‘’"


def parse_posts(text: str, limit: int | None = None) -> list[str]:
    """Pull individual posts out of whatever shape the model replied in.

    Small local models ignore output instructions often enough that this has to
    be forgiving: a JSON array, a numbered list, bullets, or bare lines all
    parse. Preamble like "Here are 5 posts:" is dropped by the colon rule.
    """
    text = (text or "").strip()
    if not text:
        return []

    # A JSON array of strings, with or without a code fence.
    unfenced = _FENCE_RE.sub("", text).strip()
    if unfenced.startswith("["):
        try:
            data = json.loads(unfenced)
            if isinstance(data, list):
                posts = [str(item).strip() for item in data if str(item).strip()]
                return posts[:limit] if limit else posts
        except ValueError:
            pass

    posts = []
    for line in unfenced.splitlines():
        line = line.strip()
        if not line:
            continue
        stripped = _LEADER_RE.sub("", line).strip()
        if not stripped:
            continue
        # "Here are five posts:" and similar scaffolding. Only before the first
        # post -- a real post may legitimately end on a colon, and dropping it
        # anywhere in the list silently ate valid output.
        if not posts and stripped.endswith(":") and len(stripped) < 60:
            continue
        stripped = stripped.strip(_WRAPPING_QUOTES).strip()
        if stripped:
            posts.append(stripped)

    return posts[:limit] if limit else posts


def parse_turn(text: str) -> tuple[str, list[str]]:
    """Split a reply into what she says and what she drafted.

    The model is asked for {"reply": ..., "posts": [...]}. Small models drift,
    so anything unparseable is treated as pure conversation rather than
    discarded -- losing her reply is worse than losing the drafts.
    """
    text = (text or "").strip()
    if not text:
        return "", []

    candidate = _FENCE_RE.sub("", text).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(candidate[start : end + 1])
        except ValueError:
            data = None
        if isinstance(data, dict):
            reply = str(data.get("reply", "")).strip()
            raw = data.get("posts")
            posts = [str(p).strip() for p in raw if str(p).strip()] if isinstance(raw, list) else []
            if reply or posts:
                return reply, posts

    return text, []
