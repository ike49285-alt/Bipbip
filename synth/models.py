"""Open-weights backends, and the cleanup that makes their output usable.

Two adapters cover nearly everything: native Ollama for a local box, and any
OpenAI-compatible `/v1/chat/completions` endpoint - which is llama.cpp's
server, vLLM, LM Studio, Together, Groq, OpenRouter, DeepInfra and most of the
rest. Both are plain stdlib HTTP, so this adds no dependency.

THE CLEANUP IS NOT AN AFTERTHOUGHT. An instruction-tuned open model asked for
one social post will hand you:

    Sure! Here's a flirty post for Sophie:

    "made a playlist for rainy afternoons. the annoying part was the ordering."

    Let me know if you'd like a different tone!

Three of those four lines are not the post. Every one of them would sail
through the content gates - it is not too long, it contradicts no canon, it
claims to be nobody - and go out with the quote marks still on. So the
completion is stripped before it reaches the pipeline, and the stripping is
the part with tests on it, because it is pure and because it is where the
real bugs are.

Local models are also the reason `temperature` defaults high here. A persona
is a voice, and a voice at temperature 0.2 is a press release.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

#: Openers a model uses to announce what it is about to do. Matched only at the
#: very start, and only when followed by the actual content, so a post that
#: genuinely begins "okay so" survives.
_PREAMBLE = re.compile(
    r"""^\s*(
        (sure|okay|ok|alright|certainly|absolutely|got\ it|here)\b[^\n:]{0,60}[:!.]\s*
        | here('?s| is)\ (a|the|one|your)\b[^\n:]{0,60}:\s*
        | (post|caption|tweet|output|response|answer)\s*:\s*
    )""",
    re.I | re.X,
)

#: Trailing offers of further service. Cut from the last line only.
_OUTRO = re.compile(
    r"""\n\s*(
        (let\ me\ know|want\ me\ to|would\ you\ like|hope\ (this|that)|
         i\ can\ (also|write)|feel\ free|shall\ i|if\ you('d|\ would)\ like)
        \b.*
    )\s*$""",
    re.I | re.X | re.S,
)

_FENCE = re.compile(r"^\s*```[a-z]*\s*\n(.*?)\n\s*```\s*$", re.S | re.I)


def clean_completion(text: str) -> str:
    """Strip everything a chat model adds around the thing you asked for.

    Deliberately conservative about the middle of the text - it removes
    wrappers, never content. Over-cleaning silently mangles a good post, which
    is harder to notice than under-cleaning, because the result still reads
    like something the character might have said.
    """
    if not text:
        return ""
    s = text.strip()

    fence = _FENCE.match(s)
    if fence:
        s = fence.group(1).strip()

    prev = None
    while prev != s:                       # "Sure! Here's a post:" is two
        prev = s
        s = _PREAMBLE.sub("", s, count=1).strip()

    s = _OUTRO.sub("", s).strip()

    # Surrounding quotes, but only a MATCHED pair wrapping the whole thing -
    # a post that merely ends on a quoted phrase keeps its punctuation.
    for open_q, close_q in (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")):
        if len(s) > 1 and s[0] == open_q and s[-1] == close_q \
                and close_q not in s[1:-1]:
            s = s[1:-1].strip()
            break

    return re.sub(r"[ \t]+", " ", s).strip()


class ModelError(RuntimeError):
    """The backend could not be reached, or returned nothing usable."""


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise ModelError(f"{url} returned {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise ModelError(
            f"could not reach {url}: {e.reason}. Is the server running?") from e


@dataclass
class OllamaCompleter:
    """Native Ollama. `ollama serve`, then `ollama pull <model>`."""
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.9
    top_p: float = 0.95
    num_predict: int = 400
    timeout: float = 120.0

    def __call__(self, prompt: str) -> str:
        data = _post(
            f"{self.host.rstrip('/')}/api/generate",
            {"model": self.model, "prompt": prompt, "stream": False,
             "options": {"temperature": self.temperature, "top_p": self.top_p,
                         "num_predict": self.num_predict}},
            {}, self.timeout)
        return clean_completion(data.get("response", ""))


@dataclass
class OpenAICompatCompleter:
    """Any `/v1/chat/completions` endpoint.

    llama.cpp's server, vLLM, LM Studio, Together, Groq, OpenRouter,
    DeepInfra - they all speak this. `base_url` is the root, without `/v1`.
    """
    model: str
    base_url: str = "http://localhost:8000"
    api_key: str = ""
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 400
    timeout: float = 120.0

    def __call__(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            {"model": self.model,
             "messages": [{"role": "user", "content": prompt}],
             "temperature": self.temperature, "top_p": self.top_p,
             "max_tokens": self.max_tokens},
            headers, self.timeout)
        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"no choices in response: {str(data)[:200]}")
        msg = choices[0].get("message") or {}
        return clean_completion(msg.get("content") or "")


def from_env(model: str | None = None):
    """Build a completer from the environment, or return None if unconfigured.

    Returning None rather than raising is deliberate: every command in this
    project must keep working with no model at all, so that the gates and the
    persona tooling can be used and reviewed before anyone stands a server up.

        SYNTH_BACKEND   ollama (default) | openai
        SYNTH_MODEL     model name
        SYNTH_BASE_URL  http://localhost:11434 or http://localhost:8000
        SYNTH_API_KEY   only for hosted open-weights providers
        SYNTH_TEMP      default 0.9 - a voice at 0.2 is a press release
    """
    name = model or os.environ.get("SYNTH_MODEL")
    if not name:
        return None
    backend = os.environ.get("SYNTH_BACKEND", "ollama").lower()
    temp = float(os.environ.get("SYNTH_TEMP", "0.9"))
    if backend in ("openai", "openai-compat", "vllm", "llamacpp", "lmstudio"):
        return OpenAICompatCompleter(
            model=name,
            base_url=os.environ.get("SYNTH_BASE_URL", "http://localhost:8000"),
            api_key=os.environ.get("SYNTH_API_KEY", ""), temperature=temp)
    return OllamaCompleter(
        model=name,
        host=os.environ.get("SYNTH_BASE_URL", "http://localhost:11434"),
        temperature=temp)
