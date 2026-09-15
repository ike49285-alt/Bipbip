"""Writing the caption.

One backend shape: POST /v1/chat/completions, which Groq, Gemini, OpenRouter,
Cerebras and a local Ollama all speak. No fallback generator -- a bot with no
model should fail loudly rather than post something worse in its name.

`pick` is a preference, not a measurement. It encodes four beliefs about what
reads well on a timeline; none of them has been tested against engagement data,
and the comment is here so nobody later mistakes the number for evidence.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

TIMEOUT = 120.0
MAX_TWEET = 280

ENV_URL = "CAPTION_URL"
ENV_KEY = "CAPTION_KEY"
ENV_MODEL = "CAPTION_MODEL"
DEFAULT_URL = "https://api.groq.com/openai/v1/chat/completions"

RULES = """\
You write captions for a photo posted to X by a fictional persona.

- One caption stands alone. Under 200 characters; shorter is better.
- No hashtags, no @-mentions, no links, no emoji.
- Do not describe the photo. The photo is already there. Say the thing the
  photo is not saying.
- Specific beats general. One idea. Cut every word doing nothing.
- Confidence, never neediness: no begging for replies, no fishing for
  compliments, nothing that reads as asking permission.
- She is an adult and fictional. Nothing explicit."""

# Phrases the timeline wore out years ago.
TIRED = (
    "game changer", "let that sink in", "read that again", "who else",
    "living rent free", "main character", "it's giving", "the way i",
    "thoughts?", "am i right", "just saying", "no because",
)

_BANNED = re.compile(r"#\w|@\w|https?://|www\.", re.IGNORECASE)
_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
_WORDS = re.compile(r"[a-z0-9']+")


class CaptionError(RuntimeError):
    """No usable caption came back."""


def _config() -> tuple[str, str, str]:
    url = os.environ.get(ENV_URL) or DEFAULT_URL
    key = os.environ.get(ENV_KEY, "")
    model = os.environ.get(ENV_MODEL, "")
    if not model:
        raise CaptionError(f"set {ENV_MODEL} to the model you want (and {ENV_KEY} if it needs one)")
    return url, key, model


def ask(persona, topic: str, n: int = 6) -> list[str]:
    """Ask the model for `n` candidate captions."""
    url, key, model = _config()
    prompt = (
        f"{RULES}\n\nHer register: {persona.voice}\n\n"
        f"She is posting about: {topic}\n\n"
        f'Reply with only a JSON array of {n} strings. Example: ["first", "second"]'
    )

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            pass
        raise CaptionError(f"{model} refused ({exc.code}){': ' + detail if detail else ''}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CaptionError(f"could not reach {url}: {exc}") from exc

    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    if not text:
        raise CaptionError("the model returned nothing")
    return parse(text)


def parse(text: str) -> list[str]:
    """Pull captions out of a JSON array, or out of a list it wrote instead."""
    unfenced = re.sub(r"^\s*```(?:\w+)?\s*|\s*```\s*$", "", text, flags=re.MULTILINE).strip()

    start, end = unfenced.find("["), unfenced.rfind("]")
    if start != -1 and end > start:
        try:
            rows = json.loads(unfenced[start : end + 1])
            if isinstance(rows, list):
                found = [str(r).strip() for r in rows if str(r).strip()]
                if found:
                    return found
        except ValueError:
            pass

    # Models drift off JSON; a numbered or bulleted list is the usual drift.
    out = []
    for line in unfenced.splitlines():
        line = re.sub(r'^\s*(?:[-*•]|\(?\d{1,2}[.):])\s*', "", line).strip()
        line = line.strip("\"'“”")
        if line and not (line.endswith(":") and len(line) < 60):
            out.append(line)
    return out


def usable(caption: str) -> bool:
    """Hard rules. A caption breaking one of these is not a candidate."""
    n = len(caption)
    if not 20 <= n <= MAX_TWEET:
        return False
    if _BANNED.search(caption) or _EMOJI.search(caption):
        return False
    if caption.count("!") > 1:
        return False
    return True


def preference(caption: str) -> float:
    """A preference, not a measurement. Four beliefs, none of them validated:
    that shorter travels further, that second person outperforms third, that a
    short closing line lands, and that worn-out phrases cost more than they pay.
    """
    words = _WORDS.findall(caption.lower())
    sentences = [s for s in re.split(r"[.!?]+", caption) if s.strip()]
    score = 0.0

    score += 2.0 if len(caption) <= 120 else (1.0 if len(caption) <= 180 else 0.0)
    score += 1.5 if any(w in {"you", "your", "youre"} for w in words) else 0.0
    if len(sentences) >= 2 and len(_WORDS.findall(sentences[-1])) <= 6:
        score += 1.5
    score -= 2.0 * sum(1 for phrase in TIRED if phrase in caption.lower())
    return score


def similar(a: str, b: str) -> float:
    """Word overlap, for not repeating last week's post."""
    wa, wb = set(_WORDS.findall(a.lower())), set(_WORDS.findall(b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def pick(candidates: list[str], avoid: list[str] | None = None) -> str:
    """The best usable caption that is not a rerun."""
    avoid = avoid or []
    fresh = [
        c for c in candidates
        if usable(c) and all(similar(c, old) < 0.5 for old in avoid)
    ]
    if not fresh:
        raise CaptionError(
            f"none of {len(candidates)} captions were usable"
            + (" and new" if avoid else "")
        )
    return max(fresh, key=preference)
