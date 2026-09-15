"""Image generation for a disclosed AI persona.

Nothing here draws anything -- every backend calls an API the operator holds a
key to. Stdlib `urllib` only, so the package still has no dependencies.

The prompt builder enforces bounds that are not configurable: the subject is an
adult, fictional, and not a likeness of any real person. Those are appended
after the persona's own description so a persona file cannot remove them.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

GENERATE_TIMEOUT = 180.0

ENV_BACKEND = "THIRSTTRAP_IMAGE_BACKEND"
ENV_KEY = "THIRSTTRAP_IMAGE_KEY"
ENV_MODEL = "THIRSTTRAP_IMAGE_MODEL"
ENV_ACCOUNT = "THIRSTTRAP_IMAGE_ACCOUNT"

# Appended to every prompt, after the persona's text, and not overridable.
BOUNDS = (
    "a fictional adult woman in her late twenties, not resembling any real or "
    "identifiable person, fully clothed, suggestive through pose and framing "
    "rather than nudity, no explicit or anatomical content, tasteful editorial "
    "photography"
)


class ImageError(RuntimeError):
    """The backend was reachable but produced no usable image."""


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_bytes(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


# Appending BOUNDS only *contradicts* a prompt that asks for something else, and
# image models resolve contradictions unpredictably. These are refused outright.
BANNED_AGE = (
    "child", "kid", "minor", "underage", "teen", "teenage", "schoolgirl", "loli",
    "young girl", "little girl", "preteen", "highschool", "high school",
)
BANNED_EXPLICIT = (
    "nude", "nudes", "naked", "topless", "bottomless", "nsfw", "explicit", "porn",
    "pornographic", "hardcore", "genital", "nipple", "areola", "spread legs",
    "sex act", "penetrat", "cum", "fellat",
)


class PersonaRejected(ValueError):
    """The persona asked for something the bounds exist to prevent."""


def validate_look(text: str) -> None:
    """Refuse a persona file that asks for a minor or for explicit content."""
    lowered = " " + " ".join(text.lower().split()) + " "
    for term in BANNED_AGE:
        if term in lowered:
            raise PersonaRejected(
                f"persona mentions {term!r}. Subjects must be adults; this is not "
                f"a setting."
            )
    for term in BANNED_EXPLICIT:
        if term in lowered:
            raise PersonaRejected(
                f"persona asks for {term!r}. This posts to a public timeline and "
                f"stays suggestive rather than explicit."
            )


def build_prompt(look: str, scene: str = "") -> str:
    """Persona description, then the scene, then the bounds that always apply."""
    validate_look(f"{look} {scene}")
    parts = [p.strip() for p in (look, scene) if p and p.strip()]
    parts.append(BOUNDS)
    return ", ".join(parts)


@dataclass
class ImageBackend:
    name: str = "image"
    key: str = ""
    model: str = ""

    def available(self) -> bool:
        raise NotImplementedError

    def render(self, prompt: str, seed: int | None = None) -> bytes:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


@dataclass
class PollinationsBackend(ImageBackend):
    """No key, no account. Good for a dry run before anyone signs up anywhere."""

    name: str = "pollinations"
    model: str = "flux"

    def available(self) -> bool:
        return True

    def render(self, prompt: str, seed: int | None = None) -> bytes:
        query = {"model": self.model, "nologo": "true", "width": 1024, "height": 1024}
        if seed is not None:
            query["seed"] = seed
        url = (
            "https://image.pollinations.ai/prompt/"
            + urllib.parse.quote(prompt, safe="")
            + "?" + urllib.parse.urlencode(query)
        )
        try:
            data = _fetch_bytes(url, GENERATE_TIMEOUT)
        except (urllib.error.URLError, OSError) as exc:
            raise ImageError(f"pollinations: {exc}") from exc
        if not data.startswith((b"\xff\xd8", b"\x89PNG")):
            raise ImageError("pollinations returned something that is not an image")
        return data

    def describe(self) -> str:
        return f"pollinations {self.model} (no key)"


@dataclass
class TogetherBackend(ImageBackend):
    name: str = "together"
    model: str = "black-forest-labs/FLUX.1-schnell-Free"

    def available(self) -> bool:
        return bool(self.key)

    def render(self, prompt: str, seed: int | None = None) -> bytes:
        payload = {"model": self.model, "prompt": prompt, "n": 1,
                   "width": 1024, "height": 1024, "response_format": "b64_json"}
        if seed is not None:
            payload["seed"] = seed
        try:
            data = _post("https://api.together.xyz/v1/images/generations", payload,
                         {"Authorization": f"Bearer {self.key}"}, GENERATE_TIMEOUT)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ImageError(f"together: {exc}") from exc
        return _decode_openai_shape(data, "together")

    def describe(self) -> str:
        return f"together {self.model}"


@dataclass
class OpenAIImageBackend(ImageBackend):
    """Anything exposing POST /v1/images/generations."""

    name: str = "openai-compat"
    host: str = "https://api.openai.com"
    model: str = "gpt-image-1"

    def available(self) -> bool:
        return bool(self.key)

    def render(self, prompt: str, seed: int | None = None) -> bytes:
        payload = {"model": self.model, "prompt": prompt, "n": 1, "size": "1024x1024"}
        try:
            data = _post(f"{self.host.rstrip('/')}/v1/images/generations", payload,
                         {"Authorization": f"Bearer {self.key}"}, GENERATE_TIMEOUT)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ImageError(f"{self.host}: {exc}") from exc
        return _decode_openai_shape(data, self.host)

    def describe(self) -> str:
        return f"openai-compat {self.model} @ {self.host}"


def _decode_openai_shape(data: dict, who: str) -> bytes:
    """Both b64_json and url come back from this endpoint shape; accept either."""
    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        raise ImageError(f"{who} returned no image")
    row = rows[0] or {}
    if row.get("b64_json"):
        try:
            return base64.b64decode(row["b64_json"])
        except (ValueError, TypeError) as exc:
            raise ImageError(f"{who} returned unreadable base64") from exc
    if row.get("url"):
        try:
            return _fetch_bytes(row["url"], GENERATE_TIMEOUT)
        except (urllib.error.URLError, OSError) as exc:
            raise ImageError(f"{who} image url: {exc}") from exc
    raise ImageError(f"{who} returned neither b64_json nor url")


def build(choice: str, key: str = "", model: str = "") -> ImageBackend:
    key = key or os.environ.get(ENV_KEY, "")
    model = model or os.environ.get(ENV_MODEL, "")

    if choice == "pollinations":
        backend = PollinationsBackend()
    elif choice == "together":
        backend = TogetherBackend(key=key)
    elif choice in {"openai", "openai-compat"}:
        backend = OpenAIImageBackend(key=key)
    else:
        raise ValueError(
            f"unknown image backend {choice!r}; try pollinations, together or openai-compat"
        )
    if model:
        backend.model = model
    return backend


def detect() -> ImageBackend:
    """Whatever is configured, else the keyless one so a dry run always works."""
    choice = os.environ.get(ENV_BACKEND, "").strip().lower()
    if choice:
        return build(choice)
    if os.environ.get(ENV_KEY):
        return build("together")
    return build("pollinations")
