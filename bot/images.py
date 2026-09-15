"""Making the picture.

Nothing here draws anything; both backends call a service. Pollinations needs
no key, so a dry run works before signing up for anything. Anything speaking
POST /v1/images/generations covers the rest.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 180.0

ENV_URL = "IMAGE_URL"
ENV_KEY = "IMAGE_KEY"
ENV_MODEL = "IMAGE_MODEL"

JPEG, PNG = b"\xff\xd8", b"\x89PNG"


class ImageError(RuntimeError):
    """No usable image came back."""


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def render(prompt: str, seed: int | None = None) -> tuple[bytes, str]:
    """Return the image bytes and a description of what made them."""
    url = os.environ.get(ENV_URL, "")
    return _openai(url, prompt) if url else _pollinations(prompt, seed)


def _pollinations(prompt: str, seed: int | None) -> tuple[bytes, str]:
    model = os.environ.get(ENV_MODEL) or "flux"
    query = {"model": model, "nologo": "true", "width": 1024, "height": 1024}
    if seed is not None:
        query["seed"] = seed
    url = ("https://image.pollinations.ai/prompt/"
           + urllib.parse.quote(prompt, safe="") + "?" + urllib.parse.urlencode(query))

    try:
        data = _get(url)
    except (urllib.error.URLError, OSError) as exc:
        raise ImageError(f"pollinations: {exc}") from exc
    if not data.startswith((JPEG, PNG)):
        raise ImageError("pollinations returned something that is not an image")
    return data, f"pollinations {model}"


def _openai(url: str, prompt: str) -> tuple[bytes, str]:
    model = os.environ.get(ENV_MODEL) or "flux"
    key = os.environ.get(ENV_KEY, "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = json.dumps({
        "model": model, "prompt": prompt, "n": 1,
        "width": 1024, "height": 1024, "response_format": "b64_json",
    }).encode()

    try:
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:300]
        except Exception:
            pass
        raise ImageError(f"image provider refused ({exc.code}){': ' + detail if detail else ''}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ImageError(f"could not reach {url}: {exc}") from exc

    return _decode(data, model), f"{model} @ {urllib.parse.urlparse(url).netloc}"


def _decode(data: dict, model: str) -> bytes:
    """This endpoint shape returns either inline base64 or a URL to fetch."""
    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        raise ImageError(f"{model} returned no image")

    row = rows[0] or {}
    if row.get("b64_json"):
        try:
            return base64.b64decode(row["b64_json"])
        except (ValueError, TypeError) as exc:
            raise ImageError(f"{model} returned unreadable base64") from exc
    if row.get("url"):
        try:
            return _get(row["url"])
        except (urllib.error.URLError, OSError) as exc:
            raise ImageError(f"{model} image url: {exc}") from exc
    raise ImageError(f"{model} returned neither b64_json nor url")
