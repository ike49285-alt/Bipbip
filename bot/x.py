"""Posting to X.

OAuth 1.0a signed with hmac and urllib, so this has no dependencies. Media
upload requires OAuth 1.0a, so it is used for everything rather than mixing
two auth schemes.

Credentials come from the environment, which on a scheduled run means GitHub
Actions secrets. `redact` is applied to every error message, because Actions
logs are public on a public repo and a traceback is a fine place to leak a key.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

MEDIA_URL = "https://upload.twitter.com/1.1/media/upload.json"
ALT_URL = "https://upload.twitter.com/1.1/media/metadata/create.json"
TWEET_URL = "https://api.x.com/2/tweets"

TIMEOUT = 120.0
MAX_TWEET = 280
MAX_IMAGE = 5 * 1024 * 1024

ENV = {
    "consumer_key": "X_API_KEY",
    "consumer_secret": "X_API_SECRET",
    "token": "X_ACCESS_TOKEN",
    "token_secret": "X_ACCESS_SECRET",
}


class XError(RuntimeError):
    """X refused the request, or it never arrived."""


def quote(value) -> str:
    """RFC 3986. quote()'s defaults leave / unescaped and encode ~; either
    breaks the signature, silently, with a 401 as the only symptom."""
    return urllib.parse.quote(str(value), safe="~-._")


@dataclass
class Credentials:
    consumer_key: str = ""
    consumer_secret: str = ""
    token: str = ""
    token_secret: str = ""

    @classmethod
    def from_env(cls) -> Credentials:
        return cls(**{field: os.environ.get(var, "") for field, var in ENV.items()})

    def complete(self) -> bool:
        return all(getattr(self, field) for field in ENV)

    def missing(self) -> list[str]:
        return [var for field, var in ENV.items() if not getattr(self, field)]

    def redact(self, text: str) -> str:
        for field in ENV:
            value = getattr(self, field)
            if value and len(value) > 6:
                text = text.replace(value, "<redacted>")
        return text


def header(creds: Credentials, method: str, url: str, params: dict | None = None) -> str:
    """An OAuth 1.0a Authorization header.

    Only query parameters are signed; a JSON or multipart body is not part of
    the base string, which is what lets the media upload work.
    """
    oauth = {
        "oauth_consumer_key": creds.consumer_key,
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": creds.token,
        "oauth_version": "1.0",
    }
    joined = "&".join(
        f"{quote(k)}={quote(v)}" for k, v in sorted({**(params or {}), **oauth}.items())
    )
    base = "&".join([method.upper(), quote(url), quote(joined)])
    key = f"{quote(creds.consumer_secret)}&{quote(creds.token_secret)}".encode()

    oauth["oauth_signature"] = base64.b64encode(
        hmac.new(key, base.encode(), hashlib.sha1).digest()
    ).decode()
    return "OAuth " + ", ".join(f'{quote(k)}="{quote(v)}"' for k, v in sorted(oauth.items()))


def _send(request, creds: Credentials, what: str) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:400]
        except Exception:
            pass
        raise XError(creds.redact(f"{what} failed ({exc.code}): {detail or exc}")) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise XError(creds.redact(f"{what} could not reach X: {exc}")) from exc

    try:
        return json.loads(raw) if raw else {}
    except ValueError as exc:
        raise XError(f"{what} returned a non-JSON reply") from exc


def upload(creds: Credentials, image: bytes, alt: str = "") -> str:
    if not image:
        raise XError("no image to upload")
    if len(image) > MAX_IMAGE:
        raise XError(f"image is {len(image) // 1024} KB; X's limit is 5 MB")

    boundary = "----bot" + secrets.token_hex(12)
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="media"; filename="image.jpg"\r\n',
        b"Content-Type: application/octet-stream\r\n\r\n",
        image,
        f"\r\n--{boundary}--\r\n".encode(),
    ])

    request = urllib.request.Request(
        MEDIA_URL, data=body, method="POST",
        headers={"Authorization": header(creds, "POST", MEDIA_URL),
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    data = _send(request, creds, "media upload")

    media_id = data.get("media_id_string") or data.get("media_id")
    if not media_id:
        raise XError("X accepted the upload but returned no media id")
    media_id = str(media_id)

    if alt:
        _alt_text(creds, media_id, alt)
    return media_id


def _alt_text(creds: Credentials, media_id: str, alt: str) -> None:
    """A courtesy, not a requirement: failing here must not lose the post."""
    payload = json.dumps({"media_id": media_id, "alt_text": {"text": alt[:1000]}}).encode()
    request = urllib.request.Request(
        ALT_URL, data=payload, method="POST",
        headers={"Authorization": header(creds, "POST", ALT_URL),
                 "Content-Type": "application/json"},
    )
    try:
        _send(request, creds, "alt text")
    except XError:
        pass


def tweet(creds: Credentials, text: str, media_ids: list[str] | None = None) -> str:
    """Publish, and return the new tweet's id."""
    if not text.strip():
        raise XError("refusing to post an empty caption")
    if len(text) > MAX_TWEET:
        raise XError(f"caption is {len(text)} characters; the limit is {MAX_TWEET}")

    payload: dict = {"text": text}
    if media_ids:
        payload["media"] = {"media_ids": list(media_ids)}

    request = urllib.request.Request(
        TWEET_URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": header(creds, "POST", TWEET_URL),
                 "Content-Type": "application/json"},
    )
    return str((_send(request, creds, "posting").get("data") or {}).get("id", ""))
