"""Posting to X.

OAuth 1.0a signed with the standard library -- hmac, base64 and urllib -- so the
package keeps its no-dependency promise. Media upload needs OAuth 1.0a, so
that is the flow used throughout rather than mixing two auth schemes.

Credentials come from the environment and are expected to be GitHub Actions
secrets. Nothing here writes them anywhere, and `redact` exists so a traceback
or a log line cannot carry one.
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
TWEET_URL = "https://api.x.com/2/tweets"
TIMEOUT = 120.0

# X's own limit, and the reason captions are scored against 280 everywhere else.
MAX_TWEET = 280
MAX_MEDIA_BYTES = 5 * 1024 * 1024

ENV = {
    "consumer_key": "X_API_KEY",
    "consumer_secret": "X_API_SECRET",
    "token": "X_ACCESS_TOKEN",
    "token_secret": "X_ACCESS_SECRET",
}


class TwitterError(RuntimeError):
    """The request reached X and was refused, or never got there."""


def _quote(value: str) -> str:
    """RFC 3986 percent-encoding, which OAuth requires and quote() does not do by default."""
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
        return all([self.consumer_key, self.consumer_secret, self.token, self.token_secret])

    def missing(self) -> list[str]:
        return [var for field, var in ENV.items() if not getattr(self, field)]

    def redact(self, text: str) -> str:
        """Strip any credential that made it into a message before it is shown."""
        for value in (self.consumer_key, self.consumer_secret, self.token, self.token_secret):
            if value and len(value) > 6:
                text = text.replace(value, "<redacted>")
        return text


def sign(creds: Credentials, method: str, url: str, params: dict | None = None) -> str:
    """Build an OAuth 1.0a Authorization header.

    Only query parameters are signed. A JSON or multipart body is not part of
    the signature base string, which is what lets media upload work at all.
    """
    oauth = {
        "oauth_consumer_key": creds.consumer_key,
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": creds.token,
        "oauth_version": "1.0",
    }

    signing_params = {**(params or {}), **oauth}
    encoded = "&".join(
        f"{_quote(k)}={_quote(v)}" for k, v in sorted(signing_params.items())
    )
    base = "&".join([method.upper(), _quote(url), _quote(encoded)])
    key = f"{_quote(creds.consumer_secret)}&{_quote(creds.token_secret)}".encode()

    digest = hmac.new(key, base.encode(), hashlib.sha1).digest()
    oauth["oauth_signature"] = base64.b64encode(digest).decode()

    return "OAuth " + ", ".join(f'{_quote(k)}="{_quote(v)}"' for k, v in sorted(oauth.items()))


def _multipart(field: str, filename: str, payload: bytes) -> tuple[bytes, str]:
    boundary = "----thirsttrap" + secrets.token_hex(12)
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        payload,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


def _send(request: urllib.request.Request, creds: Credentials, what: str) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:400]
        except Exception:
            pass
        raise TwitterError(creds.redact(f"{what} failed ({exc.code}): {detail or exc}")) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise TwitterError(creds.redact(f"{what} could not reach X: {exc}")) from exc

    try:
        return json.loads(raw) if raw else {}
    except ValueError as exc:
        raise TwitterError(f"{what} returned a non-JSON reply") from exc


def upload_media(creds: Credentials, image: bytes, alt_text: str = "") -> str:
    """Upload an image and return its media id."""
    if not image:
        raise TwitterError("no image data to upload")
    if len(image) > MAX_MEDIA_BYTES:
        raise TwitterError(f"image is {len(image) // 1024} KB; X's limit is 5 MB")

    body, content_type = _multipart("media", "image.jpg", image)
    request = urllib.request.Request(
        MEDIA_URL, data=body, method="POST",
        headers={"Authorization": sign(creds, "POST", MEDIA_URL), "Content-Type": content_type},
    )
    data = _send(request, creds, "media upload")

    media_id = data.get("media_id_string") or data.get("media_id")
    if not media_id:
        raise TwitterError("X accepted the upload but returned no media id")
    media_id = str(media_id)

    if alt_text:
        _set_alt_text(creds, media_id, alt_text)
    return media_id


def _set_alt_text(creds: Credentials, media_id: str, alt_text: str) -> None:
    """Alt text is a courtesy, not a requirement -- a failure here must not lose the post."""
    url = "https://upload.twitter.com/1.1/media/metadata/create.json"
    payload = json.dumps({"media_id": media_id, "alt_text": {"text": alt_text[:1000]}}).encode()
    request = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Authorization": sign(creds, "POST", url), "Content-Type": "application/json"},
    )
    try:
        _send(request, creds, "alt text")
    except TwitterError:
        pass


def post(creds: Credentials, text: str, media_ids: list[str] | None = None) -> dict:
    """Publish a tweet. Returns X's response."""
    if not text.strip() and not media_ids:
        raise TwitterError("refusing to post an empty tweet")
    if len(text) > MAX_TWEET:
        raise TwitterError(f"caption is {len(text)} characters; the limit is {MAX_TWEET}")

    payload: dict = {"text": text}
    if media_ids:
        payload["media"] = {"media_ids": list(media_ids)}

    request = urllib.request.Request(
        TWEET_URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": sign(creds, "POST", TWEET_URL),
                 "Content-Type": "application/json"},
    )
    return _send(request, creds, "posting")
