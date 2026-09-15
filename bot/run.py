"""One post: pick a moment, write a caption, make a picture, publish it."""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass
from pathlib import Path

from . import captions, images, x
from .persona import Persona

HISTORY = Path("posted.json")
KEEP = 30          # enough to stop reruns without growing forever
LOOKBACK = 8       # how many recent posts a new one must not resemble


@dataclass
class Post:
    scene: str
    topic: str
    caption: str
    alt: str
    prompt: str
    image: bytes | None = None
    source: str = ""
    considered: list[str] = None


def load_history(path: Path | None = None) -> list[dict]:
    path = Path(path) if path else HISTORY
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
    except (OSError, ValueError):
        return []   # a missing or damaged log must not stop a post


def save_history(rows: list[dict], path: Path | None = None) -> None:
    path = Path(path) if path else HISTORY
    path.write_text(json.dumps(rows[-KEEP:], indent=2), encoding="utf-8")


def choose(persona: Persona, history: list[dict], seed: int | None = None) -> tuple[str, str]:
    """A scene and topic, preferring ones not used recently."""
    if seed is None:
        seed = int(dt.date.today().strftime("%Y%m%d"))
    rng = random.Random(seed)

    recent = {(r.get("scene"), r.get("topic")) for r in history[-LOOKBACK:]}
    used_scenes = {r.get("scene") for r in history[-LOOKBACK:]}
    used_topics = {r.get("topic") for r in history[-LOOKBACK:]}

    scenes = [s for s in persona.scenes if s not in used_scenes] or persona.scenes
    topics = [t for t in persona.topics if t not in used_topics] or persona.topics

    for _ in range(12):
        pair = (rng.choice(scenes), rng.choice(topics))
        if pair not in recent:
            return pair
    return rng.choice(scenes), rng.choice(topics)


def build(persona: Persona, seed: int | None = None, render: bool = True,
          history: list[dict] | None = None, path: Path | None = None) -> Post:
    history = load_history(path) if history is None else history
    scene, topic = choose(persona, history, seed)

    candidates = captions.ask(persona, topic)
    caption = captions.pick(candidates, avoid=[r.get("caption", "") for r in history[-LOOKBACK:]])

    post = Post(scene=scene, topic=topic, caption=caption, alt=persona.alt_text(scene),
                prompt=persona.prompt(scene), considered=candidates)
    if render:
        post.image, post.source = images.render(post.prompt, seed=seed)
    return post


def publish(post: Post, creds: x.Credentials | None = None) -> str:
    creds = creds or x.Credentials.from_env()
    if not creds.complete():
        raise x.XError("missing credentials: " + ", ".join(creds.missing()))
    if post.image is None:
        raise x.XError("nothing to post: no image was made")

    media_id = x.upload(creds, post.image, alt=post.alt)
    return x.tweet(creds, post.caption, media_ids=[media_id])


def record(post: Post, tweet_id: str, history: list[dict] | None = None,
           path: Path | None = None) -> None:
    history = load_history(path) if history is None else history
    history.append({
        "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "id": tweet_id, "scene": post.scene, "topic": post.topic, "caption": post.caption,
    })
    save_history(history, path)
