"""The posting bot: one image, one caption, one tweet.

Wiring only. The image comes from `images`, the caption from the same generator
and ranker the interactive tool uses, and the posting from `twitter`.

Two things are deliberately not configurable. Alt text always says the image is
AI-generated, because it costs none of the 280 characters and tells anyone
using a screen reader -- or anyone who checks -- what they are looking at. And
the persona is validated before anything is rendered, so a bad persona file
fails before it costs an API call.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from . import images, personas, twitter
from .generate import propose
from .llm import BackendError, GrammarBackend, detect as detect_llm
from .rank import Ranked, rank
from .score import WEIGHTS

DEFAULT_PERSONA = Path("persona.json")
ALT_PREFIX = "AI-generated image."


@dataclass
class Persona:
    name: str = "her"
    look: str = ""
    voice: str = personas.DEFAULT_PERSONA
    disclosure: str = "AI-generated persona. Not a real person."
    scenes: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PERSONA) -> Persona:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} should contain a JSON object")

        persona = cls(
            name=str(data.get("name") or "her"),
            look=str(data.get("look") or ""),
            voice=str(data.get("voice") or personas.DEFAULT_PERSONA),
            disclosure=str(data.get("disclosure") or cls.disclosure),
            scenes=[str(s) for s in data.get("scenes", []) if str(s).strip()],
            topics=[str(t) for t in data.get("topics", []) if str(t).strip()],
        )
        if not persona.look:
            raise ValueError(f"{path} needs a 'look' describing the character")
        if not persona.scenes:
            raise ValueError(f"{path} needs at least one entry in 'scenes'")
        if not persona.topics:
            raise ValueError(f"{path} needs at least one entry in 'topics'")

        personas.get(persona.voice)          # unknown voice fails here, not mid-run
        images.validate_look(persona.look)   # and a bad look fails before any API call
        return persona


@dataclass
class Draft:
    """Everything a post is made of, before it becomes one."""

    caption: str
    scene: str
    topic: str
    prompt: str
    alt_text: str
    image: bytes | None = None
    ranked: list[Ranked] = field(default_factory=list)


def choose(persona: Persona, seed: int | None = None) -> tuple[str, str, random.Random]:
    """Pick a scene and topic. Seeded by the day, so a run is reproducible."""
    if seed is None:
        seed = int(dt.date.today().strftime("%Y%m%d"))
    rng = random.Random(seed)
    return rng.choice(persona.scenes), rng.choice(persona.topics), rng


def write_caption(
    persona: Persona, topic: str, backend=None, weights: dict | None = None, seed: int | None = None
) -> list[Ranked]:
    """Candidate captions, best first. Falls back to the grammar with no model."""
    backend = backend or detect_llm()
    try:
        candidates = propose(topic, persona=persona.voice, n=8, backend=backend, seed=seed)
    except BackendError:
        # A dead model must not stop the post; the grammar always answers.
        candidates = propose(topic, persona=persona.voice, backend=GrammarBackend(),
                             pool=300, seed=seed)
    if not candidates:
        raise ValueError("no caption could be generated for that topic")
    return rank(candidates, weights=weights or WEIGHTS)


def build(
    persona: Persona,
    image_backend=None,
    llm_backend=None,
    seed: int | None = None,
    render: bool = True,
) -> Draft:
    """Assemble a post. `render=False` skips the image call for a cheap dry run."""
    scene, topic, _ = choose(persona, seed)
    prompt = images.build_prompt(persona.look, scene)
    ranked = write_caption(persona, topic, backend=llm_backend, seed=seed)

    alt = f"{ALT_PREFIX} {persona.name}, {scene}."
    draft = Draft(caption=ranked[0].text, scene=scene, topic=topic,
                  prompt=prompt, alt_text=alt, ranked=ranked)

    if render:
        backend = image_backend or images.detect()
        draft.image = backend.render(prompt, seed=seed)
    return draft


def publish(draft: Draft, creds: twitter.Credentials | None = None) -> dict:
    """Upload the image and post the caption. Raises rather than half-posting."""
    creds = creds or twitter.Credentials.from_env()
    if not creds.complete():
        raise twitter.TwitterError(
            "missing credentials: " + ", ".join(creds.missing())
            + ". Set them as repository secrets."
        )
    if draft.image is None:
        raise twitter.TwitterError("nothing to post: the draft has no image")

    media_id = twitter.upload_media(creds, draft.image, alt_text=draft.alt_text)
    return twitter.post(creds, draft.caption, media_ids=[media_id])
