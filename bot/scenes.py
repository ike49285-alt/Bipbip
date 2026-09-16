"""Turns persona scene pools into generator-ready prompts.

Deliberately independent of *how* you generate: the same strings paste into a
browser generator, POST to a hosted API, or feed a notebook running a LoRA.

Style matters more than it looks. A prompt grammar written for photography
("natural available light", negative "airbrushed") actively fights an
illustrated character, whose smooth shading is the point. The persona picks a
StyleProfile and every prompt is composed from it.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Iterable

from bot.persona import Persona, Scene


@dataclass(frozen=True)
class StyleProfile:
    """Rendering grammar for one visual style."""

    name: str
    grammar: tuple[str, ...]
    mirror_grammar: tuple[str, ...]
    negative: tuple[str, ...]

    def grammar_for(self, activity: str) -> tuple[str, ...]:
        # A mirror shot is rear-camera with the phone visible -- the opposite of
        # an arm's-length front-camera frame. Mixing the two muddies the prompt.
        return self.mirror_grammar if "mirror" in activity.lower() else self.grammar


ILLUSTRATED = StyleProfile(
    name="illustrated",
    grammar=(
        "digital illustration",
        "anime style",
        "cel shading",
        "clean lineart",
        "selfie composition",
        "arm's length framing",
    ),
    mirror_grammar=(
        "digital illustration",
        "anime style",
        "cel shading",
        "clean lineart",
        "mirror selfie",
        "phone visible in the reflection",
    ),
    negative=(
        "photorealistic",
        "photograph",
        "3d render",
        "extra fingers",
        "deformed hands",
        "malformed limbs",
        "watermark",
        "text",
        "logo",
        "multiple people",
        "breasts",
    ),
)

PHOTO = StyleProfile(
    name="photo",
    grammar=(
        "front-facing phone camera selfie",
        "arm's length",
        "candid",
        "natural available light",
        "slight handheld motion",
        "casual framing",
    ),
    mirror_grammar=(
        "mirror selfie",
        "phone visible in the reflection",
        "natural available light",
        "slight handheld motion",
        "casual framing",
    ),
    negative=(
        "studio lighting",
        "professional photography",
        "dslr",
        "posed",
        "airbrushed",
        "extra fingers",
        "deformed hands",
        "malformed limbs",
        "watermark",
        "text",
        "logo",
        "multiple people",
    ),
)

STYLES = {p.name: p for p in (ILLUSTRATED, PHOTO)}


def style_for(persona: Persona) -> StyleProfile:
    try:
        return STYLES[persona.visual.style]
    except KeyError:
        raise ValueError(
            f"unknown visual.style {persona.visual.style!r}; known: {sorted(STYLES)}"
        ) from None


@dataclass(frozen=True)
class Shot:
    """One planned image: everything a generator needs, plus its provenance."""

    index: int
    prompt: str
    negative: str
    seed: int
    style: str
    wardrobe: str
    location: str
    time: str
    activity: str

    def as_dict(self) -> dict:
        return asdict(self)


def build_prompt(persona: Persona, scene: Scene, *, subject: str | None = None) -> str:
    """Compose one positive prompt.

    Order is deliberate: subject first (it carries identity), then the signature
    accessories that never vary, then the scene, then the style grammar.
    """
    style = style_for(persona)
    parts = [subject or persona.visual.subject]
    parts.extend(persona.visual.signature)
    parts.append(scene.as_prompt())
    parts.extend(style.grammar_for(scene.activity))
    return ", ".join(parts)


def negative_prompt(persona: Persona) -> str:
    return ", ".join(style_for(persona).negative)


def plan_shoot(
    persona: Persona,
    count: int,
    *,
    subject: str | None = None,
    recent: list[tuple] | None = None,
    rng: random.Random | None = None,
) -> list[Shot]:
    """Plan `count` distinct shots, avoiding repeats within the recency window."""
    if count < 1:
        raise ValueError("count must be at least 1")
    rng = rng or random.Random()
    style, negative = style_for(persona), negative_prompt(persona)
    seen = list(recent or [])
    shots: list[Shot] = []
    for i in range(count):
        scene = persona.sample_scene(recent=seen, rng=rng)
        seen.append(scene.key())
        shots.append(
            Shot(
                index=i,
                prompt=build_prompt(persona, scene, subject=subject),
                negative=negative,
                seed=rng.randrange(2**31),
                style=style.name,
                wardrobe=scene.wardrobe,
                location=scene.location,
                time=scene.time,
                activity=scene.activity,
            )
        )
    return shots


def as_manifest(shots: Iterable[Shot]) -> str:
    """JSON the notebook half reads and the queue builder later consumes."""
    return json.dumps([s.as_dict() for s in shots], indent=2)


def as_paste_list(shots: Iterable[Shot]) -> str:
    """Human-readable blocks for pasting into a browser generator."""
    return "\n\n".join(
        f"--- {s.index + 1} ({s.activity} / {s.location})\n"
        f"{s.prompt}\n"
        f"negative: {s.negative}\n"
        f"seed: {s.seed}"
        for s in shots
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Emit generator-ready selfie prompts.")
    ap.add_argument("-n", "--count", type=int, default=10)
    ap.add_argument("--subject", default=None, help="override the persona's anchor description")
    ap.add_argument("--seed", type=int, default=None, help="make the plan reproducible")
    ap.add_argument("--json", action="store_true", help="emit a manifest instead of paste text")
    ap.add_argument("--persona", default="persona.json")
    args = ap.parse_args()

    persona = Persona.load(args.persona)
    rng = random.Random(args.seed) if args.seed is not None else None
    shots = plan_shoot(persona, args.count, subject=args.subject, rng=rng)
    print(as_manifest(shots) if args.json else as_paste_list(shots))
