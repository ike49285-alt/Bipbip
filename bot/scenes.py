"""Turns persona scene pools into generator-ready prompts.

Deliberately independent of *how* you generate. The same strings paste into a
browser generator, POST to a hosted API, or feed a notebook running your own
LoRA -- so this module never has to change when the generation backend does.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Iterable

from bot.persona import Persona, Scene

# A selfie should read as a selfie, not a portrait session. These fragments are
# about the genre's visual language: phones held at arm's length, ordinary
# light, imperfect framing.
SELFIE_GRAMMAR = (
    "front-facing phone camera selfie",
    "arm's length",
    "candid",
    "natural available light",
    "slight handheld motion",
    "casual framing",
)

# What makes generated selfies read as fake: studio polish, and hands.
# A mirror shot is rear-camera with the phone visible -- the opposite of an
# arm's-length front-camera frame. Mixing the two yields a muddy prompt.
MIRROR_GRAMMAR = (
    "mirror selfie",
    "phone visible in the reflection",
    "natural available light",
    "slight handheld motion",
    "casual framing",
)

NEGATIVE = (
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
)


@dataclass(frozen=True)
class Shot:
    """One planned image: everything a generator needs, plus its provenance."""

    index: int
    prompt: str
    negative: str
    seed: int
    wardrobe: str
    location: str
    time: str
    activity: str

    def as_dict(self) -> dict:
        return asdict(self)


def build_prompt(persona: Persona, scene: Scene, *, subject: str | None = None) -> str:
    """Compose a single positive prompt.

    `subject` overrides the physical description of the character -- pass the
    anchor description once you have one, so every prompt anchors to the same
    person even before a LoRA exists.
    """
    who = subject or f"a woman, {persona.identity.name}"
    grammar = MIRROR_GRAMMAR if "mirror" in scene.activity.lower() else SELFIE_GRAMMAR
    return ", ".join((who, scene.as_prompt(), *grammar))


def negative_prompt() -> str:
    return ", ".join(NEGATIVE)


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
    seen = list(recent or [])
    shots: list[Shot] = []
    for i in range(count):
        scene = persona.sample_scene(recent=seen, rng=rng)
        seen.append(scene.key())
        shots.append(
            Shot(
                index=i,
                prompt=build_prompt(persona, scene, subject=subject),
                negative=negative_prompt(),
                seed=rng.randrange(2**31),
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
    """Human-readable block for pasting into a browser generator."""
    blocks = []
    for shot in shots:
        blocks.append(
            f"--- {shot.index + 1} "
            f"({shot.activity} / {shot.location})\n"
            f"{shot.prompt}\n"
            f"negative: {shot.negative}\n"
            f"seed: {shot.seed}"
        )
    return "\n\n".join(blocks)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Emit generator-ready selfie prompts.")
    ap.add_argument("-n", "--count", type=int, default=10)
    ap.add_argument("--subject", default=None, help="anchor description, once you have one")
    ap.add_argument("--seed", type=int, default=None, help="make the plan reproducible")
    ap.add_argument("--json", action="store_true", help="emit a manifest instead of paste text")
    ap.add_argument("--persona", default="persona.json")
    args = ap.parse_args()

    persona = Persona.load(args.persona)
    rng = random.Random(args.seed) if args.seed is not None else None
    shots = plan_shoot(persona, args.count, subject=args.subject, rng=rng)
    print(as_manifest(shots) if args.json else as_paste_list(shots))
