"""The character, and the limits on what she can be asked to be.

Two of those limits are not settings. The subject is an adult, and she is
fictional and resembles no real person -- both are appended to every image
prompt, and a persona file that asks otherwise is refused rather than
contradicted. Appending "adult" to a prompt asking for a child produces a
contradiction, and an image model resolves contradictions however it likes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Appended to every image prompt, after the persona's own words.
BOUNDS = (
    "a fictional adult woman in her late twenties, resembling no real or "
    "identifiable person, fully clothed, suggestive through pose and framing "
    "rather than nudity, nothing explicit or anatomical, editorial photography"
)

REFUSED_AGE = (
    "child", "kid", "minor", "underage", "teen", "teenage", "schoolgirl",
    "loli", "young girl", "little girl", "preteen", "high school", "highschool",
)
REFUSED_EXPLICIT = (
    "nude", "nudes", "naked", "topless", "bottomless", "nsfw", "explicit",
    "porn", "pornographic", "hardcore", "genital", "nipple", "areola",
    "sex act", "penetrat", "fellat",
)


class Refused(ValueError):
    """The persona asked for the thing the bounds exist to prevent."""


def check(text: str) -> None:
    lowered = " " + " ".join(text.lower().split()) + " "
    for term in REFUSED_AGE:
        if term in lowered:
            raise Refused(f"persona mentions {term!r}; subjects must be adults")
    for term in REFUSED_EXPLICIT:
        if term in lowered:
            raise Refused(f"persona asks for {term!r}; this stays suggestive, not explicit")


@dataclass
class Persona:
    name: str
    look: str
    voice: str = "playful and confident, teasing rather than pleading"
    scenes: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    def prompt(self, scene: str) -> str:
        """Image prompt: her look, the scene, then the bounds."""
        return ", ".join(p for p in (self.look, scene, BOUNDS) if p)

    def alt_text(self, scene: str) -> str:
        """Always says the image is generated. Costs none of the 280 characters."""
        return f"AI-generated image. {self.name}, {scene}."

    @classmethod
    def load(cls, path: str | Path = "persona.json") -> Persona:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} should hold a JSON object")

        for required in ("name", "look", "scenes", "topics"):
            if not data.get(required):
                raise ValueError(f"{path} needs a non-empty {required!r}")

        persona = cls(
            name=str(data["name"]),
            look=str(data["look"]),
            voice=str(data.get("voice") or cls.voice),
            scenes=[str(s) for s in data["scenes"] if str(s).strip()],
            topics=[str(t) for t in data["topics"] if str(t).strip()],
        )
        # Fail here, before anything is generated or billed.
        check(" ".join([persona.look, *persona.scenes]))
        return persona
