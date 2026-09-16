"""The persona: single source of truth for identity, voice, and bounds.

Every other module reads from a Persona and none of them keep persona state of
their own. Loading enforces the design's one hard rule -- a persona that does
not disclose itself as synthetic will not load.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


class PersonaError(ValueError):
    """Raised when a persona file is missing or internally inconsistent."""


@dataclass(frozen=True)
class Identity:
    name: str
    handle: str
    bio: str
    disclosure: str
    disclosure_short: str


@dataclass(frozen=True)
class Visual:
    style: str
    subject: str
    signature: tuple[str, ...]
    anchor_image: Path
    lora_path: Path
    identity_check: str
    identity_threshold: float
    wardrobe: tuple[str, ...]
    locations: tuple[str, ...]
    times: tuple[str, ...]
    activities: tuple[str, ...]
    mirror_locations: tuple[str, ...]
    recency_window: int


@dataclass(frozen=True)
class Voice:
    rules: tuple[str, ...]
    caption_max_chars: int
    candidates_per_caption: int
    similarity_threshold: float
    history_window: int
    constraints: dict
    max_shape_repeats: int


@dataclass(frozen=True)
class Bounds:
    always_answer_truthfully: tuple[str, ...]
    hard_block_outbound: tuple[str, ...]
    refuse_to_supply: tuple[str, ...]
    deflect: tuple[str, ...]
    terminate_thread: tuple[str, ...]
    never_claim: tuple[str, ...]
    canned: dict


@dataclass(frozen=True)
class Scene:
    """One sampled setup for a selfie. Ordered so it renders as a prompt."""

    wardrobe: str
    location: str
    time: str
    activity: str

    def as_prompt(self) -> str:
        return f"{self.activity}, wearing {self.wardrobe}, {self.location}, {self.time}"

    def key(self) -> tuple[str, str, str, str]:
        return (self.wardrobe, self.location, self.time, self.activity)


@dataclass(frozen=True)
class Persona:
    identity: Identity
    visual: Visual
    voice: Voice
    facts: tuple[str, ...]
    bounds: Bounds
    source: Path | None = field(default=None, compare=False)

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = "persona.json") -> "Persona":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PersonaError(f"no persona file at {path}") from exc
        except json.JSONDecodeError as exc:
            raise PersonaError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(raw, source=path)

    @classmethod
    def from_dict(cls, raw: dict, source: Path | None = None) -> "Persona":
        for section in ("identity", "visual", "voice", "facts", "bounds"):
            if section not in raw:
                raise PersonaError(f"persona is missing the '{section}' section")

        ident = raw["identity"]
        for key in ("name", "handle", "bio", "disclosure", "disclosure_short"):
            if not str(ident.get(key, "")).strip():
                raise PersonaError(f"identity.{key} must be a non-empty string")

        vis, voc, bnd = raw["visual"], raw["voice"], raw["bounds"]

        if not raw["facts"]:
            raise PersonaError("persona needs at least one biographical fact")
        if not voc.get("rules"):
            raise PersonaError("voice.rules must list at least one concrete rule")
        if not bnd.get("always_answer_truthfully"):
            raise PersonaError(
                "bounds.always_answer_truthfully must not be empty -- the persona has "
                "to be able to admit what it is"
            )

        for pool in ("wardrobe", "locations", "times", "activities"):
            if not vis.get(pool):
                raise PersonaError(f"visual.{pool} must not be empty")

        if not str(vis.get("subject", "")).strip():
            raise PersonaError(
                "visual.subject must describe the anchor -- it is prepended to every "
                "generation prompt and is what holds identity together"
            )

        persona = cls(
            identity=Identity(
                name=ident["name"],
                handle=ident["handle"].lstrip("@"),
                bio=ident["bio"],
                disclosure=ident["disclosure"],
                disclosure_short=ident["disclosure_short"],
            ),
            visual=Visual(
                style=vis.get("style", "photo"),
                subject=vis["subject"],
                signature=tuple(vis.get("signature", ())),
                anchor_image=Path(vis["anchor_image"]),
                lora_path=Path(vis["lora_path"]),
                identity_check=vis.get("identity_check", "clip"),
                identity_threshold=float(vis.get("identity_threshold", 0.6)),
                wardrobe=tuple(vis["wardrobe"]),
                locations=tuple(vis["locations"]),
                times=tuple(vis["times"]),
                activities=tuple(vis["activities"]),
                mirror_locations=tuple(vis.get("mirror_locations") or vis["locations"]),
                recency_window=int(vis.get("recency_window", 12)),
            ),
            voice=Voice(
                rules=tuple(voc["rules"]),
                caption_max_chars=int(voc.get("caption_max_chars", 140)),
                candidates_per_caption=int(voc.get("candidates_per_caption", 4)),
                similarity_threshold=float(voc.get("similarity_threshold", 0.82)),
                history_window=int(voc.get("history_window", 50)),
                constraints=dict(voc.get("constraints") or {}),
                max_shape_repeats=int(voc.get("max_shape_repeats", 2)),
            ),
            facts=tuple(raw["facts"]),
            bounds=Bounds(
                always_answer_truthfully=tuple(bnd["always_answer_truthfully"]),
                hard_block_outbound=tuple(bnd.get("hard_block_outbound", ())),
                refuse_to_supply=tuple(bnd.get("refuse_to_supply", ())),
                deflect=tuple(bnd.get("deflect", ())),
                terminate_thread=tuple(bnd.get("terminate_thread", ())),
                never_claim=tuple(bnd.get("never_claim", ())),
                canned=dict(bnd.get("canned") or {}),
            ),
            source=source,
        )
        persona._check_disclosure()
        return persona

    def _check_disclosure(self) -> None:
        """A persona that hides what it is does not load. See DESIGN.md."""
        blob = f"{self.identity.bio} {self.identity.disclosure} {self.identity.disclosure_short}".lower()
        if not any(marker in blob for marker in ("ai", "a.i.", "bot", "synthetic", "generated", "not a real person")):
            raise PersonaError(
                "persona does not disclose that it is synthetic. Put an explicit "
                "marker (ai / bot / synthetic / generated / not a real person) in "
                "identity.bio or identity.disclosure."
            )

    # -- prompt surfaces -------------------------------------------------

    def voice_card(self) -> str:
        """Stable prefix shared by captioning and DMs.

        Byte-identical across every call, so it belongs at the front of the
        prompt behind a cache breakpoint.
        """
        rules = "\n".join(f"- {r}" for r in self.voice.rules)
        facts = "\n".join(f"- {f}" for f in self.facts)
        return (
            f"You write as {self.identity.name} (@{self.identity.handle}).\n"
            f"{self.identity.disclosure}\n\n"
            f"Voice rules, all of them binding:\n{rules}\n\n"
            f"Things true about her:\n{facts}\n"
        )

    def sample_scene(self, *, recent: list[tuple] | None = None, rng: random.Random | None = None) -> Scene:
        """Pick a setup, avoiding exact repeats inside the recency window."""
        rng = rng or random.Random()
        recent_keys = set(tuple(k) for k in (recent or [])[-self.visual.recency_window:])
        for _ in range(40):
            activity = rng.choice(self.visual.activities)
            # A mirror shot needs somewhere with a mirror.
            pool = (
                self.visual.mirror_locations
                if "mirror" in activity.lower()
                else self.visual.locations
            )
            scene = Scene(
                wardrobe=rng.choice(self.visual.wardrobe),
                location=rng.choice(pool),
                time=rng.choice(self.visual.times),
                activity=activity,
            )
            if scene.key() not in recent_keys:
                return scene
        return scene  # pools exhausted; a repeat beats failing to post

    def combinations(self) -> int:
        """Distinct scenes available, accounting for mirror-shot constraints."""
        v = self.visual
        mirror = sum(1 for a in v.activities if "mirror" in a.lower())
        plain = len(v.activities) - mirror
        per = len(v.wardrobe) * len(v.times)
        return per * (plain * len(v.locations) + mirror * len(v.mirror_locations))
