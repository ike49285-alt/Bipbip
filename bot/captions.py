"""Step 4: turn a generated image into a caption in the persona's voice.

Two model passes. The first reads the image and returns structured facts; the
second writes candidates from those facts. Splitting them matters because the
image and the prompt that made it diverge -- caption what you got, not what you
asked for.

Everything except the model calls is pure stdlib and tested: voice enforcement,
repetition scoring, candidate selection. The LLM is injected, so none of that
needs an API key to verify.

On repetition: bots read as bots less because they repeat words than because
they repeat *construction*. Every caption being "statement. two-word fragment."
is the tell, even with entirely different vocabulary. So two signals are
scored -- trigram overlap catches near-duplicates, and a shape signature
catches an over-used sentence pattern.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from bot.persona import Persona

# The persona bot runs on a hobby budget; this is the cheap tier by an explicit
# decision recorded in DESIGN.md. Change this one line to move the whole
# pipeline up a tier.
DEFAULT_MODEL = "claude-haiku-4-5"

DESCRIBE_INSTRUCTIONS = (
    "Describe what is actually visible in this image. Report only what you can "
    "see, not what you assume. Do not name or guess at the person's identity."
)


@dataclass(frozen=True)
class SceneFacts:
    """What the vision pass actually saw. The caption is written from this."""

    setting: str
    subject_action: str
    notable_details: tuple[str, ...] = ()
    mood: str = ""
    light: str = ""

    def as_brief(self) -> str:
        lines = [f"setting: {self.setting}", f"she is: {self.subject_action}"]
        if self.light:
            lines.append(f"light: {self.light}")
        if self.mood:
            lines.append(f"mood: {self.mood}")
        if self.notable_details:
            lines.append("details: " + "; ".join(self.notable_details))
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, raw: dict) -> "SceneFacts":
        missing = [k for k in ("setting", "subject_action") if not raw.get(k)]
        if missing:
            raise ValueError(f"vision pass returned no {', '.join(missing)}")
        return cls(
            setting=str(raw["setting"]),
            subject_action=str(raw["subject_action"]),
            notable_details=tuple(raw.get("notable_details") or ()),
            mood=str(raw.get("mood") or ""),
            light=str(raw.get("light") or ""),
        )


SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "setting": {"type": "string"},
        "subject_action": {"type": "string"},
        "notable_details": {"type": "array", "items": {"type": "string"}},
        "mood": {"type": "string"},
        "light": {"type": "string"},
    },
    "required": ["setting", "subject_action"],
    "additionalProperties": False,
}


class LLM(Protocol):
    """The two calls this module makes. Injected so the logic is testable."""

    def describe_image(self, image: Path, instructions: str, schema: dict) -> dict: ...

    def write_candidates(self, system: str, prompt: str, count: int) -> list[str]: ...


# -- voice enforcement ---------------------------------------------------


def _emoji_count(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch) == "So")


def check_voice(caption: str, persona: Persona) -> list[str]:
    """Return the voice-card rules this caption breaks. Empty means it passes."""
    v = persona.voice
    limits = v.constraints
    problems: list[str] = []

    if not caption.strip():
        return ["empty caption"]
    if len(caption) > v.caption_max_chars:
        problems.append(f"{len(caption)} chars, limit {v.caption_max_chars}")
    if limits.get("lowercase", False) and caption != caption.lower():
        problems.append("contains uppercase")
    max_emoji = limits.get("max_emoji")
    if max_emoji is not None and _emoji_count(caption) > max_emoji:
        problems.append(f"{_emoji_count(caption)} emoji, limit {max_emoji}")
    for token in limits.get("forbid", ()):
        if token in caption:
            problems.append(f"contains {token!r}")
    return problems


# -- repetition ----------------------------------------------------------


def trigrams(text: str) -> set[str]:
    squashed = re.sub(r"\s+", " ", text.lower().strip())
    return {squashed[i : i + 3] for i in range(max(len(squashed) - 2, 0))}


def content_similarity(a: str, b: str) -> float:
    """Jaccard over character trigrams. Catches near-duplicate wording."""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _length_bucket(words: int) -> str:
    if words <= 3:
        return "short"
    if words <= 7:
        return "mid"
    return "long"


def shape(caption: str) -> tuple:
    """A structural signature: sentence count, coarse lengths, fragment ending.

    Two captions sharing no words at all can still share a shape, and a feed of
    one shape is what reads as automated. Lengths are bucketed rather than
    counted -- "bus is late. walking instead." and "coffee went cold. drinking
    it anyway." are the same rhythm, and an exact word count would miss that.
    """
    parts = [p.strip() for p in re.split(r"[.!?]+", caption) if p.strip()]
    buckets = tuple(_length_bucket(len(p.split())) for p in parts)
    ends_short = bool(parts) and len(parts[-1].split()) <= 3
    return (len(parts), buckets, ends_short)


def shape_repeats(caption: str, history: Sequence[str]) -> int:
    target = shape(caption)
    return sum(1 for past in history if shape(past) == target)


# -- selection -----------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    caption: str
    accepted: bool
    reasons: tuple[str, ...] = ()
    max_similarity: float = 0.0
    shape_repeats: int = 0


def assess(
    caption: str, persona: Persona, history: Sequence[str] = ()
) -> Verdict:
    """Judge one candidate against the voice card and recent history."""
    window = list(history)[-persona.voice.history_window :]
    reasons = list(check_voice(caption, persona))

    worst = max((content_similarity(caption, past) for past in window), default=0.0)
    if worst >= persona.voice.similarity_threshold:
        reasons.append(f"too close to a recent caption ({worst:.2f})")

    repeats = shape_repeats(caption, window)
    if repeats > persona.voice.max_shape_repeats:
        reasons.append(f"sentence shape used {repeats}x recently")

    return Verdict(
        caption=caption,
        accepted=not reasons,
        reasons=tuple(reasons),
        max_similarity=worst,
        shape_repeats=repeats,
    )


def pick(
    candidates: Sequence[str], persona: Persona, history: Sequence[str] = ()
) -> tuple[str | None, list[Verdict]]:
    """Choose the least-repetitive acceptable candidate.

    Returns (chosen, all verdicts). None means every candidate failed -- the
    caller regenerates rather than posting something off-voice.
    """
    verdicts = [assess(c, persona, history) for c in candidates]
    passing = [v for v in verdicts if v.accepted]
    if not passing:
        return None, verdicts
    best = min(passing, key=lambda v: (v.shape_repeats, v.max_similarity, len(v.caption)))
    return best.caption, verdicts


# -- orchestration -------------------------------------------------------


@dataclass
class Captioner:
    persona: Persona
    llm: LLM
    attempts: int = 3

    def describe(self, image: Path) -> SceneFacts:
        raw = self.llm.describe_image(image, DESCRIBE_INSTRUCTIONS, SCENE_SCHEMA)
        return SceneFacts.from_dict(raw)

    def compose_prompt(self, facts: SceneFacts, history: Sequence[str]) -> str:
        recent = "\n".join(f"- {h}" for h in list(history)[-12:])
        block = f"\n\nYour last captions, do not echo their wording or shape:\n{recent}" if recent else ""
        return (
            f"Write a caption for this photo of yourself.\n\n{facts.as_brief()}{block}\n\n"
            "One caption per line. No numbering, no quotes, no commentary."
        )

    def caption(self, image: Path, history: Sequence[str] = ()) -> tuple[str, SceneFacts]:
        """Describe, write, score. Retries before giving up rather than posting junk."""
        facts = self.describe(image)
        seen: list[Verdict] = []
        for _ in range(self.attempts):
            candidates = self.llm.write_candidates(
                self.persona.voice_card(),
                self.compose_prompt(facts, history),
                self.persona.voice.candidates_per_caption,
            )
            chosen, verdicts = pick(candidates, self.persona, history)
            seen.extend(verdicts)
            if chosen:
                return chosen, facts
        raise CaptionError(
            f"no candidate passed after {self.attempts} attempts; "
            f"last reasons: {[v.reasons for v in seen[-4:]]}"
        )


class CaptionError(RuntimeError):
    """Raised when nothing generated is fit to post."""


class AnthropicLLM:
    """Real backend. Imports the SDK lazily so bot.captions stays dependency-free."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model

    def describe_image(self, image: Path, instructions: str, schema: dict) -> dict:
        import base64
        import json

        data = base64.b64encode(image.read_bytes()).decode()
        media = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media, "data": data}},
                    {"type": "text", "text": instructions},
                ],
            }],
        )
        return json.loads(response.content[0].text)

    def write_candidates(self, system: str, prompt: str, count: int) -> list[str]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            # The voice card is byte-identical on every call; cache the prefix.
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"{prompt}\n\nGive {count} options."}],
        )
        lines = response.content[0].text.splitlines()
        return [ln.strip().lstrip("-*0123456789. ").strip('"') for ln in lines if ln.strip()]


# -- directing the voice -------------------------------------------------

REVISION_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {"type": "array", "items": {"type": "string"}},
        "changed": {"type": "string"},
    },
    "required": ["rules", "changed"],
    "additionalProperties": False,
}


class Director(Protocol):
    """Turns a note about the writing into revised rules."""

    def revise(self, prompt: str, schema: dict) -> dict: ...


def revision_prompt(persona: Persona, note: str, samples: Sequence[str]) -> str:
    """Ask for concrete rules, not adjectives -- the voice card only works
    when every line is checkable."""
    rules = "\n".join(f"- {r}" for r in persona.voice.rules)
    written = "\n".join(f"- {c}" for c in samples) or "- (nothing written yet)"
    return (
        "You direct the voice of a writer producing short social captions.\n\n"
        f"Her current rules:\n{rules}\n\n"
        f"What she wrote:\n{written}\n\n"
        f'The director\'s note: "{note}"\n\n'
        "Turn the note into rules. A rule is concrete and checkable, never an "
        'adjective: "no self-pity, state the annoyance and move on" not "be less '
        'whiny". Add at most two, and drop or reword any existing rule the note '
        "contradicts. Keep every rule the note does not touch, unchanged.\n\n"
        'Reply with only JSON: {"rules": [string], "changed": string} where '
        "changed is one short sentence naming what you altered."
    )


def direct(
    persona: Persona, note: str, director: Director, samples: Sequence[str] = ()
) -> tuple[list[str], str]:
    """Revise the voice rules from a plain-English note about the writing."""
    if not note.strip():
        raise ValueError("say what is wrong with the writing")
    raw = director.revise(revision_prompt(persona, note, samples), REVISION_SCHEMA)
    rules = [str(r).strip() for r in raw.get("rules", []) if str(r).strip()]
    if not rules:
        raise CaptionError("the revision dropped every rule; nothing was changed")
    return rules, str(raw.get("changed", "rules updated"))


class AnthropicDirector:
    """Real backend. Lazy import so bot.captions stays dependency-free."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model

    def revise(self, prompt: str, schema: dict) -> dict:
        import json

        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.content[0].text)
