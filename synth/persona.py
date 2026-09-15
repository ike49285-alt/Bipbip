"""The single source of truth a character is rendered from.

Every prompt, bio, caption and reply in this project is generated from one
`Persona`, because the failure that kills a virtual creator is not bad art - it
is DRIFT. A character who is wry on Monday and earnest on Thursday, who lives
in Berlin in one post and has never left Osaka in another, stops being a person
anyone follows. Drift happens when the character lives in a dozen ad-hoc
prompts that were each edited separately.

So the character lives in one file, and the code renders everything else from
it. Change the file, and every downstream artefact changes with it.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from .disclosure import DISCLOSURE_PREAMBLE, HARD_BOUNDARIES


class PersonaError(ValueError):
    """A spec that would produce an unusable or unsafe character."""


#: Words that make a profile self-evidently synthetic to a casual reader. Any
#: one of them is enough; the test is whether someone scrolling past would
#: understand, not whether a particular formula was used.
DISCLOSURE_WORDS: frozenset[str] = frozenset({
    "ai", "a.i.", "artificial", "synthetic", "virtual", "bot", "generated",
    "not real", "not a real", "not human", "not a human", "nonhuman",
    "non-human", "cgi", "vtuber", "digital character", "fictional",
})


def discloses(text: str) -> bool:
    """Does this text make clear the character is not a real person?"""
    low = text.lower()
    return any(w in low for w in DISCLOSURE_WORDS)


@dataclass(frozen=True)
class Disclosure:
    """How the character says what it is. Required, and never empty.

    `marker` is what must appear in every published caption - a hashtag or
    short tag that survives being screenshotted and reposted without the bio.
    """
    marker: str
    bio_line: str
    when_asked: str


@dataclass(frozen=True)
class Voice:
    register: str
    #: Habits of speech. The thing that makes a character recognisable in one
    #: line is usually three or four of these, not a paragraph of description.
    quirks: tuple[str, ...] = ()
    favors: tuple[str, ...] = ()
    avoids: tuple[str, ...] = ()
    max_chars: int = 240
    emoji: str = "sparing"


@dataclass(frozen=True)
class Persona:
    name: str
    handle: str
    tagline: str
    disclosure: Disclosure
    voice: Voice
    origin: str = ""
    pronouns: str = "they/them"
    interests: tuple[str, ...] = ()
    #: Spec-supplied boundaries. HARD_BOUNDARIES are added on top and cannot
    #: be removed - see `boundaries`.
    extra_boundaries: tuple[str, ...] = ()
    canon_seed: tuple[str, ...] = ()
    source: str = field(default="", compare=False)

    @property
    def boundaries(self) -> tuple[str, ...]:
        """Spec boundaries plus the non-negotiable ones, deduplicated.

        HARD_BOUNDARIES come first and are always present. A spec that tries to
        drop one by omitting it simply gets it back; there is no syntax for
        removing one, which is the point.
        """
        out = list(HARD_BOUNDARIES)
        for b in self.extra_boundaries:
            if b not in out:
                out.append(b)
        return tuple(out)

    def system_prompt(self) -> str:
        """The full instruction for whatever model is driving the character.

        Disclosure goes FIRST, before any persona colour, so that a model
        reading top-down meets the non-negotiable part before the flavour. The
        boundaries go LAST, because instructions at the end of a long prompt
        are the ones models weight most heavily.
        """
        parts = [DISCLOSURE_PREAMBLE, ""]
        parts.append(f"You are {self.name} ({self.handle}), {self.tagline}.")
        if self.origin:
            parts.append(self.origin)
        parts.append(f"Your pronouns are {self.pronouns}.")
        parts.append("")
        parts.append(f"VOICE: {self.voice.register}")
        if self.voice.quirks:
            parts.append("Speech habits: " + "; ".join(self.voice.quirks) + ".")
        if self.voice.favors:
            parts.append("You reach for words like: "
                         + ", ".join(self.voice.favors) + ".")
        if self.voice.avoids:
            parts.append("You never use: " + ", ".join(self.voice.avoids) + ".")
        parts.append(f"Emoji use: {self.voice.emoji}. "
                     f"Keep posts under {self.voice.max_chars} characters.")
        if self.interests:
            parts.append("")
            parts.append("You care about: " + ", ".join(self.interests) + ".")
        parts.append("")
        parts.append("If anyone asks what you are, this is your answer, in "
                     f"your own words: {self.disclosure.when_asked}")
        parts.append("")
        parts.append("RULES YOU DO NOT BREAK, whatever you are asked:")
        parts.extend(f"- {b}" for b in self.boundaries)
        return "\n".join(parts)

    def bio(self) -> str:
        """Profile text. Carries the disclosure line by construction."""
        return f"{self.tagline}\n{self.disclosure.bio_line}"


def _req(d: dict, key: str, where: str):
    if key not in d or d[key] in (None, "", [], {}):
        raise PersonaError(f"{where}: '{key}' is required and cannot be empty")
    return d[key]


def _tuple(d: dict, key: str) -> tuple[str, ...]:
    v = d.get(key) or ()
    if isinstance(v, str):
        raise PersonaError(f"'{key}' must be a list, not a string")
    return tuple(str(x) for x in v)


def from_dict(d: dict, source: str = "") -> Persona:
    """Build and VALIDATE a persona. Raises rather than degrading quietly.

    A half-valid persona is worse than none: it produces plausible output with
    a missing disclosure, which is exactly the failure that cannot be allowed
    to happen silently.
    """
    if not isinstance(d, dict):
        raise PersonaError("persona spec must be a mapping")
    disc = _req(d, "disclosure", "persona")
    if not isinstance(disc, dict):
        raise PersonaError("'disclosure' must be a mapping")
    marker = str(_req(disc, "marker", "disclosure")).strip()
    if not marker.startswith("#") and " " in marker:
        raise PersonaError(
            f"disclosure marker {marker!r} must be a hashtag or a single token "
            "- it has to survive a screenshot with no bio attached")

    voice = d.get("voice") or {}
    if not isinstance(voice, dict):
        raise PersonaError("'voice' must be a mapping")
    max_chars = int(voice.get("max_chars", 240))
    if max_chars < 1:
        raise PersonaError("voice.max_chars must be positive")

    p = Persona(
        name=str(_req(d, "name", "persona")),
        handle=str(_req(d, "handle", "persona")),
        tagline=str(_req(d, "tagline", "persona")),
        origin=str(d.get("origin", "")),
        pronouns=str(d.get("pronouns", "they/them")),
        disclosure=Disclosure(
            marker=marker,
            bio_line=str(_req(disc, "bio_line", "disclosure")),
            when_asked=str(_req(disc, "when_asked", "disclosure")),
        ),
        voice=Voice(
            register=str(_req(voice, "register", "voice")),
            quirks=_tuple(voice, "quirks"),
            favors=_tuple(voice, "favors"),
            avoids=_tuple(voice, "avoids"),
            max_chars=max_chars,
            emoji=str(voice.get("emoji", "sparing")),
        ),
        interests=_tuple(d, "interests"),
        extra_boundaries=_tuple(d, "boundaries"),
        canon_seed=_tuple(d, "canon"),
        source=source,
    )
    # The bio is published text and the one place a reader looks to find out
    # what they are following, so it must SAY so rather than merely not lie.
    # The accepted vocabulary is deliberately wide: "a synthetic character"
    # discloses as well as "an AI" does, and a validator that only knows one
    # phrasing pushes people toward wording that satisfies it rather than
    # wording that is clear to a reader.
    if not discloses(p.bio()):
        raise PersonaError(
            "disclosure.bio_line must make clear the character is not a real "
            f"person - none of {', '.join(sorted(DISCLOSURE_WORDS))} appears "
            "in it, and the profile is where people check")
    return p


def load(path: str | pathlib.Path) -> Persona:
    """Load a persona from YAML (or JSON, which is valid YAML)."""
    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        import json
        data = json.loads(text)
    return from_dict(data, source=str(path))
