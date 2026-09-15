"""Turn what you type into constraints the generator can act on.

With no model in the loop this is keyword matching, and it is honest about
that: it recognises a fixed vocabulary of adjustments and ignores everything
else. The compensation is that the grammar produces hundreds of candidates, so
a constraint is applied by filtering a large pool rather than by persuading a
model -- "shorter" always works, it just throws away everything over the cap.

A directive is standing when it is phrased as a rule ("never", "always", "from
now on") and one-off otherwise. That distinction is the native analogue of
asking a model what it learned, and it is deliberately conservative: an
unrecognised sentence changes nothing rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from . import personas
from .score import _EMOJI_RE


@dataclass(frozen=True)
class Constraints:
    """A filter over candidate posts. `None` everywhere means "no opinion"."""

    max_chars: int | None = None
    min_chars: int | None = None
    questions: bool | None = None
    numerals: bool | None = None
    max_sentences: int | None = None
    emoji: bool | None = None
    banned: tuple[str, ...] = ()

    def allows(self, text: str) -> bool:
        n = len(text)
        if self.max_chars is not None and n > self.max_chars:
            return False
        if self.min_chars is not None and n < self.min_chars:
            return False
        if self.questions is False and "?" in text:
            return False
        if self.questions is True and "?" not in text:
            return False

        has_digit = any(c.isdigit() for c in text)
        if self.numerals is False and has_digit:
            return False
        if self.numerals is True and not has_digit:
            return False

        if self.max_sentences is not None:
            sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
            if len(sentences) > self.max_sentences:
                return False

        if self.emoji is not None and bool(_EMOJI_RE.search(text)) != self.emoji:
            return False

        lowered = text.lower()
        return not any(term.lower() in lowered for term in self.banned)

    def merge(self, other: Constraints) -> Constraints:
        """Later wins per field; bans accumulate."""
        merged = {}
        for field_name in (
            "max_chars", "min_chars", "questions", "numerals", "max_sentences", "emoji",
        ):
            value = getattr(other, field_name)
            merged[field_name] = value if value is not None else getattr(self, field_name)
        bans = list(self.banned) + [b for b in other.banned if b not in self.banned]
        return replace(self, **merged, banned=tuple(bans))

    def describe(self) -> list[str]:
        out = []
        if self.max_chars is not None:
            out.append(f"at most {self.max_chars} characters")
        if self.min_chars is not None:
            out.append(f"at least {self.min_chars} characters")
        if self.questions is False:
            out.append("no questions")
        if self.questions is True:
            out.append("always a question")
        if self.numerals is False:
            out.append("no numbers")
        if self.numerals is True:
            out.append("always a number")
        if self.max_sentences is not None:
            out.append(f"at most {self.max_sentences} sentence(s)")
        if self.emoji is False:
            out.append("no emoji")
        if self.emoji is True:
            out.append("always an emoji")
        out += [f"never say {b!r}" for b in self.banned]
        return out

    def __bool__(self) -> bool:
        return bool(self.describe())


@dataclass(frozen=True)
class Directive:
    """What one typed line asked for."""

    constraints: Constraints = Constraints()
    persona: str | None = None
    like: int | None = None
    standing: bool = False
    recognised: bool = False


# Ordered: the first pattern that matches a phrase wins, so "much shorter"
# is not swallowed by "shorter".
_RULES: tuple[tuple[str, dict], ...] = (
    (r"\b(much|way|far)\s+(shorter|tighter|briefer)\b", {"max_chars": 60}),
    (r"\b(shorter|tighter|briefer|trim|cut it down|less wordy)\b", {"max_chars": 95}),
    (r"\b(much|way|far)\s+longer\b", {"min_chars": 160}),
    (r"\b(longer|more detail|flesh it out)\b", {"min_chars": 120}),
    (r"\b(one|single)\s+(line|sentence)\b|\bone-liner\b", {"max_sentences": 1}),
    (r"\bno\s+questions?\b|\bstop\s+asking\b|\bdon'?t\s+ask\b", {"questions": False}),
    (r"\bask\s+(a\s+)?questions?\b|\bmake\s+it\s+a\s+question\b", {"questions": True}),
    (r"\bno\s+(numbers?|numerals?|digits?)\b", {"numerals": False}),
    (r"\b(use|with)\s+(a\s+)?numbers?\b", {"numerals": True}),
)

# A negation within a short distance of the thing being banned, so "no
# exclamation marks", "never use exclamation marks" and "stop shouting" all land.
_NEGATION = r"\b(?:no|never|not|stop|without|don'?t|drop the|lose the)\b[^.?!]{0,24}?"

_BAN_RULES: tuple[tuple[str, str], ...] = (
    (_NEGATION + r"\bexclamation|\bstop\s+(?:shouting|yelling)\b", "!"),
    (_NEGATION + r"\bellips[ei]s\b|\bno\s+dot\s*dot\s*dot\b", "..."),
    (_NEGATION + r"\bhashtags?\b", "#"),
)

# Emoji is a character class, not a substring, so it gets a field rather than a ban.
_FIELD_RULES: tuple[tuple[str, dict], ...] = (
    (_NEGATION + r"\bemoji\w*\b", {"emoji": False}),
)

# "never", "always", "from now on" -- phrasing that outlives the current topic.
_STANDING_RE = re.compile(
    r"\b(never|always|from now on|every time|going forward|stop using|"
    r"i hate|i don'?t want|don'?t ever)\b"
)

_LIKE_RE = re.compile(r"\b(?:more\s+like|like|similar to)\s+#?(\d+)\b")
_BAN_PHRASE_RE = re.compile(
    r"\b(?:never|stop|don'?t|avoid)\s+(?:say(?:ing)?|us(?:e|ing)|writ(?:e|ing))\s+"
    r"[\"']([^\"']+)[\"']"
)


def parse(text: str) -> Directive:
    """Read one typed line. Unrecognised input yields an inert directive."""
    lowered = text.lower()
    fields: dict = {}
    bans: list[str] = []
    recognised = False

    for pattern, effect in _RULES:
        if re.search(pattern, lowered):
            for key, value in effect.items():
                fields.setdefault(key, value)
            recognised = True

    for pattern, effect in _FIELD_RULES:
        if re.search(pattern, lowered):
            for key, value in effect.items():
                fields.setdefault(key, value)
            recognised = True

    for pattern, term in _BAN_RULES:
        if re.search(pattern, lowered) and term not in bans:
            bans.append(term)
            recognised = True

    quoted = _BAN_PHRASE_RE.search(lowered)
    if quoted:
        bans.append(quoted.group(1).strip())
        recognised = True

    persona = None
    for name in personas.names():
        # "gym" as a topic word should not switch voice, so require an intent verb.
        if re.search(rf"\b(try|switch to|be|go|sound|more)\s+(the\s+)?{name}\b", lowered):
            persona = name
            recognised = True
            break

    like = None
    match = _LIKE_RE.search(lowered)
    if match:
        like = int(match.group(1))
        recognised = True

    return Directive(
        constraints=Constraints(**fields, banned=tuple(bans)),
        persona=persona,
        like=like,
        standing=bool(_STANDING_RE.search(lowered)),
        recognised=recognised,
    )
