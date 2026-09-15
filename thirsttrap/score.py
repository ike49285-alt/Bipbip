"""Heuristic engagement scoring for short-form social posts.

This is a PRIOR, not a measurement. Every weight here encodes a belief about
what makes a post travel on X -- drawn from platform behaviour that is widely
reported (hashtags suppress reach, off-platform links suppress reach, second
person outperforms third) rather than from engagement data this package has
fitted. Nothing in it has been validated against your account.

Treat the score as a way to rank candidates against each other, not as a
prediction of any particular number of likes. The README's "Calibrating this"
section describes what it would take to replace the prior with a fit.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Each component returns 0.0-1.0; weights sum to 1.0.
WEIGHTS: dict[str, float] = {
    "hook": 0.20,
    "length": 0.14,
    "direct_address": 0.12,
    "curiosity": 0.12,
    "restraint": 0.12,
    "concreteness": 0.10,
    "rhythm": 0.10,
    "freshness": 0.10,
}

MAX_CHARS = 280

# Openers that do work in the first few words.
_CONTRARIAN = {
    "nobody", "everyone", "most", "stop", "never", "unpopular", "worst",
    "hardest", "truth", "reminder", "psa", "hot",
}
_QUESTION_WORDS = {"what", "why", "how", "who", "when", "where", "which"}
# Openers that bleed authority before the sentence has started.
_HEDGES = {
    "i", "just", "maybe", "perhaps", "kinda", "sorta", "honestly", "basically",
    "literally", "actually", "personally", "arguably", "probably", "somewhat",
}

_SECOND_PERSON = {"you", "your", "youre", "yours", "yourself", "u", "ur"}

_OPEN_LOOP = (
    "nobody tells you", "no one tells you", "here's why", "heres why",
    "the reason", "what happens", "turns out", "and yet", "until you",
    "the part nobody", "watch what", "wait for", "the trick",
)

# Phrases that were fresh in 2016 and now read as filler.
_CLICHES = (
    "game changer", "game-changer", "let that sink in", "who else", "this 👆",
    "read that again", "needless to say", "at the end of the day",
    "thoughts?", "am i right", "drop a", "like and retweet", "rt if",
    "you won't believe", "you wont believe", "this is everything",
    "living rent free", "main character", "it's giving", "its giving",
    "no because", "the way i", "i can't even", "i cant even",
)

_ABSTRACT_SUFFIXES = ("ness", "ity", "tion", "sion", "ment", "ance", "ence", "ism")

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0000FE0F\U00002B00-\U00002BFF]"
)
_URL_RE = re.compile(r"https?://\S+|\bwww\.\S+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"(?:^|\s)#\w+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_RE = re.compile(r"[.!?\n]+")


@dataclass
class Score:
    """A post's score plus the per-component breakdown that produced it."""

    total: float
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = " ".join(f"{k}={v:.2f}" for k, v in sorted(self.components.items()))
        return f"{self.total:.1f}  {parts}"


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _normalize(word: str) -> str:
    """Lowercase and strip accents/apostrophes so `You're` and `youre` match."""
    stripped = unicodedata.normalize("NFKD", word.lower())
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return stripped.replace("'", "")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ramp(value: float, lo: float, hi: float) -> float:
    """Linear 0 at `lo`, 1 at `hi`. Handles hi < lo (descending ramp)."""
    if lo == hi:
        return 1.0 if value >= hi else 0.0
    return _clamp((value - lo) / (hi - lo))


def score_length(text: str) -> float:
    """Reward the punchy band; punish the unreadably short and the overlong."""
    n = len(text.strip())
    if n > MAX_CHARS:
        return 0.0
    if n < 60:
        return _ramp(n, 12, 60)
    if n <= 140:
        return 1.0
    return _ramp(n, MAX_CHARS, 140)


def score_hook(text: str) -> float:
    """Judge the first eight words -- that is all the timeline shows before a scroll."""
    words = [_normalize(w) for w in _words(text)[:8]]
    if not words:
        return 0.0

    score = 0.35
    first = words[0]

    if first in _SECOND_PERSON:
        score += 0.35
    elif first in _CONTRARIAN:
        score += 0.30
    elif first in _QUESTION_WORDS:
        score += 0.25
    elif first.isdigit():
        score += 0.25
    elif first in _HEDGES:
        score -= 0.25

    # Any of these anywhere in the opening clause still pulls the reader in.
    if any(w in _SECOND_PERSON for w in words[1:]):
        score += 0.15
    if any(w in _CONTRARIAN for w in words[1:]):
        score += 0.10
    if any(w.isdigit() for w in words):
        score += 0.10

    return _clamp(score)


def score_direct_address(text: str) -> float:
    """Second person is the whole mechanic; more is better until it is nagging."""
    hits = sum(1 for w in _words(text) if _normalize(w) in _SECOND_PERSON)
    if hits == 0:
        return 0.15
    if hits <= 3:
        return 1.0
    return _clamp(1.0 - 0.15 * (hits - 3))


def score_curiosity(text: str) -> float:
    """Open loops and a single question mark; two questions is an interrogation."""
    lowered = text.lower()
    score = 0.3

    questions = text.count("?")
    if questions == 1:
        score += 0.3
    elif questions >= 2:
        score += 0.1

    loops = sum(1 for phrase in _OPEN_LOOP if phrase in lowered)
    score += min(0.35, 0.18 * loops)

    if "..." in text or "…" in text:
        score += 0.1

    return _clamp(score)


def score_concreteness(text: str) -> float:
    """Specifics travel; abstraction nouns do not."""
    words = _words(text)
    if not words:
        return 0.0

    score = 0.4
    if any(w.isdigit() for w in words):
        score += 0.3

    abstract = sum(
        1 for w in words if len(w) > 6 and _normalize(w).endswith(_ABSTRACT_SUFFIXES)
    )
    score -= 0.2 * abstract

    # Mid-sentence capitals stand in for named, specific things.
    proper = sum(1 for w in words[1:] if w[0].isupper() and not w.isupper())
    score += min(0.2, 0.1 * proper)

    return _clamp(score)


def score_rhythm(text: str) -> float:
    """Setup then punch: a short closing sentence after a longer one."""
    sentences = _sentences(text)
    if not sentences:
        return 0.0
    if len(sentences) == 1:
        return 0.6

    lengths = [len(_words(s)) for s in sentences]
    if len(sentences) > 4:
        return 0.35

    earlier = sum(lengths[:-1]) / len(lengths[:-1])
    closing = lengths[-1]
    if closing == 0:
        return 0.5
    if closing < earlier * 0.7:
        return 1.0
    if closing < earlier:
        return 0.8
    return 0.55


def score_restraint(text: str) -> tuple[float, list[str]]:
    """Start clean and subtract for every reach-suppressing tic."""
    score = 1.0
    notes: list[str] = []

    emoji = len(_EMOJI_RE.findall(text))
    if emoji > 2:
        score -= 0.15 * (emoji - 2)
        notes.append(f"{emoji} emoji (2 is the ceiling)")

    hashtags = len(_HASHTAG_RE.findall(text))
    if hashtags:
        score -= 0.30 * hashtags
        notes.append(f"{hashtags} hashtag(s) -- these suppress reach on X")

    links = len(_URL_RE.findall(text))
    if links:
        score -= 0.35 * links
        notes.append("off-platform link -- put it in a reply instead")

    exclaims = text.count("!")
    if exclaims > 1:
        score -= 0.12 * (exclaims - 1)
        notes.append(f"{exclaims} exclamation marks")

    shouty = [w for w in _words(text) if len(w) > 2 and w.isupper()]
    if len(shouty) > 1:
        score -= 0.12 * len(shouty)
        notes.append(f"{len(shouty)} ALL-CAPS words")

    return _clamp(score), notes


def score_freshness(text: str) -> tuple[float, list[str]]:
    """Penalise phrases the timeline has already worn out."""
    lowered = text.lower()
    hits = [phrase for phrase in _CLICHES if phrase in lowered]
    return _clamp(1.0 - 0.3 * len(hits)), [f"cliche: {h!r}" for h in hits]


def normalise(weights: dict[str, float]) -> dict[str, float]:
    """Clamp to non-negative and rescale to sum 1, so totals stay on 0-100."""
    clean = {k: max(0.0, float(weights.get(k, 0.0))) for k in WEIGHTS}
    total = sum(clean.values())
    if total <= 0:
        return dict(WEIGHTS)
    return {k: v / total for k, v in clean.items()}


def score_post(text: str, weights: dict[str, float] | None = None) -> Score:
    """Score one post 0-100 with the component breakdown that produced it.

    `weights` overrides the default prior -- this is how a learned profile makes
    the ranking personal. Components are unchanged; only their importance moves.
    """
    restraint, restraint_notes = score_restraint(text)
    freshness, freshness_notes = score_freshness(text)

    components = {
        "hook": score_hook(text),
        "length": score_length(text),
        "direct_address": score_direct_address(text),
        "curiosity": score_curiosity(text),
        "restraint": restraint,
        "concreteness": score_concreteness(text),
        "rhythm": score_rhythm(text),
        "freshness": freshness,
    }

    notes = restraint_notes + freshness_notes
    if len(text.strip()) > MAX_CHARS:
        notes.append(f"{len(text.strip())} chars -- over the {MAX_CHARS} limit")

    active = normalise(weights) if weights else WEIGHTS
    total = 100.0 * sum(components[k] * active[k] for k in WEIGHTS)
    return Score(total=total, components=components, notes=notes)
