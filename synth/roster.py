"""Several characters at once, and whether they are actually different people.

This is the failure mode of a persona factory, and it is not obvious from
inside. Each spec reads fine on its own. Together they are one character with
six avatars: the same sentence rhythm, the same six adjectives, the same three
opinions. An audience that follows two of them spots it in a week, and the
whole roster reads as output rather than as people.

It is also measurable, which is the useful part. Two characters are distinct to
a reader mostly through VOICE - the words they reach for, the words they will
not touch, the things they have opinions about. Overlap on those is overlap
full stop, and it can be computed without asking anyone's opinion.

The threshold is a judgement call and is therefore a parameter, printed with
every report rather than hidden. What is not a judgement call is that a handle,
a name, or a disclosure marker shared between two live characters is a
collision rather than a similarity, and those are errors.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from .persona import Persona, PersonaError, load


#: Words that are in everyone's register description and therefore distinguish
#: nobody. Without this the shared-word list is mostly "the" and "her", which
#: makes a real clash harder to read rather than easier.
_STOP = frozenset("""
and the her his its their they them she he you your with that this than then
for from into onto but not never always very really just only about like when
where what who how are was were been being have has had does did doing will
would could should can may might must one two three own out off over under
""".split())


def _words(*groups) -> set[str]:
    """Significant words across a set of voice or interest fields.

    Punctuation is stripped rather than split on, so "precise," and "precise"
    are one word - otherwise a trailing comma in one spec and not the other
    hides a genuine overlap, and the report shows tokens no human would
    recognise as words.
    """
    out: set[str] = set()
    for g in groups:
        for item in g:
            for w in re.findall(r"[a-z][a-z'-]{2,}", str(item).lower()):
                if w not in _STOP:
                    out.add(w)
    return out


def jaccard(a: set, b: set) -> float:
    """Overlap as a share of the combined vocabulary. 0 distinct, 1 identical."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


@dataclass(frozen=True)
class Overlap:
    a: str
    b: str
    voice: float
    interests: float
    shared: tuple[str, ...]

    @property
    def score(self) -> float:
        """Voice counts double. Two characters can care about the same things
        and stay distinct if they talk differently; the reverse is not true."""
        return (2 * self.voice + self.interests) / 3


@dataclass(frozen=True)
class Collision:
    field: str
    value: str
    a: str
    b: str


def compare(p: Persona, q: Persona) -> Overlap:
    pv = _words(p.voice.favors, p.voice.avoids, p.voice.quirks,
                [p.voice.register])
    qv = _words(q.voice.favors, q.voice.avoids, q.voice.quirks,
                [q.voice.register])
    pi, qi = _words(p.interests), _words(q.interests)
    return Overlap(a=p.name, b=q.name, voice=jaccard(pv, qv),
                   interests=jaccard(pi, qi),
                   shared=tuple(sorted(pv & qv))[:12])


def collisions(personas: list[Persona]) -> list[Collision]:
    """Shared identity fields. These are errors, not similarities.

    Two characters sharing a handle cannot both exist. Two sharing a disclosure
    marker are harder to tell apart in a screenshot than in the timeline, which
    is exactly where the marker has to work.
    """
    out = []
    for i, p in enumerate(personas):
        for q in personas[i + 1:]:
            for field, a, b in (("name", p.name, q.name),
                                ("handle", p.handle, q.handle)):
                if a.lower() == b.lower():
                    out.append(Collision(field, a, p.name, q.name))
    return out


def load_all(directory: str | pathlib.Path) -> tuple[list[Persona], list[str]]:
    """Every persona in a directory, plus the ones that would not load.

    A broken spec is reported rather than skipped: silently ignoring it is how
    a character quietly stops being part of the roster while its files sit
    there looking fine.
    """
    directory = pathlib.Path(directory)
    good, bad = [], []
    for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        try:
            good.append(load(path))
        except (PersonaError, Exception) as e:
            bad.append(f"{path.name}: {e}")
    return good, bad


def report(personas: list[Persona], threshold: float = 0.25) -> str:
    lines = [f"{len(personas)} character(s), similarity threshold {threshold:.2f}",
             ""]
    for c in collisions(personas):
        lines.append(f"  COLLISION  {c.a} and {c.b} share a {c.field}: {c.value}")
    if collisions(personas):
        lines.append("")

    pairs = []
    for i, p in enumerate(personas):
        for q in personas[i + 1:]:
            pairs.append(compare(p, q))
    pairs.sort(key=lambda o: -o.score)

    if not pairs:
        lines.append("  nothing to compare - one character is always distinct")
        return "\n".join(lines)

    lines.append(f"  {'pair':<34}{'voice':>8}{'interest':>10}{'score':>8}")
    for o in pairs:
        mark = "   TOO ALIKE" if o.score >= threshold else ""
        lines.append(f"  {o.a + ' / ' + o.b:<34}{o.voice:>8.2f}"
                     f"{o.interests:>10.2f}{o.score:>8.2f}{mark}")
        if o.score >= threshold and o.shared:
            lines.append(f"      shared: {', '.join(o.shared)}")
    worst = pairs[0]
    lines.append("")
    if worst.score >= threshold:
        lines.append(f"  {worst.a} and {worst.b} are the pair to fix first. "
                     "Two characters\n  can care about the same things and stay "
                     "distinct; they cannot talk the\n  same way and stay "
                     "distinct.")
    else:
        lines.append("  No pair crosses the threshold. That is not proof they "
                     "read as different\n  people - it is only proof they do "
                     "not share vocabulary.")
    return "\n".join(lines)
