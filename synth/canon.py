"""Everything the character has publicly established, and nothing else.

A character becomes real to an audience through ACCUMULATED CONSISTENCY - she
mentioned her flatmate in March, so when the flatmate comes up in September it
lands. The same mechanism destroys her: one post saying she has never left the
city, another describing last year in Lisbon, and the audience stops treating
her as a person and starts treating her as output. They are right to.

A language model cannot prevent this. It has no memory between calls, and a
long prompt full of "remember that you said" degrades everything else in it. So
canon lives outside the model: an append-only log of what has been said in
public, queried for the few facts relevant to what is being written now, and
checked afterwards for contradictions.

Append-only is deliberate. Editing the past is how a character quietly becomes
incoherent while every individual post looks fine; a retcon has to be an
explicit, recorded act.
"""
from __future__ import annotations

import json
import pathlib
import re
import time
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Fact:
    """One thing established in public, and where it was established."""
    text: str
    #: Free-form grouping - "home", "family", "work". Facts in the same topic
    #: are the ones worth checking a new claim against.
    topic: str = "general"
    #: Where it was said. A fact with no source cannot be verified later and is
    #: usually a seed from the spec rather than something the audience saw.
    source: str = ""
    at: float = field(default_factory=time.time)
    #: A superseded fact is kept, not deleted - the audience saw it.
    retconned_by: str = ""

    @property
    def live(self) -> bool:
        return not self.retconned_by


#: Claims that are inherently mutually exclusive. Crude on purpose: this is a
#: cheap first pass that catches the obvious self-contradictions, not a
#: reasoner. The expensive check is `contradiction_prompt`, which asks a model.
NEGATORS: tuple[tuple[str, str], ...] = (
    ("never", "always"), ("never", "used to"), ("never been", "went to"),
    ("never been", "lived in"), ("can't", "can "), ("cannot", "can "),
    ("doesn't", "does "), ("don't", "do "), ("no longer", "still"),
    ("hate", "love"), ("dislike", "adore"), ("only child", "sister"),
    ("only child", "brother"), ("vegetarian", "steak"), ("vegan", "cheese"),
)


class Canon:
    """An append-only fact store, persisted as JSON lines."""

    def __init__(self, path: str | pathlib.Path | None = None):
        self.path = pathlib.Path(path) if path else None
        self.facts: list[Fact] = []
        if self.path and self.path.exists():
            self.load()

    # -- storage ----------------------------------------------------------
    def load(self) -> None:
        self.facts = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                self.facts.append(Fact(**json.loads(line)))

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".partial")
        tmp.write_text(
            "".join(json.dumps(asdict(f)) + "\n" for f in self.facts),
            encoding="utf-8")
        tmp.replace(self.path)

    # -- writing ----------------------------------------------------------
    def add(self, text: str, topic: str = "general", source: str = "") -> Fact:
        f = Fact(text=text.strip(), topic=topic, source=source)
        self.facts.append(f)
        self.save()
        return f

    def retcon(self, old: str, by: str) -> bool:
        """Mark a fact superseded WITHOUT deleting it.

        The audience saw the original. Keeping it is what lets you answer
        "didn't you say..." honestly instead of gaslighting people who were
        paying attention, which is the fastest way to lose them.
        """
        for i, f in enumerate(self.facts):
            if f.text == old and f.live:
                self.facts[i] = Fact(text=f.text, topic=f.topic,
                                     source=f.source, at=f.at, retconned_by=by)
                self.save()
                return True
        return False

    # -- reading ----------------------------------------------------------
    def live_facts(self, topic: str | None = None) -> list[Fact]:
        return [f for f in self.facts
                if f.live and (topic is None or f.topic == topic)]

    def relevant(self, draft: str, limit: int = 12) -> list[Fact]:
        """The established facts worth showing the model for THIS draft.

        Scored by shared significant words. A prompt carrying every fact ever
        recorded drowns the instruction that matters; a prompt carrying the
        dozen related ones is what stops the contradiction.
        """
        words = _keywords(draft)
        scored = []
        for f in self.live_facts():
            overlap = len(words & _keywords(f.text))
            if overlap:
                scored.append((overlap, f))
        scored.sort(key=lambda x: (-x[0], x[1].at))
        return [f for _, f in scored[:limit]]

    def conflicts(self, draft: str) -> list[Fact]:
        """Cheap first pass: established facts this draft may contradict.

        Scans EVERY live fact, not just the keyword-relevant ones. The first
        version filtered through `relevant` first and silently missed the whole
        class of contradiction it was built for: "I am vegetarian" and "best
        steak of my life" share no words at all, so relevance scoring dropped
        the fact before the negator check could see it. Semantic opposites are
        exactly the pairs that do not overlap lexically.

        Deliberately noisy. A flagged draft costs a second look; an unflagged
        contradiction ships and stays in the timeline forever.
        """
        low = draft.lower()
        out = []
        for f in self.live_facts():
            fl = f.text.lower()
            for a, b in NEGATORS:
                if (a in low and b in fl) or (b in low and a in fl):
                    out.append(f)
                    break
        return out

    def brief(self, draft: str, limit: int = 12) -> str:
        """Canon block to paste into a generation prompt."""
        rel = self.relevant(draft, limit=limit)
        if not rel:
            return ""
        lines = ["ESTABLISHED FACTS - do not contradict these:"]
        lines += [f"- {f.text}" for f in rel]
        return "\n".join(lines)


_STOP = frozenset("""
a an and are as at be been but by can cant do does dont for from had has have
he her hers him his how i if in is it its me my no not of on or our out she so
that the their them then there these they this to too us was we were what when
where which who why will with you your im ive dont thats just really very
""".split())


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']{3,}", text.lower()) if w not in _STOP}
