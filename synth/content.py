"""Idea to publishable post, with the checks that cannot be skipped in between.

The pipeline is deliberately boring: generate, then GATE. Every interesting
failure of a synthetic creator happens between "the model wrote something good"
and "it went out", so that gap is where the code lives.

    idea -> draft -> canon check -> disclosure check -> boundary check -> queue

Generation takes a `complete` callable - anything that maps a prompt to text -
so the pipeline runs offline in tests with a stub, and against a real model in
production, without the gates knowing or caring which. The gates are the
product; the model is swappable.

Nothing here posts. `Draft.approved` only means it passed the automatic checks;
publishing is a separate, explicit act through a platform adapter, because an
autonomous poster with no human in the loop is how an account dies at 4am.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

from .canon import Canon, Fact
from .disclosure import missing_disclosure
from .persona import Persona

#: Anything that maps a prompt to a completion.
Completer = Callable[[str], str]


@dataclass
class Draft:
    text: str
    idea: str = ""
    #: Reasons this draft is not publishable. Empty means it passed.
    blockers: list[str] = field(default_factory=list)
    #: Established facts it may contradict, for a human to glance at.
    flagged_facts: list[str] = field(default_factory=list)
    at: float = field(default_factory=time.time)
    published_to: str = ""

    @property
    def approved(self) -> bool:
        return not self.blockers

    def render(self, persona: Persona) -> str:
        """The caption as it would actually go out, marker attached."""
        if missing_disclosure(self.text, persona.disclosure.marker):
            return f"{self.text} {persona.disclosure.marker}"
        return self.text


def check(draft_text: str, persona: Persona, canon: Canon) -> Draft:
    """Run every gate over one candidate post and report all failures.

    All of them, not the first: a draft with three problems should come back
    with three, so one pass of edits fixes it instead of three.
    """
    d = Draft(text=draft_text.strip())
    if not d.text:
        d.blockers.append("empty draft")
        return d

    # Length is measured on the RENDERED caption, marker included, because the
    # marker is not optional and the platform counts it.
    rendered = d.render(persona)
    if len(rendered) > persona.voice.max_chars:
        d.blockers.append(
            f"{len(rendered)} chars over the {persona.voice.max_chars} limit "
            f"(the disclosure marker counts and cannot be dropped to fit)")

    for word in persona.voice.avoids:
        if word.lower() in d.text.lower():
            d.blockers.append(f"uses avoided word {word!r}")

    conflicts = canon.conflicts(d.text)
    d.flagged_facts = [f.text for f in conflicts]
    if conflicts:
        d.blockers.append(
            f"may contradict {len(conflicts)} established fact(s)")

    # A draft that asserts humanity is the one failure with no acceptable
    # version, so it is checked on the text itself rather than trusted to the
    # prompt that produced it.
    for claim in ("i'm a real person", "im a real person", "i am a real person",
                  "i'm human", "im human", "i am human", "not a bot",
                  "not an ai", "i'm not ai", "im not ai"):
        if claim in d.text.lower():
            d.blockers.append(f"claims to be human: {claim!r}")
    return d


def generate(persona: Persona, canon: Canon, idea: str, complete: Completer,
             attempts: int = 3) -> Draft:
    """Draft a post for `idea`, retrying with the gate's own complaints.

    A rejected draft is fed back with the reason, which is far more effective
    than regenerating blind - the model that wrote a 300-character post can
    shorten it if told that is the problem.
    """
    feedback = ""
    last = Draft(text="", blockers=["no attempt produced a draft"])
    for _ in range(max(attempts, 1)):
        prompt = _prompt(persona, canon, idea, feedback)
        draft = check(complete(prompt), persona, canon)
        draft.idea = idea
        if draft.approved:
            return draft
        last = draft
        feedback = ("Your previous attempt was rejected for: "
                    + "; ".join(draft.blockers)
                    + ". Write a new version that fixes this.")
    return last


def _prompt(persona: Persona, canon: Canon, idea: str, feedback: str) -> str:
    parts = [persona.system_prompt(), ""]
    brief = canon.brief(idea)
    if brief:
        parts += [brief, ""]
    parts.append(f"Write one post about: {idea}")
    parts.append(f"Under {persona.voice.max_chars} characters including the "
                 f"tag {persona.disclosure.marker}, which must appear in it.")
    parts.append("Output the post text only - no preamble, no quotation marks.")
    if feedback:
        parts += ["", feedback]
    return "\n".join(parts)


class Queue:
    """Approved drafts waiting for a human to release them."""

    def __init__(self, path: str | pathlib.Path | None = None):
        self.path = pathlib.Path(path) if path else None
        self.drafts: list[Draft] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.drafts.append(Draft(**json.loads(line)))

    def add(self, draft: Draft) -> None:
        self.drafts.append(draft)
        self.save()

    def pending(self) -> list[Draft]:
        return [d for d in self.drafts if d.approved and not d.published_to]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".partial")
        tmp.write_text(
            "".join(json.dumps(asdict(d)) + "\n" for d in self.drafts),
            encoding="utf-8")
        tmp.replace(self.path)
