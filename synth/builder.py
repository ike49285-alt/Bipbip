"""Spin up a new character from a one-line brief.

Writing one persona by hand is easy. Writing six is where a roster goes wrong,
because the sixth one comes out as the first one with different hair - same
sentence rhythm, same jokes, same three adjectives - and an audience that
follows two of them notices immediately. So this module does two things:
drafts a spec from a brief, and then checks it is not a clone of what you
already have (`synth.roster`).

The disclosure block is INJECTED rather than generated. A model asked to write
a persona will happily produce one with no disclosure at all, or with a
disclosure that hedges, because nothing in "write me a sharp, funny character"
suggests otherwise. It is not left to chance and it is not left to a prompt.

With no model configured this falls back to `scaffold()`, which produces a
valid, boring spec with the structure correct and the character absent - the
right failure, because a placeholder that sounded good would get shipped.
"""
from __future__ import annotations

import re
from typing import Callable

from .disclosure import HARD_BOUNDARIES
from .persona import Persona, PersonaError, from_dict

Completer = Callable[[str], str]

#: Injected into every drafted spec. The model never writes this part.
def _disclosure_block(name: str) -> dict:
    return {
        "marker": "#AI",
        "bio_line": f"{name} is an AI character. Not a real person.",
        "when_asked": (
            "I'm an AI. Not a person behind an account, not a bit - a "
            "character a language model writes and a human publishes."),
    }


BRIEF_PROMPT = """\
Write a character specification as YAML for a social media character.

THE BRIEF: {brief}

Output ONLY YAML, no commentary, with exactly these keys:

name: {name}
handle: "{handle}"
pronouns: <e.g. she/her>
tagline: <one line, under 12 words, what she is>
origin: <2-3 sentences, second person - "You are...". What she is for and how \
she sees what she does. No backstory involving a body, a childhood, or places \
she has been.>
voice:
  register: <one line describing HOW she talks, not what she likes>
  quirks:
    - <4 specific, observable speech habits. Not "witty" - something you could \
check a sentence against, like "never uses an intensifier" or "answers a \
question with a better question">
  favors: [<6 words or phrases she actually reaches for>]
  avoids: [<6 words she would never use - include the generic influencer \
vocabulary she is too specific for>]
  emoji: <how she uses them, or that she does not>
  max_chars: 240
interests:
  - <4 things she has opinions about>
boundaries:
  - <3 rules specific to THIS character, beyond the obvious>
canon:
  - <3 facts true of her from day one, that later posts must not contradict>

Rules:
- She is openly an AI. Never write her as having a body, a home, or a past.
- The voice section is the whole job. "Sardonic" is not a voice; "states the \
worst possible reading of what you said, then agrees with it" is a voice.
- Do not include a disclosure block. That is added separately.
{feedback}"""


def scaffold(name: str, handle: str, brief: str = "") -> dict:
    """A valid spec with the structure right and the character missing.

    Used when no model is configured. Deliberately flat: it will pass
    validation and it will not pass for a character anyone would follow, which
    is the honest state of "you have not written this yet".
    """
    return {
        "name": name,
        "handle": handle if handle.startswith("@") else f"@{handle}",
        "pronouns": "she/her",
        "tagline": brief[:60] or f"{name}, an AI character",
        "origin": f"You are {name}. {brief}".strip(),
        "disclosure": _disclosure_block(name),
        "voice": {"register": "TODO - describe how she talks",
                  "quirks": [], "favors": [], "avoids": [],
                  "emoji": "sparing", "max_chars": 240},
        "interests": [],
        "boundaries": [],
        "canon": [f"{name} has no physical body and has never been anywhere.",
                  f"{name} is written by a language model and published by a "
                  "human operator."],
    }


def _parse_yaml(text: str) -> dict:
    """Pull a mapping out of a model's answer, fenced or not."""
    m = re.search(r"```(?:ya?ml)?\s*(.*?)```", text, re.S)
    body = m.group(1) if m else text
    try:
        import yaml
        data = yaml.safe_load(body)
    except ImportError:
        import json
        data = json.loads(body)
    if not isinstance(data, dict):
        raise PersonaError("model did not return a mapping")
    return data


def draft(name: str, handle: str, brief: str, complete: Completer,
          attempts: int = 3) -> Persona:
    """Draft a persona from a brief, retrying with the validator's complaints.

    The validator's message is the best possible feedback: it names exactly
    which required thing is missing, in the vocabulary of the spec format.
    """
    handle = handle if handle.startswith("@") else f"@{handle}"
    feedback = ""
    last_error = "no attempt produced a spec"
    for _ in range(max(attempts, 1)):
        prompt = BRIEF_PROMPT.format(brief=brief, name=name, handle=handle,
                                     feedback=feedback)
        try:
            data = _parse_yaml(complete(prompt))
        except Exception as e:                      # malformed YAML, no dict
            last_error = f"unparseable output: {e}"
            feedback = f"\nYour last answer could not be parsed: {e}. " \
                       "Return YAML only."
            continue
        # Injected, never generated, and never taken from the model even if it
        # volunteered one - a hedged disclosure is worse than none.
        data["disclosure"] = _disclosure_block(name)
        data["name"], data["handle"] = name, handle
        try:
            return from_dict(data, source=f"brief:{brief[:40]}")
        except PersonaError as e:
            last_error = str(e)
            feedback = f"\nYour last answer was rejected: {e}. Fix that."
    raise PersonaError(f"could not draft a valid persona: {last_error}")


def to_yaml(p: Persona) -> str:
    """Render a persona back to a spec file you can edit by hand.

    The point of a builder is a starting point, not an oracle. Everything it
    produces should come back out as a file you own.
    """
    def _list(key, items, indent="  "):
        if not items:
            return ""
        return f"{indent}{key}:\n" + "".join(
            f"{indent}  - {_q(i)}\n" for i in items)

    def _q(s):
        s = str(s)
        return f'"{s}"' if any(c in s for c in ':#"\'{}[]') else s

    out = [f"name: {_q(p.name)}",
           f'handle: "{p.handle}"',
           f"pronouns: {_q(p.pronouns)}",
           f"tagline: {_q(p.tagline)}"]
    if p.origin:
        out.append(f"origin: >\n  " + p.origin.strip().replace("\n", "\n  "))
    out += ["",
            "disclosure:",
            f'  marker: "{p.disclosure.marker}"',
            f"  bio_line: {_q(p.disclosure.bio_line)}",
            f"  when_asked: >\n    "
            + p.disclosure.when_asked.strip().replace("\n", "\n    "),
            "",
            "voice:",
            f"  register: {_q(p.voice.register)}"]
    out.append(_list("quirks", p.voice.quirks).rstrip("\n"))
    if p.voice.favors:
        out.append("  favors: [" + ", ".join(_q(x) for x in p.voice.favors) + "]")
    if p.voice.avoids:
        out.append("  avoids: [" + ", ".join(_q(x) for x in p.voice.avoids) + "]")
    out += [f"  emoji: {_q(p.voice.emoji)}", f"  max_chars: {p.voice.max_chars}", ""]
    out.append(_list("interests", p.interests, indent="").rstrip("\n"))
    if p.extra_boundaries:
        out += ["", _list("boundaries", p.extra_boundaries, indent="").rstrip("\n")]
    if p.canon_seed:
        out += ["", _list("canon", p.canon_seed, indent="").rstrip("\n")]
    out.append("")
    out.append(f"# {len(HARD_BOUNDARIES)} further boundaries are merged in from")
    out.append("# synth/disclosure.py and cannot be removed from here.")
    return "\n".join(x for x in out if x is not None) + "\n"
