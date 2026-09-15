"""The generator: a slot grammar over rhetorical frames.

A frame is a sentence shape that has a reason to work -- withheld knowledge,
second-person assertion, setup and punch. Slots resolve from the topic or from
persona-tagged lexicons, recursively, so one frame yields thousands of surface
forms.

What this is not: it does not understand your topic. It arranges your words
inside shapes that read well. Output is more formulaic than a language model's,
and the honest mitigation is volume -- generate hundreds, let the ranker throw
most of them away.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .topic import Topic

_SLOT_RE = re.compile(r"\{(\w+)\}")
_MAX_DEPTH = 6

ANY = "*"


@dataclass(frozen=True)
class Frame:
    template: str
    personas: frozenset = frozenset()  # empty means every persona

    def allows(self, persona: str) -> bool:
        return not self.personas or persona in self.personas


def _frames(*templates: str, personas: tuple = ()) -> list[Frame]:
    tagged = frozenset(personas)
    return [Frame(template=t, personas=tagged) for t in templates]


FRAMES: list[Frame] = [
    # Withheld knowledge -- the reader is told something is being kept from them.
    *_frames(
        "Nobody tells you about {phrase}. {punch}",
        "Nobody warns you about {phrase}. {punch}",
        "Nobody mentions the {quiet_part} of {gerund}.",
        "The part nobody posts about {phrase}: {consequence}.",
        "Everyone talks about {head}. Nobody mentions {consequence}.",
        "They never tell you {phrase} is mostly {consequence}.",
    ),
    # Second person -- assert something about the reader and refuse to soften it.
    *_frames(
        "You already know about {head}. You're just {stalling}.",
        "You don't need permission for {phrase}. You need {need}.",
        "You keep {circling} {head}. {punch}",
        "You want {head}. Say it properly.",
        "You'll think about {head} at {time}. You always do.",
        "Your version of {phrase} is {consequence}. Mine isn't.",
    ),
    # Setup and punch -- a longer line, then a short one.
    *_frames(
        "{num} {unit} of {gerund}. {punch}",
        "{time}. {sentence}. {punch}",
        "Took me {num} {unit} to {settle} {head}. {punch}",
        "{sentence} at {time}. Home by {time}. {punch}",
        "Spent {num} {unit} {circling} {head}. {punch}",
    ),
    # Imperative -- instruct the reader, never plead.
    *_frames(
        "Stop apologising for {phrase}.",
        "Stop explaining {head} to people who {dismiss} it.",
        "Stop waiting for someone to approve {phrase}.",
        "Say it out loud: {phrase}.",
    ),
    # Reversal -- set an expectation and break it.
    *_frames(
        "Thought {phrase} would be the hard part. It wasn't.",
        "Everyone said {phrase} would {dismiss} me. {punch}",
        "{sentence} was supposed to be the ending. {punch}",
        "I was wrong about {head}. Gloriously.",
    ),
    # Withholding -- imply a story and refuse to resolve it.
    *_frames(
        "{sentence}. That's all you're getting.",
        "{sentence}. Ask me in person.",
        "Something happened at {time}. {phrase} is involved. That's the post.",
        "Still thinking about {head}. Still not saying why.",
        personas=("soft_launch", "flirt", "deadpan"),
    ),
    # Dry -- one line, no wind-up.
    *_frames(
        "{sentence}. Still not sorry.",
        "{sentence}, and I'd do it again at {time}.",
        "Turns out {phrase} was the easy bit.",
        personas=("deadpan", "chaos", "flex"),
    ),
    # Effort as the flex.
    *_frames(
        "{num} {unit} in and {head} still {dismiss} me. Good.",
        "{time} for {head}. Every day. That's it, that's the secret.",
        "Nobody's coming to make {phrase} easier. {punch}",
        personas=("gym", "flex"),
    ),
    # Question -- exactly one, pointed at the reader.
    *_frames(
        "How long have you been {circling} {head}?",
        "What would you do about {phrase} if nobody was watching?",
        "Who told you {phrase} needed {need}?",
    ),
]


# Slot -> persona -> options. ANY applies to every persona and is always pooled in.
LEXICON: dict[str, dict[str, tuple[str, ...]]] = {
    "punch": {
        ANY: (
            "It gets quieter.", "I'd do it again.", "No notes.",
            "That's the whole story.", "It was never close.",
            "Took long enough.", "Worth every minute.",
        ),
        "flirt": ("You'd have stayed.", "Ask me why.", "You already knew.",
                  "Come find out.", "You'd look good here."),
        "soft": ("It was warm.", "Nobody saw.", "I kept it.",
                 "It still is.", "I didn't tell anyone."),
        "chaos": ("Anyway.", "Unwell about it.", "No further questions.",
                  "I am normal.", "Seek help, not from me."),
        "deadpan": ("Fine.", "Sure.", "It's fine.", "Not a metaphor.", "Anyway."),
        "flex": ("Twice.", "On the first try.", "Still is.", "Nobody asked. Telling you."),
        "gym": ("Again tomorrow.", "Same time tomorrow.", "Unbroken.", "Still going."),
        "soft_launch": ("That's all.", "Don't ask.", "You'll find out.", "Not yet."),
    },
    "quiet_part": {
        ANY: ("quiet part", "best part", "second day", "morning after",
              "cost", "aftermath", "silence"),
    },
    "consequence": {
        ANY: ("waiting", "silence", "admin", "cold coffee", "logistics",
              "other people's opinions", "the bit nobody films", "small print"),
        "soft": ("quiet", "the light at 6am", "somebody's playlist"),
        "chaos": ("psychic damage", "a group chat", "unexplained crying"),
        "gym": ("sore hamstrings", "the drive home", "chalk everywhere"),
    },
    "stalling": {
        ANY: ("stalling", "waiting for permission", "rehearsing it",
              "drafting the message", "circling the airport", "being polite about it"),
    },
    "circling": {
        ANY: ("circling", "rehearsing", "avoiding", "overthinking",
              "drafting a message about", "putting off"),
    },
    "need": {
        ANY: ("a Tuesday", "a reason you like", "an hour", "nerve",
              "one honest sentence", "less advice"),
    },
    "dismiss": {
        ANY: ("humble", "flatten", "outlast", "underestimate", "test", "bore"),
    },
    "settle": {
        ANY: ("admit", "say", "stop arguing about", "forgive", "finish with"),
    },
    "time": {
        ANY: ("5am", "4am", "6am", "midnight", "11pm", "a Tuesday",
              "half past nothing", "Sunday", "closing time"),
        "gym": ("5am", "6am", "5:40", "before work", "after work"),
    },
    "num": {
        ANY: ("three", "four", "six", "nine", "eleven", "two", "seven"),
    },
    "unit": {
        ANY: ("years", "months", "weeks", "days", "sessions", "attempts"),
        "gym": ("weeks", "sessions", "sets", "months"),
    },
}


def options(slot: str, persona: str) -> tuple[str, ...]:
    """Everything this persona may put in the slot: its own entries plus the shared ones."""
    bank = LEXICON.get(slot)
    if bank is None:
        return ()
    return tuple(bank.get(persona, ())) + tuple(bank.get(ANY, ()))


def _tidy(text: str) -> str:
    """Repair the seams: spacing, duplicated stops, and sentence capitals."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    if text and text[-1] not in ".!?":
        text += "."

    # Capitalise the first letter of the post and of each following sentence.
    out = []
    capitalise = True
    for char in text:
        out.append(char.upper() if capitalise and char.isalpha() else char)
        if char.isalpha() or char.isdigit():
            capitalise = False
        elif char in ".!?":
            capitalise = True
    return "".join(out)


def expand(template: str, topic: Topic, persona: str, rng: random.Random) -> str:
    """Resolve every slot, following slots that appear inside lexicon entries."""
    text = template
    for _ in range(_MAX_DEPTH):
        if not _SLOT_RE.search(text):
            break

        def replace(match: re.Match) -> str:
            slot = match.group(1)
            value = getattr(topic, slot, None)
            if isinstance(value, str):
                return value
            choices = options(slot, persona)
            # An unknown slot leaves its own name rather than vanishing, so a
            # typo in a frame shows up in output instead of hiding.
            return rng.choice(choices) if choices else match.group(0)

        text = _SLOT_RE.sub(replace, text)
    return _tidy(text)


def generate(
    topic: Topic,
    persona: str,
    count: int = 400,
    seed: int | None = None,
    frames: list[Frame] | None = None,
) -> list[str]:
    """Produce up to `count` distinct posts. Fewer if the grammar runs dry."""
    rng = random.Random(seed)
    usable = [f for f in (frames or FRAMES) if f.allows(persona)]
    if not usable:
        return []

    seen: set[str] = set()
    out: list[str] = []
    # Bounded: identical expansions are common once a frame's slots are exhausted.
    for _ in range(count * 8):
        if len(out) >= count:
            break
        text = expand(rng.choice(usable).template, topic, persona, rng)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out
