"""Turn a raw topic string into the grammatical forms the frames need.

Deliberately conservative. Telling a verb from a noun needs a lexicon this
package does not carry, and guessing produces "gyming at 5am" and "legging day".
So the gerund slot converts only words on a small known-verb list, leaves an
existing "-ing" word alone, and otherwise hands back the phrase untouched --
which reads correctly in every frame that uses it ("the best part of leg day").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LEADERS = ("about ", "on ", "re ", "regarding ")

_DETERMINERS = frozenset(
    "a an the my your his her their our its this that these those".split()
)

_VOWELS = "aeiou"
_WORD_RE = re.compile(r"[A-Za-z0-9'\-]+")

# Gerunds that are not formed by the default suffix rules.
_IRREGULAR = {
    "run": "running", "sit": "sitting", "quit": "quitting", "get": "getting",
    "put": "putting", "cut": "cutting", "win": "winning", "begin": "beginning",
    "swim": "swimming", "plan": "planning", "stop": "stopping", "drop": "dropping",
    "lie": "lying", "die": "dying", "tie": "tying", "be": "being", "see": "seeing",
    "shut": "shutting", "set": "setting", "let": "letting", "forget": "forgetting",
}

# Only these convert. Anything else is left alone rather than guessed at.
_KNOWN_VERBS = frozenset(
    list(_IRREGULAR)
    + """leave love hate want need miss text call date meet kiss chase block delete
    post ghost move walk drive train lift work wake sleep eat drink cook clean
    learn teach read write build make take give send wait watch listen talk speak
    ask answer say tell think know feel try start finish fail pass fall rise
    dance sing swim climb ride travel pack unpack apply resign retire""".split()
)

_PREPOSITIONS = frozenset(
    """at in on of for with from to about after before during under over into
    through against between among around near past""".split()
)

_FUNCTION_WORDS = frozenset(
    """a an the my your his her their our its this that these those and or but is
    are was were be been am do does did done not no so as if then than when
    while again still just very really quite finally""".split()
) | _PREPOSITIONS


def is_gerund(word: str) -> bool:
    """A word already in participle form. False positives here are harmless --
    an "-ing" word is left unchanged either way."""
    return len(word) >= 5 and word.lower().endswith("ing")


def to_gerund(word: str) -> str:
    """Present participle of a known verb. Unknown words come back unchanged."""
    lowered = word.lower()
    if is_gerund(lowered):
        return lowered
    if lowered in _IRREGULAR:
        return _IRREGULAR[lowered]
    if lowered not in _KNOWN_VERBS:
        return lowered
    if lowered.endswith("ie"):
        return lowered[:-2] + "ying"
    if lowered.endswith("e") and not lowered.endswith(("ee", "oe", "ye")):
        return lowered[:-1] + "ing"
    if (
        len(lowered) >= 3
        and lowered[-1] not in _VOWELS
        and lowered[-1] not in "wxy"
        and lowered[-2] in _VOWELS
        and lowered[-3] not in _VOWELS
    ):
        return lowered + lowered[-1] + "ing"
    return lowered + "ing"


@dataclass(frozen=True)
class Topic:
    """The forms a frame can slot in.

    raw     -- what was typed, whitespace normalised
    phrase  -- raw with a leading "about" stripped, article intact
    gerund  -- phrase led by a participle where that is safe, else phrase
    head    -- the head noun with its determiner and modifiers ("the job"),
               because frames slot it after prepositions and "about job" reads wrong
    """

    raw: str
    phrase: str
    gerund: str
    head: str

    @property
    def sentence(self) -> str:
        """The phrase as a capitalised standalone opener."""
        return self.phrase[:1].upper() + self.phrase[1:] if self.phrase else ""


def parse(text: str) -> Topic:
    raw = " ".join(text.split())
    lowered = raw.lower()

    # Only "about"/"re" are stripped. A leading article is kept: every frame
    # slots the phrase after a preposition or at a sentence start, and both want
    # "the gym at 5am" rather than "gym at 5am".
    phrase = lowered
    for leader in _LEADERS:
        if phrase.startswith(leader):
            phrase = phrase[len(leader):]
            break

    words = _WORD_RE.findall(phrase)
    if not words:
        return Topic(raw=raw, phrase=raw, gerund=raw, head=raw)

    converted = to_gerund(words[0])
    gerund = " ".join([converted, *words[1:]]) if converted != words[0] else phrase

    # Build the head phrase from the text before the article was stripped, so the
    # determiner survives: "finally quitting the job" -> "the job".
    full = _WORD_RE.findall(lowered)
    head_zone = []
    for word in full:
        if word in _PREPOSITIONS and head_zone:
            break
        head_zone.append(word)

    content = [i for i, w in enumerate(head_zone) if w not in _FUNCTION_WORDS]
    if not content:
        return Topic(raw=raw, phrase=phrase, gerund=gerund, head=head_zone[-1])

    end = content[-1]
    start = end
    # Absorb preceding modifiers ("leg day"), then at most one determiner ("the job").
    while start > 0 and head_zone[start - 1] not in _FUNCTION_WORDS:
        start -= 1
    if start > 0 and head_zone[start - 1] in _DETERMINERS:
        start -= 1

    return Topic(raw=raw, phrase=phrase, gerund=gerund, head=" ".join(head_zone[start : end + 1]))
