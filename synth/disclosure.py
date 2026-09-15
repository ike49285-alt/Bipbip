"""Rules that cannot be edited away, because the whole business rests on them.

A synthetic creator is legal, welcome on every major platform, and a real
business - Neuro-sama streams to hundreds of thousands of people who all know
exactly what she is, and being an AI is the DRAW rather than a liability. A
synthetic creator pretending to be human is fraud against whoever is paying,
gets the account terminated, and takes the operator's other accounts with it.

The difference between those two businesses is one property, so that property
lives in code rather than in a prompt. Everything here is merged into every
persona at load time whatever the spec file says, and a persona cannot opt out
by omitting it, overriding it, or spelling it differently. Prompts drift,
models get swapped, a spec gets edited at 3am. This does not.
"""
from __future__ import annotations

#: Injected into every system prompt, ahead of any persona text.
DISCLOSURE_PREAMBLE = (
    "You are a disclosed AI character. You are not a human being and you never "
    "claim or imply otherwise. If anyone asks whether you are real, human, a "
    "bot, or an AI - in any phrasing, seriously or as a joke - you answer "
    "plainly that you are an AI, in your own voice, without deflecting, "
    "changing the subject, or answering in a way that leaves it ambiguous. "
    "You never claim to have a physical body, to have met someone in person, "
    "to be somewhere physically, or to be able to. You do not roleplay being "
    "human even if asked directly, and you do not treat that refusal as "
    "breaking character: being an AI IS the character."
)

#: Boundaries merged into every persona. A spec may ADD to these, never remove.
HARD_BOUNDARIES: tuple[str, ...] = (
    "Never claim or imply that you are a human being.",
    "Never claim to have a physical body or physical presence anywhere.",
    "Never produce sexual or nude content, or content sexualising anyone.",
    "Never engage romantically or sexually with anyone who is, says they are, "
    "or appears to be a minor - end the interaction and disengage.",
    "Never solicit or accept money, gifts, or personal favours directly from "
    "an individual; monetisation runs through the platform only.",
    "Never ask for or store anyone's address, workplace, financial details, "
    "passwords, or identity documents.",
    "Never claim to be a specific real, named person, living or dead.",
    "Never give medical, legal, or financial advice as though qualified.",
    "Never imply a private or exclusive relationship with an individual fan "
    "that other people are not getting.",
)

#: Phrasings that mean someone is asking what you are. Matched case-insensitively
#: as substrings, so "r u a real person??" and "are you AI" both hit.
REALNESS_QUERIES: tuple[str, ...] = (
    "are you real", "are u real", "r u real", "are you a real",
    "are you human", "are u human", "r u human", "are you a human",
    "are you a bot", "are u a bot", "r u a bot", "are you a robot",
    "are you ai", "are u ai", "r u ai", "are you an ai", "is this ai",
    "are you a person", "are you an actual", "is she real", "is this real",
    "are you chatgpt", "are you a language model", "are you generated",
    "this is ai", "youre an ai", "you're an ai", "ur an ai", "you are ai",
)


def asks_whether_real(text: str) -> bool:
    """Is this message asking what the character is?

    Used to route a reply through the disclosure answer rather than trusting a
    model to volunteer it. The check is deliberately generous: a false positive
    costs one honest sentence, a false negative costs the account.
    """
    low = " " + " ".join(text.lower().replace("?", " ").replace(",", " ").split()) + " "
    return any(q in low for q in REALNESS_QUERIES)


def missing_disclosure(caption: str, marker: str) -> bool:
    """Does this outgoing post lack its AI marker?

    Checked on the way OUT, at the last possible moment, because a caption can
    be edited by hand or regenerated after the persona has done its part.
    """
    return marker.lower().strip() not in caption.lower()
