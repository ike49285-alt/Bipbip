"""Voice presets. Each one is a prompt fragment, not a template.

Templates produce posts that read like templates. These describe a register and
let the model write inside it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    summary: str
    directive: str


PERSONAS: dict[str, Persona] = {
    "flirt": Persona(
        name="flirt",
        summary="playful, confident, suggestive without being explicit",
        directive=(
            "Playful and confident. Teasing rather than pleading -- the voice of "
            "someone who assumes interest rather than asking for it. Suggestive "
            "through implication and restraint, never through description."
        ),
    ),
    "soft": Persona(
        name="soft",
        summary="warm, intimate, low volume",
        directive=(
            "Quiet and intimate, like a text sent at 1am rather than a broadcast. "
            "Small concrete details over grand statements. No exclamation marks."
        ),
    ),
    "chaos": Persona(
        name="chaos",
        summary="unhinged, funny, self-aware",
        directive=(
            "Unhinged but in control of the joke. Self-aware about being online. "
            "Willing to look ridiculous, never willing to look earnest. The humour "
            "carries the attention, not the flattery."
        ),
    ),
    "deadpan": Persona(
        name="deadpan",
        summary="dry, understated, one-line",
        directive=(
            "Dry and understated. One line, no wind-up, no explanation after the "
            "punch. Trust the reader to get it. Understatement is the entire move."
        ),
    ),
    "flex": Persona(
        name="flex",
        summary="confident brag that stays likeable",
        directive=(
            "Confident about something real and specific. The brag is buried in a "
            "detail rather than announced. Never self-deprecating to soften it -- "
            "that reads as fishing."
        ),
    ),
    "soft_launch": Persona(
        name="soft_launch",
        summary="the vague, loaded, deliberately unexplained post",
        directive=(
            "Deliberately under-explained. Implies a story the reader has to "
            "reconstruct. The withheld information is the point -- never resolve it."
        ),
    ),
    "gym": Persona(
        name="gym",
        summary="physique-adjacent confidence, effort as the flex",
        directive=(
            "Training and physical effort as the source of the confidence. The "
            "discipline is the attractive part, not the body. Specific numbers, "
            "specific sessions, no motivational-poster language."
        ),
    ),
}

DEFAULT_PERSONA = "flirt"


def get(name: str) -> Persona:
    """Look up a persona, with the valid names in the error when it misses."""
    key = name.strip().lower()
    if key not in PERSONAS:
        available = ", ".join(sorted(PERSONAS))
        raise KeyError(f"unknown persona {name!r}; available: {available}")
    return PERSONAS[key]


def names() -> list[str]:
    return sorted(PERSONAS)
