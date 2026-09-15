"""Propose candidate posts. Entirely local -- no network, no API, no model.

Generation is a slot grammar (`grammar.py`) over a parsed topic (`topic.py`).
Selection is the scorer. That pairing has a consequence worth stating plainly:

**When posts are generated here, the displayed score is the objective being
optimised, not an independent opinion of quality.** Picking the top of a pool by
score guarantees a high score the same way choosing the tallest person in a room
guarantees height. It is still the right way to pick -- but read it as "this
matched your preferences best", never as evidence the post is good. The score
only functions as an outside judge on text the grammar did not write, which is
what `thirsttrap score` is for.
"""

from __future__ import annotations

from . import grammar, personas
from .directives import Constraints
from .topic import parse as parse_topic

DEFAULT_POOL = 400


def propose(
    topic: str,
    persona: str = personas.DEFAULT_PERSONA,
    pool: int = DEFAULT_POOL,
    seed: int | None = None,
    constraints: Constraints | None = None,
) -> list[str]:
    """Generate a pool of distinct candidates and keep the ones that qualify.

    Constraints are applied by filtering rather than by steering: the grammar
    cannot be asked for a shorter post, so it is asked for many and the long
    ones are discarded. A pool that filters down to nothing returns nothing --
    the caller is expected to say so rather than quietly relax the constraint.
    """
    if pool < 1:
        raise ValueError(f"pool must be at least 1, got {pool}")
    personas.get(persona)  # raises with the valid names if it misses

    candidates = grammar.generate(parse_topic(topic), persona, count=pool, seed=seed)
    if constraints is None:
        return candidates
    return [text for text in candidates if constraints.allows(text)]
