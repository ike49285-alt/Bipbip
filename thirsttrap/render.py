"""Formatting for scored and ranked posts.

These return strings rather than printing, so the one-shot CLI and the
interactive session can share them without either owning stdout.
"""

from __future__ import annotations

from .rank import Ranked
from .score import WEIGHTS, Score

BAR_WIDTH = 24


def bar(fraction: float) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * BAR_WIDTH)
    return "#" * filled + "." * (BAR_WIDTH - filled)


def _components(score: Score, indent: str) -> list[str]:
    lines = []
    for name in sorted(score.components, key=lambda k: -WEIGHTS[k]):
        value = score.components[name]
        lines.append(f"{indent}{name:<15} {bar(value)} {value:.2f}  (w {WEIGHTS[name]:.2f})")
    return lines


def render_score(text: str, score: Score, breakdown: bool = True) -> str:
    """One post, its total, and optionally why."""
    lines = [f"{score.total:.1f}   {len(text)} chars"]
    if breakdown:
        lines += _components(score, indent="  ")
    lines += [f"  ! {note}" for note in score.notes]
    return "\n".join(lines)


def render_ranked(items: list[Ranked], breakdown: bool = False, start: int = 1) -> str:
    """A ranked batch. Numbering starts at `start` and follows displayed order.

    The numbers shown here are the numbers the model is told it produced, so a
    follow-up like "more like 2" points at the same post for both of you.
    """
    if not items:
        return "(nothing to show)"

    blocks = []
    for i, item in enumerate(items, start=start):
        flag = "  [near-duplicate]" if item.duplicate_of else ""
        lines = [f"{i}. {item.final:5.1f}{flag}", f"   {item.text}"]
        if breakdown:
            lines.append(f"   {len(item.text)} chars")
            lines += _components(item.score, indent="     ")
        lines += [f"     ! {note}" for note in item.score.notes]
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def render_weights(profile) -> str:
    """Current component weights against the shipped prior, biggest movers first."""
    from .score import WEIGHTS, normalise

    current = normalise(profile.weights)
    lines = [profile.confidence()]
    for name, delta in profile.drift():
        arrow = "+" if delta > 0 else ("-" if delta < 0 else " ")
        lines.append(
            f"  {name:<15} {bar(current[name] / max(current.values()))} "
            f"{current[name]:.3f}  (prior {WEIGHTS[name]:.2f}, {arrow}{abs(delta):.3f})"
        )
    return "\n".join(lines)
