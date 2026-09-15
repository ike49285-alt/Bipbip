"""Rank scored candidates, penalising the ones that repeat each other.

A batch of twelve posts that are all the same joke in different words is one
post. The novelty penalty is applied at rank time rather than inside `score_post`
because it is a property of the batch, not of the post.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .score import Score, score_post

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Too common to say anything about whether two posts are the same post.
_STOPWORDS = frozenset(
    """a an and are as at be but by for from had has have he her his i if in is it
    its me my not of on or she so that the their them they this to was we were what
    when who will with you your""".split()
)


@dataclass
class Ranked:
    """One candidate with its intrinsic score and its batch-adjusted final score."""

    text: str
    score: Score
    novelty: float
    final: float

    @property
    def duplicate_of(self) -> bool:
        return self.novelty < 0.75


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap on content words. 1.0 means the same words in any order."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def rank(candidates: list[str], novelty_weight: float = 0.35) -> list[Ranked]:
    """Score every candidate, then discount each for resembling a better one.

    Candidates are scored intrinsically first, then walked best-first: each post
    is compared only against the posts already accepted above it, so the
    strongest member of a near-duplicate cluster keeps its score and the weaker
    ones absorb the penalty.
    """
    scored = [(text, score_post(text)) for text in candidates]
    scored.sort(key=lambda pair: pair[1].total, reverse=True)

    ranked: list[Ranked] = []
    for text, score in scored:
        worst_overlap = max(
            (similarity(text, kept.text) for kept in ranked), default=0.0
        )
        novelty = 1.0 - worst_overlap
        final = score.total * (1.0 - novelty_weight * worst_overlap)
        ranked.append(Ranked(text=text, score=score, novelty=novelty, final=final))

    ranked.sort(key=lambda r: r.final, reverse=True)
    return ranked
