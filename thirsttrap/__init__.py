"""thirsttrap -- generate candidate posts for X and rank them by a stated prior."""

from .personas import PERSONAS, Persona
from .rank import Ranked, rank, similarity
from .score import Score, score_post

__version__ = "0.1.0"

__all__ = [
    "PERSONAS",
    "Persona",
    "Ranked",
    "Score",
    "rank",
    "score_post",
    "similarity",
]
