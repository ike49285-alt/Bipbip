"""thirsttrap -- generate candidate posts locally and rank them by a learned prior."""

from .generate import propose
from .personas import PERSONAS, Persona
from .profile import Profile
from .rank import Ranked, rank, similarity
from .score import Score, score_post
from .topic import Topic

__version__ = "0.2.0"

__all__ = [
    "PERSONAS",
    "Persona",
    "Profile",
    "Ranked",
    "Score",
    "Topic",
    "propose",
    "rank",
    "score_post",
    "similarity",
]
