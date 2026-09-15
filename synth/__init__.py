"""Tooling for a DISCLOSED synthetic creator.

The whole project rests on one distinction. A synthetic character who is open
about what she is, is legal, welcome on every major platform, and a real
business - Neuro-sama streams to hundreds of thousands of people who all know
exactly what they are watching, and being an AI is the draw. A synthetic
character passing as human is fraud against whoever is paying for it, and ends
in a terminated account that takes the operator's other accounts with it.

That distinction is not left to a prompt. `synth.disclosure` holds rules that
are merged into every persona at load time, cannot be removed by editing a
spec, are re-checked on the way out to any platform, and are covered by tests.
Everything else here - voice, canon, the content pipeline - exists to make a
character worth following once that part is settled.
"""
from .canon import Canon, Fact
from .content import Draft, Queue, check, generate
from .disclosure import asks_whether_real, HARD_BOUNDARIES
from .persona import Persona, PersonaError, load

__all__ = ["Canon", "Fact", "Draft", "Queue", "check", "generate",
           "asks_whether_real", "HARD_BOUNDARIES", "Persona", "PersonaError",
           "load"]
