"""What she remembers between sessions.

Three things persist, and they are different in kind:

- `weights` -- how much each scoring component matters, learned from which
  candidate you keep out of the batch you were shown. This is the tuning.
- `standing` -- constraints you stated as rules ("never use exclamation
  marks"), parsed by `directives.py` rather than inferred.
- `examples` -- the posts you kept, which is the record the weights were
  learned from.

The weight update is contrastive: keeping a post is a statement that it beat
the others on screen, so components where it scored above the batch average
gain weight and the rest lose it. With a handful of keeps this is badly
underdetermined -- `confidence()` exists so the interface can say so rather
than implying the numbers mean more than they do.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import personas
from .directives import Constraints
from .score import WEIGHTS, normalise

MAX_EXAMPLES = 24
LEARNING_RATE = 0.06

# Below this many keeps the learned weights are noise dressed as preference.
CONFIDENT_AFTER = 20

ENV_OVERRIDE = "THIRSTTRAP_PROFILE"

_CONSTRAINT_FIELDS = (
    "max_chars", "min_chars", "questions", "numerals", "max_sentences", "emoji",
)


def default_path() -> Path:
    """Respect an explicit override, then XDG, then the conventional fallback."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return Path(override).expanduser()
    config = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config).expanduser() if config else Path.home() / ".config"
    return base / "thirsttrap" / "profile.json"


@dataclass
class Profile:
    persona: str = personas.DEFAULT_PERSONA
    weights: dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))
    standing: Constraints = field(default_factory=Constraints)
    examples: list[str] = field(default_factory=list)
    keeps: int = 0
    batches: int = 0
    path: Path | None = None

    # -- learning --------------------------------------------------------

    def learn_from_keep(self, chosen: dict[str, float], others: list[dict[str, float]]) -> dict[str, float]:
        """Shift weights toward what distinguished the kept post. Returns the movement.

        `chosen` and `others` are component dicts. A keep from a batch of one
        teaches nothing -- there was no alternative to prefer it over.
        """
        self.keeps += 1
        if not others:
            return {}

        moved = {}
        updated = dict(self.weights)
        for name in WEIGHTS:
            average = sum(o.get(name, 0.0) for o in others) / len(others)
            delta = LEARNING_RATE * (chosen.get(name, 0.0) - average)
            updated[name] = updated.get(name, WEIGHTS[name]) + delta
            moved[name] = delta

        before = normalise(self.weights)
        self.weights = normalise(updated)
        # Report the movement after renormalisation, which is what actually applies.
        return {k: self.weights[k] - before[k] for k in WEIGHTS}

    def confidence(self) -> str:
        """An honest label for how much the learned weights are worth."""
        if self.keeps == 0:
            return "untuned"
        if self.keeps < CONFIDENT_AFTER:
            return f"barely tuned ({self.keeps}/{CONFIDENT_AFTER} keeps)"
        return f"tuned on {self.keeps} keeps"

    def drift(self) -> list[tuple[str, float]]:
        """How far each weight has moved from the shipped prior, largest first."""
        current = normalise(self.weights)
        deltas = [(name, current[name] - WEIGHTS[name]) for name in WEIGHTS]
        return sorted(deltas, key=lambda pair: -abs(pair[1]))

    def reset_weights(self) -> None:
        self.weights = dict(WEIGHTS)
        self.keeps = 0

    # -- standing constraints --------------------------------------------

    def add_standing(self, constraints: Constraints) -> list[str]:
        """Adopt a stated rule. Returns the clauses that were genuinely new."""
        before = set(self.standing.describe())
        self.standing = self.standing.merge(constraints)
        return [clause for clause in self.standing.describe() if clause not in before]

    def clear_standing(self) -> int:
        count = len(self.standing.describe())
        self.standing = Constraints()
        return count

    # -- examples --------------------------------------------------------

    def remember(self, text: str) -> bool:
        if text in self.examples:
            return False
        self.examples.append(text)
        if len(self.examples) > MAX_EXAMPLES:
            del self.examples[: len(self.examples) - MAX_EXAMPLES]
        return True

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        standing = {f: getattr(self.standing, f) for f in _CONSTRAINT_FIELDS}
        standing["banned"] = list(self.standing.banned)
        return {
            "persona": self.persona,
            "weights": self.weights,
            "standing": standing,
            "examples": self.examples,
            "keeps": self.keeps,
            "batches": self.batches,
        }

    def save(self) -> None:
        """Write atomically. A crash mid-write must not cost the whole profile."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temp.replace(self.path)

    @classmethod
    def load(cls, path: Path | None = None) -> Profile:
        """Load, or return an empty profile. A damaged file never blocks a session."""
        path = path or default_path()
        profile = cls(path=path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return profile
        if not isinstance(data, dict):
            return profile

        persona = data.get("persona")
        if isinstance(persona, str) and persona in personas.PERSONAS:
            profile.persona = persona

        weights = data.get("weights")
        if isinstance(weights, dict):
            numeric = {
                k: float(v)
                for k, v in weights.items()
                if k in WEIGHTS and isinstance(v, (int, float))
            }
            if numeric:
                profile.weights = normalise({**WEIGHTS, **numeric})

        standing = data.get("standing")
        if isinstance(standing, dict):
            # .get, not [f] -- a file written by an older version, or by hand,
            # may simply be missing a field.
            fields = {
                f: standing.get(f)
                for f in _CONSTRAINT_FIELDS
                if isinstance(standing.get(f), (int, bool, type(None)))
            }
            banned = standing.get("banned")
            fields["banned"] = (
                tuple(str(b) for b in banned) if isinstance(banned, list) else ()
            )
            try:
                profile.standing = Constraints(**fields)
            except TypeError:
                profile.standing = Constraints()

        examples = data.get("examples")
        if isinstance(examples, list):
            profile.examples = [str(e) for e in examples if str(e).strip()][:MAX_EXAMPLES]

        for name in ("keeps", "batches"):
            value = data.get(name)
            setattr(profile, name, value if isinstance(value, int) and value >= 0 else 0)

        return profile
