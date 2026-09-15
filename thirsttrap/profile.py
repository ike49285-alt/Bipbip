"""What she remembers between sessions.

Two kinds of memory, and the distinction matters:

- `rules` are standing preferences in the user's own words, inferred from
  ordinary instructions ("I never want exclamation marks").
- `examples` are posts the user actually kept -- ground truth about what lands,
  not an opinion about it.

Deliberately absent: anything derived from the scorer. Feeding "you tend to keep
high-rhythm posts" back into generation would hand the model the rubric it is
being judged against, which is the one thing the split in generate.py exists to
prevent. The user's words and the user's choices are evidence; the scorer's
opinion of them is not.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import personas

# Every rule and example is resent on every turn, so both are capped. The limits
# are about prompt weight, not significance -- the oldest go first.
MAX_RULES = 40
MAX_EXAMPLES = 12

ENV_OVERRIDE = "THIRSTTRAP_PROFILE"


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
    rules: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    batches: int = 0
    path: Path | None = None

    # -- learning --------------------------------------------------------

    def learn(self, rules: list[str]) -> list[str]:
        """Add standing rules, returning only the genuinely new ones.

        The caller shows what came back, so a wrong inference is visible on the
        turn it happens rather than quietly steering every batch afterwards.
        """
        added = []
        for rule in rules:
            cleaned = " ".join(str(rule).split())
            if not cleaned:
                continue
            if any(cleaned.lower() == existing.lower() for existing in self.rules):
                continue
            self.rules.append(cleaned)
            added.append(cleaned)

        if len(self.rules) > MAX_RULES:
            del self.rules[: len(self.rules) - MAX_RULES]
        return added

    def remember(self, text: str) -> bool:
        """Record a kept post as an example. False if it was already there."""
        if any(text == existing for existing in self.examples):
            return False
        self.examples.append(text)
        if len(self.examples) > MAX_EXAMPLES:
            del self.examples[: len(self.examples) - MAX_EXAMPLES]
        return True

    def forget(self, index: int) -> str:
        """Drop rule `index` (1-based). The correction path for a bad inference."""
        if not self.rules:
            raise ValueError("nothing learned yet")
        if not 1 <= index <= len(self.rules):
            raise ValueError(f"pick 1-{len(self.rules)}, got {index}")
        return self.rules.pop(index - 1)

    def forget_all(self) -> int:
        count = len(self.rules)
        self.rules.clear()
        return count

    # -- prompt ----------------------------------------------------------

    def brief(self) -> str:
        """The fragment injected into the system prompt. Empty when she knows nothing."""
        blocks = []
        if self.rules:
            lines = "\n".join(f"- {r}" for r in self.rules)
            blocks.append(
                "Standing preferences learned from this user in earlier sessions. "
                "Apply them unless the current instruction overrides one:\n" + lines
            )
        if self.examples:
            lines = "\n".join(f"- {e}" for e in self.examples)
            blocks.append(
                "Posts this user chose to keep. Match what these have in common -- "
                "do not reuse their wording:\n" + lines
            )
        return "\n\n".join(blocks)

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "persona": self.persona,
            "rules": self.rules,
            "examples": self.examples,
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
        profile.rules = [str(r) for r in data.get("rules", []) if str(r).strip()][:MAX_RULES]
        profile.examples = [
            str(e) for e in data.get("examples", []) if str(e).strip()
        ][:MAX_EXAMPLES]
        batches = data.get("batches")
        profile.batches = batches if isinstance(batches, int) and batches >= 0 else 0
        return profile
