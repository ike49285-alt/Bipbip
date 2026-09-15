"""Publishing, with the disclosure check at the last possible moment.

Every adapter inherits `publish`, which re-checks the marker immediately before
the network call and refuses without it. That check already ran in the content
gate, and it runs again here on purpose: between the gate and the wire a
caption can be hand-edited, truncated to fit, reordered by a scheduler, or
assembled by code written later by someone who did not read this file. The
check is cheap and the failure is unrecoverable, so it happens twice.

`DryRun` is the default adapter and prints instead of posting. A project like
this should be able to run end to end, for weeks, without an API key - so that
what gets reviewed is the actual output rather than a description of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..disclosure import missing_disclosure
from ..persona import Persona


class PublishError(RuntimeError):
    """Refused before anything left the machine."""


@dataclass
class Platform:
    """Base adapter. Subclasses implement `_send` only."""
    name: str = "base"
    #: Hard cap the platform itself imposes, independent of the persona's.
    max_chars: int = 280
    sent: list = field(default_factory=list)

    def publish(self, caption: str, persona: Persona) -> str:
        caption = caption.strip()
        if not caption:
            raise PublishError("refusing to publish an empty caption")
        if missing_disclosure(caption, persona.disclosure.marker):
            raise PublishError(
                f"refusing to publish without {persona.disclosure.marker!r}. "
                "The marker is what tells a reader who screenshots this post, "
                "with no profile attached, what they are looking at.")
        if len(caption) > self.max_chars:
            raise PublishError(
                f"{len(caption)} chars exceeds {self.name}'s {self.max_chars} "
                "limit - shorten the text, never the marker")
        ref = self._send(caption, persona)
        self.sent.append((ref, caption))
        return ref

    def _send(self, caption: str, persona: Persona) -> str:
        raise NotImplementedError


@dataclass
class DryRun(Platform):
    """Prints what would go out. The default, and the one used in tests."""
    name: str = "dryrun"
    max_chars: int = 280
    verbose: bool = True

    def _send(self, caption: str, persona: Persona) -> str:
        ref = f"dryrun://{persona.handle.lstrip('@')}/{len(self.sent) + 1}"
        if self.verbose:
            print(f"  [{self.name}] {caption}")
        return ref
