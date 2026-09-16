"""Step 5: replying to DMs in character, inside hard boundaries.

Two rules shape this file.

**Deterministic first, model second.** Every boundary is enforced by string
matching that works regardless of which model sits behind it. The model writes
the words; it does not decide whether a boundary applies. This matters because
the plan is to run a cheap model here, and a cheap model must not be able to
argue its way past a limit.

**Outbound is checked, not just inbound.** Classifying the incoming message is
half the job. Whatever the model generates is checked again before it is sent,
and thrown away for a canned reply if it breaks a rule. That is what makes
"answers truthfully when asked if it's a bot" a guarantee rather than a hope:
worst case, the persona's own disclosure line goes out verbatim.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from bot.persona import Persona

# Highest priority first. A message can trip several; the first one here wins
# for choosing behaviour, but every tripped policy still constrains the reply.
TERMINATE = "terminate"
DISCLOSE = "disclose"
BLOCK_MONEY = "block_money"
REFUSE_PII = "refuse_pii"
DEFLECT = "deflect"
NORMAL = "normal"

PRIORITY = (TERMINATE, DISCLOSE, BLOCK_MONEY, REFUSE_PII, DEFLECT)

# Markers that count as admitting to being synthetic.
DISCLOSURE_MARKERS = ("ai", "a.i.", "bot", "synthetic", "generated", "not a real person",
                      "not real", "not human")

# Outbound violation codes. Dispatch on these, never on the message text.
EMPTY = "empty"
CLAIMS_HUMAN = "claims_human"
SOLICITS = "solicits"
NO_DISCLOSURE = "no_disclosure"


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}" if self.detail else self.code


def normalise(text: str) -> str:
    """Lowercase, straighten quotes, collapse whitespace and punctuation."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def squash(text: str) -> str:
    """Alphanumerics only -- defeats 'c a s h a p p' and 'cash-app'."""
    return re.sub(r"[^a-z0-9]", "", normalise(text))


def matches(text: str, phrases: Sequence[str]) -> tuple[str, ...]:
    """Which of `phrases` appear in `text`, checked plainly and squashed."""
    flat, tight = normalise(text), squash(text)
    hits = []
    for phrase in phrases:
        if normalise(phrase) in flat or squash(phrase) in tight:
            hits.append(phrase)
    return tuple(hits)


@dataclass(frozen=True)
class Assessment:
    """What the incoming message tripped, and why. Kept for audit."""

    action: str
    policies: tuple[str, ...] = ()
    matched: dict = field(default_factory=dict)

    def tripped(self, policy: str) -> bool:
        return policy in self.policies


def classify(message: str, persona: Persona) -> Assessment:
    """Decide what an incoming message requires. Pure string matching."""
    b = persona.bounds
    found: dict[str, tuple[str, ...]] = {}
    for policy, phrases in (
        (TERMINATE, b.terminate_thread),
        (DISCLOSE, b.always_answer_truthfully),
        (BLOCK_MONEY, b.hard_block_outbound),
        (REFUSE_PII, b.refuse_to_supply),
        (DEFLECT, b.deflect),
    ):
        hits = matches(message, phrases)
        if hits:
            found[policy] = hits

    policies = tuple(p for p in PRIORITY if p in found)
    return Assessment(
        action=policies[0] if policies else NORMAL,
        policies=policies,
        matched=found,
    )


def check_outbound(
    text: str, persona: Persona, *, must_disclose: bool = False
) -> list[Violation]:
    """Reasons this reply must not be sent. Empty means it may go."""
    b = persona.bounds
    problems: list[Violation] = []

    if not text.strip():
        return [Violation(EMPTY)]

    lied = matches(text, b.never_claim)
    if lied:
        problems.append(Violation(CLAIMS_HUMAN, repr(lied[0])))

    solicits = matches(text, b.hard_block_outbound)
    if solicits:
        problems.append(Violation(SOLICITS, repr(solicits[0])))

    if must_disclose:
        flat = normalise(text)
        if not any(marker in flat for marker in DISCLOSURE_MARKERS):
            problems.append(Violation(NO_DISCLOSURE, "asked directly"))

    return problems


class DMLLM(Protocol):
    """The one call this module makes. Injected so the gates are testable."""

    def reply(self, system: str, history: Sequence[dict], guidance: str) -> str: ...


@dataclass(frozen=True)
class Reply:
    """The outcome of one inbound message."""

    text: str | None
    action: str
    terminated: bool = False
    flagged: bool = False
    canned: bool = False
    notes: tuple[str, ...] = ()

    @property
    def sends(self) -> bool:
        return self.text is not None


@dataclass
class DMAgent:
    persona: Persona
    llm: DMLLM
    attempts: int = 2

    # -- guidance the model receives in addition to the voice card ------

    def guidance_for(self, assessment: Assessment) -> str:
        p, b = self.persona, self.persona.bounds
        lines = ["Reply in your own voice, one or two short lines."]
        if assessment.tripped(DISCLOSE):
            lines.append(
                "They asked whether you are real or a bot. Say plainly that you "
                "are an AI persona. Do not soften it into a joke or a dodge."
            )
        if assessment.tripped(BLOCK_MONEY):
            lines.append(
                "They raised money, payment or promotion. Decline plainly. Never "
                "name a payment method and never offer a link."
            )
        if assessment.tripped(REFUSE_PII):
            lines.append("They asked for identifying details. Decline, in character.")
        if assessment.tripped(DEFLECT):
            lines.append(
                "They proposed meeting or a call. You have no physical existence "
                "to offer. Say so without being cold about it."
            )
        return "\n".join(lines)

    def context(self, history: Sequence[dict], recent_posts: Sequence[str] = ()) -> list[dict]:
        window = list(history)[-20:]
        if recent_posts:
            posted = "; ".join(recent_posts[:3])
            window = [{"role": "system", "content": f"You recently posted: {posted}"}] + window
        return window

    # -- the fallback that makes the guarantee hold ---------------------

    def canned_reply(
        self, assessment: Assessment, violations: Sequence[Violation] = ()
    ) -> str:
        """Pick a safe reply.

        Outbound violations come first: the gate can catch something the
        inbound classifier had no reason to flag, and answering *that* is the
        point. Falling back to the tripped inbound policy would, for example,
        answer a money solicitation with a disclosure line.
        """
        canned = self.persona.bounds.canned
        codes = {v.code for v in violations}
        if CLAIMS_HUMAN in codes or NO_DISCLOSURE in codes:
            return self.persona.identity.disclosure
        if SOLICITS in codes:
            return canned.get("money", "no.")

        if assessment.tripped(DISCLOSE):
            return self.persona.identity.disclosure
        if assessment.tripped(BLOCK_MONEY):
            return canned.get("money", "no.")
        if assessment.tripped(REFUSE_PII):
            return canned.get("pii", "no.")
        if assessment.tripped(DEFLECT):
            return canned.get("deflect", "i only exist here.")
        return self.persona.identity.disclosure

    def respond(
        self,
        message: str,
        history: Sequence[dict] = (),
        recent_posts: Sequence[str] = (),
    ) -> Reply:
        assessment = classify(message, self.persona)

        if assessment.action == TERMINATE:
            # No reply at all. Engaging is the failure mode here.
            return Reply(
                text=None, action=TERMINATE, terminated=True, flagged=True,
                notes=(f"matched {assessment.matched[TERMINATE]!r}", "needs human review"),
            )

        must_disclose = assessment.tripped(DISCLOSE)
        notes: list[str] = []
        problems: list[Violation] = []

        for _ in range(self.attempts):
            draft = self.llm.reply(
                self.persona.voice_card(),
                self.context(history, recent_posts),
                self.guidance_for(assessment),
            )
            problems = check_outbound(draft, self.persona, must_disclose=must_disclose)
            if not problems:
                return Reply(text=draft, action=assessment.action, notes=tuple(notes))
            notes.append("rejected draft: " + "; ".join(str(p) for p in problems))

        # Every draft broke a rule. Send the persona's own safe line instead of
        # sending nothing -- silence after "are you a bot?" is its own answer.
        codes = {v.code for v in problems}
        return Reply(
            text=self.canned_reply(assessment, problems),
            action=assessment.action,
            canned=True,
            flagged=must_disclose or bool(codes & {CLAIMS_HUMAN, SOLICITS}),
            notes=tuple(notes + ["fell back to canned reply"]),
        )


class AnthropicDMLLM:
    """Real backend. Lazy import so bot.dm stays dependency-free."""

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model

    def reply(self, system: str, history: Sequence[dict], guidance: str) -> str:
        messages = [m for m in history if m.get("role") in ("user", "assistant")]
        messages.append({"role": "user", "content": guidance})
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            # Voice card is identical every call; cache the prefix.
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        return response.content[0].text.strip()
