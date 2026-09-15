"""Candidate generation via the Claude API.

Design note -- the generator is deliberately NOT told the scoring rubric.
If it were, the score would mostly measure compliance with instructions the
generator had just been handed, and every batch would look excellent. Keeping
the two halves ignorant of each other is what makes the ranking worth reading.
The one overlap is format (no hashtags, no links), which is a platform fact
rather than a scoring trick -- see the README for what that costs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import personas

MODEL = "claude-opus-5"

SYSTEM = """You write short-form posts for X (Twitter). You are good at it \
because you understand that attention is won in the first five words and lost \
by trying too hard.

Hard rules, no exceptions:
- Every post stands alone. No threads, no numbering, no "1/".
- No hashtags. No @-mentions. No URLs.
- Under 280 characters. Most should be well under.
- Subjects are adults. Never write anything sexual about anyone who could read \
as underage, and never imply an age below adult.
- Suggestive through implication and restraint only. Nothing sexually explicit, \
nothing anatomical.
- Do not write as, quote, or reference any real named person.
- Confidence, never neediness. No begging for engagement, no "like if you agree", \
no fishing for compliments through self-deprecation.
- Never demean anyone, including the reader, and never negging.

Craft:
- Specific beats general. A real detail outperforms an adjective.
- Say one thing. A post that makes two points makes neither.
- Cut every word that is doing nothing. Then cut the first sentence.
- Vary the candidates genuinely -- different angles and structures, not the \
same sentence rephrased. Rephrasings are worthless to the caller."""


class Candidate(BaseModel):
    text: str = Field(description="The post exactly as it would be published.")
    angle: str = Field(description="The angle taken, in three or four words.")


class Batch(BaseModel):
    candidates: list[Candidate]


def build_prompt(topic: str, persona: personas.Persona, n: int) -> str:
    return (
        f"Write {n} candidate posts about: {topic}\n\n"
        f"Voice -- {persona.summary}:\n{persona.directive}\n\n"
        f"Give {n} genuinely different takes. If two of them could be edited into "
        f"each other, one of them is wasted."
    )


def generate(
    topic: str,
    persona: str = personas.DEFAULT_PERSONA,
    n: int = 12,
    client=None,
) -> list[Candidate]:
    """Generate `n` candidate posts. Returns them unscored and unranked.

    `client` is injectable so tests and callers can supply their own configured
    client (or a fake) instead of this function reaching for the environment.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    voice = personas.get(persona)

    if client is None:
        import anthropic

        # Credentials resolve from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile -- the SDK checks all three, so don't pre-check one.
        client = anthropic.Anthropic()

    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": build_prompt(topic, voice, n)}],
        output_format=Batch,
    )

    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", None) or "no explanation given"
        raise RuntimeError(f"the model declined this topic: {detail}")

    return response.parsed_output.candidates
