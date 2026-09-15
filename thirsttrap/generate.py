"""Propose candidate posts from a locally running model.

Generation goes to whatever local backend `llm.detect()` finds -- Ollama, an
OpenAI-compatible server, or an in-process GGUF. Nothing leaves the machine and
there is no API key anywhere in this package.

The generator is never told the scoring rubric. If it were, the score would
mostly measure how well it followed instructions it had just been handed, and
every batch would look excellent. Format rules (length ceiling, no hashtags, no
links) are stated because they are platform facts, and the README says what that
costs the `restraint` and `length` components.

Constraints are applied twice on purpose: asked for in the prompt, then enforced
by filtering. Small local models ignore instructions often enough that asking
alone does not work, and filtering alone wastes most of a slow batch.
"""

from __future__ import annotations

from . import grammar, personas
from .directives import Constraints
from .llm import Backend, BackendError, GrammarBackend, detect, parse_posts, parse_turn
from .topic import parse as parse_topic

DEFAULT_N = 12
DEFAULT_POOL = 400  # grammar fallback only; a model is asked for far fewer
MAX_ATTEMPTS = 2

# Small local models have small context windows, so the conversation is
# trimmed rather than allowed to grow until it silently truncates.
MAX_HISTORY = 12

POST_RULES = """\
Rules for anything you draft:
- Every post stands alone. No threads, no numbering, no "1/".
- No hashtags. No @-mentions. No URLs.
- Under 280 characters. Most should be well under.
- Subjects are adults. Never write anything sexual about anyone who could read \
as underage, and never imply an age below adult.
- Suggestive through implication and restraint only. Nothing sexually explicit, \
nothing anatomical.
- Do not write as, quote, or reference any real named person.
- Confidence, never neediness. No begging for engagement, no fishing for \
compliments through self-deprecation.
- Never demean anyone, including the reader.
- Specific beats general. Say one thing. Cut every word doing nothing."""

CONVERSATION = """\
You are someone's writing partner. They talk to you about their life; you talk \
back, and when something in it is worth posting, you draft a few.

Talk like a person, not a service. React to what they actually said. Ask the \
question that gets the good detail out of them -- the specific one, not "tell \
me more". Keep it to a sentence or two; you are texting, not writing an essay.

You are NOT a form that takes a topic and returns tweets. Never ask them for a \
topic, never say "here are some options for you", never list what you can do.

Draft only when there is something real to draft from. If they are mid-story, \
or just venting, or you still do not know the detail that would make the post \
land, reply with no drafts and ask. An empty posts list is a normal turn.

When you do draft, draft from what THEY said -- their detail, their words, \
their situation. Two or three is plenty."""

RULES = """\
You write short-form posts for X (Twitter). You are good at it because you \
understand that attention is won in the first five words and lost by trying \
too hard.

Hard rules, no exceptions:
- Every post stands alone. No threads, no numbering across posts, no "1/".
- No hashtags. No @-mentions. No URLs.
- Under 280 characters. Most should be well under.
- Subjects are adults. Never write anything sexual about anyone who could read \
as underage, and never imply an age below adult.
- Suggestive through implication and restraint only. Nothing sexually explicit, \
nothing anatomical.
- Do not write as, quote, or reference any real named person.
- Confidence, never neediness. No begging for engagement, no "like if you \
agree", no fishing for compliments through self-deprecation.
- Never demean anyone, including the reader.

Craft:
- Specific beats general. A real detail outperforms an adjective.
- Say one thing. A post that makes two points makes neither.
- Cut every word doing nothing, then cut the first sentence.
- Make the candidates genuinely different from each other. Two posts that could \
be edited into one another means one of them is wasted.

Output format:
- One post per line. Nothing else -- no numbering, no commentary, no preamble.
- Do not wrap posts in quotes."""


def build_conversation_system(persona: str = personas.DEFAULT_PERSONA, profile=None) -> str:
    """Who she is, how she talks, and what this user has taught her."""
    voice = personas.get(persona)
    parts = [
        CONVERSATION,
        f"Your register, in conversation and in what you draft -- {voice.summary}:\n{voice.directive}",
        POST_RULES,
    ]
    if profile is not None:
        clauses = profile.standing.describe()
        if clauses:
            parts.append(
                "Standing preferences this user has stated. Apply them to every "
                "draft unless they override one:\n" + "\n".join(f"- {c}" for c in clauses)
            )
        if profile.examples:
            parts.append(
                "Posts they kept before. Match what these have in common -- do not "
                "reuse their wording:\n" + "\n".join(f"- {e}" for e in profile.examples[-6:])
            )
    parts.append(
        'Reply with only JSON: {"reply": "what you say back", "posts": ["draft", ...]}\n'
        'Leave "posts" empty when there is nothing worth drafting yet.'
    )
    return "\n\n".join(parts)


def converse(
    backend: Backend,
    history: list[dict],
    persona: str = personas.DEFAULT_PERSONA,
    profile=None,
    constraints: Constraints | None = None,
) -> tuple[str, list[str]]:
    """One conversational turn: what she says, and anything she drafted.

    Drafts that break a constraint are dropped rather than shown -- but her
    reply is kept either way, because a turn where she only talks is a normal
    turn, not a failure.
    """
    system = build_conversation_system(persona, profile)
    if constraints and constraints.describe():
        system += "\n\nRight now they want every draft to be: " + ", ".join(constraints.describe())

    raw = backend.chat(system, history[-MAX_HISTORY:], json_mode=True)
    reply, posts = parse_turn(raw)

    seen, kept = set(), []
    for text in posts:
        if text in seen:
            continue
        seen.add(text)
        if constraints is None or constraints.allows(text):
            kept.append(text)
    return reply, kept


def build_system(persona: str = personas.DEFAULT_PERSONA, profile=None) -> str:
    """Rules and voice for the one-shot `gen` command."""
    voice = personas.get(persona)
    parts = [RULES, f"Voice -- {voice.summary}:\n{voice.directive}"]

    if profile is not None:
        clauses = profile.standing.describe()
        if clauses:
            parts.append(
                "Standing preferences from this user. Apply them unless the current "
                "instruction overrides one:\n" + "\n".join(f"- {c}" for c in clauses)
            )
        if profile.examples:
            recent = profile.examples[-6:]
            parts.append(
                "Posts this user kept. Match what these have in common -- do not "
                "reuse their wording:\n" + "\n".join(f"- {e}" for e in recent)
            )
    return "\n\n".join(parts)


def build_prompt(
    topic: str,
    n: int = DEFAULT_N,
    constraints: Constraints | None = None,
    like_text: str | None = None,
) -> str:
    parts = [f"Write {n} candidate posts about: {topic}"]

    clauses = constraints.describe() if constraints else []
    if clauses:
        parts.append("Requirements for every post:\n" + "\n".join(f"- {c}" for c in clauses))
    if like_text:
        parts.append(f"Aim for the register of this one, without repeating it:\n{like_text}")

    parts.append(f"{n} posts, one per line, nothing else.")
    return "\n\n".join(parts)


def _from_grammar(
    topic: str, persona: str, pool: int, seed: int | None, constraints: Constraints | None
) -> list[str]:
    candidates = grammar.generate(parse_topic(topic), persona, count=pool, seed=seed)
    if constraints is None:
        return candidates
    return [text for text in candidates if constraints.allows(text)]


def propose(
    topic: str,
    persona: str = personas.DEFAULT_PERSONA,
    n: int = DEFAULT_N,
    backend: Backend | None = None,
    profile=None,
    constraints: Constraints | None = None,
    seed: int | None = None,
    pool: int = DEFAULT_POOL,
    like_text: str | None = None,
) -> list[str]:
    """Generate candidates, keeping only those the constraints allow.

    A constraint nothing satisfies returns an empty list rather than quietly
    relaxing itself -- the caller is expected to say so.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if pool < 1:
        raise ValueError(f"pool must be at least 1, got {pool}")
    personas.get(persona)  # raises with the valid names if it misses

    backend = backend or detect()
    if isinstance(backend, GrammarBackend):
        return _from_grammar(topic, persona, pool, seed, constraints)

    system = build_system(persona, profile)
    prompt = build_prompt(topic, n, constraints, like_text)

    seen: set[str] = set()
    kept: list[str] = []
    errors: list[str] = []

    # A small model often returns fewer usable posts than asked for, especially
    # once constraints bite. One retry is worth it; a loop is not.
    for _ in range(MAX_ATTEMPTS):
        if len(kept) >= n:
            break
        try:
            raw = backend.complete(system, prompt)
        except BackendError as exc:
            errors.append(str(exc))
            break

        for text in parse_posts(raw):
            if text in seen:
                continue
            seen.add(text)
            if constraints is None or constraints.allows(text):
                kept.append(text)

    if not kept and errors:
        raise BackendError(errors[0])
    return kept[:n]
