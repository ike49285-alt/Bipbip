"""Interactive session: refine a batch by talking to it, and have it stick.

Three objects. `Session` owns the conversation with Claude. `Profile` (in
profile.py) owns what survives between sessions. `Repl` owns the commands and
turning a typed line into output -- it returns strings instead of printing, so
the loop in `cli.py` stays short and the whole thing is testable without stdin.

Tuning is meant to be passive: say "stop using exclamation marks" once and it
becomes a standing rule, rather than something you set with a flag. The cost of
that is inference, which is sometimes wrong -- so every newly learned rule is
printed on the turn it is learned, and `/forget` removes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from . import personas
from .generate import MODEL, SYSTEM, Candidate
from .profile import Profile
from .rank import Ranked, rank
from .render import render_ranked, render_score
from .score import score_post


class ChatBatch(BaseModel):
    """The one-shot `Batch` plus whatever the turn revealed about lasting taste."""

    candidates: list[Candidate]
    learned: list[str] = Field(
        default_factory=list,
        description=(
            "Durable style preferences revealed by the latest instruction, each a "
            "short rule. Usually empty."
        ),
    )


CHAT_SYSTEM = """
You are in an interactive session. Each turn, return a fresh batch of \
candidates responding to the latest instruction.

The numbers in your previous turn are the numbers the user can see. When they \
say "more like 2" or "shorter than 3", that is the post you wrote with that \
number.

Refinement means rewriting the batch, not appending to it. If the user asks for \
shorter, every candidate gets shorter -- do not return one short post and five \
of the previous ones.

Also return `learned`: standing preferences the instruction reveals about this \
user's taste, phrased as short rules for your future self.

This is for preferences that should outlive the current topic. "Make it \
shorter" is about these posts and is not durable. "I never want exclamation \
marks" or "always end on the short line" are. Most turns reveal nothing \
durable, so most turns should return an empty list -- a rule you invent from a \
one-off request will steer every future batch wrongly."""

HELP = """\
Type anything to talk to her. The first thing you type is the topic; after that
you are refining ("shorter", "more like 2", "less earnest", "try a gym angle").

Standing preferences are picked up as you talk -- anything learned is printed
when it happens, and /forget removes it.

  /rules           what she has learned    /forget N   drop a rule (/forget all)
  /persona NAME    switch voice            /personas   list voices
  /n N             candidates per batch    /top K      how many to show
  /breakdown       toggle component bars   /score TEXT score text locally
  /keep N          pin a candidate         /kept       show pinned
  /drop N          unpin                   /save PATH  write pinned to a file
  /profile         where memory is stored
  /again           re-run the last instruction
  /reset           forget the conversation (keeps what she has learned)
  /help  /quit"""


@dataclass
class Session:
    """The conversation with Claude. Stateless API, so history is resent each turn."""

    profile: Profile = field(default_factory=Profile)
    n: int = 12
    client: object | None = None
    history: list[dict] = field(default_factory=list)
    last_learned: list[str] = field(default_factory=list)

    @property
    def persona(self) -> str:
        return self.profile.persona

    @persona.setter
    def persona(self, name: str) -> None:
        self.profile.persona = name

    def _client(self):
        if self.client is None:
            import anthropic

            # Credentials resolve from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or
            # an `ant auth login` profile -- the SDK checks all three.
            self.client = anthropic.Anthropic()
        return self.client

    def system(self) -> str:
        voice = personas.get(self.persona)
        parts = [
            SYSTEM,
            CHAT_SYSTEM,
            f"Return exactly {self.n} candidates every turn.",
            f"Voice -- {voice.summary}:\n{voice.directive}",
        ]
        brief = self.profile.brief()
        if brief:
            parts.append(brief)
        return "\n\n".join(parts)

    def send(self, message: str) -> list[Ranked]:
        """One turn: send, rank the reply, record it, and absorb what it taught."""
        self.history.append({"role": "user", "content": message})

        response = self._client().messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=self.system(),
            thinking={"type": "adaptive"},
            # Caches the longest stable prefix. Early turns are below the minimum
            # cacheable length and simply won't hit; it earns its keep as history grows.
            cache_control={"type": "ephemeral"},
            messages=self.history,
            output_format=ChatBatch,
        )

        if response.stop_reason == "refusal":
            self.history.pop()  # don't leave a turn that got no reply
            detail = getattr(response.stop_details, "explanation", None) or "no explanation given"
            raise RuntimeError(f"the model declined that: {detail}")

        batch = response.parsed_output
        ranked = rank([c.text for c in batch.candidates])

        # Feed the batch back in ranked order and numbered, so "2" means the same
        # post to the model as it does on screen.
        self.history.append(
            {
                "role": "assistant",
                "content": "\n".join(f"{i}. {r.text}" for i, r in enumerate(ranked, 1)),
            }
        )

        self.last_learned = self.profile.learn(batch.learned)
        self.profile.batches += 1
        self.profile.save()
        return ranked

    def reset(self) -> None:
        """Clear the conversation. What she has learned is not part of it."""
        self.history.clear()
        self.last_learned = []


class Repl:
    """Turns a typed line into output. Holds no file handles and never prints."""

    def __init__(self, session: Session | None = None, top: int = 3, breakdown: bool = False):
        self.session = session or Session()
        self.top = top
        self.breakdown = breakdown
        self.kept: list[Ranked] = []
        self.shown: list[Ranked] = []
        self.last_message: str | None = None

    @property
    def profile(self) -> Profile:
        return self.session.profile

    # -- helpers ---------------------------------------------------------

    def _pick(self, arg: str) -> Ranked:
        """Resolve a 1-based reference into the batch on screen."""
        if not self.shown:
            raise ValueError("no batch yet -- say something first")
        try:
            index = int(arg)
        except ValueError:
            raise ValueError(f"expected a number, got {arg!r}") from None
        if not 1 <= index <= len(self.shown):
            raise ValueError(f"pick 1-{len(self.shown)}, got {index}")
        return self.shown[index - 1]

    def _turn(self, message: str) -> str:
        ranked = self.session.send(message)
        self.last_message = message
        self.shown = ranked[: self.top] if self.top > 0 else ranked

        header = f"{len(ranked)} candidates, showing top {len(self.shown)}  [{self.session.persona}]"
        out = f"{header}\n\n{render_ranked(self.shown, breakdown=self.breakdown)}"

        # Surface inference immediately -- a rule learned silently is a rule that
        # steers every later batch without anyone knowing to correct it.
        for rule in self.session.last_learned:
            out += f"\n\n+ learned: {rule}   (/forget to drop it)"
        return out

    # -- commands --------------------------------------------------------

    def _command(self, line: str) -> tuple[str, bool]:
        word, _, arg = line[1:].strip().partition(" ")
        arg = arg.strip()
        cmd = word.lower()

        if cmd in {"quit", "exit", "q"}:
            return "", False
        if cmd in {"help", "h", "?"}:
            return HELP, True
        if cmd == "personas":
            return "\n".join(
                f"{n:<14} {personas.PERSONAS[n].summary}" for n in personas.names()
            ), True

        if cmd == "persona":
            if not arg:
                return f"persona is {self.session.persona}", True
            personas.get(arg)  # raises with the valid names if it misses
            self.session.persona = arg.strip().lower()
            self.profile.save()
            return f"persona -> {self.session.persona}", True

        if cmd == "rules":
            if not self.profile.rules:
                return "nothing learned yet -- just talk to her", True
            listed = "\n".join(f"{i}. {r}" for i, r in enumerate(self.profile.rules, 1))
            return f"learned so far:\n{listed}", True

        if cmd == "forget":
            if arg.lower() == "all":
                count = self.profile.forget_all()
                self.profile.save()
                return f"forgot {count} rule(s)", True
            if not arg:
                return "usage: /forget N   (or /forget all)", True
            dropped = self.profile.forget(int(arg))
            self.profile.save()
            return f"forgot: {dropped}", True

        if cmd == "profile":
            where = self.profile.path or "(not saved to disk)"
            return (
                f"stored at {where}\n"
                f"{len(self.profile.rules)} rule(s), "
                f"{len(self.profile.examples)} example(s), "
                f"{self.profile.batches} batch(es) so far",
                True,
            )

        if cmd == "n":
            self.session.n = max(1, int(arg))
            return f"batch size -> {self.session.n}", True

        if cmd == "top":
            self.top = max(0, int(arg))
            return f"showing -> {'all' if self.top == 0 else self.top}", True

        if cmd == "breakdown":
            self.breakdown = not self.breakdown
            return f"breakdown -> {'on' if self.breakdown else 'off'}", True

        if cmd == "score":
            if not arg:
                return "usage: /score some text to score", True
            return render_score(arg, score_post(arg), breakdown=True), True

        if cmd == "keep":
            item = self._pick(arg)
            if any(k.text == item.text for k in self.kept):
                return "already kept", True
            self.kept.append(item)
            # Keeping is the strongest signal there is, so it outlives the session.
            self.profile.remember(item.text)
            self.profile.save()
            return f"kept ({len(self.kept)} total, and remembered)", True

        if cmd == "kept":
            if not self.kept:
                return "nothing kept yet", True
            return render_ranked(self.kept, breakdown=self.breakdown), True

        if cmd == "drop":
            if not self.kept:
                return "nothing kept yet", True
            index = int(arg)
            if not 1 <= index <= len(self.kept):
                raise ValueError(f"pick 1-{len(self.kept)}, got {index}")
            dropped = self.kept.pop(index - 1)
            return f"dropped: {dropped.text[:48]}", True

        if cmd == "save":
            if not self.kept:
                return "nothing kept to save", True
            if not arg:
                return "usage: /save path/to/file.txt", True
            with open(arg, "w", encoding="utf-8") as handle:
                handle.write("\n".join(k.text for k in self.kept) + "\n")
            return f"wrote {len(self.kept)} to {arg}", True

        if cmd == "again":
            if not self.last_message:
                return "nothing to re-run yet", True
            return self._turn(self.last_message), True

        if cmd == "reset":
            self.session.reset()
            self.shown = []
            self.last_message = None
            return "conversation cleared (she keeps what she learned)", True

        return f"unknown command {word!r} -- /help for the list", True

    # -- entry point -----------------------------------------------------

    def handle(self, line: str) -> tuple[str, bool]:
        """Return (output, keep_going). Never raises for ordinary user error."""
        line = line.strip()
        if not line:
            return "", True

        try:
            if line.startswith("/"):
                return self._command(line)
            return self._turn(line), True
        except (ValueError, KeyError) as exc:
            message = exc.args[0] if exc.args else str(exc)
            return f"error: {message}", True
        except OSError as exc:
            return f"error: {exc}", True
        except Exception as exc:  # API, auth, refusal -- keep the session alive
            return f"error: {exc}", True
