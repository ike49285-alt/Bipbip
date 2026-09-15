"""Interactive session: refine a batch by talking to it.

Two objects. `Session` owns the conversation with Claude and nothing else.
`Repl` owns the commands, the kept list, and turning a typed line into output --
it returns strings instead of printing, so the loop in `cli.py` stays four lines
and the whole thing is testable without stdin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import personas
from .generate import MODEL, SYSTEM, Batch
from .rank import Ranked, rank
from .render import render_ranked, render_score
from .score import score_post

CHAT_SYSTEM = """
You are in an interactive session. Each turn, return a fresh batch of \
candidates responding to the latest instruction.

The numbers in your previous turn are the numbers the user can see. When they \
say "more like 2" or "shorter than 3", that is the post you wrote with that \
number.

Refinement means rewriting the batch, not appending to it. If the user asks for \
shorter, every candidate gets shorter -- do not return one short post and five \
of the previous ones."""

HELP = """\
Type anything to talk to it. The first thing you type is the topic; after that
you are refining ("shorter", "more like 2", "less earnest", "try a gym angle").

  /persona NAME    switch voice          /personas   list voices
  /n N             candidates per batch  /top K      how many to show
  /breakdown       toggle component bars /score TEXT score text locally
  /keep N          pin a candidate       /kept       show pinned
  /drop N          unpin                 /save PATH  write pinned to a file
  /again           re-run the last instruction
  /reset           forget the conversation
  /help  /quit"""


@dataclass
class Session:
    """The conversation with Claude. Stateless API, so history is resent each turn."""

    persona: str = personas.DEFAULT_PERSONA
    n: int = 12
    client: object | None = None
    history: list[dict] = field(default_factory=list)

    def _client(self):
        if self.client is None:
            import anthropic

            # Credentials resolve from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or
            # an `ant auth login` profile -- the SDK checks all three.
            self.client = anthropic.Anthropic()
        return self.client

    def system(self) -> str:
        voice = personas.get(self.persona)
        return (
            f"{SYSTEM}\n{CHAT_SYSTEM}\n\n"
            f"Return exactly {self.n} candidates every turn.\n\n"
            f"Voice -- {voice.summary}:\n{voice.directive}"
        )

    def send(self, message: str) -> list[Ranked]:
        """One turn: send, rank the reply, and record it in the numbering the user sees."""
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
            output_format=Batch,
        )

        if response.stop_reason == "refusal":
            self.history.pop()  # don't leave a turn that got no reply
            detail = getattr(response.stop_details, "explanation", None) or "no explanation given"
            raise RuntimeError(f"the model declined that: {detail}")

        ranked = rank([c.text for c in response.parsed_output.candidates])

        # Feed the batch back in ranked order and numbered, so "2" means the same
        # post to the model as it does on screen.
        self.history.append(
            {
                "role": "assistant",
                "content": "\n".join(f"{i}. {r.text}" for i, r in enumerate(ranked, 1)),
            }
        )
        return ranked

    def reset(self) -> None:
        self.history.clear()


class Repl:
    """Turns a typed line into output. Holds no file handles and never prints."""

    def __init__(self, session: Session | None = None, top: int = 3, breakdown: bool = False):
        self.session = session or Session()
        self.top = top
        self.breakdown = breakdown
        self.kept: list[Ranked] = []
        self.shown: list[Ranked] = []
        self.last_message: str | None = None

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
        return f"{header}\n\n{render_ranked(self.shown, breakdown=self.breakdown)}"

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
            return f"persona -> {self.session.persona}", True

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
            return f"kept ({len(self.kept)} total)", True

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
            return "conversation cleared (kept posts survive)", True

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
