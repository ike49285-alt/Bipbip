"""The conversation.

You talk to her; she talks back, and drafts when there is something worth
drafting. That framing is the point: the model is not a form that takes a topic
and returns tweets, so `Session` keeps a real message history and a turn where
she only asks a question is a normal turn.

`Session` owns the conversation and the model. `Profile` owns what survives
between runs. `Repl` turns a typed line into output and returns strings rather
than printing, so the loop in `cli.py` stays short and this is all testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import directives, personas
from .directives import Constraints
from .generate import DEFAULT_POOL, converse, propose
from .llm import Backend, BackendError, GrammarBackend, detect
from .profile import Profile
from .rank import Ranked, rank, similarity
from .render import render_ranked, render_score, render_weights
from .score import score_post

# How hard "more like 2" pulls the ranking toward the named draft.
AFFINITY = 60.0

NO_MODEL_REPLY = (
    "No local model is running, so I can't actually talk back. These are from "
    "the grammar -- start Ollama and I'll do better."
)

HELP = """\
Just talk. Tell her what happened; she'll ask, and draft when there's something
worth drafting. Say "shorter" or "never use exclamation marks" the way you'd
say it to a person.

  /rules           standing rules      /forget         drop them
  /weights         what she's learned  /untune         reset to the shipped prior
  /persona NAME    her register        /personas       list them
  /top K           drafts to show      /breakdown      component bars
  /keep N          keep one (teaches)  /kept  /drop N  /save PATH
  /score TEXT      score text yourself /backend        which model is answering
  /again           ask her again       /reset          forget the conversation
  /help  /quit"""


@dataclass
class Turn:
    """One exchange: what she said, and what she drafted."""

    reply: str
    drafts: list[Ranked] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)


@dataclass
class Session:
    """The conversation, the voice, and the constraints in force."""

    profile: Profile = field(default_factory=Profile)
    backend: Backend | None = None
    pool: int = DEFAULT_POOL
    history: list[dict] = field(default_factory=list)
    turn_constraints: Constraints = field(default_factory=Constraints)
    seed: int | None = None
    turns: int = 0

    def engine(self) -> Backend:
        """Resolve the backend once, so a probe does not run every turn."""
        if self.backend is None:
            self.backend = detect()
        return self.backend

    @property
    def persona(self) -> str:
        return self.profile.persona

    @persona.setter
    def persona(self, name: str) -> None:
        self.profile.persona = name

    def constraints(self) -> Constraints:
        """Standing rules first, then whatever this turn asked for."""
        return self.profile.standing.merge(self.turn_constraints)

    def last_said(self) -> str:
        for message in reversed(self.history):
            if message["role"] == "user":
                return message["content"]
        return ""

    def _grammar_turn(self, constraints: Constraints) -> Turn:
        """No model: draft from the last thing they said and say so."""
        topic = self.last_said()
        seed = None if self.seed is None else self.seed + self.turns
        drafts = propose(
            topic, persona=self.persona, backend=GrammarBackend(),
            pool=self.pool, seed=seed, constraints=constraints,
        )
        if not drafts and constraints.describe():
            # Silence would read as "nothing to say"; the constraint is the reason.
            return Turn(reply=(
                "Nothing I can write fits " + ", ".join(constraints.describe())
                + ". Loosen one of those and ask again."
            ))
        return Turn(reply=NO_MODEL_REPLY, drafts=rank(drafts, weights=self.profile.weights))

    def send(self, message: str, like_text: str | None = None) -> Turn:
        """One exchange. Raises ValueError for anything the caller should explain."""
        directive = directives.parse(message)
        adopted: list[str] = []

        if directive.persona:
            self.persona = directive.persona

        if directive.constraints.describe():
            if directive.standing:
                adopted = self.profile.add_standing(directive.constraints)
            else:
                self.turn_constraints = self.turn_constraints.merge(directive.constraints)

        # She never saw our numbering, so a reference is substituted for the text.
        said = message
        if like_text:
            said = f'{message}\n(referring to this draft: "{like_text}")'

        self.history.append({"role": "user", "content": said})
        constraints = self.constraints()

        try:
            if isinstance(self.engine(), GrammarBackend):
                turn = self._grammar_turn(constraints)
            else:
                reply, posts = converse(
                    self.engine(), self.history, persona=self.persona,
                    profile=self.profile, constraints=constraints,
                )
                ranked = rank(posts, weights=self.profile.weights) if posts else []
                if like_text and ranked:
                    for item in ranked:
                        item.final += AFFINITY * similarity(item.text, like_text)
                    ranked.sort(key=lambda r: r.final, reverse=True)
                turn = Turn(reply=reply, drafts=ranked)
        except BackendError as exc:
            self.history.pop()  # don't resend a turn that got no reply
            raise ValueError(str(exc)) from exc

        turn.adopted = adopted
        # Record what she said so the next turn has it, drafts included -- she
        # wrote them and may be asked to change them.
        spoken = turn.reply
        if turn.drafts:
            spoken += "\n" + "\n".join(f"- {d.text}" for d in turn.drafts)
        self.history.append({"role": "assistant", "content": spoken.strip() or "(no reply)"})

        self.turns += 1
        self.profile.batches += 1
        self.profile.save()
        return turn

    def rewind(self) -> str:
        """Drop the last exchange and hand back what they had said."""
        said = self.last_said()
        while self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
        if self.history:
            self.history.pop()
        return said

    def reset(self) -> None:
        self.history.clear()
        self.turn_constraints = Constraints()


class Repl:
    """Turns a typed line into output. Holds no file handles and never prints."""

    def __init__(self, session: Session | None = None, top: int = 3, breakdown: bool = False):
        self.session = session or Session()
        self.top = top
        self.breakdown = breakdown
        self.kept: list[Ranked] = []
        self.shown: list[Ranked] = []
        # The last exchange, for front ends that render it rather than print it.
        self.last_reply: str = ""
        self.last_adopted: list[str] = []

    @property
    def profile(self) -> Profile:
        return self.session.profile

    # -- helpers ---------------------------------------------------------

    def _pick(self, arg: str, pool: list[Ranked], what: str) -> int:
        if not pool:
            raise ValueError(f"no {what} yet")
        try:
            index = int(arg)
        except ValueError:
            raise ValueError(f"expected a number, got {arg!r}") from None
        if not 1 <= index <= len(pool):
            raise ValueError(f"pick 1-{len(pool)}, got {index}")
        return index - 1

    def _say(self, message: str) -> str:
        like_text = None
        directive = directives.parse(message)
        if directive.like is not None:
            if not self.shown:
                raise ValueError("no drafts yet")
            if not 1 <= directive.like <= len(self.shown):
                raise ValueError(f"pick 1-{len(self.shown)}, got {directive.like}")
            like_text = self.shown[directive.like - 1].text

        turn = self.session.send(message, like_text=like_text)
        self.last_reply = turn.reply
        self.last_adopted = turn.adopted
        out = turn.reply or "(she said nothing)"

        if turn.drafts:
            self.shown = turn.drafts[: self.top] if self.top > 0 else turn.drafts
            out += "\n\n" + render_ranked(self.shown, breakdown=self.breakdown)
        else:
            self.shown = []

        active = self.session.constraints().describe()
        if turn.drafts and active:
            out += "\n\n(" + ", ".join(active) + ")"
        for clause in turn.adopted:
            out += f"\n\n+ standing rule: {clause}   (/forget to drop it)"
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
                return f"she's set to {self.session.persona}", True
            personas.get(arg)
            self.session.persona = arg.strip().lower()
            self.profile.save()
            return f"persona -> {self.session.persona}", True

        if cmd == "rules":
            clauses = self.profile.standing.describe()
            if not clauses:
                return "no standing rules -- say one, like 'never use exclamation marks'", True
            return "standing rules:\n" + "\n".join(f"- {c}" for c in clauses), True

        if cmd == "forget":
            count = self.profile.clear_standing()
            self.profile.save()
            return f"dropped {count} standing rule(s)", True

        if cmd == "weights":
            return render_weights(self.profile), True

        if cmd == "untune":
            self.profile.reset_weights()
            self.profile.save()
            return "weights back to the shipped prior", True

        if cmd == "backend":
            engine = self.session.engine()
            note = "" if not isinstance(engine, GrammarBackend) else \
                "\nno local model found -- start Ollama, or see --backend in --help"
            return f"talking to {engine.describe()}{note}", True

        if cmd == "profile":
            where = self.profile.path or "(not saved to disk)"
            return (
                f"stored at {where}\n"
                f"{self.profile.confidence()}, "
                f"{len(self.profile.standing.describe())} standing rule(s), "
                f"{len(self.profile.examples)} example(s), "
                f"{self.profile.batches} turn(s)",
                True,
            )

        if cmd == "top":
            self.top = max(0, int(arg))
            return f"showing -> {'all' if self.top == 0 else self.top}", True

        if cmd == "breakdown":
            self.breakdown = not self.breakdown
            return f"breakdown -> {'on' if self.breakdown else 'off'}", True

        if cmd == "score":
            if not arg:
                return "usage: /score some text to score", True
            return render_score(arg, score_post(arg, self.profile.weights)), True

        if cmd == "keep":
            index = self._pick(arg, self.shown, "drafts")
            item = self.shown[index]
            if any(k.text == item.text for k in self.kept):
                return "already kept", True

            self.kept.append(item)
            self.profile.remember(item.text)
            others = [s.score.components for i, s in enumerate(self.shown) if i != index]
            moved = self.profile.learn_from_keep(item.score.components, others)
            self.profile.save()

            out = f"kept ({len(self.kept)} total)"
            if moved:
                top = sorted(moved.items(), key=lambda pair: -abs(pair[1]))[:2]
                shifts = ", ".join(f"{k} {v:+.3f}" for k, v in top if abs(v) > 0.0005)
                if shifts:
                    out += f"\nlearned: {shifts}   ({self.profile.confidence()})"
            return out, True

        if cmd == "kept":
            if not self.kept:
                return "nothing kept yet", True
            return render_ranked(self.kept, breakdown=self.breakdown), True

        if cmd == "drop":
            index = self._pick(arg, self.kept, "kept posts")
            return f"dropped: {self.kept.pop(index).text[:48]}", True

        if cmd == "save":
            if not self.kept:
                return "nothing kept to save", True
            if not arg:
                return "usage: /save path/to/file.txt", True
            with open(arg, "w", encoding="utf-8") as handle:
                handle.write("\n".join(k.text for k in self.kept) + "\n")
            return f"wrote {len(self.kept)} to {arg}", True

        if cmd == "again":
            said = self.session.rewind()
            if not said:
                return "nothing to ask again yet", True
            return self._say(said), True

        if cmd == "reset":
            self.session.reset()
            self.shown = []
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
            return self._say(line), True
        except (ValueError, KeyError) as exc:
            return f"error: {exc.args[0] if exc.args else exc}", True
        except OSError as exc:
            return f"error: {exc}", True
