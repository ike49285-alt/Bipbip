"""Interactive session. Nothing here leaves the machine.

`Session` holds the topic and the constraints in force. `Profile` holds what
survives between runs. `Repl` turns a typed line into output and returns
strings rather than printing, so the loop in `cli.py` stays short and the whole
surface is testable.

Tuning is passive in two ways. Rules you state ("never use exclamation marks")
are parsed into standing constraints. Everything else is learned from which
candidate you keep: the kept post is a statement that it beat the others on
screen, and the weights move toward whatever distinguished it. Both are
visible -- a new rule prints when adopted, `/weights` shows the drift, and both
are reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import directives, personas
from .directives import Constraints
from .generate import DEFAULT_N, DEFAULT_POOL, propose
from .llm import Backend, BackendError, GrammarBackend, detect
from .profile import Profile
from .rank import Ranked, rank, similarity
from .render import render_ranked, render_score, render_weights
from .score import score_post

# How hard "more like 2" pulls the ranking toward the named post. Large enough
# that an explicit instruction beats a score gap -- at 25 the request lost to
# whatever happened to score highest, which is not what the user asked for.
AFFINITY = 60.0

HELP = """\
Type a topic to start. After that, type adjustments -- "shorter", "no
questions", "one line", "more like 2", "try deadpan".

Say a rule ("never use exclamation marks") and it sticks across sessions.
Everything else she learns from what you /keep.

  /rules           standing rules      /forget         drop them
  /weights         what she learned    /untune         reset to the shipped prior
  /persona NAME    switch voice        /personas       list voices
  /pool N          candidates to draw  /top K          how many to show
  /breakdown       component bars      /score TEXT     score text yourself
  /keep N          keep one (teaches)  /kept  /drop N  /save PATH
  /clear           drop this turn's adjustments
  /profile         where memory lives    /backend  which model is answering
  /again  /help  /quit"""


@dataclass
class Session:
    """Topic, voice, and the constraints currently in force."""

    profile: Profile = field(default_factory=Profile)
    backend: Backend | None = None
    n: int = DEFAULT_N
    pool: int = DEFAULT_POOL
    topic: str | None = None
    turn_constraints: Constraints = field(default_factory=Constraints)
    turns: int = 0
    last_adopted: list[str] = field(default_factory=list)
    seed: int | None = None

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

    def send(self, message: str, like_text: str | None = None) -> list[Ranked]:
        """Interpret one line and return a ranked batch."""
        directive = directives.parse(message)
        self.last_adopted = []

        if directive.persona:
            self.persona = directive.persona

        if directive.constraints:
            if directive.standing:
                self.last_adopted = self.profile.add_standing(directive.constraints)
            else:
                self.turn_constraints = self.turn_constraints.merge(directive.constraints)

        # An unrecognised line is a subject, not an instruction. That is what
        # makes the first thing you type the topic without any ceremony.
        if not directive.recognised:
            self.topic = message
            self.turn_constraints = Constraints()
        elif self.topic is None:
            raise ValueError("say what the posts should be about first")

        constraints = self.constraints()
        # Vary the draw per turn so /again is a fresh look, but stay reproducible
        # for a caller that pinned a seed.
        seed = None if self.seed is None else self.seed + self.turns
        try:
            candidates = propose(
                self.topic,
                persona=self.persona,
                n=self.n,
                backend=self.engine(),
                profile=self.profile,
                constraints=constraints,
                seed=seed,
                pool=self.pool,
                like_text=like_text,
            )
        except BackendError as exc:
            raise ValueError(f"{exc}") from exc

        if not candidates:
            clauses = ", ".join(constraints.describe()) or "the current constraints"
            raise ValueError(f"nothing survived: {clauses}. /clear to relax this turn")

        if like_text:
            # Handing back the same post is not "more like" it.
            candidates = [c for c in candidates if c != like_text] or candidates

        ranked = rank(candidates, weights=self.profile.weights)
        if like_text:
            for item in ranked:
                item.final += AFFINITY * similarity(item.text, like_text)
            ranked.sort(key=lambda r: r.final, reverse=True)

        self.turns += 1
        self.profile.batches += 1
        self.profile.save()
        return ranked

    def clear(self) -> None:
        self.turn_constraints = Constraints()


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

    def _pick(self, arg: str, pool: list[Ranked], what: str) -> int:
        """Resolve a 1-based reference, or explain the range."""
        if not pool:
            raise ValueError(f"no {what} yet")
        try:
            index = int(arg)
        except ValueError:
            raise ValueError(f"expected a number, got {arg!r}") from None
        if not 1 <= index <= len(pool):
            raise ValueError(f"pick 1-{len(pool)}, got {index}")
        return index - 1

    def _turn(self, message: str) -> str:
        like_text = None
        directive = directives.parse(message)
        if directive.like is not None:
            if not self.shown:
                raise ValueError("no batch yet -- say a topic first")
            if not 1 <= directive.like <= len(self.shown):
                raise ValueError(f"pick 1-{len(self.shown)}, got {directive.like}")
            like_text = self.shown[directive.like - 1].text

        ranked = self.session.send(message, like_text=like_text)
        self.last_message = message
        self.shown = ranked[: self.top] if self.top > 0 else ranked

        head = f"{len(ranked)} candidates, showing top {len(self.shown)}  [{self.session.persona}]"
        active = self.session.constraints().describe()
        if active:
            head += "\n" + ", ".join(active)

        out = f"{head}\n\n{render_ranked(self.shown, breakdown=self.breakdown)}"
        for clause in self.session.last_adopted:
            out += f"\n\n+ standing rule: {clause}   (/forget to drop it)"
        if not directive.recognised and directive.constraints:
            out += "\n\n(that read as a topic, not an instruction)"
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
            return f"generating with {engine.describe()}{note}", True

        if cmd == "profile":
            where = self.profile.path or "(not saved to disk)"
            return (
                f"stored at {where}\n"
                f"{self.profile.confidence()}, "
                f"{len(self.profile.standing.describe())} standing rule(s), "
                f"{len(self.profile.examples)} example(s), "
                f"{self.profile.batches} batch(es)",
                True,
            )

        if cmd == "pool":
            self.session.pool = max(1, int(arg))
            return f"pool -> {self.session.pool}", True

        if cmd == "top":
            self.top = max(0, int(arg))
            return f"showing -> {'all' if self.top == 0 else self.top}", True

        if cmd == "breakdown":
            self.breakdown = not self.breakdown
            return f"breakdown -> {'on' if self.breakdown else 'off'}", True

        if cmd == "clear":
            self.session.clear()
            return "this turn's adjustments dropped (standing rules kept)", True

        if cmd == "score":
            if not arg:
                return "usage: /score some text to score", True
            return render_score(arg, score_post(arg, self.profile.weights)), True

        if cmd == "keep":
            index = self._pick(arg, self.shown, "batch")
            item = self.shown[index]
            if any(k.text == item.text for k in self.kept):
                return "already kept", True

            self.kept.append(item)
            self.profile.remember(item.text)
            # Keeping is the training signal: it says this one beat the others shown.
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
            if not self.last_message:
                return "nothing to re-run yet", True
            return self._turn(self.last_message), True

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
            return f"error: {exc.args[0] if exc.args else exc}", True
        except OSError as exc:
            return f"error: {exc}", True
