"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from . import personas
from .rank import rank
from .render import render_ranked, render_score
from .score import score_post


def _backend(args: argparse.Namespace):
    from .llm import build, detect

    if args.backend:
        return build(args.backend, host=args.host, model=args.model, gguf=args.gguf)
    return detect()


def cmd_gen(args: argparse.Namespace) -> int:
    from .generate import propose
    from .llm import BackendError

    try:
        engine = _backend(args)
        candidates = propose(
            args.topic, persona=args.persona, n=args.n, backend=engine,
            pool=args.pool, seed=args.seed,
        )
    except (KeyError, ValueError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not candidates:
        print("error: nothing was generated for that topic", file=sys.stderr)
        return 1

    ranked = rank(candidates)
    top = ranked[: args.top] if args.top > 0 else ranked

    print(
        f"{len(candidates)} candidates, showing top {len(top)}  "
        f"[{args.persona} | {engine.describe()}]"
    )
    print()
    print(render_ranked(top, breakdown=args.breakdown))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    texts = args.text or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        print("error: no text given (pass arguments or pipe lines on stdin)", file=sys.stderr)
        return 2

    if len(texts) == 1:
        print(render_score(texts[0], score_post(texts[0])))
        return 0

    print(render_ranked(rank(texts), breakdown=args.breakdown))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .chat import Repl, Session
    from .profile import Profile

    path = Path(args.profile).expanduser() if args.profile else None
    # --fresh still writes; it starts from nothing rather than refusing to learn.
    profile = Profile(path=path) if args.fresh else Profile.load(path)
    if args.persona:
        profile.persona = args.persona

    repl = Repl(
        session=Session(
            profile=profile, backend=_backend(args), n=args.n,
            pool=args.pool, seed=args.seed,
        ),
        top=args.top,
        breakdown=args.breakdown,
    )

    known = f" -- {profile.confidence()}"
    rules = len(profile.standing.describe())
    if rules:
        known += f", {rules} standing rule(s)"
    print(f"thirsttrap [{profile.persona}]{known}")
    print(f"generating with {repl.session.engine().describe()}")
    print("/help for commands, /quit to leave")
    if args.topic:
        output, _ = repl.handle(" ".join(args.topic))
        print(output)

    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        output, keep_going = repl.handle(line)
        if output:
            print(output)
        if not keep_going:
            return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .profile import Profile
    from .serve import serve

    profile = Profile(path=None) if args.fresh else Profile.load()
    if args.persona:
        profile.persona = args.persona
    try:
        serve(
            host=args.host_bind, port=args.port, backend=_backend(args),
            profile=profile, n=args.n, top=args.top, open_browser=not args.no_browser,
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_personas(_: argparse.Namespace) -> int:
    for name in personas.names():
        print(f"{name:<14} {personas.PERSONAS[name].summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thirsttrap",
        description=(
            "Talk to a local model about what happened; it drafts posts and ranks "
            "them by a prior that learns from what you keep."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="one-shot: draft from a topic, no conversation")
    gen.add_argument("topic", help="what the posts should be about")
    gen.add_argument(
        "--pool", type=int, default=400, help="candidates to draw before ranking (default 400)"
    )
    gen.add_argument("--seed", type=int, default=None, help="fix the draw for a reproducible batch")
    gen.add_argument("-n", type=int, default=12, help="candidates to ask the model for")
    gen.add_argument("--backend", help="ollama, openai-compat, llama-cpp or grammar")
    gen.add_argument("--model", help="model name the local server should load")
    gen.add_argument("--host", help="base URL of the local server")
    gen.add_argument("--gguf", help="path to a .gguf for the in-process backend")

    gen.add_argument("-k", "--top", type=int, default=3, help="candidates to show (0 = all)")
    gen.add_argument(
        "-p", "--persona", default=personas.DEFAULT_PERSONA, choices=personas.names(),
        help=f"voice preset (default {personas.DEFAULT_PERSONA})",
    )
    gen.add_argument("-b", "--breakdown", action="store_true", help="show component scores")
    gen.set_defaults(func=cmd_gen)

    sc = sub.add_parser("score", help="score text you wrote yourself (no model needed)")
    sc.add_argument("text", nargs="*", help="one or more posts; omit to read lines from stdin")
    sc.add_argument("-b", "--breakdown", action="store_true", help="show component scores")
    sc.set_defaults(func=cmd_score)

    ch = sub.add_parser("chat", help="converse in the terminal (needs a local model)")
    ch.add_argument("topic", nargs="*", help="optional opening line")
    ch.add_argument(
        "--pool", type=int, default=400, help="candidates to draw before ranking (default 400)"
    )
    ch.add_argument("--seed", type=int, default=None, help="fix the draw for a reproducible session")
    ch.add_argument("-n", type=int, default=12, help="candidates to ask the model for per turn")
    ch.add_argument("--backend", help="ollama, openai-compat, llama-cpp or grammar")
    ch.add_argument("--model", help="model name the local server should load")
    ch.add_argument("--host", help="base URL of the local server")
    ch.add_argument("--gguf", help="path to a .gguf for the in-process backend")

    ch.add_argument("-k", "--top", type=int, default=3, help="candidates to show (0 = all)")
    ch.add_argument(
        "-p", "--persona", default=None, choices=personas.names(),
        help="voice preset (default: whatever the profile remembers)",
    )
    ch.add_argument("-b", "--breakdown", action="store_true", help="show component scores")
    ch.add_argument("--profile", metavar="PATH", help="use a specific profile file")
    ch.add_argument(
        "--fresh", action="store_true",
        help="start from nothing instead of loading the saved profile",
    )
    ch.set_defaults(func=cmd_chat)

    sv = sub.add_parser("serve", help="converse in your browser (needs a local model)")
    sv.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    sv.add_argument(
        "--host-bind", default="127.0.0.1",
        help="interface to bind (default 127.0.0.1 -- this endpoint is unauthenticated)",
    )
    sv.add_argument("--no-browser", action="store_true", help="do not open a browser")
    sv.add_argument("-n", type=int, default=12, help="candidates per batch")
    sv.add_argument("-k", "--top", type=int, default=5, help="candidates to show")
    sv.add_argument(
        "-p", "--persona", default=None, choices=personas.names(),
        help="voice preset (default: whatever the profile remembers)",
    )
    sv.add_argument("--fresh", action="store_true", help="ignore the saved profile")
    sv.add_argument("--backend", help="ollama, openai-compat, llama-cpp or grammar")
    sv.add_argument("--model", help="model name the local server should load")
    sv.add_argument("--host", help="base URL of the local model server")
    sv.add_argument("--gguf", help="path to a .gguf for the in-process backend")
    sv.set_defaults(func=cmd_serve)

    pl = sub.add_parser("personas", help="list the voice presets")
    pl.set_defaults(func=cmd_personas)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
