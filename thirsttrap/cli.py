"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from . import personas
from .rank import Ranked, rank
from .score import WEIGHTS, score_post

BAR_WIDTH = 24


def _bar(fraction: float) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * BAR_WIDTH)
    return "#" * filled + "." * (BAR_WIDTH - filled)


def _print_ranked(items: list[Ranked], show_breakdown: bool) -> None:
    for i, item in enumerate(items, start=1):
        flag = "  [near-duplicate]" if item.duplicate_of else ""
        print(f"\n{i}. {item.final:5.1f}{flag}")
        print(f"   {item.text}")

        if show_breakdown:
            print(f"   {len(item.text)} chars")
            for name in sorted(item.score.components, key=lambda k: -WEIGHTS[k]):
                value = item.score.components[name]
                print(f"     {name:<15} {_bar(value)} {value:.2f}  (w {WEIGHTS[name]:.2f})")
        for note in item.score.notes:
            print(f"     ! {note}")


def cmd_gen(args: argparse.Namespace) -> int:
    from .generate import generate

    try:
        candidates = generate(args.topic, persona=args.persona, n=args.n)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    except Exception as exc:  # network, auth, refusal -- all user-actionable
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ranked = rank([c.text for c in candidates])
    top = ranked[: args.top] if args.top > 0 else ranked

    print(f"{len(candidates)} candidates, showing top {len(top)}  [{args.persona}]")
    _print_ranked(top, show_breakdown=args.breakdown)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    texts = args.text or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        print("error: no text given (pass arguments or pipe lines on stdin)", file=sys.stderr)
        return 2

    if len(texts) == 1:
        score = score_post(texts[0])
        print(f"{score.total:.1f}   {len(texts[0])} chars")
        for name in sorted(score.components, key=lambda k: -WEIGHTS[k]):
            value = score.components[name]
            print(f"  {name:<15} {_bar(value)} {value:.2f}  (w {WEIGHTS[name]:.2f})")
        for note in score.notes:
            print(f"  ! {note}")
        return 0

    _print_ranked(rank(texts), show_breakdown=args.breakdown)
    return 0


def cmd_personas(_: argparse.Namespace) -> int:
    for name in personas.names():
        print(f"{name:<14} {personas.PERSONAS[name].summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thirsttrap",
        description="Generate candidate posts for X and rank them by a stated prior.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="generate candidates and rank them (needs API access)")
    gen.add_argument("topic", help="what the posts should be about")
    gen.add_argument("-n", type=int, default=12, help="candidates to generate (default 12)")
    gen.add_argument("-k", "--top", type=int, default=3, help="candidates to show (0 = all)")
    gen.add_argument(
        "-p", "--persona", default=personas.DEFAULT_PERSONA, choices=personas.names(),
        help=f"voice preset (default {personas.DEFAULT_PERSONA})",
    )
    gen.add_argument("-b", "--breakdown", action="store_true", help="show component scores")
    gen.set_defaults(func=cmd_gen)

    sc = sub.add_parser("score", help="score text you already have (no API access needed)")
    sc.add_argument("text", nargs="*", help="one or more posts; omit to read lines from stdin")
    sc.add_argument("-b", "--breakdown", action="store_true", help="show component scores")
    sc.set_defaults(func=cmd_score)

    pl = sub.add_parser("personas", help="list the voice presets")
    pl.set_defaults(func=cmd_personas)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
