"""`python -m bot` -- build a post, and publish it only if asked."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import captions, images, run, x
from .persona import Persona, Refused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bot", description="Post an image and a caption as a disclosed AI persona."
    )
    parser.add_argument("--persona", default="persona.json", help="persona file")
    parser.add_argument("--post", action="store_true", help="actually publish")
    parser.add_argument("--no-image", action="store_true", help="skip image generation")
    parser.add_argument("--out", metavar="PATH", help="save the image to a file")
    parser.add_argument("--seed", type=int, default=None, help="fix the draw (default: today)")
    parser.add_argument("--show", action="store_true", help="show every caption considered")
    parser.add_argument("--models", action="store_true",
                        help="list the models this key can reach, then stop")
    args = parser.parse_args(argv)

    if args.models:
        try:
            available = captions.models()
        except captions.CaptionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"{len(available)} models available -- set CAPTION_MODEL to one:")
        for name in available:
            print(f"  {name}")
        return 0

    try:
        persona = Persona.load(args.persona)
    except (OSError, ValueError, Refused) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        post = run.build(persona, seed=args.seed, render=not args.no_image)
    except (captions.CaptionError, images.ImageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"persona  {persona.name}")
    print(f"scene    {post.scene}")
    print(f"topic    {post.topic}")
    print(f"image    {post.source or 'skipped'}"
          + (f" -- {len(post.image):,} bytes" if post.image else ""))
    print(f"alt      {post.alt}")
    print()
    print(f"caption  {post.caption}   ({len(post.caption)}/280)")

    if args.show:
        print("\nconsidered:")
        for candidate in post.considered:
            mark = "  " if captions.usable(candidate) else "x "
            print(f"  {mark}{candidate}")

    if args.out and post.image:
        Path(args.out).write_bytes(post.image)
        print(f"\nwrote {args.out}")

    if not args.post:
        print("\ndry run -- nothing posted. Add --post to publish.")
        return 0

    try:
        tweet_id = run.publish(post)
    except x.XError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    run.record(post, tweet_id)
    print(f"\nposted: https://x.com/i/status/{tweet_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
