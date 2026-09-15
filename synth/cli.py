"""Command line for building and checking a character before it goes anywhere.

    python3 -m synth validate personas/example.yaml
    python3 -m synth prompt   personas/example.yaml
    python3 -m synth draft    personas/example.yaml --idea "a playlist"
    python3 -m synth canon    personas/example.yaml --add "Vee likes rain"
    python3 -m synth new      --name Sophie --handle @sophie --brief "..."
    python3 -m synth roster   personas/
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from .builder import draft, scaffold, to_yaml
from .canon import Canon
from .content import Queue, check, generate
from .disclosure import asks_whether_real
from .models import ModelError, from_env
from .persona import PersonaError, from_dict, load
from .roster import load_all, report
from .platforms import DryRun


def _canon_for(path: pathlib.Path, persona) -> Canon:
    """Canon lives beside the persona file, seeded from it on first use."""
    c = Canon(path.with_suffix(".canon.jsonl"))
    if not c.facts and persona.canon_seed:
        for fact in persona.canon_seed:
            c.add(fact, topic="seed", source=str(path))
    return c


def _echo_completer(_prompt: str) -> str:
    """Stand-in for a model, so every command runs with nothing configured.

    It returns something deliberately unusable, because a placeholder that
    produced plausible posts would get mistaken for the real thing.
    """
    return "[no model configured - set SYNTH_MODEL or pass --model] #AI"


def _completer(model_name):
    """(completer, is_real). The flag matters.

    The placeholder passes every content gate - it is short, contradicts no
    canon and claims to be nobody - so without this it gets stamped APPROVED
    and queued for publication. The gates are SAFETY gates, not quality gates,
    and asking them to recognise a placeholder would be asking the wrong
    component. The caller knows whether a model was configured; it should not
    queue work produced by nothing.
    """
    try:
        c = from_env(model_name)
    except Exception as e:                        # bad env, not worth a crash
        print(f"  model config ignored: {e}", file=sys.stderr)
        c = None
    return (c, True) if c else (_echo_completer, False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="synth", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # `new` and `roster` do not take a persona FILE, so they are built first
    # and the loop below skips them.
    n = sub.add_parser("new", help="draft a new character from a brief")
    n.add_argument("--name", required=True)
    n.add_argument("--handle", required=True)
    n.add_argument("--brief", default="", help="one line: who she is")
    n.add_argument("--out", help="write the spec here instead of stdout")
    n.add_argument("--force", action="store_true",
                   help="overwrite an existing spec file")
    n.add_argument("--model", help="override SYNTH_MODEL")

    r = sub.add_parser("roster", help="check a directory of characters")
    r.add_argument("directory", nargs="?", default="personas")
    r.add_argument("--threshold", type=float, default=0.25)

    for name in ("validate", "prompt", "bio", "draft", "canon"):
        s = sub.add_parser(name)
        s.add_argument("persona")
        if name == "draft":
            s.add_argument("--idea", required=True)
            s.add_argument("--model", help="override SYNTH_MODEL")
            s.add_argument("--text", help="check this text instead of generating")
            s.add_argument("--publish", action="store_true",
                           help="send an approved draft to the dry-run adapter")
        if name == "canon":
            s.add_argument("--add")
            s.add_argument("--topic", default="general")
            s.add_argument("--check", help="flag facts this text may contradict")
    args = p.parse_args(argv)

    if args.cmd == "roster":
        personas, broken = load_all(args.directory)
        for b in broken:
            print(f"  WILL NOT LOAD  {b}", file=sys.stderr)
        if not personas:
            print(f"no valid personas in {args.directory}", file=sys.stderr)
            return 2
        print(report(personas, threshold=args.threshold))
        return 1 if broken else 0

    if args.cmd == "new":
        try:
            persona = draft(args.name, args.handle, args.brief,
                            _completer(args.model)[0])
        except (PersonaError, ModelError):
            # No model configured, so fall back to the empty scaffold rather
            # than inventing a character nobody wrote.
            persona = from_dict(scaffold(args.name, args.handle, args.brief))
            print("  no model configured - writing a SCAFFOLD. The "
                  "structure is right and the character is missing; fill "
                  "in voice, interests and canon by hand, or wire a "
                  "Completer.", file=sys.stderr)
        text = to_yaml(persona)
        if args.out:
            out = pathlib.Path(args.out)
            if out.exists() and not args.force:
                print(f"{out} exists - pass --force to overwrite",
                      file=sys.stderr)
                return 2
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote {out}")
        else:
            print(text)
        return 0

    path = pathlib.Path(args.persona)
    try:
        persona = load(path)
    except PersonaError as e:
        print(f"INVALID persona: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"no such persona file: {path}", file=sys.stderr)
        return 2

    if args.cmd == "validate":
        canon = _canon_for(path, persona)
        print(f"OK  {persona.name} ({persona.handle})")
        print(f"    {len(persona.boundaries)} boundaries "
              f"({len(persona.extra_boundaries)} from the spec, the rest "
              f"non-removable)")
        print(f"    marker {persona.disclosure.marker!r} "
              f"| {persona.voice.max_chars} chars "
              f"| {len(canon.live_facts())} canon facts")
        print(f"    answers the realness question: "
              f"{asks_whether_real('are you real?')}")
        return 0

    if args.cmd == "prompt":
        print(persona.system_prompt())
        return 0

    if args.cmd == "bio":
        print(persona.bio())
        return 0

    canon = _canon_for(path, persona)

    if args.cmd == "canon":
        if args.add:
            f = canon.add(args.add, topic=args.topic, source="cli")
            print(f"added [{f.topic}] {f.text}")
        elif args.check:
            hits = canon.conflicts(args.check)
            if hits:
                print("MAY CONTRADICT:")
                for f in hits:
                    print(f"  - {f.text}")
            else:
                print("no conflict found (a clean pass is not a guarantee)")
        else:
            for f in canon.live_facts():
                print(f"[{f.topic}] {f.text}")
        return 0

    # draft
    if args.text:
        d = check(args.text, persona, canon)
        d.idea = args.idea
    else:
        completer, real = _completer(args.model)
        try:
            d = generate(persona, canon, args.idea, completer)
        except ModelError as e:
            print(f"model error: {e}", file=sys.stderr)
            return 3
        if not real:
            print(f"idea: {d.idea}")
            print(f"text: {d.text}")
            print("NOT A DRAFT - no model is configured, so nothing was "
                  "written and nothing\nwas queued. Set SYNTH_MODEL (and "
                  "SYNTH_BACKEND/SYNTH_BASE_URL) or pass --model.")
            return 2
    print(f"idea: {d.idea}")
    print(f"text: {d.text}")
    print(f"post: {d.render(persona)}")
    if d.flagged_facts:
        print("flagged facts:")
        for f in d.flagged_facts:
            print(f"  - {f}")
    if d.approved:
        print("APPROVED by the automatic gates (a human still releases it)")
        if args.publish:
            print(f"published: {DryRun().publish(d.render(persona), persona)}")
        Queue(path.with_suffix(".queue.jsonl")).add(d)
    else:
        print("BLOCKED:")
        for b in d.blockers:
            print(f"  - {b}")
    return 0 if d.approved else 1
