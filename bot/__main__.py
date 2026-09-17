"""One front door for the character studio.

    python -m bot                 chat with her and tune her voice
    python -m bot generate -n 20  render her pictures on this GPU
    python -m bot curate          score them and pick a LoRA training set
    python -m bot train           train her LoRA
    python -m bot prompts -n 20   just the prompts, if you generate elsewhere
    python -m bot caption DIR     caption a folder of images into the timeline
    python -m bot direct "..."    tell the writer what is wrong with the captions
    python -m bot timeline        print what has been captioned so far
    python -m bot check           validate persona.json
    python -m bot doctor          what this machine can run

`caption` is the step that joins the halves: it reads images you generated,
writes captions in her voice, and stores them so the timeline and the chat
window both see them. Repetition scoring runs against everything already
captioned, so running it twice does not produce two versions of one joke.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import json

from bot.captions import CaptionError, Captioner, direct
from bot.driver import LocalDriver
from bot.persona import Persona, PersonaError
from bot.scenes import as_manifest, as_paste_list, plan_shoot

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _persona(args) -> Persona:
    try:
        return Persona.load(args.persona)
    except PersonaError as exc:
        sys.exit(f"persona: {exc}")


def cmd_chat(args) -> None:
    from bot.ui import serve

    serve(args.host, args.port, args.db, args.persona)


def cmd_prompts(args) -> None:
    persona = _persona(args)
    rng = random.Random(args.seed) if args.seed is not None else None
    shots = plan_shoot(persona, args.count, subject=args.subject, rng=rng)
    print(as_manifest(shots) if args.json else as_paste_list(shots))


def _generator(args, persona, lora=None):
    from bot.bootstrap import FAMILIES, family_for_vram
    from bot.doctor import diagnose

    if args.family:
        family = FAMILIES[args.family]
    else:
        report = diagnose()
        family = family_for_vram(report.vram_mb)
        print(f"{report.gpu or 'no GPU'}: using {family.name} at {family.resolution}px")

    try:
        from bot.images import Generator
    except ImportError as exc:
        sys.exit(f"image generation needs torch and diffusers ({exc}).\n"
                 "  run: python -m bot doctor")
    return Generator(persona, family=family, scale=args.scale, lora=lora)


def cmd_generate(args) -> None:
    """Render the shot list onto disk, resuming anything already there."""
    from bot.bootstrap import run_batch

    persona = _persona(args)
    rng = random.Random(args.seed) if args.seed is not None else None
    shots = [s.as_dict() for s in plan_shoot(persona, args.count, rng=rng)]
    gen = _generator(args, persona)
    out = Path(args.out)

    if args.sweep:
        for path in gen.sweep(shots[0], out):
            print(f"  {path}")
        print("look at those, then rerun with --scale set to the one you liked")
        return

    result = run_batch(shots, gen.generate, out,
                       on_progress=lambda n, total: print(f"  {n}/{total}", flush=True))
    print(result.summary())
    for index, reason in result.failed:
        print(f"  shot {index}: {reason}", file=sys.stderr)


def cmd_curate(args) -> None:
    """Score what was generated against the anchor and pick a training set."""
    from bot.bootstrap import assemble_training_set, gate_images, write_report
    from bot.qc import summarise

    persona = _persona(args)
    images = sorted(p for p in Path(args.out).iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("sweep-"))
    if not images:
        sys.exit(f"nothing to curate in {args.out}")

    try:
        from bot.qc import ClipEmbedder
    except ImportError as exc:
        sys.exit(f"curation needs torch and transformers ({exc})")

    judgements, vectors = gate_images(
        images, persona.visual.anchor_image, ClipEmbedder(), persona.visual.identity_threshold)
    print(summarise(judgements))
    picked = assemble_training_set(judgements, vectors, Path(args.out) / "train", keep=args.keep)
    write_report(judgements, picked, Path(args.out) / "qc-report.json")
    print(f"{len(picked)} curated into {Path(args.out) / 'train'}")
    print("look through that folder before training -- the gate catches off-model, not ugly")


def cmd_train(args) -> None:
    """Train the character LoRA on the curated set."""
    import subprocess

    from bot.bootstrap import FAMILIES, family_for_vram, train_command
    from bot.doctor import diagnose

    train_dir = Path(args.out) / "train"
    images = [p for p in train_dir.iterdir()] if train_dir.exists() else []
    if not images:
        sys.exit(f"no training images in {train_dir} -- run generate and curate first")

    family = FAMILIES[args.family] if args.family else family_for_vram(diagnose().vram_mb)
    trainer = Path(args.trainer).expanduser()
    if not trainer.exists():
        sys.exit(
            f"trainer not found at {trainer}. Clone the examples once:\n"
            f"  git clone --depth 1 --branch v0.31.0 "
            f"https://github.com/huggingface/diffusers ~/diffusers"
        )

    cmd = train_command(train_dir, Path(args.lora_out), instance_token=args.token,
                        family=family, trainer=str(trainer), steps=args.steps, rank=args.rank)
    print(f"{len(images)} images, ~{args.steps // max(len(images), 1)} steps each")
    print(" \\\n  ".join(cmd))
    subprocess.run(cmd, check=True)

    weights = list(Path(args.lora_out).glob("*.safetensors"))
    if not weights:
        sys.exit(f"training produced no weights in {args.lora_out}")
    print(f"trained: {weights[0]} ({weights[0].stat().st_size / 1e6:.1f} MB)")
    print(f"point visual.lora_path at it, then: python -m bot generate --lora {weights[0]}")


def cmd_caption(args) -> None:
    persona = _persona(args)
    images = sorted(
        p for p in Path(args.directory).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        sys.exit(f"no images in {args.directory}")

    try:
        from bot.captions import AnthropicLLM

        llm = AnthropicLLM(args.model)
    except Exception as exc:
        sys.exit(
            f"captioning needs the Anthropic SDK and a key ({exc}).\n"
            "  pip install anthropic && export ANTHROPIC_API_KEY=..."
        )

    with LocalDriver(args.db) as driver:
        captioner = Captioner(persona, llm)
        history = [p.caption for p in driver.recent_posts(persona.voice.history_window)]
        already = {p.image for p in driver.recent_posts(10_000)}
        written = skipped = failed = 0

        for image in images:
            if str(image) in already:
                skipped += 1
                continue
            try:
                caption, _ = captioner.caption(image, history)
            except CaptionError as exc:
                print(f"  !! {image.name}: {exc}", file=sys.stderr)
                failed += 1
                continue
            driver.post(image, caption)
            history.append(caption)
            written += 1
            print(f"  {image.name}: {caption}")

    summary = f"{written} captioned"
    if skipped:
        summary += f", {skipped} already done"
    if failed:
        summary += f", {failed} failed"
    print(summary)


def cmd_direct(args) -> None:
    """Tell the caption writer what is wrong with the writing."""
    persona = _persona(args)
    try:
        from bot.captions import AnthropicDirector

        director = AnthropicDirector(args.model or None)
    except Exception as exc:
        sys.exit(
            f"directing needs the Anthropic SDK and a key ({exc}).\n"
            "  pip install anthropic && export ANTHROPIC_API_KEY=..."
        )

    with LocalDriver(args.db) as driver:
        samples = [p.caption for p in driver.recent_posts(12)]

    try:
        rules, changed = direct(persona, args.note, director, samples)
    except (ValueError, CaptionError) as exc:
        sys.exit(str(exc))

    raw = json.loads(Path(args.persona).read_text(encoding="utf-8"))
    before = list(raw["voice"]["rules"])
    raw["voice"]["rules"] = rules
    Path(args.persona).write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    print(changed)
    for rule in rules:
        print(f"  {'+' if rule not in before else ' '} {rule}")
    for rule in before:
        if rule not in rules:
            print(f"  - {rule}")


def cmd_timeline(args) -> None:
    with LocalDriver(args.db) as driver:
        posts = driver.timeline(args.limit)
    if not posts:
        print("timeline is empty. run: python -m bot caption <dir>")
        return
    for post in posts:
        print(f"{post.created_at}  {post.caption}")
        print(f"{'':19}  {post.image}")


def cmd_doctor(args) -> None:
    """What this machine can run: GPU, image model, who writes the words."""
    from bot.doctor import diagnose, render

    print(render(diagnose()))


def cmd_check(args) -> None:
    persona = _persona(args)
    v = persona.visual
    print(f"{persona.identity.name} (@{persona.identity.handle})")
    print(f"  discloses    : {persona.identity.disclosure_short}")
    print(f"  style        : {v.style}, identity check via {v.identity_check}")
    print(f"  scenes       : {persona.combinations()} combinations")
    print(f"  voice        : {len(persona.voice.rules)} rules, {len(persona.facts)} facts")
    print(f"  boundaries   : {sum(len(x) for x in (persona.bounds.always_answer_truthfully, persona.bounds.hard_block_outbound, persona.bounds.refuse_to_supply, persona.bounds.deflect, persona.bounds.terminate_thread))} phrases")
    anchor = v.anchor_image
    print(f"  anchor       : {anchor} {'(present)' if anchor.exists() else '(MISSING)'}")
    print("ok")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bot", description="Local character studio.")
    ap.add_argument("--persona", default="persona.json")
    ap.add_argument("--db", default="content/local.db")
    sub = ap.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="chat with her and tune her voice")
    chat.add_argument("--host", default="127.0.0.1")
    chat.add_argument("--port", type=int, default=8000)
    chat.set_defaults(func=cmd_chat)

    prompts = sub.add_parser("prompts", help="emit generation prompts")
    prompts.add_argument("-n", "--count", type=int, default=10)
    prompts.add_argument("--subject", default=None)
    prompts.add_argument("--seed", type=int, default=None)
    prompts.add_argument("--json", action="store_true")
    prompts.set_defaults(func=cmd_prompts)

    def image_args(parser):
        parser.add_argument("--out", default="content/img", help="where the images live")
        parser.add_argument("--family", choices=("sd15", "sdxl"), default=None,
                            help="default: chosen from your VRAM")
        return parser

    generate = image_args(sub.add_parser("generate", help="render her pictures on this GPU"))
    generate.add_argument("-n", "--count", type=int, default=20)
    generate.add_argument("--scale", type=float, default=0.65,
                          help="how strongly the anchor pulls (sweep this first)")
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--sweep", action="store_true",
                          help="render one prompt at four anchor strengths and stop")
    generate.set_defaults(func=cmd_generate)

    curate = image_args(sub.add_parser("curate", help="score images and pick a training set"))
    curate.add_argument("--keep", type=int, default=40)
    curate.set_defaults(func=cmd_curate)

    train = image_args(sub.add_parser("train", help="train her LoRA on the curated set"))
    train.add_argument("--token", default="rmyx", help="a rare word, not her name")
    train.add_argument("--steps", type=int, default=1200)
    train.add_argument("--rank", type=int, default=16)
    train.add_argument("--lora-out", default="content/lora")
    train.add_argument("--trainer",
                       default="~/diffusers/examples/dreambooth/train_dreambooth_lora.py")
    train.set_defaults(func=cmd_train)

    caption = sub.add_parser("caption", help="caption a folder of images")
    caption.add_argument("directory")
    caption.add_argument("--model", default=None)
    caption.set_defaults(func=cmd_caption)

    direct_p = sub.add_parser("direct", help="tell the writer what is wrong with the captions")
    direct_p.add_argument("note", help='e.g. "too whiny"')
    direct_p.add_argument("--model", default=None)
    direct_p.set_defaults(func=cmd_direct)

    timeline = sub.add_parser("timeline", help="print captioned posts")
    timeline.add_argument("--limit", type=int, default=50)
    timeline.set_defaults(func=cmd_timeline)

    doctor = sub.add_parser("doctor", help="what can this machine run?")
    doctor.set_defaults(func=cmd_doctor)

    check = sub.add_parser("check", help="validate persona.json")
    check.set_defaults(func=cmd_check)
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command is None:          # bare `python -m bot` opens the studio
        args = ap.parse_args(["chat"] + (argv or []))
    if args.command == "caption" and args.model is None:
        from bot.captions import DEFAULT_MODEL

        args.model = DEFAULT_MODEL
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
