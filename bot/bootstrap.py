"""Orchestration for the bootstrap run: generate, gate, curate.

The GPU work is injected as a callable, so everything around it -- resume after
a timed-out session, QC wiring, curation, the report -- is ordinary tested code
rather than something that only runs on Kaggle. `notebooks/bootstrap.py` builds
a real diffusers pipeline and hands it in; the tests hand in a stub.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from bot.qc import Embedder, Judgement, QCGate, curate, summarise

# Given one shot dict and a destination, write an image there.
GenerateFn = Callable[[dict, Path], None]

# Pinned so a rerun months later reproduces this one. The trainer script lives
# in the diffusers examples; writing our own training loop would be strictly
# worse than the maintained one.
DIFFUSERS_VERSION = "0.31.0"
TRAINER = "examples/dreambooth/train_dreambooth_lora_sdxl.py"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# SDXL's stock VAE produces NaNs in fp16; this is the standard replacement.
FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"


def train_command(
    train_dir: Path,
    output_dir: Path,
    *,
    instance_token: str,
    trainer: str = TRAINER,
    steps: int = 1200,
    rank: int = 16,
    learning_rate: float = 1e-4,
    resolution: int = 1024,
    seed: int = 0,
) -> list[str]:
    """Build the accelerate invocation for a character LoRA.

    Defaults are sized for a free 16GB Kaggle GPU: fp16, gradient checkpointing
    and 8-bit Adam are what keep SDXL training inside that budget. `rank` 16 is
    the usual range for a character (identity, not style); raise it only if the
    likeness is still soft after a full run.
    """
    if not instance_token.strip():
        raise ValueError("instance_token must be a rare word the model has no prior for")
    return [
        "accelerate", "launch", trainer,
        f"--pretrained_model_name_or_path={BASE_MODEL}",
        f"--pretrained_vae_model_name_or_path={FP16_VAE}",
        f"--instance_data_dir={train_dir}",
        f"--output_dir={output_dir}",
        f"--instance_prompt=a picture of {instance_token}",
        f"--resolution={resolution}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--mixed_precision=fp16",
        f"--learning_rate={learning_rate}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--rank={rank}",
        f"--max_train_steps={steps}",
        "--checkpointing_steps=400",
        f"--seed={seed}",
    ]


@dataclass(frozen=True)
class BatchResult:
    generated: tuple[Path, ...]
    skipped: tuple[Path, ...]
    failed: tuple[tuple[int, str], ...]

    @property
    def images(self) -> tuple[Path, ...]:
        return tuple(sorted(set(self.generated) | set(self.skipped)))

    def summary(self) -> str:
        parts = [f"{len(self.generated)} generated"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} already present")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def shot_path(out_dir: Path, shot: dict) -> Path:
    return Path(out_dir) / f"shot-{int(shot['index']):04d}.png"


def run_batch(
    shots: Sequence[dict],
    generate: GenerateFn,
    out_dir: Path,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Generate every shot, skipping any already on disk.

    Kaggle sessions time out mid-run, so an existing file is treated as done.
    A single failing shot does not abandon the batch -- an hour of GPU time is
    too expensive to throw away over one bad prompt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    skipped: list[Path] = []
    failed: list[tuple[int, str]] = []

    for n, shot in enumerate(shots, 1):
        target = shot_path(out_dir, shot)
        if target.exists() and target.stat().st_size > 0:
            skipped.append(target)
        else:
            try:
                generate(shot, target)
            except Exception as exc:  # noqa: BLE001 - one bad shot must not end the run
                failed.append((int(shot["index"]), str(exc)))
                continue
            if not target.exists():
                failed.append((int(shot["index"]), "generator wrote no file"))
                continue
            generated.append(target)
        if on_progress:
            on_progress(n, len(shots))

    return BatchResult(tuple(generated), tuple(skipped), tuple(failed))


def gate_images(
    images: Iterable[Path],
    anchor: Path,
    embedder: Embedder,
    threshold: float,
) -> tuple[list[Judgement], dict[Path, list[float]]]:
    """Score every image against the anchor. Returns verdicts and embeddings."""
    gate = QCGate(embedder.embed(anchor), threshold)
    vectors = {image: embedder.embed(image) for image in images}
    return [gate.judge(img, vec) for img, vec in vectors.items()], vectors


def assemble_training_set(
    judgements: Sequence[Judgement],
    vectors: dict,
    train_dir: Path,
    *,
    keep: int,
    max_similarity: float = 0.98,
) -> list[Judgement]:
    """Copy the curated picks into a clean directory for the trainer."""
    picked = curate(judgements, vectors, keep=keep, max_similarity=max_similarity)
    train_dir = Path(train_dir)
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True)
    for judgement in picked:
        shutil.copy(judgement.image, train_dir / judgement.image.name)
    return picked


def write_report(judgements: Sequence[Judgement], picked: Sequence[Judgement], path: Path) -> None:
    chosen = {j.image for j in picked}
    Path(path).write_text(json.dumps({
        "summary": summarise(judgements),
        "kept": len(picked),
        "images": [
            {"image": str(j.image), "score": round(j.score, 4),
             "passed": j.passed, "trained_on": j.image in chosen}
            for j in sorted(judgements, key=lambda j: j.score, reverse=True)
        ],
    }, indent=2), encoding="utf-8")
