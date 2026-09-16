"""Quality gate: does a generated image still look like the persona?

Nothing here imports torch. The embedding backend is injected, so the gate's
logic -- thresholding, ranking, diversity -- is testable on a laptop with no ML
stack, and the heavy backend only loads inside the notebook that needs it.

For an illustrated persona the embedder must be CLIP, not a face-recognition
model: ArcFace/InsightFace are trained on photographs and frequently fail to
detect an anime face at all, let alone score it. See DESIGN.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

Embedding = Sequence[float]


class Embedder(Protocol):
    """Maps an image to a vector. CLIP for illustration, a face model for photo."""

    def embed(self, image: Path) -> Embedding: ...


def cosine(a: Embedding, b: Embedding) -> float:
    """Cosine similarity, pure stdlib so this module stays dependency-free."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cannot compare a zero vector")
    return dot / (na * nb)


@dataclass(frozen=True)
class Judgement:
    """One image's verdict, keeping the score so thresholds can be retuned."""

    image: Path
    score: float
    passed: bool
    reason: str

    def __str__(self) -> str:
        mark = "pass" if self.passed else "REJECT"
        return f"{mark} {self.score:.3f}  {self.image.name}  ({self.reason})"


class QCGate:
    """Scores candidates against the anchor and rejects the off-model ones."""

    def __init__(self, anchor: Embedding, threshold: float) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        if not anchor:
            raise ValueError("anchor embedding is empty")
        self.anchor = anchor
        self.threshold = threshold

    def judge(self, image: Path, embedding: Embedding) -> Judgement:
        score = cosine(self.anchor, embedding)
        passed = score >= self.threshold
        reason = (
            f"at or above threshold {self.threshold:.2f}"
            if passed
            else f"below threshold {self.threshold:.2f}"
        )
        return Judgement(image=image, score=score, passed=passed, reason=reason)

    def judge_all(self, images: Iterable[Path], embedder: Embedder) -> list[Judgement]:
        return [self.judge(img, embedder.embed(img)) for img in images]


def curate(
    judgements: Iterable[Judgement],
    embeddings: dict[Path, Embedding],
    *,
    keep: int,
    max_similarity: float = 0.98,
) -> list[Judgement]:
    """Pick the training set: on-model, but not N copies of the same picture.

    Greedy by anchor score, skipping any candidate too close to one already
    chosen. A LoRA trained on near-duplicates overfits to that one pose.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    ranked = sorted((j for j in judgements if j.passed), key=lambda j: j.score, reverse=True)
    chosen: list[Judgement] = []
    for candidate in ranked:
        if len(chosen) >= keep:
            break
        vec = embeddings[candidate.image]
        if any(cosine(vec, embeddings[c.image]) > max_similarity for c in chosen):
            continue
        chosen.append(candidate)
    return chosen


def summarise(judgements: Sequence[Judgement]) -> str:
    if not judgements:
        return "no candidates"
    passed = [j for j in judgements if j.passed]
    scores = [j.score for j in judgements]
    return (
        f"{len(passed)}/{len(judgements)} passed  "
        f"(best {max(scores):.3f}, worst {min(scores):.3f}, "
        f"mean {sum(scores) / len(scores):.3f})"
    )


class ClipEmbedder:
    """Real backend. Imports torch/transformers lazily so bot.qc stays light."""

    def __init__(self, model: str = "openai/clip-vit-large-patch14") -> None:
        from PIL import Image  # noqa: F401  (validated at construction, used in embed)
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._model = CLIPModel.from_pretrained(model).eval()
        self._processor = CLIPProcessor.from_pretrained(model)

    def embed(self, image: Path) -> Embedding:
        from PIL import Image

        with Image.open(image) as img:
            inputs = self._processor(images=img.convert("RGB"), return_tensors="pt")
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return features[0].tolist()
