import math
from pathlib import Path

import pytest

from bot.qc import Judgement, QCGate, cosine, curate, summarise


def vec(*xs):
    return list(map(float, xs))


# -- cosine --------------------------------------------------------------


def test_identical_vectors_score_one():
    assert cosine(vec(1, 2, 3), vec(1, 2, 3)) == pytest.approx(1.0)


def test_scale_does_not_change_similarity():
    assert cosine(vec(1, 2, 3), vec(10, 20, 30)) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine(vec(1, 0), vec(0, 1)) == pytest.approx(0.0)


def test_opposite_vectors_score_minus_one():
    assert cosine(vec(1, 0), vec(-1, 0)) == pytest.approx(-1.0)


def test_dimension_mismatch_is_an_error():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine(vec(1, 2), vec(1, 2, 3))


def test_zero_vector_is_an_error():
    """A zero embedding means the backend failed; don't silently score it 0."""
    with pytest.raises(ValueError, match="zero vector"):
        cosine(vec(0, 0), vec(1, 0))


# -- gate ----------------------------------------------------------------


@pytest.fixture
def gate():
    return QCGate(vec(1, 0, 0), threshold=0.62)


def test_on_model_image_passes(gate):
    j = gate.judge(Path("a.png"), vec(0.99, 0.1, 0))
    assert j.passed and j.score > 0.62


def test_off_model_image_is_rejected(gate):
    j = gate.judge(Path("b.png"), vec(0, 1, 0))
    assert not j.passed
    assert "below threshold" in j.reason


def test_threshold_is_inclusive(gate):
    """Exactly at threshold should pass, not sit in a dead zone."""
    theta = math.acos(0.62)
    j = gate.judge(Path("edge.png"), vec(math.cos(theta), math.sin(theta), 0))
    assert j.score == pytest.approx(0.62)
    assert j.passed


def test_score_is_kept_so_thresholds_can_be_retuned(gate):
    j = gate.judge(Path("a.png"), vec(0.9, 0.4, 0))
    assert 0.0 < j.score < 1.0


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_nonsense_thresholds_are_rejected(bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        QCGate(vec(1, 0), threshold=bad)


def test_empty_anchor_is_rejected():
    with pytest.raises(ValueError, match="anchor embedding is empty"):
        QCGate([], threshold=0.5)


def test_judge_all_uses_the_injected_embedder(gate):
    table = {Path("a.png"): vec(1, 0, 0), Path("b.png"): vec(0, 1, 0)}

    class Fake:
        def embed(self, image):
            return table[image]

    results = gate.judge_all(table, Fake())
    assert [j.passed for j in results] == [True, False]


def test_judgement_renders_readably(gate):
    assert "REJECT" in str(gate.judge(Path("b.png"), vec(0, 1, 0)))
    assert "pass" in str(gate.judge(Path("a.png"), vec(1, 0, 0)))


# -- curation ------------------------------------------------------------


def make(name, score, passed=True):
    return Judgement(Path(name), score, passed, "")


def test_curate_takes_the_highest_scoring_first():
    embeds = {Path("a.png"): vec(1, 0), Path("b.png"): vec(0, 1), Path("c.png"): vec(1, 1)}
    picked = curate([make("a.png", 0.9), make("b.png", 0.7), make("c.png", 0.8)], embeds, keep=2)
    assert [j.image.name for j in picked] == ["a.png", "c.png"]


def test_curate_skips_rejected_candidates():
    embeds = {Path("a.png"): vec(1, 0), Path("b.png"): vec(0, 1)}
    picked = curate([make("a.png", 0.9, passed=False), make("b.png", 0.7)], embeds, keep=5)
    assert [j.image.name for j in picked] == ["b.png"]


def test_curate_drops_near_duplicates():
    """A LoRA trained on near-identical frames overfits to that one pose."""
    embeds = {
        Path("a.png"): vec(1, 0, 0),
        Path("dup.png"): vec(1, 0.001, 0),   # essentially the same image
        Path("c.png"): vec(0, 1, 0),
    }
    picked = curate(
        [make("a.png", 0.95), make("dup.png", 0.94), make("c.png", 0.7)],
        embeds, keep=3, max_similarity=0.98,
    )
    assert [j.image.name for j in picked] == ["a.png", "c.png"]


def test_curate_respects_keep_limit():
    embeds = {Path(f"{i}.png"): vec(1, i) for i in range(10)}
    js = [make(f"{i}.png", 0.9 - i * 0.01) for i in range(10)]
    assert len(curate(js, embeds, keep=4)) == 4


def test_curate_rejects_nonsense_keep():
    with pytest.raises(ValueError, match="at least 1"):
        curate([], {}, keep=0)


def test_curate_on_an_empty_set_returns_empty():
    assert curate([], {}, keep=5) == []


# -- reporting -----------------------------------------------------------


def test_summarise_counts_passes():
    assert summarise([make("a.png", 0.9), make("b.png", 0.2, passed=False)]).startswith("1/2 passed")


def test_summarise_handles_no_candidates():
    assert summarise([]) == "no candidates"
