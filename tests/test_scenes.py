import json
import random

import pytest

from bot.persona import Persona, Scene
from bot.scenes import (
    MIRROR_GRAMMAR,
    NEGATIVE,
    SELFIE_GRAMMAR,
    as_manifest,
    as_paste_list,
    build_prompt,
    negative_prompt,
    plan_shoot,
)


@pytest.fixture
def persona():
    return Persona.load("persona.json")


def test_prompt_mentions_every_scene_axis(persona):
    scene = Scene("a green jacket", "a laundromat", "golden hour", "mid-laugh")
    prompt = build_prompt(persona, scene)
    for part in ("a green jacket", "a laundromat", "golden hour", "mid-laugh"):
        assert part in prompt


def test_prompt_carries_selfie_grammar(persona):
    prompt = build_prompt(persona, Scene("a tee", "a diner", "dusk", "half-asleep"))
    assert "front-facing phone camera selfie" in prompt


def test_subject_override_replaces_the_default_description(persona):
    scene = Scene("a tee", "a diner", "dusk", "half-asleep")
    prompt = build_prompt(persona, scene, subject="a woman with cropped dark hair, 20s")
    assert "cropped dark hair" in prompt
    assert persona.identity.name not in prompt


def test_mirror_shots_use_mirror_grammar_not_arms_length(persona):
    """A mirror shot is rear-camera; the two grammars contradict each other."""
    prompt = build_prompt(persona, Scene("a tee", "a bedroom", "dusk", "mirror shot"))
    assert "mirror selfie" in prompt
    assert "front-facing phone camera selfie" not in prompt
    assert "arm's length" not in prompt


def test_non_mirror_shots_do_not_get_mirror_grammar(persona):
    prompt = build_prompt(persona, Scene("a tee", "a bedroom", "dusk", "mid-laugh"))
    assert "mirror selfie" not in prompt
    assert "front-facing phone camera selfie" in prompt


@pytest.mark.parametrize("marker", ["dusk", "dawn", "morning", "afternoon", "night", "evening"])
def test_locations_carry_no_time_of_day(persona, marker):
    """Regression: 'rooftop at dusk' collided with the time axis, producing
    prompts like 'rooftop at dusk, early morning'."""
    for location in persona.visual.locations:
        assert marker not in location.lower(), f"{location!r} encodes a time of day"


def test_negative_prompt_targets_the_usual_tells():
    neg = negative_prompt()
    for tell in ("studio lighting", "extra fingers", "watermark", "multiple people"):
        assert tell in neg


def test_plan_shoot_returns_requested_count(persona):
    assert len(plan_shoot(persona, 7)) == 7


def test_plan_shoot_rejects_nonsense_counts(persona):
    with pytest.raises(ValueError, match="at least 1"):
        plan_shoot(persona, 0)


def test_plan_shoot_avoids_repeats_within_the_window(persona):
    shots = plan_shoot(persona, 12, rng=random.Random(3))
    keys = [(s.wardrobe, s.location, s.time, s.activity) for s in shots]
    assert len(set(keys)) == len(keys)


def test_plan_shoot_is_reproducible_with_a_seed(persona):
    a = plan_shoot(persona, 5, rng=random.Random(42))
    b = plan_shoot(persona, 5, rng=random.Random(42))
    assert [s.as_dict() for s in a] == [s.as_dict() for s in b]


def test_plan_shoot_indexes_sequentially(persona):
    assert [s.index for s in plan_shoot(persona, 4)] == [0, 1, 2, 3]


def test_shots_carry_a_seed_for_reproducing_a_good_one(persona):
    for shot in plan_shoot(persona, 3):
        assert isinstance(shot.seed, int) and shot.seed >= 0


def test_manifest_is_valid_json_with_full_provenance(persona):
    shots = plan_shoot(persona, 3, rng=random.Random(1))
    parsed = json.loads(as_manifest(shots))
    assert len(parsed) == 3
    for entry in parsed:
        assert set(entry) == {
            "index", "prompt", "negative", "seed",
            "wardrobe", "location", "time", "activity",
        }


def test_paste_list_contains_one_block_per_shot(persona):
    text = as_paste_list(plan_shoot(persona, 4, rng=random.Random(2)))
    assert text.count("negative:") == 4
    assert text.count("seed:") == 4


def test_grammars_are_disjoint_on_the_camera_facing_claim():
    assert "arm's length" in SELFIE_GRAMMAR
    assert "arm's length" not in MIRROR_GRAMMAR
    assert not (set(NEGATIVE) & set(SELFIE_GRAMMAR))
