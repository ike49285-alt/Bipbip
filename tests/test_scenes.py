import json
import random

import pytest

from bot.persona import Persona, Scene
from bot.scenes import (
    ILLUSTRATED,
    PHOTO,
    STYLES,
    as_manifest,
    as_paste_list,
    build_prompt,
    negative_prompt,
    plan_shoot,
    style_for,
)


@pytest.fixture
def persona():
    return Persona.load("persona.json")


# -- composition ---------------------------------------------------------


def test_prompt_mentions_every_scene_axis(persona):
    scene = Scene("a green jacket", "a laundromat", "golden hour", "mid-laugh")
    prompt = build_prompt(persona, scene)
    for part in ("a green jacket", "a laundromat", "golden hour", "mid-laugh"):
        assert part in prompt


def test_prompt_leads_with_the_subject(persona):
    """Subject carries identity, so it goes first."""
    prompt = build_prompt(persona, Scene("a tee", "a diner", "dusk", "half-asleep"))
    assert prompt.startswith(persona.visual.subject)


def test_prompt_always_carries_signature_accessories(persona):
    prompt = build_prompt(persona, Scene("a tee", "a diner", "dusk", "half-asleep"))
    for item in persona.visual.signature:
        assert item in prompt


def test_subject_override_replaces_the_persona_description(persona):
    prompt = build_prompt(
        persona, Scene("a tee", "a diner", "dusk", "half-asleep"), subject="a tall redhead"
    )
    assert prompt.startswith("a tall redhead")
    assert persona.visual.subject not in prompt


# -- style profiles ------------------------------------------------------


def test_illustrated_persona_gets_illustrated_grammar(persona):
    prompt = build_prompt(persona, Scene("a tee", "a diner", "dusk", "mid-laugh"))
    assert "digital illustration" in prompt
    assert "natural available light" not in prompt


def test_illustrated_negative_excludes_photo_realism(persona):
    neg = negative_prompt(persona)
    assert "photorealistic" in neg
    # "airbrushed" fights cel shading; it belongs only to the photo profile.
    assert "airbrushed" not in neg


def test_photo_profile_is_still_intact():
    assert "front-facing phone camera selfie" in PHOTO.grammar
    assert "airbrushed" in PHOTO.negative


def test_unknown_style_is_rejected(persona):
    raw = json.loads(Persona.load().source.read_text())
    raw["visual"]["style"] = "claymation"
    with pytest.raises(ValueError, match="unknown visual.style"):
        style_for(Persona.from_dict(raw))


def test_every_registered_style_is_self_consistent():
    for name, profile in STYLES.items():
        assert profile.name == name
        assert not (set(profile.negative) & set(profile.grammar))
        assert not (set(profile.negative) & set(profile.mirror_grammar))


# -- conflicts caught in review -----------------------------------------


def test_mirror_shots_use_mirror_grammar(persona):
    prompt = build_prompt(persona, Scene("a tee", "cluttered bedroom", "dusk", "mirror shot"))
    assert "mirror selfie" in prompt
    assert "arm's length framing" not in prompt


def test_no_style_asserts_gaze_direction(persona):
    """Regression: 'looking at viewer' contradicted the 'looking off-camera' activity."""
    for profile in STYLES.values():
        for fragment in (*profile.grammar, *profile.mirror_grammar):
            assert "looking at" not in fragment


def test_mirror_shots_only_land_where_a_mirror_could_be(persona):
    """Regression: 'mirror shot / bus window seat' has no mirror in it."""
    rng = random.Random(5)
    for _ in range(80):
        scene = persona.sample_scene(rng=rng)
        if "mirror" in scene.activity.lower():
            assert scene.location in persona.visual.mirror_locations


@pytest.mark.parametrize("marker", ["dusk", "dawn", "morning", "afternoon", "night", "evening"])
def test_locations_carry_no_time_of_day(persona, marker):
    """Regression: 'rooftop at dusk' collided with the time axis."""
    for location in persona.visual.locations:
        assert marker not in location.lower(), f"{location!r} encodes a time of day"


# -- planning ------------------------------------------------------------


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


def test_shots_record_the_style_they_were_planned_for(persona):
    assert {s.style for s in plan_shoot(persona, 4)} == {"illustrated"}


def test_manifest_is_valid_json_with_full_provenance(persona):
    parsed = json.loads(as_manifest(plan_shoot(persona, 3, rng=random.Random(1))))
    assert len(parsed) == 3
    for entry in parsed:
        assert set(entry) == {
            "index", "prompt", "negative", "seed", "style",
            "wardrobe", "location", "time", "activity",
        }


def test_paste_list_contains_one_block_per_shot(persona):
    text = as_paste_list(plan_shoot(persona, 4, rng=random.Random(2)))
    assert text.count("negative:") == 4
    assert text.count("seed:") == 4
