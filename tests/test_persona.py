import copy
import json
import random
from pathlib import Path

import pytest

from bot.persona import Persona, PersonaError, Scene

RAW = json.loads(Path("persona.json").read_text())


def make(**overrides):
    raw = copy.deepcopy(RAW)
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if key:
            raw[section][key] = value
        else:
            raw[section] = value
    return raw


def test_ships_persona_loads():
    p = Persona.load("persona.json")
    assert p.identity.name == "Remy"
    assert p.identity.handle == "remy_synthetic"


def test_handle_strips_at_sign():
    p = Persona.from_dict(make(**{"identity.handle": "@remy_synthetic"}))
    assert p.identity.handle == "remy_synthetic"


def test_missing_section_is_rejected():
    raw = copy.deepcopy(RAW)
    del raw["voice"]
    with pytest.raises(PersonaError, match="voice"):
        Persona.from_dict(raw)


def test_missing_file_is_rejected():
    with pytest.raises(PersonaError, match="no persona file"):
        Persona.load("does-not-exist.json")


def test_bad_json_is_rejected(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text("{not json")
    with pytest.raises(PersonaError, match="not valid JSON"):
        Persona.load(bad)


@pytest.mark.parametrize("field", ["name", "handle", "bio", "disclosure", "disclosure_short"])
def test_blank_identity_fields_are_rejected(field):
    with pytest.raises(PersonaError, match=field):
        Persona.from_dict(make(**{f"identity.{field}": "   "}))


def test_undisclosed_persona_will_not_load():
    """The design's one hard rule, enforced at load time."""
    raw = make(
        **{
            "identity.bio": "just a girl in a cold apartment",
            "identity.disclosure": "born in november, allegedly",
            "identity.disclosure_short": "she/her",
        }
    )
    with pytest.raises(PersonaError, match="does not disclose"):
        Persona.from_dict(raw)


@pytest.mark.parametrize(
    "marker",
    ["ai persona here", "i am a bot", "fully synthetic", "generated nightly", "not a real person"],
)
def test_any_disclosure_marker_satisfies_the_check(marker):
    Persona.from_dict(make(**{"identity.bio": marker}))


def test_empty_truthfulness_list_is_rejected():
    with pytest.raises(PersonaError, match="admit what it is"):
        Persona.from_dict(make(**{"bounds.always_answer_truthfully": []}))


def test_empty_voice_rules_rejected():
    with pytest.raises(PersonaError, match="voice.rules"):
        Persona.from_dict(make(**{"voice.rules": []}))


def test_empty_facts_rejected():
    with pytest.raises(PersonaError, match="biographical fact"):
        Persona.from_dict(make(facts=[]))


def test_empty_visual_pool_rejected():
    with pytest.raises(PersonaError, match="visual.wardrobe"):
        Persona.from_dict(make(**{"visual.wardrobe": []}))


def test_voice_card_carries_rules_facts_and_disclosure():
    p = Persona.load()
    card = p.voice_card()
    assert p.identity.disclosure in card
    for rule in p.voice.rules:
        assert rule in card
    for fact in p.facts:
        assert fact in card


def test_voice_card_is_byte_stable():
    """It sits in the cached prompt prefix, so it must not vary between calls."""
    p = Persona.load()
    assert p.voice_card() == p.voice_card()
    assert Persona.load().voice_card() == p.voice_card()


def test_sample_scene_avoids_recent_repeats():
    p = Persona.load()
    rng = random.Random(0)
    seen: list[tuple] = []
    for _ in range(12):
        scene = p.sample_scene(recent=seen, rng=rng)
        assert scene.key() not in seen
        seen.append(scene.key())


def test_sample_scene_returns_something_when_pools_exhausted():
    """A repeat is better than failing to post."""
    raw = make(
        **{
            "visual.wardrobe": ["only"],
            "visual.locations": ["one"],
            "visual.times": ["combination"],
            "visual.activities": ["exists"],
        }
    )
    p = Persona.from_dict(raw)
    blocked = [Scene("only", "one", "combination", "exists").key()]
    assert p.sample_scene(recent=blocked) is not None


def test_scene_prompt_mentions_every_axis():
    s = Scene("a tee", "a diner", "golden hour", "mid-laugh")
    prompt = s.as_prompt()
    for part in ("a tee", "a diner", "golden hour", "mid-laugh"):
        assert part in prompt


def test_combinations_is_the_product_of_the_pools():
    p = Persona.load()
    v = p.visual
    assert p.combinations() == len(v.wardrobe) * len(v.locations) * len(v.times) * len(v.activities)
