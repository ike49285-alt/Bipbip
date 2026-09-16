from pathlib import Path

import pytest

from bot.captions import (
    CaptionError,
    Captioner,
    SceneFacts,
    assess,
    check_voice,
    content_similarity,
    pick,
    shape,
    shape_repeats,
    trigrams,
)
from bot.persona import Persona


@pytest.fixture
def persona():
    return Persona.load("persona.json")


# -- voice enforcement ---------------------------------------------------


def test_a_good_caption_breaks_no_rules(persona):
    assert check_voice("third coffee. black. milk is an errand.", persona) == []


def test_uppercase_is_caught(persona):
    """Regression: a missing dataclass field silently disabled every constraint."""
    assert any("uppercase" in p for p in check_voice("Posting Anyway.", persona))


def test_over_length_is_caught(persona):
    assert any("limit" in p for p in check_voice("a" * 200, persona))


def test_hashtags_are_caught(persona):
    assert any("'#'" in p for p in check_voice("laundry day #vibes", persona))


def test_em_dash_is_caught(persona):
    assert any("—" in p for p in check_voice("coffee — black", persona))


def test_links_are_caught(persona):
    assert any("http" in p for p in check_voice("see http://x.com", persona))


def test_one_emoji_is_allowed_two_is_not(persona):
    assert check_voice("radiator again \U0001f331", persona) == []
    assert any("emoji" in p for p in check_voice("radiator \U0001f331\U0001f319", persona))


def test_empty_caption_is_rejected(persona):
    assert check_voice("   ", persona) == ["empty caption"]


def test_constraints_actually_loaded(persona):
    """The bug was a silently-absent field, so assert it is present."""
    assert persona.voice.constraints["lowercase"] is True
    assert "#" in persona.voice.constraints["forbid"]


# -- similarity ----------------------------------------------------------


def test_identical_text_is_maximally_similar():
    assert content_similarity("coffee. black.", "coffee. black.") == pytest.approx(1.0)


def test_unrelated_text_is_barely_similar():
    assert content_similarity("radiator clanked all night", "six flights up") < 0.2


def test_near_duplicates_score_high():
    assert content_similarity("room's a disaster. posting anyway.",
                              "rooms a disaster. posting anyway.") > 0.8


def test_similarity_ignores_whitespace_and_case():
    assert content_similarity("Coffee.  Black.", "coffee. black.") == pytest.approx(1.0)


def test_trigrams_of_short_text_do_not_crash():
    assert trigrams("ab") == set()


# -- shape ---------------------------------------------------------------


def test_same_construction_different_words_share_a_shape():
    """The real bot tell: identical rhythm, unrelated vocabulary."""
    assert shape("room's a disaster. posting anyway.") == shape("bus is late. walking instead.")


def test_different_construction_differs_in_shape():
    assert shape("one. two.") != shape("a single longer sentence with no break")


def test_shape_repeats_counts_matching_history():
    history = ["room's a disaster. posting anyway.", "bus is late. walking instead."]
    assert shape_repeats("coffee went cold. drinking it anyway.", history) == 2
    assert shape_repeats("a single longer sentence running on and on here", history) == 0


def test_overused_shape_is_rejected_even_when_wording_is_fresh(persona):
    history = ["room's a disaster. posting anyway.",
               "bus is late. walking instead.",
               "sink is full. ignoring it."]
    verdict = assess("phone died. whatever now.", persona, history)
    assert not verdict.accepted
    assert any("shape" in r for r in verdict.reasons)


# -- assessment and selection -------------------------------------------


def test_near_duplicate_of_history_is_rejected(persona):
    v = assess("rooms a disaster. posting anyway.", persona,
               ["room's a disaster. posting anyway."])
    assert not v.accepted
    assert v.max_similarity > 0.8


def test_clean_caption_against_empty_history_passes(persona):
    assert assess("six flights up for this. worth it, probably.", persona).accepted


def test_history_window_is_respected(persona):
    """An old caption outside the window should not block a new one."""
    old = "rooms a disaster. posting anyway."
    filler = [f"caption number {i} running along here for a while" for i in range(60)]
    assert assess(old, persona, [old] + filler).accepted


def test_pick_returns_none_when_everything_fails(persona):
    chosen, verdicts = pick(["BAD CAPS", "also #bad"], persona)
    assert chosen is None
    assert len(verdicts) == 2
    assert all(not v.accepted for v in verdicts)


def test_pick_prefers_the_least_repetitive_candidate(persona):
    history = ["room's a disaster. posting anyway."]
    chosen, _ = pick(
        ["rooms a disaster. posting anyway.",          # near-duplicate, rejected
         "six flights up for this and the view is worth the climb today"],
        persona, history,
    )
    assert chosen.startswith("six flights")


def test_pick_skips_voice_violations(persona):
    chosen, _ = pick(["Uppercase Option", "lowercase option here."], persona)
    assert chosen == "lowercase option here."


# -- scene facts ---------------------------------------------------------


def test_scene_facts_brief_includes_what_was_seen():
    facts = SceneFacts("a laundromat", "leaning on a machine",
                       ("a single sock",), "bored", "fluorescent")
    brief = facts.as_brief()
    for part in ("laundromat", "leaning", "single sock", "bored", "fluorescent"):
        assert part in brief


def test_scene_facts_require_setting_and_action():
    with pytest.raises(ValueError, match="setting"):
        SceneFacts.from_dict({"subject_action": "sitting"})
    with pytest.raises(ValueError, match="subject_action"):
        SceneFacts.from_dict({"setting": "a diner"})


def test_scene_facts_tolerate_missing_optional_fields():
    facts = SceneFacts.from_dict({"setting": "a diner", "subject_action": "sitting"})
    assert facts.notable_details == () and facts.mood == ""


# -- orchestration -------------------------------------------------------


class FakeLLM:
    def __init__(self, batches):
        self.batches = list(batches)
        self.described = 0
        self.systems = []

    def describe_image(self, image, instructions, schema):
        self.described += 1
        return {"setting": "a laundromat", "subject_action": "waiting"}

    def write_candidates(self, system, prompt, count):
        self.systems.append(system)
        return self.batches.pop(0)


def test_captioner_returns_a_passing_caption(persona):
    llm = FakeLLM([["third coffee. black. milk is an errand."]])
    caption, facts = Captioner(persona, llm).caption(Path("x.png"))
    assert caption == "third coffee. black. milk is an errand."
    assert facts.setting == "a laundromat"


def test_captioner_retries_before_giving_up(persona):
    llm = FakeLLM([["BAD"], ["ALSO BAD"], ["good one. finally here."]])
    caption, _ = Captioner(persona, llm).caption(Path("x.png"))
    assert caption == "good one. finally here."
    assert llm.described == 1, "should describe the image once, not per attempt"


def test_captioner_raises_rather_than_posting_junk(persona):
    llm = FakeLLM([["BAD"], ["STILL BAD"], ["NOPE"]])
    with pytest.raises(CaptionError, match="no candidate passed"):
        Captioner(persona, llm).caption(Path("x.png"))


def test_captioner_sends_the_stable_voice_card_as_system(persona):
    llm = FakeLLM([["fine caption here. good."]])
    Captioner(persona, llm).caption(Path("x.png"))
    assert llm.systems[0] == persona.voice_card()


def test_prompt_includes_recent_captions_to_avoid(persona):
    captioner = Captioner(persona, FakeLLM([[]]))
    facts = SceneFacts("a diner", "sitting")
    prompt = captioner.compose_prompt(facts, ["an earlier caption."])
    assert "an earlier caption." in prompt
    assert "a diner" in prompt
