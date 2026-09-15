import pytest

from thirsttrap.score import (
    MAX_CHARS,
    WEIGHTS,
    score_concreteness,
    score_direct_address,
    score_freshness,
    score_hook,
    score_length,
    score_post,
    score_restraint,
    score_rhythm,
)


def test_weights_sum_to_one():
    # The 0-100 range of score_post depends on this holding.
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_total_is_bounded_and_uses_full_range():
    strong = score_post("You already know the answer. You just want permission.")
    weak = score_post("just some thoughts on stuff #content #viral https://example.com")
    assert 0.0 <= weak.total <= strong.total <= 100.0
    assert strong.total > weak.total


def test_every_weighted_component_is_reported():
    score = score_post("You look better when you stop explaining yourself.")
    assert set(score.components) == set(WEIGHTS)


class TestLength:
    def test_punchy_band_scores_full(self):
        assert score_length("x" * 90) == 1.0

    def test_over_the_limit_is_zero(self):
        assert score_length("x" * (MAX_CHARS + 1)) == 0.0

    def test_too_short_is_penalised(self):
        assert score_length("hey") < 0.4

    def test_score_is_monotonic_approaching_the_band(self):
        assert score_length("x" * 20) < score_length("x" * 40) < score_length("x" * 60)

    def test_over_limit_adds_a_note(self):
        notes = score_post("x" * (MAX_CHARS + 5)).notes
        assert any("over the" in n for n in notes)


class TestHook:
    def test_second_person_opener_beats_hedge(self):
        assert score_hook("You never needed the approval") > score_hook(
            "I just think maybe it was fine"
        )

    def test_contrarian_opener_scores_well(self):
        assert score_hook("Nobody tells you this part") > 0.6

    def test_empty_text_scores_zero(self):
        assert score_hook("") == 0.0

    def test_only_the_opening_words_count(self):
        # "you" arriving in word 20 should not rescue a hedged opener.
        late = "Honestly I was basically just sitting there thinking about nothing "
        late += "much at all and then eventually it occurred to me that you exist"
        assert score_hook(late) < 0.4


class TestDirectAddress:
    def test_absence_is_heavily_penalised(self):
        assert score_direct_address("the weather turned cold today") < 0.3

    def test_a_few_hits_score_full(self):
        assert score_direct_address("you know what you did") == 1.0

    def test_nagging_is_penalised(self):
        many = "you you your you're yours yourself you you"
        assert score_direct_address(many) < 1.0

    def test_contractions_and_accents_normalise(self):
        assert score_direct_address("you’re fine") == score_direct_address("youre fine")


class TestRestraint:
    def test_clean_text_is_unpenalised(self):
        score, notes = score_restraint("A perfectly ordinary sentence.")
        assert score == 1.0
        assert notes == []

    def test_hashtags_are_penalised_and_noted(self):
        score, notes = score_restraint("look at this #viral #content")
        assert score < 0.5
        assert any("hashtag" in n for n in notes)

    def test_links_are_penalised(self):
        score, notes = score_restraint("more here https://example.com/thing")
        assert score < 0.7
        assert any("link" in n for n in notes)

    def test_two_emoji_are_fine_but_five_are_not(self):
        assert score_restraint("nice \U0001F642\U0001F525")[0] == 1.0
        assert score_restraint("nice \U0001F642\U0001F525\U0001F60D\U0001F44F\U0001F4AF")[0] < 1.0

    def test_shouting_is_penalised(self):
        assert score_restraint("THIS IS THE BEST THING")[0] < 1.0

    def test_a_single_acronym_is_not_shouting(self):
        assert score_restraint("the API shipped today")[0] == 1.0


class TestFreshness:
    def test_cliche_is_penalised_and_named(self):
        score, notes = score_freshness("let that sink in")
        assert score < 1.0
        assert any("sink in" in n for n in notes)

    def test_clean_text_is_unpenalised(self):
        assert score_freshness("an ordinary original sentence")[0] == 1.0

    def test_penalty_stacks(self):
        one = score_freshness("let that sink in")[0]
        two = score_freshness("let that sink in, read that again")[0]
        assert two < one


class TestConcreteness:
    def test_numerals_help(self):
        assert score_concreteness("3 sets of 12") > score_concreteness("a few sets")

    def test_abstraction_hurts(self):
        assert score_concreteness("effectiveness and productivity") < score_concreteness(
            "cold coffee and a broken chair"
        )


class TestRhythm:
    def test_short_punch_after_setup_scores_best(self):
        setup_punch = "I spent four years learning the rules of this. Then I left."
        flat = "I spent four years learning the rules of this and then I left it all."
        assert score_rhythm(setup_punch) > score_rhythm(flat)

    def test_rambling_is_penalised(self):
        rambling = "One. Two. Three. Four. Five. Six."
        assert score_rhythm(rambling) < 0.5


def test_empty_and_whitespace_input_do_not_raise():
    for text in ("", "   ", "\n"):
        score = score_post(text)
        assert 0.0 <= score.total <= 100.0
