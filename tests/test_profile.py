import json

import pytest

from thirsttrap.directives import Constraints
from thirsttrap.profile import (
    CONFIDENT_AFTER,
    ENV_OVERRIDE,
    MAX_EXAMPLES,
    Profile,
    default_path,
)
from thirsttrap.score import WEIGHTS, score_post


def components(text):
    return score_post(text).components


class TestDefaultPath:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path / "mine.json"))
        assert default_path() == tmp_path / "mine.json"

    def test_falls_back_to_xdg(self, monkeypatch, tmp_path):
        monkeypatch.delenv(ENV_OVERRIDE, raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert default_path() == tmp_path / "thirsttrap" / "profile.json"

    def test_falls_back_to_home_config(self, monkeypatch, tmp_path):
        monkeypatch.delenv(ENV_OVERRIDE, raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert default_path() == tmp_path / ".config" / "thirsttrap" / "profile.json"


class TestLearning:
    def test_a_fresh_profile_carries_the_shipped_prior(self):
        assert Profile().weights == WEIGHTS

    def test_weights_stay_normalised_after_learning(self):
        profile = Profile()
        profile.learn_from_keep(
            components("You already know. You're stalling."),
            [components("Three years of this and it remains somewhat difficult, honestly.")],
        )
        assert sum(profile.weights.values()) == pytest.approx(1.0)

    def test_weight_moves_toward_what_distinguished_the_kept_post(self):
        profile = Profile()
        chosen = components("You already know about you and your choices.")
        other = components("The weather turned cold this week.")
        profile.learn_from_keep(chosen, [other])
        # The kept post is the one using second person, so that component gains.
        assert profile.weights["direct_address"] > WEIGHTS["direct_address"]

    def test_keeping_from_a_batch_of_one_teaches_nothing(self):
        profile = Profile()
        assert profile.learn_from_keep(components("a post"), []) == {}
        assert profile.weights == WEIGHTS

    def test_a_keep_is_counted_even_when_it_teaches_nothing(self):
        profile = Profile()
        profile.learn_from_keep(components("a post"), [])
        assert profile.keeps == 1

    def test_learning_is_reversible(self):
        profile = Profile()
        profile.learn_from_keep(components("You know you."), [components("Cold weather.")])
        profile.reset_weights()
        assert profile.weights == WEIGHTS and profile.keeps == 0

    def test_drift_reports_the_biggest_mover_first(self):
        profile = Profile()
        profile.learn_from_keep(components("You know you."), [components("Cold weather.")])
        drift = profile.drift()
        assert abs(drift[0][1]) >= abs(drift[-1][1])

    def test_an_untouched_profile_has_no_drift(self):
        assert all(delta == pytest.approx(0.0) for _, delta in Profile().drift())


class TestConfidence:
    def test_it_says_untuned_before_any_keep(self):
        assert Profile().confidence() == "untuned"

    def test_a_few_keeps_are_labelled_as_barely_tuned(self):
        profile = Profile()
        profile.keeps = 3
        assert "barely tuned" in profile.confidence()

    def test_enough_keeps_drops_the_caveat(self):
        profile = Profile()
        profile.keeps = CONFIDENT_AFTER
        assert "barely" not in profile.confidence()


class TestStanding:
    def test_a_rule_is_adopted_and_reported(self):
        profile = Profile()
        added = profile.add_standing(Constraints(banned=("!",)))
        assert added == ["never say '!'"]

    def test_readopting_the_same_rule_reports_nothing_new(self):
        profile = Profile()
        profile.add_standing(Constraints(banned=("!",)))
        assert profile.add_standing(Constraints(banned=("!",))) == []

    def test_rules_accumulate(self):
        profile = Profile()
        profile.add_standing(Constraints(banned=("!",)))
        profile.add_standing(Constraints(questions=False))
        assert len(profile.standing.describe()) == 2

    def test_clearing_reports_the_count(self):
        profile = Profile()
        profile.add_standing(Constraints(banned=("!",), questions=False))
        assert profile.clear_standing() == 2
        assert profile.standing.describe() == []


class TestExamples:
    def test_an_example_is_recorded_once(self):
        profile = Profile()
        assert profile.remember("a post") is True
        assert profile.remember("a post") is False

    def test_examples_are_capped_dropping_the_oldest(self):
        profile = Profile()
        for i in range(MAX_EXAMPLES + 5):
            profile.remember(f"post {i}")
        assert len(profile.examples) == MAX_EXAMPLES
        assert "post 0" not in profile.examples


class TestPersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "profile.json"
        original = Profile(path=path, persona="gym")
        original.add_standing(Constraints(max_chars=90, banned=("!",)))
        original.learn_from_keep(components("You know you."), [components("Cold weather.")])
        original.remember("a kept post")
        original.save()

        loaded = Profile.load(path)
        assert loaded.persona == "gym"
        assert loaded.standing.max_chars == 90
        assert loaded.standing.banned == ("!",)
        assert loaded.examples == ["a kept post"]
        assert loaded.keeps == 1
        assert loaded.weights == pytest.approx(original.weights)

    def test_saving_creates_missing_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "profile.json"
        Profile(path=path).save()
        assert path.exists()

    def test_saving_leaves_no_temp_file_behind(self, tmp_path):
        Profile(path=tmp_path / "profile.json").save()
        assert [p.name for p in tmp_path.iterdir()] == ["profile.json"]

    def test_a_pathless_profile_saves_nowhere_without_error(self):
        Profile().save()

    def test_missing_file_loads_an_empty_profile(self, tmp_path):
        loaded = Profile.load(tmp_path / "absent.json")
        assert loaded.weights == WEIGHTS and loaded.keeps == 0

    def test_corrupt_file_never_blocks_a_session(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text("{not json at all")
        assert Profile.load(path).weights == WEIGHTS

    def test_a_json_list_is_rejected_without_crashing(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text("[1, 2, 3]")
        assert Profile.load(path).weights == WEIGHTS

    def test_an_unknown_persona_on_disk_falls_back(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"persona": "smoulder", "keeps": 4}))
        loaded = Profile.load(path)
        assert loaded.persona == Profile().persona
        assert loaded.keeps == 4

    def test_garbage_weights_are_ignored(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"weights": {"hook": "lots", "nonsense": 3}}))
        assert Profile.load(path).weights == WEIGHTS

    def test_partial_weights_are_completed_from_the_prior(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"weights": {"hook": 0.5}}))
        loaded = Profile.load(path)
        assert set(loaded.weights) == set(WEIGHTS)
        assert sum(loaded.weights.values()) == pytest.approx(1.0)

    def test_a_malformed_standing_block_is_survived(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"standing": {"max_chars": "loads", "banned": "nope"}}))
        assert Profile.load(path).standing.describe() == []

    def test_negative_counters_are_rejected(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"keeps": -4, "batches": "many"}))
        loaded = Profile.load(path)
        assert loaded.keeps == 0 and loaded.batches == 0
