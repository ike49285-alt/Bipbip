import json

import pytest

from thirsttrap.profile import (
    ENV_OVERRIDE,
    MAX_EXAMPLES,
    MAX_RULES,
    Profile,
    default_path,
)


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


class TestLearn:
    def test_new_rules_are_returned(self):
        profile = Profile()
        assert profile.learn(["no exclamation marks"]) == ["no exclamation marks"]

    def test_duplicates_are_ignored_case_insensitively(self):
        profile = Profile()
        profile.learn(["No Exclamation Marks"])
        assert profile.learn(["no exclamation marks"]) == []
        assert len(profile.rules) == 1

    def test_blank_rules_are_dropped(self):
        assert Profile().learn(["  ", "", "\n"]) == []

    def test_whitespace_is_normalised(self):
        profile = Profile()
        profile.learn(["no    exclamation\n marks"])
        assert profile.rules == ["no exclamation marks"]

    def test_rules_are_capped_dropping_the_oldest(self):
        profile = Profile()
        profile.learn([f"rule {i}" for i in range(MAX_RULES + 10)])
        assert len(profile.rules) == MAX_RULES
        assert profile.rules[-1] == f"rule {MAX_RULES + 9}"
        assert "rule 0" not in profile.rules


class TestRemember:
    def test_an_example_is_recorded_once(self):
        profile = Profile()
        assert profile.remember("a post") is True
        assert profile.remember("a post") is False
        assert profile.examples == ["a post"]

    def test_examples_are_capped_dropping_the_oldest(self):
        profile = Profile()
        for i in range(MAX_EXAMPLES + 5):
            profile.remember(f"post {i}")
        assert len(profile.examples) == MAX_EXAMPLES
        assert profile.examples[-1] == f"post {MAX_EXAMPLES + 4}"
        assert "post 0" not in profile.examples


class TestForget:
    def test_removes_and_returns_the_rule(self):
        profile = Profile()
        profile.learn(["one", "two"])
        assert profile.forget(1) == "one"
        assert profile.rules == ["two"]

    def test_out_of_range_names_the_range(self):
        profile = Profile()
        profile.learn(["only"])
        with pytest.raises(ValueError, match="pick 1-1"):
            profile.forget(5)

    def test_forgetting_from_nothing_says_so(self):
        with pytest.raises(ValueError, match="nothing learned"):
            Profile().forget(1)

    def test_forget_all_reports_the_count(self):
        profile = Profile()
        profile.learn(["one", "two", "three"])
        assert profile.forget_all() == 3
        assert profile.rules == []


class TestBrief:
    def test_empty_profile_contributes_nothing(self):
        assert Profile().brief() == ""

    def test_rules_and_examples_both_appear(self):
        profile = Profile()
        profile.learn(["no exclamation marks"])
        profile.remember("You already know the answer.")
        brief = profile.brief()
        assert "no exclamation marks" in brief
        assert "You already know the answer." in brief

    def test_examples_warn_against_reuse(self):
        profile = Profile()
        profile.remember("a kept post")
        assert "do not reuse" in profile.brief().lower()


class TestPersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "profile.json"
        original = Profile(path=path, persona="gym")
        original.learn(["no exclamation marks"])
        original.remember("a kept post")
        original.batches = 4
        original.save()

        loaded = Profile.load(path)
        assert loaded.persona == "gym"
        assert loaded.rules == ["no exclamation marks"]
        assert loaded.examples == ["a kept post"]
        assert loaded.batches == 4

    def test_saving_creates_missing_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "profile.json"
        Profile(path=path).save()
        assert path.exists()

    def test_saving_leaves_no_temp_file_behind(self, tmp_path):
        path = tmp_path / "profile.json"
        Profile(path=path).save()
        assert [p.name for p in tmp_path.iterdir()] == ["profile.json"]

    def test_a_pathless_profile_saves_nowhere_without_error(self):
        Profile().save()  # must not raise

    def test_missing_file_loads_an_empty_profile(self, tmp_path):
        loaded = Profile.load(tmp_path / "absent.json")
        assert loaded.rules == [] and loaded.examples == []
        assert loaded.path == tmp_path / "absent.json"

    def test_corrupt_file_never_blocks_a_session(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text("{not json at all")
        loaded = Profile.load(path)
        assert loaded.rules == []
        assert loaded.persona == Profile().persona

    def test_a_json_list_is_rejected_without_crashing(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text("[1, 2, 3]")
        assert Profile.load(path).rules == []

    def test_an_unknown_persona_on_disk_falls_back(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"persona": "smoulder", "rules": ["keep me"]}))
        loaded = Profile.load(path)
        assert loaded.persona == Profile().persona
        assert loaded.rules == ["keep me"]

    def test_garbage_field_types_are_survived(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"rules": "not a list", "batches": "lots"}))
        loaded = Profile.load(path)
        assert loaded.batches == 0
        assert all(isinstance(r, str) for r in loaded.rules)

    def test_oversized_files_are_truncated_on_load(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"rules": [f"r{i}" for i in range(MAX_RULES + 20)]}))
        assert len(Profile.load(path).rules) == MAX_RULES
