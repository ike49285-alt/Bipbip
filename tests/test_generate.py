import pytest

from thirsttrap.directives import Constraints, parse as parse_directive
from thirsttrap.generate import build_prompt, build_system, propose
from thirsttrap.llm import Backend, BackendError, GrammarBackend
from thirsttrap.profile import Profile
from thirsttrap.score import WEIGHTS


class FakeBackend(Backend):
    """Returns canned completions and records what it was asked."""

    def __init__(self, *replies):
        self.name = "fake"
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def available(self):
        return True

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        if not self.replies:
            return ""
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def describe(self):
        return "fake backend"


class TestBuildSystem:
    def test_hard_rules_are_always_present(self):
        system = build_system()
        for rule in ("hashtags", "URLs", "280", "adults", "explicit"):
            assert rule in system

    def test_the_persona_directive_is_included(self):
        from thirsttrap import personas

        assert personas.get("gym").directive in build_system("gym")

    def test_standing_rules_reach_the_model(self):
        profile = Profile()
        profile.add_standing(parse_directive("never use exclamation marks").constraints)
        assert "never say '!'" in build_system("flirt", profile)

    def test_kept_examples_reach_the_model(self):
        profile = Profile()
        profile.remember("You already know. You're stalling.")
        system = build_system("flirt", profile)
        assert "You already know. You're stalling." in system
        assert "do not reuse" in system.lower()

    def test_an_empty_profile_adds_nothing(self):
        assert build_system("flirt", Profile()) == build_system("flirt")

    def test_the_scoring_rubric_never_leaks(self):
        # Telling the generator what the scorer rewards makes the score measure
        # instruction-following instead of quality.
        profile = Profile()
        profile.remember("a kept post")
        haystack = build_system("flirt", profile).lower()
        for component in WEIGHTS:
            if component in {"length", "restraint"}:
                continue  # format facts, stated on purpose
            assert component.replace("_", " ") not in haystack


class TestBuildPrompt:
    def test_topic_and_count_are_stated(self):
        prompt = build_prompt("leg day", 7)
        assert "leg day" in prompt and "7" in prompt

    def test_constraints_are_asked_for(self):
        prompt = build_prompt("leg day", 5, Constraints(max_chars=80, questions=False))
        assert "at most 80 characters" in prompt and "no questions" in prompt

    def test_a_reference_post_is_included_without_asking_for_a_copy(self):
        prompt = build_prompt("leg day", 5, like_text="A kept one.")
        assert "A kept one." in prompt and "without repeating" in prompt


class TestProposeWithAModel:
    def test_parsed_posts_come_back(self):
        backend = FakeBackend("first post\nsecond post")
        assert propose("leg day", backend=backend, n=5) == ["first post", "second post"]

    def test_the_count_is_capped(self):
        backend = FakeBackend("a\nb\nc\nd\ne")
        assert len(propose("leg day", backend=backend, n=2)) == 2

    def test_duplicates_across_attempts_are_dropped(self):
        backend = FakeBackend("same post", "same post\nnew post")
        assert propose("leg day", backend=backend, n=5) == ["same post", "new post"]

    def test_it_retries_when_the_first_reply_is_short(self):
        backend = FakeBackend("only one", "two\nthree")
        propose("leg day", backend=backend, n=3)
        assert len(backend.calls) == 2

    def test_it_stops_once_enough_came_back(self):
        backend = FakeBackend("a\nb\nc")
        propose("leg day", backend=backend, n=3)
        assert len(backend.calls) == 1

    def test_a_backend_failure_propagates_rather_than_returning_nothing(self):
        backend = FakeBackend(BackendError("model not pulled"))
        with pytest.raises(BackendError, match="model not pulled"):
            propose("leg day", backend=backend, n=3)

    def test_an_unknown_persona_raises_before_calling_the_model(self):
        backend = FakeBackend("a post")
        with pytest.raises(KeyError):
            propose("leg day", persona="smoulder", backend=backend)
        assert backend.calls == []

    @pytest.mark.parametrize("n", [0, -1])
    def test_a_non_positive_count_raises(self, n):
        with pytest.raises(ValueError):
            propose("leg day", backend=FakeBackend("a"), n=n)


class TestConstraintsAreEnforcedNotJustRequested:
    def test_a_model_ignoring_the_length_cap_is_filtered(self):
        backend = FakeBackend("tiny\n" + "x" * 200)
        kept = propose("leg day", backend=backend, n=5,
                       constraints=Constraints(max_chars=50))
        assert kept == ["tiny"]

    def test_the_cap_is_also_asked_for_in_the_prompt(self):
        backend = FakeBackend("tiny")
        propose("leg day", backend=backend, n=5, constraints=Constraints(max_chars=50))
        assert "at most 50 characters" in backend.calls[0][1]

    def test_everything_filtered_out_returns_nothing_rather_than_relaxing(self):
        backend = FakeBackend("a question?", "another question?")
        assert propose("leg day", backend=backend, n=5,
                       constraints=Constraints(questions=False)) == []


class TestGrammarFallback:
    def test_it_generates_without_any_model(self):
        posts = propose("leg day", backend=GrammarBackend(), pool=50, seed=1)
        assert len(posts) == 50

    def test_the_seed_makes_it_reproducible(self):
        a = propose("leg day", backend=GrammarBackend(), pool=30, seed=2)
        b = propose("leg day", backend=GrammarBackend(), pool=30, seed=2)
        assert a == b

    def test_constraints_filter_the_pool(self):
        posts = propose("leg day", backend=GrammarBackend(), pool=200, seed=3,
                        constraints=Constraints(max_chars=70))
        assert posts and all(len(p) <= 70 for p in posts)

    def test_an_impossible_constraint_returns_nothing(self):
        assert propose("leg day", backend=GrammarBackend(), pool=80, seed=4,
                       constraints=Constraints(max_chars=1)) == []

    def test_no_network_module_is_ever_imported(self):
        import sys

        propose("leg day", backend=GrammarBackend(), pool=10, seed=1)
        assert "anthropic" not in sys.modules
