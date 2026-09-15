import pytest

from thirsttrap.directives import Constraints
from thirsttrap.generate import propose


class TestPropose:
    def test_it_returns_a_pool(self):
        assert len(propose("finally quitting the job", pool=50, seed=1)) == 50

    def test_the_same_seed_reproduces_the_batch(self):
        a = propose("leg day", pool=30, seed=2)
        b = propose("leg day", pool=30, seed=2)
        assert a == b

    def test_an_unknown_persona_raises_with_the_valid_names(self):
        with pytest.raises(KeyError) as excinfo:
            propose("a topic", persona="smoulder")
        assert "flirt" in excinfo.value.args[0]

    @pytest.mark.parametrize("pool", [0, -5])
    def test_a_non_positive_pool_raises(self, pool):
        with pytest.raises(ValueError):
            propose("a topic", pool=pool)

    def test_no_network_module_is_imported(self):
        # The whole point of this version: nothing reaches out.
        import sys

        propose("a topic", pool=10, seed=1)
        assert "anthropic" not in sys.modules


class TestConstraints:
    def test_a_length_cap_is_enforced(self):
        posts = propose("finally quitting the job", pool=200, seed=3,
                        constraints=Constraints(max_chars=70))
        assert posts and all(len(p) <= 70 for p in posts)

    def test_questions_can_be_excluded(self):
        posts = propose("finally quitting the job", pool=200, seed=4,
                        constraints=Constraints(questions=False))
        assert posts and all("?" not in p for p in posts)

    def test_numerals_can_be_excluded(self):
        posts = propose("finally quitting the job", pool=200, seed=5,
                        constraints=Constraints(numerals=False))
        assert posts and not any(c.isdigit() for p in posts for c in p)

    def test_filtering_narrows_rather_than_replaces(self):
        seed = 6
        everything = propose("leg day", pool=150, seed=seed)
        filtered = propose("leg day", pool=150, seed=seed,
                           constraints=Constraints(max_chars=60))
        assert set(filtered) <= set(everything)
        assert len(filtered) < len(everything)

    def test_an_impossible_constraint_returns_nothing_rather_than_relaxing(self):
        assert propose("leg day", pool=100, seed=7,
                       constraints=Constraints(max_chars=1)) == []
