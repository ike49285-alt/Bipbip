import random

import pytest

from thirsttrap import personas
from thirsttrap.grammar import ANY, FRAMES, Frame, expand, generate, options
from thirsttrap.topic import parse

TOPIC = parse("finally quitting the job")


class TestFrames:
    def test_every_frame_resolves_for_every_persona_it_allows(self):
        rng = random.Random(0)
        for frame in FRAMES:
            for name in personas.names():
                if not frame.allows(name):
                    continue
                text = expand(frame.template, TOPIC, name, rng)
                assert "{" not in text, f"unresolved slot in {frame.template!r}"

    def test_an_untagged_frame_allows_everything(self):
        frame = Frame(template="x")
        assert all(frame.allows(name) for name in personas.names())

    def test_a_tagged_frame_is_restricted(self):
        frame = Frame(template="x", personas=frozenset({"gym"}))
        assert frame.allows("gym") and not frame.allows("soft")


class TestOptions:
    def test_persona_entries_are_pooled_with_the_shared_ones(self):
        shared = set(options("punch", "__nonexistent__"))
        gym = set(options("punch", "gym"))
        assert shared and shared < gym

    def test_an_unknown_slot_has_no_options(self):
        assert options("not_a_slot", "flirt") == ()


class TestExpand:
    def test_topic_slots_win_over_the_lexicon(self):
        rng = random.Random(0)
        assert expand("{head}", TOPIC, "flirt", rng) == "The job."

    def test_output_is_tidied(self):
        rng = random.Random(0)
        text = expand("{sentence}  .  and again", TOPIC, "flirt", rng)
        assert "  " not in text
        assert " ." not in text
        assert text.endswith(".")

    def test_sentences_are_capitalised(self):
        rng = random.Random(0)
        assert expand("one thing. two thing", TOPIC, "flirt", rng) == "One thing. Two thing."

    def test_terminal_punctuation_is_not_doubled(self):
        rng = random.Random(0)
        assert expand("already done.", TOPIC, "flirt", rng) == "Already done."

    def test_an_unknown_slot_stays_visible_rather_than_vanishing(self):
        # A typo in a frame should show up in output, not silently disappear.
        rng = random.Random(0)
        assert "{nonsense}" in expand("a {nonsense} b", TOPIC, "flirt", rng)


class TestGenerate:
    def test_the_same_seed_gives_the_same_batch(self):
        assert generate(TOPIC, "flirt", count=20, seed=5) == generate(
            TOPIC, "flirt", count=20, seed=5
        )

    def test_different_seeds_diverge(self):
        assert generate(TOPIC, "flirt", count=20, seed=1) != generate(
            TOPIC, "flirt", count=20, seed=2
        )

    def test_candidates_are_distinct(self):
        posts = generate(TOPIC, "flirt", count=200, seed=3)
        assert len(posts) == len(set(posts))

    def test_it_produces_a_large_pool(self):
        assert len(generate(TOPIC, "flirt", count=300, seed=4)) == 300

    @pytest.mark.parametrize("name", personas.names())
    def test_every_persona_can_generate(self, name):
        assert len(generate(TOPIC, name, count=25, seed=6)) == 25

    def test_a_persona_with_no_frames_returns_nothing_rather_than_raising(self):
        frames = [Frame(template="x", personas=frozenset({"gym"}))]
        assert generate(TOPIC, "soft", count=5, frames=frames) == []

    def test_a_grammar_that_runs_dry_returns_what_it_had(self):
        frames = [Frame(template="one fixed post")]
        assert generate(TOPIC, "flirt", count=50, frames=frames) == ["One fixed post."]

    def test_every_post_ends_in_terminal_punctuation(self):
        for post in generate(TOPIC, "flirt", count=100, seed=7):
            assert post[-1] in ".!?"

    def test_the_topic_reaches_the_output(self):
        posts = generate(TOPIC, "flirt", count=60, seed=8)
        assert any("job" in p.lower() for p in posts)
