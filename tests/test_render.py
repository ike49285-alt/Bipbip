from thirsttrap.rank import rank
from thirsttrap.render import BAR_WIDTH, bar, render_ranked, render_score
from thirsttrap.score import WEIGHTS, score_post


class TestBar:
    def test_is_always_the_declared_width(self):
        for fraction in (0.0, 0.33, 1.0, -5.0, 9.0):
            assert len(bar(fraction)) == BAR_WIDTH

    def test_empty_and_full_are_distinguishable(self):
        assert bar(0.0) == "." * BAR_WIDTH
        assert bar(1.0) == "#" * BAR_WIDTH

    def test_out_of_range_values_clamp(self):
        assert bar(-1.0) == bar(0.0)
        assert bar(2.0) == bar(1.0)


class TestRenderScore:
    def test_shows_total_length_and_every_component(self):
        text = "You already know the answer."
        out = render_score(text, score_post(text))
        assert f"{len(text)} chars" in out
        for component in WEIGHTS:
            assert component in out

    def test_breakdown_can_be_suppressed(self):
        text = "You already know the answer."
        assert "hook" not in render_score(text, score_post(text), breakdown=False)

    def test_notes_survive_without_the_breakdown(self):
        text = "look at this #viral"
        assert "hashtag" in render_score(text, score_post(text), breakdown=False)


class TestRenderRanked:
    def test_numbers_follow_displayed_order(self):
        ranked = rank(
            [
                "You already know the answer. You want permission.",
                "just some thoughts #viral https://example.com",
            ]
        )
        out = render_ranked(ranked)
        assert out.startswith("1. ")
        assert out.index("1. ") < out.index("2. ")
        assert out.index(ranked[0].text) < out.index(ranked[1].text)

    def test_start_offset_is_respected(self):
        out = render_ranked(rank(["a single post here"]), start=4)
        assert out.startswith("4. ")

    def test_near_duplicates_are_flagged(self):
        out = render_ranked(
            rank(
                [
                    "You already know the answer. You want permission.",
                    "You already know the answer. You just want permission.",
                ]
            )
        )
        assert "[near-duplicate]" in out

    def test_distinct_posts_are_not_flagged(self):
        out = render_ranked(
            rank(["You already know the answer.", "Nobody warned me about the lift."])
        )
        assert "[near-duplicate]" not in out

    def test_breakdown_adds_components(self):
        ranked = rank(["You already know the answer."])
        assert "hook" not in render_ranked(ranked)
        assert "hook" in render_ranked(ranked, breakdown=True)

    def test_empty_batch_renders_a_placeholder(self):
        assert render_ranked([]) == "(nothing to show)"
