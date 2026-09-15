from thirsttrap.rank import rank, similarity, tokens


class TestSimilarity:
    def test_identical_text_is_one(self):
        assert similarity("the same words here", "the same words here") == 1.0

    def test_disjoint_text_is_zero(self):
        assert similarity("cold coffee", "broken elevator") == 0.0

    def test_word_order_does_not_matter(self):
        assert similarity("coffee cold", "cold coffee") == 1.0

    def test_stopwords_do_not_create_similarity(self):
        # These share only "the", "of", "a" -- nothing that says they are alike.
        assert similarity("the end of a rope", "the smell of a garden") == 0.0

    def test_empty_text_is_zero_not_an_error(self):
        assert similarity("", "anything") == 0.0

    def test_is_symmetric(self):
        a, b = "cold coffee and rain", "rain and warm coffee"
        assert similarity(a, b) == similarity(b, a)


def test_tokens_drop_stopwords():
    assert tokens("the cold coffee") == {"cold", "coffee"}


class TestRank:
    def test_output_is_sorted_by_final_score(self):
        ranked = rank(
            [
                "You already know. You are just waiting for permission.",
                "some thoughts about things #viral https://example.com",
                "Nobody tells you the quiet part. You find it yourself.",
            ]
        )
        finals = [r.final for r in ranked]
        assert finals == sorted(finals, reverse=True)

    def test_every_candidate_survives_ranking(self):
        candidates = ["first post here", "second post there", "third post everywhere"]
        assert len(rank(candidates)) == len(candidates)

    def test_near_duplicates_are_discounted(self):
        original = "You already know the answer. You want permission."
        echo = "You already know the answer. You just want permission."
        ranked = rank([original, echo])

        # The stronger member of the pair keeps its intrinsic score...
        assert ranked[0].final == ranked[0].score.total
        # ...and the echo absorbs the penalty.
        assert ranked[1].final < ranked[1].score.total
        assert ranked[1].duplicate_of

    def test_distinct_candidates_are_not_discounted(self):
        ranked = rank(
            [
                "You already know the answer.",
                "Nobody warned me about the elevator.",
            ]
        )
        for item in ranked:
            assert item.final == item.score.total
            assert not item.duplicate_of

    def test_novelty_weight_of_zero_disables_the_penalty(self):
        pair = [
            "You already know the answer. You want permission.",
            "You already know the answer. You just want permission.",
        ]
        for item in rank(pair, novelty_weight=0.0):
            assert item.final == item.score.total

    def test_a_weak_original_still_yields_to_a_strong_echo(self):
        # Ranking walks best-first, so the penalty lands on the weaker twin
        # regardless of which one was passed in first.
        strong = "You already know the answer. You want permission."
        weak = "you already know the answer, you want permission, i guess #thoughts"
        ranked = rank([weak, strong])
        assert ranked[0].text == strong
        assert ranked[0].final == ranked[0].score.total

    def test_single_candidate_is_never_a_duplicate(self):
        ranked = rank(["You already know the answer."])
        assert ranked[0].final == ranked[0].score.total
        assert not ranked[0].duplicate_of

    def test_empty_input_returns_empty(self):
        assert rank([]) == []
