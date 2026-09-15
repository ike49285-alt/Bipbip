import pytest

from thirsttrap.topic import is_gerund, parse, to_gerund


class TestToGerund:
    @pytest.mark.parametrize(
        "word,expected",
        [
            ("quit", "quitting"), ("run", "running"), ("sit", "sitting"),
            ("leave", "leaving"), ("lie", "lying"), ("be", "being"),
            ("train", "training"), ("apply", "applying"),
        ],
    )
    def test_known_verbs_convert(self, word, expected):
        assert to_gerund(word) == expected

    @pytest.mark.parametrize("word", ["gym", "leg", "coffee", "marathon", "finally"])
    def test_unknown_words_are_left_alone(self, word):
        # Guessing here produced "gyming" and "legging"; not guessing is the fix.
        assert to_gerund(word) == word

    def test_an_existing_participle_is_unchanged(self):
        assert to_gerund("quitting") == "quitting"
        assert to_gerund("being") == "being"

    def test_is_gerund_needs_length_as_well_as_suffix(self):
        assert is_gerund("being") and is_gerund("quitting")
        assert not is_gerund("ing")


class TestPhrase:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("about my ex", "my ex"),
            ("re the gym", "the gym"),
            ("leg day", "leg day"),
        ],
    )
    def test_a_leading_about_is_stripped(self, raw, expected):
        assert parse(raw).phrase == expected

    @pytest.mark.parametrize("raw", ["the gym at 5am", "a long walk"])
    def test_a_leading_article_is_kept(self, raw):
        # Frames read "about the gym at 5am", never "about gym at 5am".
        assert parse(raw).phrase == raw.lower()

    def test_whitespace_is_normalised(self):
        assert parse("  finally   quitting  ").phrase == "finally quitting"

    def test_sentence_form_is_capitalised(self):
        assert parse("the gym at 5am").sentence == "The gym at 5am"


class TestHead:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("finally quitting the job", "the job"),
            ("the gym at 5am", "the gym"),
            ("my ex", "my ex"),
            ("leg day", "leg day"),
            ("the best coffee in town", "the best coffee"),
        ],
    )
    def test_head_keeps_its_determiner_and_modifiers(self, raw, expected):
        # Frames slot this after prepositions, and "about job" reads wrong.
        assert parse(raw).head == expected

    def test_head_stops_at_the_first_preposition(self):
        assert "5am" not in parse("the gym at 5am").head


class TestGerundSlot:
    def test_a_known_verb_leads_with_a_participle(self):
        assert parse("quit my job").gerund == "quitting my job"

    def test_an_unknown_lead_word_passes_through_unchanged(self):
        assert parse("leg day").gerund == "leg day"
        assert parse("the gym at 5am").gerund == "the gym at 5am"


class TestDegenerateInput:
    @pytest.mark.parametrize("raw", ["", "   ", "!!!", "the"])
    def test_never_raises_and_always_fills_every_slot(self, raw):
        topic = parse(raw)
        for slot in ("raw", "phrase", "gerund", "head"):
            assert isinstance(getattr(topic, slot), str)
