import pytest

from thirsttrap.directives import Constraints, parse


class TestConstraintsAllow:
    def test_char_bounds(self):
        assert Constraints(max_chars=10).allows("short")
        assert not Constraints(max_chars=4).allows("longer text")
        assert not Constraints(min_chars=50).allows("short")

    def test_questions_can_be_required_or_forbidden(self):
        assert not Constraints(questions=False).allows("really?")
        assert Constraints(questions=False).allows("really.")
        assert not Constraints(questions=True).allows("really.")

    def test_numerals_can_be_required_or_forbidden(self):
        assert not Constraints(numerals=False).allows("3 years")
        assert Constraints(numerals=False).allows("three years")
        assert not Constraints(numerals=True).allows("three years")

    def test_sentence_cap(self):
        assert Constraints(max_sentences=1).allows("One thing.")
        assert not Constraints(max_sentences=1).allows("One thing. Two things.")

    def test_emoji_is_a_character_class_not_a_substring(self):
        forbidden = Constraints(emoji=False)
        assert not forbidden.allows("nice \U0001F525")
        assert not forbidden.allows("nice \U0001F642")
        assert forbidden.allows("nice")

    def test_banned_terms_are_case_insensitive(self):
        assert not Constraints(banned=("game changer",)).allows("a GAME CHANGER here")

    def test_an_empty_constraint_allows_anything(self):
        assert Constraints().allows("literally anything at all")

    def test_emptiness_is_falsey(self):
        assert not Constraints()
        assert Constraints(max_chars=90)


class TestMerge:
    def test_later_values_win(self):
        merged = Constraints(max_chars=120).merge(Constraints(max_chars=60))
        assert merged.max_chars == 60

    def test_unset_fields_do_not_erase_earlier_ones(self):
        merged = Constraints(max_chars=90).merge(Constraints(questions=False))
        assert merged.max_chars == 90 and merged.questions is False

    def test_bans_accumulate_without_duplicating(self):
        merged = Constraints(banned=("!",)).merge(Constraints(banned=("!", "#")))
        assert merged.banned == ("!", "#")


class TestParse:
    @pytest.mark.parametrize(
        "line,field,expected",
        [
            ("shorter", "max_chars", 95),
            ("much shorter", "max_chars", 60),
            ("tighter please", "max_chars", 95),
            ("longer", "min_chars", 120),
            ("way longer", "min_chars", 160),
            ("make it one line", "max_sentences", 1),
            ("no questions", "questions", False),
            ("ask a question", "questions", True),
            ("no numbers", "numerals", False),
        ],
    )
    def test_adjustments_map_to_fields(self, line, field, expected):
        assert getattr(parse(line).constraints, field) == expected

    def test_a_stronger_phrase_is_not_swallowed_by_a_weaker_one(self):
        assert parse("much shorter").constraints.max_chars == 60

    @pytest.mark.parametrize(
        "line",
        ["never use exclamation marks", "no exclamation marks", "stop shouting"],
    )
    def test_exclamation_bans_are_recognised_however_phrased(self, line):
        assert "!" in parse(line).constraints.banned

    def test_a_quoted_phrase_can_be_banned(self):
        assert "game changer" in parse('stop using "game changer"').constraints.banned

    def test_emoji_becomes_a_field_not_a_ban(self):
        constraints = parse("no emoji please").constraints
        assert constraints.emoji is False
        assert constraints.banned == ()

    @pytest.mark.parametrize(
        "line", ["never use exclamation marks", "always one line", "from now on, shorter"]
    )
    def test_rule_phrasing_marks_a_directive_standing(self, line):
        assert parse(line).standing

    def test_a_plain_adjustment_is_not_standing(self):
        assert not parse("shorter").standing

    def test_more_like_captures_the_number(self):
        assert parse("more like 2").like == 2
        assert parse("more like #3").like == 3

    def test_persona_switches_need_an_intent_verb(self):
        assert parse("try deadpan").persona == "deadpan"
        # A bare topic word must not hijack the voice.
        assert parse("leg day at the gym").persona is None

    def test_an_unrecognised_line_changes_nothing(self):
        directive = parse("finally quitting the job")
        assert not directive.recognised
        assert not directive.constraints
        assert directive.persona is None and directive.like is None

    def test_several_adjustments_combine_in_one_line(self):
        constraints = parse("shorter, no questions, and one line").constraints
        assert constraints.max_chars == 95
        assert constraints.questions is False
        assert constraints.max_sentences == 1
