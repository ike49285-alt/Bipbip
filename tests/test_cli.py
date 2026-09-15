import io

import pytest

from thirsttrap import personas
from thirsttrap.cli import build_parser, main


class TestPersonasCommand:
    def test_lists_every_persona_with_its_summary(self, capsys):
        assert main(["personas"]) == 0
        out = capsys.readouterr().out
        for name in personas.names():
            assert name in out
            assert personas.PERSONAS[name].summary in out


class TestScoreCommand:
    def test_single_post_prints_a_total_and_every_component(self, capsys):
        assert main(["score", "You already know the answer."]) == 0
        out = capsys.readouterr().out
        from thirsttrap.score import WEIGHTS

        for component in WEIGHTS:
            assert component in out

    def test_multiple_posts_are_ranked(self, capsys):
        code = main(
            [
                "score",
                "You already know the answer. You want permission.",
                "some thoughts #viral https://example.com",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert out.index("You already know") < out.index("some thoughts")

    def test_notes_surface_without_the_breakdown_flag(self, capsys):
        main(["score", "look at this #viral"])
        assert "hashtag" in capsys.readouterr().out

    def test_breakdown_flag_adds_component_bars(self, capsys):
        main(["score", "-b", "a post here", "another post there"])
        assert "hook" in capsys.readouterr().out

    def test_reads_lines_from_stdin_when_no_arguments(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("first post\n\nsecond post\n"))
        assert main(["score"]) == 0
        out = capsys.readouterr().out
        assert "first post" in out and "second post" in out

    def test_empty_stdin_is_an_error(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("   \n\n"))
        assert main(["score"]) == 2
        assert "no text" in capsys.readouterr().err


class TestGenCommand:
    def test_it_generates_locally_and_truncates_to_top_k(self, capsys):
        assert main(["gen", "finally quitting the job", "--pool", "40", "-k", "2"]) == 0
        out = capsys.readouterr().out
        assert "40 candidates, showing top 2" in out
        assert "1. " in out and "2. " in out

    def test_a_seed_makes_the_output_reproducible(self, capsys):
        main(["gen", "leg day", "--pool", "30", "--seed", "5"])
        first = capsys.readouterr().out
        main(["gen", "leg day", "--pool", "30", "--seed", "5"])
        assert capsys.readouterr().out == first

    def test_an_invalid_pool_exits_nonzero_with_the_reason(self, capsys):
        assert main(["gen", "a topic", "--pool", "0"]) == 2
        assert "pool must be at least 1" in capsys.readouterr().err

    def test_the_breakdown_flag_adds_components(self, capsys):
        main(["gen", "leg day", "--pool", "20", "-k", "1", "-b"])
        assert "hook" in capsys.readouterr().out


class TestParser:
    def test_no_command_is_a_usage_error(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_persona_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["gen", "topic", "-p", "smoulder"])

    def test_gen_defaults(self):
        args = build_parser().parse_args(["gen", "a topic"])
        assert args.pool == 400
        assert args.top == 3
        assert args.seed is None
        assert args.persona == personas.DEFAULT_PERSONA

    def test_chat_defaults_to_the_remembered_persona(self):
        # None here means "whatever the profile says", not the shipped default.
        assert build_parser().parse_args(["chat"]).persona is None
