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
    def test_api_failure_exits_nonzero_with_the_reason(self, capsys, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("the model declined this topic")

        monkeypatch.setattr("thirsttrap.generate.generate", boom)
        assert main(["gen", "a topic"]) == 1
        assert "declined" in capsys.readouterr().err

    def test_ranks_and_truncates_to_the_requested_top_k(self, capsys, monkeypatch):
        import types

        def fake_generate(topic, persona, n):
            return [types.SimpleNamespace(text=f"candidate number {i}") for i in range(n)]

        monkeypatch.setattr("thirsttrap.generate.generate", fake_generate)
        assert main(["gen", "a topic", "-n", "6", "-k", "2"]) == 0
        out = capsys.readouterr().out
        assert "6 candidates, showing top 2" in out


class TestParser:
    def test_no_command_is_a_usage_error(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_persona_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["gen", "topic", "-p", "smoulder"])

    def test_gen_defaults(self):
        args = build_parser().parse_args(["gen", "a topic"])
        assert args.n == 12
        assert args.top == 3
        assert args.persona == personas.DEFAULT_PERSONA
