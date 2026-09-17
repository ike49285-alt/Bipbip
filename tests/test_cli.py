import pytest

from bot.__main__ import build_parser, main


def run(capsys, *argv):
    main(list(argv))
    return capsys.readouterr().out


def test_check_reports_the_persona(capsys):
    out = run(capsys, "check")
    assert "Remy" in out and "ok" in out
    assert "352 combinations" in out


def test_check_flags_a_missing_anchor(capsys, tmp_path):
    import json, shutil
    p = tmp_path / "persona.json"
    raw = json.loads(open("persona.json").read())
    raw["visual"]["anchor_image"] = "content/anchor/nope.png"
    p.write_text(json.dumps(raw))
    out = run(capsys, "--persona", str(p), "check")
    assert "MISSING" in out


def test_check_rejects_a_broken_persona(tmp_path):
    p = tmp_path / "persona.json"
    p.write_text("{not json")
    with pytest.raises(SystemExit, match="persona:"):
        main(["--persona", str(p), "check"])


def test_prompts_emits_the_requested_count(capsys):
    out = run(capsys, "prompts", "-n", "3", "--seed", "1")
    assert out.count("negative:") == 3


def test_prompts_json_is_machine_readable(capsys):
    import json
    out = run(capsys, "prompts", "-n", "2", "--seed", "1", "--json")
    assert len(json.loads(out)) == 2


def test_prompts_subject_override_reaches_the_prompt(capsys):
    out = run(capsys, "prompts", "-n", "1", "--subject", "a tall redhead")
    assert "a tall redhead" in out


def test_timeline_is_empty_on_a_fresh_db(capsys, tmp_path):
    out = run(capsys, "--db", str(tmp_path / "t.db"), "timeline")
    assert "timeline is empty" in out


def test_timeline_prints_captions(capsys, tmp_path):
    from pathlib import Path
    from bot.driver import LocalDriver

    db = tmp_path / "t.db"
    with LocalDriver(db) as d:
        d.post(Path("a.png"), "third coffee. black.")
    out = run(capsys, "--db", str(db), "timeline")
    assert "third coffee. black." in out


def test_caption_refuses_an_empty_directory(tmp_path):
    with pytest.raises(SystemExit, match="no images"):
        main(["caption", str(tmp_path)])


def test_bare_invocation_defaults_to_chat():
    args = build_parser().parse_args(["chat"])
    assert args.command == "chat"


def test_every_subcommand_is_wired():
    parser = build_parser()
    for command in ("chat", "prompts", "caption", "timeline", "check"):
        args = parser.parse_args([command] + (["d"] if command == "caption" else []))
        assert callable(args.func)


def test_direct_is_wired_with_a_note():
    parser = build_parser()
    args = parser.parse_args(["direct", "too whiny"])
    assert args.note == "too whiny" and callable(args.func)
