import copy
import types

import pytest

from thirsttrap.chat import Repl, Session
from thirsttrap.generate import Batch, Candidate


class FakeMessages:
    def __init__(self, batches):
        self._batches = list(batches)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        # Snapshot: Session passes its live history list, so recording the
        # reference would let later turns rewrite what this call "sent".
        self.calls.append(copy.deepcopy(kwargs))
        batch = self._batches.pop(0) if self._batches else ["fallback post"]
        if isinstance(batch, Exception):
            raise batch
        if isinstance(batch, dict):  # a refusal
            return types.SimpleNamespace(
                stop_reason="refusal",
                stop_details=types.SimpleNamespace(explanation=batch["refusal"]),
                parsed_output=None,
            )
        return types.SimpleNamespace(
            stop_reason="end_turn",
            stop_details=None,
            parsed_output=Batch(
                candidates=[Candidate(text=t, angle="a") for t in batch]
            ),
        )


class FakeClient:
    def __init__(self, *batches):
        self.messages = FakeMessages(batches)


def make_repl(*batches, **kwargs):
    client = FakeClient(*batches)
    return Repl(session=Session(client=client), **kwargs), client


class TestSession:
    def test_history_alternates_user_and_assistant(self):
        session = Session(client=FakeClient(["one post", "two post"], ["three post"]))
        session.send("a topic")
        session.send("shorter")
        assert [m["role"] for m in session.history] == [
            "user", "assistant", "user", "assistant",
        ]

    def test_assistant_turn_is_numbered_in_ranked_order(self):
        # The weak post must come back numbered 2, matching what the user sees.
        strong = "You already know the answer. You want permission."
        weak = "just some thoughts #viral https://example.com"
        session = Session(client=FakeClient([weak, strong]))

        ranked = session.send("a topic")
        assistant = session.history[-1]["content"]

        assert ranked[0].text == strong
        assert assistant.startswith("1. " + strong)
        assert "2. " + weak in assistant

    def test_full_history_is_resent_each_turn(self):
        client = FakeClient(["one"], ["two"])
        session = Session(client=client)
        session.send("a topic")
        session.send("shorter")
        assert len(client.messages.calls[1]["messages"]) == 3

    def test_persona_and_batch_size_reach_the_system_prompt(self):
        client = FakeClient(["one"])
        session = Session(persona="gym", n=7, client=client)
        session.send("a topic")

        system = client.messages.calls[0]["system"]
        from thirsttrap import personas

        assert personas.get("gym").directive in system
        assert "exactly 7" in system

    def test_caching_is_requested(self):
        client = FakeClient(["one"])
        Session(client=client).send("a topic")
        assert client.messages.calls[0]["cache_control"] == {"type": "ephemeral"}

    def test_sampling_parameters_are_never_sent(self):
        client = FakeClient(["one"])
        Session(client=client).send("a topic")
        for param in ("temperature", "top_p", "top_k"):
            assert param not in client.messages.calls[0]

    def test_refusal_raises_and_leaves_no_dangling_turn(self):
        session = Session(client=FakeClient({"refusal": "declined that"}))
        with pytest.raises(RuntimeError, match="declined that"):
            session.send("a topic")
        assert session.history == []

    def test_reset_clears_history(self):
        session = Session(client=FakeClient(["one"]))
        session.send("a topic")
        session.reset()
        assert session.history == []


class TestRepl:
    def test_plain_text_runs_a_turn(self):
        repl, _ = make_repl(["You already know the answer."])
        output, keep_going = repl.handle("a topic")
        assert keep_going
        assert "You already know the answer." in output

    def test_top_k_limits_what_is_shown(self):
        repl, _ = make_repl(["post one here", "post two here", "post three here"], top=2)
        output, _ = repl.handle("a topic")
        assert "showing top 2" in output
        assert len(repl.shown) == 2

    def test_blank_line_is_a_noop(self):
        repl, client = make_repl(["one"])
        assert repl.handle("   ") == ("", True)
        assert client.messages.calls == []

    def test_quit_stops_the_loop(self):
        repl, _ = make_repl()
        assert repl.handle("/quit")[1] is False

    def test_unknown_command_is_reported_not_sent(self):
        repl, client = make_repl(["one"])
        output, keep_going = repl.handle("/nonsense")
        assert keep_going and "unknown command" in output
        assert client.messages.calls == []

    def test_help_lists_commands(self):
        repl, _ = make_repl()
        assert "/persona" in repl.handle("/help")[0]

    def test_api_error_keeps_the_session_alive(self):
        repl, _ = make_repl(RuntimeError("network died"))
        output, keep_going = repl.handle("a topic")
        assert keep_going
        assert "network died" in output


class TestReplCommands:
    def test_persona_switch_reaches_the_next_request(self):
        repl, client = make_repl(["one"])
        assert "gym" in repl.handle("/persona gym")[0]
        repl.handle("a topic")

        from thirsttrap import personas

        assert personas.get("gym").directive in client.messages.calls[0]["system"]

    def test_invalid_persona_is_rejected_with_the_valid_names(self):
        repl, _ = make_repl()
        output, _ = repl.handle("/persona smoulder")
        assert "smoulder" in output and "flirt" in output
        assert repl.session.persona == "flirt"

    def test_bare_persona_reports_the_current_one(self):
        repl, _ = make_repl()
        assert "flirt" in repl.handle("/persona")[0]

    def test_score_is_local_and_costs_no_request(self):
        repl, client = make_repl()
        output, _ = repl.handle("/score You already know the answer.")
        assert "hook" in output
        assert client.messages.calls == []

    def test_breakdown_toggles(self):
        repl, _ = make_repl()
        assert "on" in repl.handle("/breakdown")[0]
        assert "off" in repl.handle("/breakdown")[0]

    def test_n_and_top_are_settable(self):
        repl, _ = make_repl()
        repl.handle("/n 5")
        repl.handle("/top 1")
        assert repl.session.n == 5
        assert repl.top == 1

    def test_non_numeric_argument_is_an_error_not_a_crash(self):
        repl, _ = make_repl()
        output, keep_going = repl.handle("/n lots")
        assert keep_going and output.startswith("error:")

    def test_keep_then_kept_then_drop(self):
        repl, _ = make_repl(["You already know the answer."])
        repl.handle("a topic")

        assert "kept (1 total)" in repl.handle("/keep 1")[0]
        assert "You already know" in repl.handle("/kept")[0]
        assert "dropped" in repl.handle("/drop 1")[0]
        assert repl.kept == []

    def test_keeping_the_same_post_twice_is_refused(self):
        repl, _ = make_repl(["You already know the answer."])
        repl.handle("a topic")
        repl.handle("/keep 1")
        assert "already kept" in repl.handle("/keep 1")[0]
        assert len(repl.kept) == 1

    def test_keep_before_any_batch_is_an_error(self):
        repl, _ = make_repl()
        assert "no batch yet" in repl.handle("/keep 1")[0]

    def test_keep_out_of_range_names_the_range(self):
        repl, _ = make_repl(["only post here"])
        repl.handle("a topic")
        assert "pick 1-1" in repl.handle("/keep 4")[0]

    def test_keep_resolves_against_displayed_numbering(self):
        strong = "You already know the answer. You want permission."
        weak = "just some thoughts #viral"
        repl, _ = make_repl([weak, strong])
        repl.handle("a topic")
        repl.handle("/keep 1")
        assert repl.kept[0].text == strong

    def test_save_writes_kept_posts(self, tmp_path):
        repl, _ = make_repl(["You already know the answer."])
        repl.handle("a topic")
        repl.handle("/keep 1")

        target = tmp_path / "kept.txt"
        assert "wrote 1" in repl.handle(f"/save {target}")[0]
        assert target.read_text().strip() == "You already know the answer."

    def test_save_with_nothing_kept_says_so(self):
        repl, _ = make_repl()
        assert "nothing kept" in repl.handle("/save out.txt")[0]

    def test_save_to_an_unwritable_path_is_an_error_not_a_crash(self, tmp_path):
        repl, _ = make_repl(["a post here"])
        repl.handle("a topic")
        repl.handle("/keep 1")
        output, keep_going = repl.handle(f"/save {tmp_path / 'missing' / 'x.txt'}")
        assert keep_going and output.startswith("error:")

    def test_again_reuses_the_last_instruction(self):
        repl, client = make_repl(["first"], ["second"])
        repl.handle("a topic")
        repl.handle("/again")
        assert client.messages.calls[1]["messages"][-1]["content"] == "a topic"

    def test_again_before_anything_says_so(self):
        repl, _ = make_repl()
        assert "nothing to re-run" in repl.handle("/again")[0]

    def test_reset_clears_the_conversation_but_not_the_kept_list(self):
        repl, _ = make_repl(["You already know the answer."])
        repl.handle("a topic")
        repl.handle("/keep 1")
        repl.handle("/reset")

        assert repl.session.history == []
        assert repl.shown == []
        assert len(repl.kept) == 1
