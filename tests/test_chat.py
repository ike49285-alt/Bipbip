import copy
import types

import pytest

from thirsttrap.chat import ChatBatch, Repl, Session
from thirsttrap.generate import Candidate
from thirsttrap.profile import Profile


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
        if isinstance(batch, dict) and "refusal" in batch:
            return types.SimpleNamespace(
                stop_reason="refusal",
                stop_details=types.SimpleNamespace(explanation=batch["refusal"]),
                parsed_output=None,
            )

        learned = []
        if isinstance(batch, dict):  # {"texts": [...], "learned": [...]}
            learned = batch.get("learned", [])
            batch = batch["texts"]

        return types.SimpleNamespace(
            stop_reason="end_turn",
            stop_details=None,
            parsed_output=ChatBatch(
                candidates=[Candidate(text=t, angle="a") for t in batch],
                learned=learned,
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
        session = Session(profile=Profile(persona="gym"), n=7, client=client)
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

        assert "kept (1 total" in repl.handle("/keep 1")[0]
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


class TestPassiveLearning:
    def test_durable_rule_is_absorbed_and_announced(self):
        repl, _ = make_repl({"texts": ["a post here"], "learned": ["no exclamation marks"]})
        output, _ = repl.handle("stop shouting at me")

        assert "+ learned: no exclamation marks" in output
        assert repl.profile.rules == ["no exclamation marks"]

    def test_learned_rules_steer_the_next_request(self):
        repl, client = make_repl(
            {"texts": ["one"], "learned": ["no exclamation marks"]},
            ["two"],
        )
        repl.handle("stop shouting")
        repl.handle("another topic")

        assert "no exclamation marks" in client.messages.calls[1]["system"]

    def test_turns_that_teach_nothing_announce_nothing(self):
        repl, _ = make_repl(["a post here"])
        output, _ = repl.handle("a topic")

        assert "+ learned" not in output
        assert repl.profile.rules == []

    def test_a_repeated_rule_is_not_announced_twice(self):
        repl, _ = make_repl(
            {"texts": ["one"], "learned": ["no exclamation marks"]},
            {"texts": ["two"], "learned": ["no exclamation marks"]},
        )
        repl.handle("stop shouting")
        second, _ = repl.handle("again please")

        assert "+ learned" not in second
        assert repl.profile.rules == ["no exclamation marks"]

    def test_rules_command_lists_them_numbered(self):
        repl, _ = make_repl({"texts": ["one"], "learned": ["rule one", "rule two"]})
        repl.handle("a topic")
        output, _ = repl.handle("/rules")

        assert "1. rule one" in output and "2. rule two" in output

    def test_rules_command_before_anything_is_learned(self):
        repl, _ = make_repl()
        assert "nothing learned yet" in repl.handle("/rules")[0]

    def test_forget_removes_a_rule_and_stops_sending_it(self):
        repl, client = make_repl(
            {"texts": ["one"], "learned": ["no exclamation marks"]},
            ["two"],
        )
        repl.handle("stop shouting")
        assert "forgot: no exclamation marks" in repl.handle("/forget 1")[0]

        repl.handle("another topic")
        assert "no exclamation marks" not in client.messages.calls[1]["system"]

    def test_forget_out_of_range_names_the_range(self):
        repl, _ = make_repl({"texts": ["one"], "learned": ["only rule"]})
        repl.handle("a topic")
        assert "pick 1-1" in repl.handle("/forget 9")[0]

    def test_forget_all_clears_them(self):
        repl, _ = make_repl({"texts": ["one"], "learned": ["rule one", "rule two"]})
        repl.handle("a topic")
        assert "forgot 2" in repl.handle("/forget all")[0]
        assert repl.profile.rules == []

    def test_forget_without_an_argument_explains_itself(self):
        repl, _ = make_repl()
        assert "usage:" in repl.handle("/forget")[0]

    def test_keeping_a_post_remembers_it_as_an_example(self):
        repl, _ = make_repl(["You already know the answer."])
        repl.handle("a topic")
        repl.handle("/keep 1")

        assert repl.profile.examples == ["You already know the answer."]

    def test_kept_examples_reach_the_next_request(self):
        repl, client = make_repl(["You already know the answer."], ["two"])
        repl.handle("a topic")
        repl.handle("/keep 1")
        repl.handle("another topic")

        assert "You already know the answer." in client.messages.calls[1]["system"]

    def test_the_scorer_never_leaks_into_the_learned_profile(self):
        # Feeding component names back into generation would hand the model the
        # rubric it is judged against -- the same trap generate.py avoids.
        from thirsttrap.score import WEIGHTS

        repl, client = make_repl(["You already know the answer."], ["two"])
        repl.handle("a topic")
        repl.handle("/keep 1")
        repl.handle("another topic")

        brief = repl.profile.brief()
        for component in WEIGHTS:
            if component in {"length", "restraint"}:
                continue  # stated on purpose as format constraints
            assert component.replace("_", " ") not in brief.lower()

    def test_reset_keeps_what_she_learned(self):
        repl, _ = make_repl({"texts": ["one"], "learned": ["no exclamation marks"]})
        repl.handle("stop shouting")
        repl.handle("/reset")

        assert repl.session.history == []
        assert repl.profile.rules == ["no exclamation marks"]

    def test_persona_switch_persists_to_the_profile(self):
        repl, _ = make_repl()
        repl.handle("/persona gym")
        assert repl.profile.persona == "gym"

    def test_batch_count_tracks_turns_not_commands(self):
        repl, _ = make_repl(["one"], ["two"])
        repl.handle("a topic")
        repl.handle("/rules")
        repl.handle("another topic")

        assert repl.profile.batches == 2

    def test_a_refusal_teaches_nothing(self):
        repl, _ = make_repl({"refusal": "declined that"})
        repl.handle("a topic")

        assert repl.profile.batches == 0
        assert repl.profile.rules == []

    def test_profile_command_reports_location_and_size(self):
        repl, _ = make_repl({"texts": ["one"], "learned": ["a rule"]})
        repl.handle("a topic")
        output, _ = repl.handle("/profile")

        assert "1 rule(s)" in output and "1 batch(es)" in output
