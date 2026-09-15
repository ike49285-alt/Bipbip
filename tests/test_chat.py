import json

import pytest

from thirsttrap.chat import NO_MODEL_REPLY, Repl, Session, Turn
from thirsttrap.llm import Backend, BackendError, GrammarBackend
from thirsttrap.profile import Profile
from thirsttrap.score import WEIGHTS


class Fake(Backend):
    """A model that returns canned turns and records what it was sent."""

    def __init__(self, *replies):
        self.name = "fake"
        self.replies = list(replies)
        self.calls: list[dict] = []

    def available(self):
        return True

    def chat(self, system, messages, json_mode=False):
        self.calls.append(
            {"system": system, "messages": [dict(m) for m in messages], "json": json_mode}
        )
        if not self.replies:
            return json.dumps({"reply": "mm.", "posts": []})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    def describe(self):
        return "fake model"


def turn(reply, posts=()):
    return json.dumps({"reply": reply, "posts": list(posts)})


def make(*replies, **kwargs):
    top = kwargs.pop("top", 3)
    backend = Fake(*replies)
    session = Session(profile=Profile(path=None), backend=backend, **kwargs)
    return Repl(session=session, top=top), backend


def grammar_repl(**kwargs):
    top = kwargs.pop("top", 3)
    return Repl(
        session=Session(profile=Profile(path=None), backend=GrammarBackend(),
                        pool=120, seed=5, **kwargs),
        top=top,
    )


class TestConversation:
    def test_she_replies_without_being_given_a_topic(self):
        repl, _ = make(turn("about time. how did it feel walking out?"))
        output, keep_going = repl.handle("ugh I finally quit today")
        assert keep_going
        assert "how did it feel" in output

    def test_a_turn_with_no_drafts_is_normal(self):
        repl, _ = make(turn("how did it feel?"))
        repl.handle("I quit today")
        assert repl.shown == []

    def test_drafts_appear_when_she_writes_them(self):
        repl, _ = make(turn("that's the post.", ["You already know. You're stalling."]))
        output, _ = repl.handle("it felt anticlimactic")
        assert "You already know" in output
        assert len(repl.shown) == 1

    def test_history_accumulates_across_turns(self):
        repl, backend = make(turn("and then?"), turn("that's the post.", ["a draft"]))
        repl.handle("I quit today")
        repl.handle("it felt anticlimactic")
        roles = [m["role"] for m in backend.calls[1]["messages"]]
        assert roles == ["user", "assistant", "user"]

    def test_her_own_words_come_back_to_her(self):
        repl, backend = make(turn("how did it feel?"), turn("right.", []))
        repl.handle("I quit today")
        repl.handle("weird")
        assert backend.calls[1]["messages"][1]["content"].startswith("how did it feel?")

    def test_her_drafts_come_back_to_her_too(self):
        repl, backend = make(turn("here.", ["draft one"]), turn("ok", []))
        repl.handle("I quit today")
        repl.handle("hm")
        assert "draft one" in backend.calls[1]["messages"][1]["content"]

    def test_json_mode_is_requested(self):
        repl, backend = make(turn("hi"))
        repl.handle("I quit today")
        assert backend.calls[0]["json"] is True

    def test_a_plain_text_reply_is_kept_as_conversation(self):
        # Small models drift off JSON; losing her reply is worse than losing drafts.
        repl, _ = make("that's rough, what happened?")
        output, _ = repl.handle("I quit today")
        assert "what happened" in output
        assert repl.shown == []

    def test_history_is_trimmed_so_a_small_context_does_not_overflow(self):
        from thirsttrap.generate import MAX_HISTORY

        repl, backend = make(*[turn("mm") for _ in range(12)])
        for i in range(12):
            repl.handle(f"thing number {i}")
        assert len(backend.calls[-1]["messages"]) <= MAX_HISTORY

    def test_reset_clears_the_conversation_but_not_what_she_learned(self):
        repl, _ = make(turn("noted.", []))
        repl.handle("never use exclamation marks")
        repl.handle("/reset")
        assert repl.session.history == []
        assert repl.profile.standing.banned == ("!",)

    def test_again_re_asks_the_last_thing_said(self):
        repl, backend = make(turn("first answer"), turn("second answer"))
        repl.handle("I quit today")
        output, _ = repl.handle("/again")
        assert "second answer" in output
        assert backend.calls[1]["messages"][-1]["content"] == "I quit today"

    def test_again_before_anything_says_so(self):
        repl, _ = make()
        assert "nothing to ask again" in repl.handle("/again")[0]


class TestConstraints:
    def test_an_adjustment_is_passed_to_her_and_enforced(self):
        long_post = "x" * 200
        repl, backend = make(turn("sure.", ["tiny one", long_post]))
        repl.handle("much shorter")
        assert "at most 60 characters" in backend.calls[0]["system"]
        assert [d.text for d in repl.shown] == ["tiny one"]

    def test_a_stated_rule_is_adopted_and_announced(self):
        repl, _ = make(turn("noted.", []))
        assert "+ standing rule" in repl.handle("never use exclamation marks")[0]
        assert repl.profile.standing.banned == ("!",)

    def test_a_standing_rule_reaches_her_on_later_turns(self):
        repl, backend = make(turn("noted.", []), turn("here.", ["clean draft"]))
        repl.handle("never use exclamation marks")
        repl.handle("I quit today")
        assert "never say '!'" in backend.calls[1]["system"]

    def test_a_draft_breaking_a_standing_rule_is_dropped(self):
        repl, _ = make(turn("noted.", []), turn("here.", ["clean draft", "shouty draft!"]))
        repl.handle("never use exclamation marks")
        repl.handle("I quit today")
        assert [d.text for d in repl.shown] == ["clean draft"]

    def test_her_reply_survives_even_when_every_draft_is_dropped(self):
        repl, _ = make(turn("noted.", []), turn("still talking", ["nope!"]))
        repl.handle("never use exclamation marks")
        output, _ = repl.handle("I quit today")
        assert "still talking" in output


class TestReferences:
    def test_more_like_sends_her_the_text_not_the_number(self):
        # She never saw our numbering, so a bare "2" would mean nothing to her.
        repl, backend = make(turn("here.", ["alpha draft", "beta draft"]), turn("ok", []))
        repl.handle("I quit today")
        repl.handle("more like 2")
        sent = backend.calls[1]["messages"][-1]["content"]
        assert repl.shown is not None
        assert "draft" in sent and "referring to this draft" in sent

    def test_more_like_without_drafts_is_an_error(self):
        repl, _ = make()
        assert "no drafts yet" in repl.handle("more like 2")[0]

    def test_more_like_out_of_range_names_the_range(self):
        repl, _ = make(turn("here.", ["only one"]))
        repl.handle("I quit today")
        assert "pick 1-1" in repl.handle("more like 4")[0]


class TestFailures:
    def test_a_dead_model_keeps_the_session_alive(self):
        repl, _ = make(BackendError("connection refused"))
        output, keep_going = repl.handle("I quit today")
        assert keep_going and "connection refused" in output

    def test_a_failed_turn_is_not_left_in_the_history(self):
        repl, _ = make(BackendError("nope"), turn("fine"))
        repl.handle("I quit today")
        assert repl.session.history == []

    def test_the_backend_command_names_the_model(self):
        repl, _ = make()
        assert "fake model" in repl.handle("/backend")[0]


class TestWithoutAModel:
    def test_it_says_so_and_still_drafts(self):
        repl = grammar_repl()
        output, _ = repl.handle("finally quitting the job")
        assert NO_MODEL_REPLY.split(",")[0] in output
        assert repl.shown

    def test_the_backend_command_flags_the_fallback(self):
        assert "no local model found" in grammar_repl().handle("/backend")[0]

    def test_a_constraint_nothing_satisfies_is_explained_not_silent(self):
        from thirsttrap.directives import Constraints

        repl = grammar_repl()
        repl.handle("finally quitting the job")
        repl.profile.add_standing(Constraints(max_chars=1))
        output, keep_going = repl.handle("go on")
        assert keep_going
        assert "Nothing I can write fits" in output
        assert repl.shown == []


class TestKeeping:
    def test_keeping_teaches_and_persists(self):
        repl, _ = make(turn("here.", ["You already know. You're stalling.",
                                      "The weather turned cold this week."]))
        repl.handle("I quit today")
        output, _ = repl.handle("/keep 1")
        assert "kept (1 total)" in output
        assert repl.profile.keeps == 1
        assert repl.profile.weights != WEIGHTS

    def test_kept_drafts_reach_her_on_a_later_turn(self):
        repl, backend = make(turn("here.", ["a memorable draft"]), turn("ok", []))
        repl.handle("I quit today")
        repl.handle("/keep 1")
        repl.handle("what else")
        assert "a memorable draft" in backend.calls[1]["system"]

    def test_keep_before_any_drafts_is_an_error(self):
        repl, _ = make()
        assert "no drafts yet" in repl.handle("/keep 1")[0]

    def test_keep_out_of_range_names_the_range(self):
        repl, _ = make(turn("here.", ["only one"]))
        repl.handle("I quit today")
        assert "pick 1-1" in repl.handle("/keep 5")[0]

    def test_save_writes_kept_drafts(self, tmp_path):
        repl, _ = make(turn("here.", ["a draft worth keeping"]))
        repl.handle("I quit today")
        repl.handle("/keep 1")
        target = tmp_path / "kept.txt"
        assert "wrote 1" in repl.handle(f"/save {target}")[0]
        assert target.read_text().strip() == "a draft worth keeping"


class TestCommands:
    def test_blank_line_is_a_noop(self):
        repl, backend = make()
        assert repl.handle("   ") == ("", True)
        assert backend.calls == []

    def test_quit_stops_the_loop(self):
        repl, _ = make()
        assert repl.handle("/quit")[1] is False

    def test_unknown_command_is_not_sent_to_her(self):
        repl, backend = make()
        assert "unknown command" in repl.handle("/nonsense")[0]
        assert backend.calls == []

    def test_score_is_local_and_costs_no_turn(self):
        repl, backend = make()
        assert "hook" in repl.handle("/score You already know the answer.")[0]
        assert backend.calls == []

    def test_persona_switch_persists(self):
        repl, _ = make()
        repl.handle("/persona gym")
        assert repl.profile.persona == "gym"

    def test_persona_reaches_her_register(self):
        from thirsttrap import personas

        repl, backend = make(turn("hi"))
        repl.handle("/persona deadpan")
        repl.handle("I quit today")
        assert personas.get("deadpan").directive in backend.calls[0]["system"]

    def test_untune_restores_the_prior(self):
        repl, _ = make(turn("here.", ["draft a", "draft b"]))
        repl.handle("I quit today")
        repl.handle("/keep 1")
        repl.handle("/untune")
        assert repl.profile.weights == WEIGHTS


class TestPersistence:
    def test_rules_weights_and_voice_survive_a_restart(self, tmp_path):
        path = tmp_path / "profile.json"
        backend = Fake(turn("noted.", []), turn("here.", ["draft a", "draft b"]))
        first = Repl(session=Session(profile=Profile.load(path), backend=backend), top=3)
        first.handle("never use exclamation marks")
        first.handle("/persona gym")
        first.handle("I quit today")
        first.handle("/keep 1")

        second = Profile.load(path)
        assert second.persona == "gym"
        assert second.standing.banned == ("!",)
        assert second.keeps == 1
        assert second.weights == pytest.approx(first.profile.weights)
