import pytest

from thirsttrap.chat import Repl, Session
from thirsttrap.llm import Backend, GrammarBackend
from thirsttrap.profile import Profile
from thirsttrap.score import WEIGHTS


def make_repl(**kwargs):
    """Pinned to the grammar backend so these stay deterministic even on a
    machine with Ollama running."""
    top = kwargs.pop("top", 3)
    kwargs.setdefault("backend", GrammarBackend())
    return Repl(session=Session(pool=120, seed=11, **kwargs), top=top)


class FakeBackend(Backend):
    def __init__(self, *replies):
        self.name = "fake"
        self.replies = list(replies)
        self.calls = []

    def available(self):
        return True

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        return self.replies.pop(0) if self.replies else ""

    def describe(self):
        return "fake backend"


class TestSession:
    def test_the_first_line_becomes_the_topic(self):
        session = Session(pool=60, seed=1, backend=GrammarBackend())
        session.send("finally quitting the job")
        assert session.topic == "finally quitting the job"

    def test_an_adjustment_before_a_topic_is_an_error(self):
        with pytest.raises(ValueError, match="what the posts should be about"):
            Session(pool=60, seed=1, backend=GrammarBackend()).send("shorter")

    def test_an_adjustment_keeps_the_topic(self):
        session = Session(pool=120, seed=1, backend=GrammarBackend())
        session.send("finally quitting the job")
        session.send("shorter")
        assert session.topic == "finally quitting the job"

    def test_an_adjustment_constrains_the_batch(self):
        session = Session(pool=200, seed=2, backend=GrammarBackend())
        session.send("finally quitting the job")
        ranked = session.send("much shorter")
        assert all(len(r.text) <= 60 for r in ranked)

    def test_adjustments_accumulate_within_a_topic(self):
        session = Session(pool=250, seed=3, backend=GrammarBackend())
        session.send("finally quitting the job")
        session.send("shorter")
        ranked = session.send("no questions")
        assert all(len(r.text) <= 95 and "?" not in r.text for r in ranked)

    def test_a_new_topic_drops_the_previous_adjustments(self):
        session = Session(pool=200, seed=4, backend=GrammarBackend())
        session.send("finally quitting the job")
        session.send("much shorter")
        session.send("leg day")
        assert session.turn_constraints.max_chars is None

    def test_a_standing_rule_survives_a_new_topic(self):
        session = Session(pool=200, seed=5, backend=GrammarBackend())
        session.send("finally quitting the job")
        session.send("never use exclamation marks")
        ranked = session.send("leg day")
        assert all("!" not in r.text for r in ranked)

    def test_impossible_constraints_say_so_rather_than_relaxing(self):
        session = Session(pool=80, seed=6, backend=GrammarBackend())
        session.send("finally quitting the job")
        with pytest.raises(ValueError, match="nothing survived"):
            session.send('never say "the"')

    def test_a_persona_switch_is_recognised_mid_conversation(self):
        session = Session(pool=80, seed=7, backend=GrammarBackend())
        session.send("finally quitting the job")
        session.send("try deadpan")
        assert session.persona == "deadpan"

    def test_clear_drops_turn_constraints_but_not_standing_ones(self):
        session = Session(pool=120, seed=8, backend=GrammarBackend())
        session.send("finally quitting the job")
        session.send("shorter")
        session.send("never use exclamation marks")
        session.clear()
        assert session.turn_constraints.max_chars is None
        assert session.constraints().banned == ("!",)

    def test_a_pinned_seed_makes_a_session_reproducible(self):
        first = Session(pool=100, seed=9, backend=GrammarBackend()).send("finally quitting the job")
        second = Session(pool=100, seed=9, backend=GrammarBackend()).send("finally quitting the job")
        assert [r.text for r in first] == [r.text for r in second]

    def test_repeating_a_turn_redraws(self):
        session = Session(pool=150, seed=10, backend=GrammarBackend())
        first = session.send("finally quitting the job")
        again = session.send("finally quitting the job")
        assert [r.text for r in first] != [r.text for r in again]

    def test_batches_are_counted(self):
        session = Session(pool=60, seed=1, backend=GrammarBackend())
        session.send("finally quitting the job")
        session.send("shorter")
        assert session.profile.batches == 2


class TestRepl:
    def test_a_topic_produces_a_ranked_batch(self):
        repl = make_repl()
        output, keep_going = repl.handle("finally quitting the job")
        assert keep_going and "showing top 3" in output
        assert len(repl.shown) == 3

    def test_active_constraints_are_shown_in_the_header(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        assert "at most 95 characters" in repl.handle("shorter")[0]

    def test_blank_line_is_a_noop(self):
        assert make_repl().handle("   ") == ("", True)

    def test_quit_stops_the_loop(self):
        assert make_repl().handle("/quit")[1] is False

    def test_unknown_command_is_reported(self):
        assert "unknown command" in make_repl().handle("/nonsense")[0]

    def test_an_impossible_ask_keeps_the_session_alive(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        output, keep_going = repl.handle('never say "the"')
        assert keep_going and output.startswith("error:")

    def test_more_like_needs_a_batch_first(self):
        assert "no batch yet" in make_repl().handle("more like 2")[0]

    def test_more_like_out_of_range_names_the_range(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        assert "pick 1-3" in repl.handle("more like 9")[0]

    def test_more_like_pulls_the_batch_toward_that_post(self):
        from thirsttrap.rank import similarity

        repl = make_repl(top=5)
        repl.handle("finally quitting the job")
        target = repl.shown[4].text
        repl.handle("more like 5")
        assert similarity(repl.shown[0].text, target) > 0.5

    def test_more_like_does_not_just_hand_back_the_same_post(self):
        repl = make_repl(top=5)
        repl.handle("finally quitting the job")
        target = repl.shown[4].text
        repl.handle("more like 5")
        assert repl.shown[0].text != target


class TestReplCommands:
    def test_persona_switch_persists_to_the_profile(self):
        repl = make_repl()
        repl.handle("/persona gym")
        assert repl.profile.persona == "gym"

    def test_invalid_persona_lists_the_valid_ones(self):
        repl = make_repl()
        assert "flirt" in repl.handle("/persona smoulder")[0]

    def test_rules_before_any_are_stated(self):
        assert "no standing rules" in make_repl().handle("/rules")[0]

    def test_a_stated_rule_is_adopted_announced_and_listed(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        assert "+ standing rule" in repl.handle("never use exclamation marks")[0]
        assert "never say '!'" in repl.handle("/rules")[0]

    def test_forget_clears_standing_rules(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        repl.handle("never use exclamation marks")
        assert "dropped 1" in repl.handle("/forget")[0]
        assert repl.profile.standing.describe() == []

    def test_weights_render_with_the_confidence_label(self):
        assert "untuned" in make_repl().handle("/weights")[0]

    def test_keeping_teaches_and_says_so(self):
        repl = make_repl(top=5)
        repl.handle("finally quitting the job")
        output, _ = repl.handle("/keep 5")
        assert "kept (1 total)" in output
        assert repl.profile.keeps == 1

    def test_keeping_moves_the_weights(self):
        repl = make_repl(top=5)
        repl.handle("finally quitting the job")
        repl.handle("/keep 5")
        assert repl.profile.weights != WEIGHTS

    def test_untune_restores_the_shipped_prior(self):
        repl = make_repl(top=5)
        repl.handle("finally quitting the job")
        repl.handle("/keep 5")
        repl.handle("/untune")
        assert repl.profile.weights == WEIGHTS

    def test_keeping_the_same_post_twice_is_refused(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        repl.handle("/keep 1")
        assert "already kept" in repl.handle("/keep 1")[0]
        assert repl.profile.keeps == 1

    def test_keep_before_a_batch_is_an_error(self):
        assert "no batch yet" in make_repl().handle("/keep 1")[0]

    def test_keep_out_of_range_names_the_range(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        assert "pick 1-3" in repl.handle("/keep 9")[0]

    def test_kept_posts_are_remembered_as_examples(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        repl.handle("/keep 1")
        assert repl.profile.examples == [repl.kept[0].text]

    def test_kept_then_dropped(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        repl.handle("/keep 1")
        assert "1." in repl.handle("/kept")[0]
        assert "dropped" in repl.handle("/drop 1")[0]
        assert repl.kept == []

    def test_save_writes_kept_posts(self, tmp_path):
        repl = make_repl()
        repl.handle("finally quitting the job")
        repl.handle("/keep 1")
        target = tmp_path / "kept.txt"
        assert "wrote 1" in repl.handle(f"/save {target}")[0]
        assert target.read_text().strip() == repl.kept[0].text

    def test_save_to_an_unwritable_path_is_an_error_not_a_crash(self, tmp_path):
        repl = make_repl()
        repl.handle("finally quitting the job")
        repl.handle("/keep 1")
        output, keep_going = repl.handle(f"/save {tmp_path / 'missing' / 'x.txt'}")
        assert keep_going and output.startswith("error:")

    def test_score_uses_the_learned_weights(self):
        repl = make_repl()
        assert "hook" in repl.handle("/score You already know the answer.")[0]

    def test_pool_and_top_are_settable(self):
        repl = make_repl()
        repl.handle("/pool 50")
        repl.handle("/top 1")
        assert repl.session.pool == 50 and repl.top == 1

    def test_non_numeric_argument_is_an_error_not_a_crash(self):
        output, keep_going = make_repl().handle("/pool lots")
        assert keep_going and output.startswith("error:")

    def test_clear_relaxes_this_turn(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        repl.handle("much shorter")
        assert "dropped" in repl.handle("/clear")[0]

    def test_again_reruns_the_last_line(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        assert "showing top" in repl.handle("/again")[0]

    def test_again_before_anything_says_so(self):
        assert "nothing to re-run" in make_repl().handle("/again")[0]

    def test_profile_reports_location_and_state(self):
        repl = make_repl()
        repl.handle("finally quitting the job")
        assert "batch(es)" in repl.handle("/profile")[0]


class TestPersistenceAcrossSessions:
    def test_rules_weights_and_voice_survive_a_restart(self, tmp_path):
        path = tmp_path / "profile.json"

        first = Repl(session=Session(profile=Profile.load(path), pool=150, seed=1, backend=GrammarBackend()), top=5)
        first.handle("finally quitting the job")
        first.handle("never use exclamation marks")
        first.handle("/persona gym")
        first.handle("/keep 5")

        second = Profile.load(path)
        assert second.persona == "gym"
        assert second.standing.banned == ("!",)
        assert second.keeps == 1
        assert second.weights == pytest.approx(first.profile.weights)


class TestWithAModelBackend:
    def test_a_topic_is_generated_by_the_model(self):
        backend = FakeBackend("You already know. You're stalling.\nNobody tells you.")
        repl = Repl(session=Session(backend=backend, n=5), top=3)
        output, _ = repl.handle("finally quitting the job")
        assert "You already know" in output
        assert backend.calls, "the model was never called"

    def test_the_topic_reaches_the_prompt(self):
        backend = FakeBackend("a post\nanother post")
        repl = Repl(session=Session(backend=backend, n=5))
        repl.handle("finally quitting the job")
        assert "finally quitting the job" in backend.calls[0][1]

    def test_standing_rules_reach_the_model_on_the_next_turn(self):
        backend = FakeBackend("a post\nanother", "third post\nfourth")
        repl = Repl(session=Session(backend=backend, n=2))
        repl.handle("finally quitting the job")
        repl.handle("never use exclamation marks")
        assert "never say '!'" in backend.calls[1][0]

    def test_kept_examples_reach_the_model_on_the_next_turn(self):
        backend = FakeBackend("a post\nanother", "third post\nfourth")
        repl = Repl(session=Session(backend=backend, n=2))
        repl.handle("finally quitting the job")
        repl.handle("/keep 1")
        repl.handle("leg day")
        assert "a post" in backend.calls[1][0]

    def test_an_adjustment_reaches_the_prompt_and_is_enforced(self):
        backend = FakeBackend("a post\nanother", "short\ntiny\n" + "x" * 300)
        repl = Repl(session=Session(backend=backend, n=2))
        repl.handle("finally quitting the job")
        repl.handle("much shorter")
        assert "at most 60 characters" in backend.calls[1][1]
        assert all(len(item.text) <= 60 for item in repl.shown)

    def test_a_dead_backend_keeps_the_session_alive(self):
        from thirsttrap.llm import BackendError

        class Dead(Backend):
            def available(self):
                return True

            def complete(self, system, prompt):
                raise BackendError("connection refused")

            def describe(self):
                return "dead"

        repl = Repl(session=Session(backend=Dead(), n=5))
        output, keep_going = repl.handle("finally quitting the job")
        assert keep_going and "connection refused" in output

    def test_the_backend_command_names_the_engine(self):
        repl = Repl(session=Session(backend=FakeBackend(), n=5))
        assert "fake backend" in repl.handle("/backend")[0]

    def test_the_grammar_fallback_is_flagged_as_such(self):
        assert "no local model found" in make_repl().handle("/backend")[0]
