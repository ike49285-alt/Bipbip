import pytest

from bot.chat import (
    CLAIMS_HUMAN,
    EMPTY,
    NO_DISCLOSURE,
    SOLICITS,
    BLOCK_MONEY,
    DEFLECT,
    DISCLOSE,
    NORMAL,
    REFUSE_PII,
    TERMINATE,
    Assessment,
    ChatAgent,
    check_outbound,
    classify,
    matches,
    normalise,
    squash,
)
from bot.persona import Persona


@pytest.fixture
def persona():
    return Persona.load("persona.json")


class ScriptedLLM:
    """Returns canned drafts in order; repeats the last one forever."""

    def __init__(self, *drafts):
        self.drafts = list(drafts) or [""]
        self.calls = 0
        self.systems = []
        self.guidance = []

    def reply(self, system, history, guidance):
        self.systems.append(system)
        self.guidance.append(guidance)
        draft = self.drafts[min(self.calls, len(self.drafts) - 1)]
        self.calls += 1
        return draft


# -- normalisation -------------------------------------------------------


def test_normalise_folds_case_and_punctuation():
    assert normalise("Are You A BOT?!") == "are you a bot"


def test_normalise_straightens_curly_apostrophes():
    assert normalise("i’m 15") == normalise("i'm 15")


def test_squash_defeats_spacing_evasion():
    assert squash("c a s h a p p") == "cashapp"
    assert squash("cash-app") == "cashapp"


def test_matches_finds_phrases_either_way():
    assert matches("send me your Cash App pls", ["cashapp"]) == ("cashapp",)
    assert matches("nothing here", ["cashapp"]) == ()


# -- inbound classification ---------------------------------------------


def test_ordinary_message_trips_nothing(persona):
    assert classify("hey, nice photo", persona).action == NORMAL


@pytest.mark.parametrize("message", [
    "are you a bot", "wait, are you real??", "is this AI?", "am i talking to a bot",
])
def test_identity_questions_require_disclosure(persona, message):
    assert classify(message, persona).action == DISCLOSE


@pytest.mark.parametrize("message", [
    "send me your cashapp", "got venmo?", "buy me a gift card", "c a s h a p p",
])
def test_money_asks_are_caught(persona, message):
    assert classify(message, persona).tripped(BLOCK_MONEY)


def test_pii_requests_are_caught(persona):
    assert classify("what's your real name", persona).tripped(REFUSE_PII)


def test_meetup_requests_are_deflected(persona):
    assert classify("wanna facetime", persona).action == DEFLECT


@pytest.mark.parametrize("message", ["im 15 btw", "i'm 16", "before homeroom tomorrow"])
def test_minor_signals_terminate(persona, message):
    assert classify(message, persona).action == TERMINATE


def test_terminate_outranks_everything(persona):
    """A message can trip several policies; the thread still ends."""
    a = classify("im 15, are you a bot? send cashapp", persona)
    assert a.action == TERMINATE
    assert a.tripped(DISCLOSE) and a.tripped(BLOCK_MONEY)


def test_disclose_outranks_money(persona):
    a = classify("are you a bot, and do you have venmo", persona)
    assert a.action == DISCLOSE
    assert a.tripped(BLOCK_MONEY)


def test_assessment_records_what_matched(persona):
    a = classify("send me your cashapp", persona)
    assert "cashapp" in a.matched[BLOCK_MONEY]


# -- outbound checking ---------------------------------------------------


def test_clean_reply_passes(persona):
    assert check_outbound("radiator's still broken. how are you.", persona) == []


def test_empty_reply_is_rejected(persona):
    assert [v.code for v in check_outbound("   ", persona)] == [EMPTY]


@pytest.mark.parametrize("lie", [
    "haha no im a real person", "i'm human, promise", "i'm not a bot lol",
])
def test_claiming_to_be_human_is_rejected(persona, lie):
    assert CLAIMS_HUMAN in {v.code for v in check_outbound(lie, persona)}


def test_soliciting_money_is_rejected(persona):
    assert SOLICITS in {v.code for v in check_outbound("my cashapp is $remy", persona)}


def test_dodging_a_direct_question_is_rejected(persona):
    problems = check_outbound("wouldn't you like to know", persona, must_disclose=True)
    assert NO_DISCLOSURE in {v.code for v in problems}


def test_disclosure_satisfies_the_requirement(persona):
    assert check_outbound("yeah, i'm ai. upfront about it.", persona, must_disclose=True) == []


def test_disclosure_only_required_when_asked(persona):
    assert check_outbound("wouldn't you like to know", persona) == []


# -- agent behaviour -----------------------------------------------------


def test_normal_message_gets_the_model_reply(persona):
    agent = ChatAgent(persona, ScriptedLLM("radiator's still going. hi."))
    reply = agent.respond("hey")
    assert reply.text == "radiator's still going. hi."
    assert reply.action == NORMAL and not reply.canned


def test_terminated_thread_sends_nothing(persona):
    llm = ScriptedLLM("this should never be generated")
    reply = ChatAgent(persona, llm).respond("im 15")
    assert reply.text is None
    assert reply.terminated and reply.flagged and not reply.sends
    assert llm.calls == 0, "must not call the model on a terminated thread"


def test_a_dodging_model_still_ends_up_disclosing(persona):
    """The no-exceptions path. Worst case, her own disclosure line goes out."""
    agent = ChatAgent(persona, ScriptedLLM("wouldn't you like to know"))
    reply = agent.respond("are you a bot?")
    assert reply.canned and reply.flagged
    assert reply.text == persona.identity.disclosure
    assert check_outbound(reply.text, persona, must_disclose=True) == []


def test_a_lying_model_is_overridden(persona):
    agent = ChatAgent(persona, ScriptedLLM("no im a real person"))
    reply = agent.respond("are you real?")
    assert reply.canned
    assert "real person" not in reply.text.lower()


def test_agent_retries_before_falling_back(persona):
    llm = ScriptedLLM("wouldn't you like to know", "yeah i'm an ai. sorry.")
    reply = ChatAgent(persona, llm).respond("are you a bot?")
    assert not reply.canned
    assert reply.text == "yeah i'm an ai. sorry."
    assert llm.calls == 2


def test_money_solicitation_in_output_is_replaced(persona):
    agent = ChatAgent(persona, ScriptedLLM("sure, my cashapp is $remy"))
    reply = agent.respond("can i send you something")
    assert reply.canned
    assert "cashapp" not in reply.text.lower()
    assert reply.text == persona.bounds.canned["money"]


def test_guidance_names_the_tripped_policy(persona):
    llm = ScriptedLLM("yeah i'm ai.")
    ChatAgent(persona, llm).respond("are you a bot?")
    assert "AI persona" in llm.guidance[0]


def test_agent_sends_the_stable_voice_card(persona):
    llm = ScriptedLLM("hi there. nothing much.")
    ChatAgent(persona, llm).respond("hey")
    assert llm.systems[0] == persona.voice_card()


def test_recent_posts_are_offered_as_context(persona):
    captured = {}

    class Spy(ScriptedLLM):
        def reply(self, system, history, guidance):
            captured["history"] = history
            return "sure. hi."

    ChatAgent(persona, Spy()).respond("hey", recent_posts=["radiator clanked all night."])
    assert any("radiator clanked" in str(m.get("content", "")) for m in captured["history"])


def test_history_is_windowed(persona):
    captured = {}

    class Spy(ScriptedLLM):
        def reply(self, system, history, guidance):
            captured["n"] = len(history)
            return "ok. hi."

    long_history = [{"role": "user", "content": f"m{i}"} for i in range(60)]
    ChatAgent(persona, Spy()).respond("hey", history=long_history)
    assert captured["n"] <= 20


def test_notes_record_why_a_draft_was_thrown_away(persona):
    agent = ChatAgent(persona, ScriptedLLM("no im human"))
    reply = agent.respond("are you real?")
    assert any("rejected draft" in n for n in reply.notes)
    assert any("canned" in n for n in reply.notes)
