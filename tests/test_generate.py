import types

import pytest

from thirsttrap import personas
from thirsttrap.generate import SYSTEM, Batch, Candidate, build_prompt, generate


class FakeMessages:
    """Stands in for client.messages, recording the kwargs it was called with."""

    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def make_response(texts, stop_reason="end_turn", explanation=None):
    return types.SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=types.SimpleNamespace(explanation=explanation),
        parsed_output=Batch(
            candidates=[Candidate(text=t, angle="test angle") for t in texts]
        ),
    )


class TestBuildPrompt:
    def test_prompt_carries_topic_count_and_voice(self):
        prompt = build_prompt("late trains", personas.get("deadpan"), 7)
        assert "late trains" in prompt
        assert "7" in prompt
        assert personas.get("deadpan").directive in prompt

    def test_prompt_does_not_leak_the_scoring_rubric(self):
        # The generator must stay ignorant of the scorer, or the score just
        # measures instruction-following. See generate.py's module docstring.
        from thirsttrap.score import WEIGHTS

        prompt = build_prompt("anything", personas.get("flirt"), 5)
        haystack = (prompt + SYSTEM).lower()
        for component in WEIGHTS:
            if component in {"length", "restraint"}:
                continue  # format constraints are stated on purpose
            assert component.replace("_", " ") not in haystack


class TestSystemPrompt:
    @pytest.mark.parametrize(
        "constraint", ["hashtags", "URLs", "280", "adults", "explicit"]
    )
    def test_hard_constraints_are_stated(self, constraint):
        assert constraint in SYSTEM


class TestGenerate:
    def test_returns_the_parsed_candidates(self):
        client = FakeClient(make_response(["one", "two"]))
        result = generate("a topic", n=2, client=client)
        assert [c.text for c in result] == ["one", "two"]

    def test_request_pins_the_model_and_voice(self):
        client = FakeClient(make_response(["one"]))
        generate("a topic", persona="gym", n=1, client=client)

        call = client.messages.calls[0]
        assert call["model"] == "claude-opus-5"
        assert call["output_format"] is Batch
        assert call["thinking"] == {"type": "adaptive"}
        assert personas.get("gym").directive in call["messages"][0]["content"]

    def test_sampling_parameters_are_never_sent(self):
        # temperature/top_p/top_k are rejected with a 400 on this model family.
        client = FakeClient(make_response(["one"]))
        generate("a topic", n=1, client=client)
        call = client.messages.calls[0]
        for param in ("temperature", "top_p", "top_k"):
            assert param not in call

    def test_unknown_persona_raises_before_any_request(self):
        client = FakeClient(make_response(["one"]))
        with pytest.raises(KeyError):
            generate("a topic", persona="smoulder", n=1, client=client)
        assert client.messages.calls == []

    @pytest.mark.parametrize("n", [0, -1])
    def test_non_positive_n_raises_before_any_request(self, n):
        client = FakeClient(make_response([]))
        with pytest.raises(ValueError):
            generate("a topic", n=n, client=client)
        assert client.messages.calls == []

    def test_refusal_raises_with_the_explanation(self):
        client = FakeClient(
            make_response([], stop_reason="refusal", explanation="declined the topic")
        )
        with pytest.raises(RuntimeError, match="declined the topic"):
            generate("a topic", n=1, client=client)

    def test_refusal_without_an_explanation_still_raises(self):
        response = make_response([], stop_reason="refusal")
        response.stop_details = None
        with pytest.raises(RuntimeError, match="declined"):
            generate("a topic", n=1, client=FakeClient(response))
