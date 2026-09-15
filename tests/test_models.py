"""Backends, and the cleanup that decides whether their output is usable.

The cleaner is the only part of a model integration that can be tested hard
without a model, and it is also where the real bugs are. An instruction-tuned
open model asked for one post returns the post wrapped in an announcement, a
pair of quote marks and an offer of further service - and every one of those
wrappers would pass the content gates untouched, because none of them is too
long, contradicts canon, or claims to be human. They would simply go out.

So this file pushes in both directions. Under-cleaning ships "Sure! Here's a
post:" to a timeline. OVER-cleaning is worse, because it mangles a good post
into something that still reads like the character and nobody notices.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from synth.models import (ModelError, OllamaCompleter, OpenAICompatCompleter,
                          clean_completion, from_env)


# -- things that must be stripped ------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("Sure! Here's a post:\n\nmade a playlist", "made a playlist"),
    ("Here is the post:\n\nmade a playlist", "made a playlist"),
    ("Here's your caption:\nmade a playlist", "made a playlist"),
    ("Okay! made a playlist", "made a playlist"),
    ("Post: made a playlist", "made a playlist"),
    ("Caption:\nmade a playlist", "made a playlist"),
    ('"made a playlist"', "made a playlist"),
    ("“made a playlist”", "made a playlist"),
    ("made a playlist\n\nLet me know if you want another!", "made a playlist"),
    ("made a playlist\nHope this helps!", "made a playlist"),
    ("made a playlist\n\nWould you like a different tone?", "made a playlist"),
    ("```\nmade a playlist\n```", "made a playlist"),
    ("```text\nmade a playlist\n```", "made a playlist"),
])
def test_wrappers_are_stripped(raw, want):
    assert clean_completion(raw) == want


def test_several_layers_at_once():
    """The realistic case - a model does all of them in one answer."""
    raw = ('Sure! Here\'s a flirty post for Sophie:\n\n'
           '"made a playlist for rainy afternoons."\n\n'
           'Let me know if you\'d like a different tone!')
    assert clean_completion(raw) == "made a playlist for rainy afternoons."


# -- things that must SURVIVE ----------------------------------------------

@pytest.mark.parametrize("text", [
    "made a playlist. the annoying part was the ordering.",
    "okay so I have thoughts about your playlist",         # legit "okay so"
    "here's the thing though, that song is a war crime",   # legit "here's"
    'she said "no" and I respected that',                  # internal quotes
    "sure, and I'd do it again",                           # legit "sure,"
    "listen. LISTEN. the bridge is the whole song",
    "post-rock is not a genre it is a threat",             # starts with "post"
])
def test_real_posts_are_not_mangled(text):
    assert clean_completion(text) == text


def test_an_unmatched_quote_is_left_alone():
    """A post ending on a quoted phrase keeps its punctuation - only a matched
    pair wrapping the WHOLE thing is a wrapper."""
    assert clean_completion('he called it "fine"') == 'he called it "fine"'


def test_a_multi_line_post_keeps_its_lines():
    raw = "two things.\n\none: no.\ntwo: absolutely not."
    assert clean_completion(raw) == raw


def test_empty_input_is_empty_output():
    assert clean_completion("") == ""
    assert clean_completion("   \n  ") == ""


# -- the backends ----------------------------------------------------------

def test_ollama_sends_the_documented_shape(monkeypatch):
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen.update(url=url, payload=payload)
        return {"response": "Here's a post:\n\nmade a playlist"}

    monkeypatch.setattr("synth.models._post", fake_post)
    out = OllamaCompleter(model="llama3.1:8b")("write something")
    assert seen["url"].endswith("/api/generate")
    assert seen["payload"]["stream"] is False
    assert seen["payload"]["model"] == "llama3.1:8b"
    assert out == "made a playlist"          # cleaned on the way out


def test_openai_compatible_sends_the_documented_shape(monkeypatch):
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": '"made a playlist"'}}]}

    monkeypatch.setattr("synth.models._post", fake_post)
    out = OpenAICompatCompleter(model="mistral", base_url="http://x:8000/",
                                api_key="k")("write something")
    assert seen["url"] == "http://x:8000/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert seen["payload"]["messages"][0]["role"] == "user"
    assert out == "made a playlist"


def test_no_api_key_means_no_auth_header(monkeypatch):
    seen = {}
    monkeypatch.setattr("synth.models._post",
                        lambda u, p, h, t: seen.update(headers=h) or
                        {"choices": [{"message": {"content": "x"}}]})
    OpenAICompatCompleter(model="m")("hi")
    assert "Authorization" not in seen["headers"]


def test_an_empty_choices_list_RAISES_rather_than_returning_nothing(monkeypatch):
    """A silent empty string would flow into the pipeline as an empty draft
    and be blocked there, which reports the wrong problem."""
    monkeypatch.setattr("synth.models._post", lambda *a: {"choices": []})
    with pytest.raises(ModelError, match="no choices"):
        OpenAICompatCompleter(model="m")("hi")


# -- configuration ---------------------------------------------------------

def test_no_configuration_returns_None_rather_than_raising(monkeypatch):
    """Every command must keep working with no model, so the gates and the
    persona tooling can be reviewed before anyone stands a server up."""
    monkeypatch.delenv("SYNTH_MODEL", raising=False)
    assert from_env() is None


def test_the_backend_is_chosen_by_env(monkeypatch):
    monkeypatch.setenv("SYNTH_MODEL", "llama3.1:8b")
    monkeypatch.delenv("SYNTH_BACKEND", raising=False)
    monkeypatch.delenv("SYNTH_BASE_URL", raising=False)
    assert isinstance(from_env(), OllamaCompleter)
    monkeypatch.setenv("SYNTH_BACKEND", "openai")
    assert isinstance(from_env(), OpenAICompatCompleter)


def test_an_explicit_model_name_beats_the_env(monkeypatch):
    monkeypatch.setenv("SYNTH_MODEL", "from-env")
    monkeypatch.delenv("SYNTH_BACKEND", raising=False)
    assert from_env("explicit").model == "explicit"


def test_temperature_defaults_high_because_a_voice_needs_it(monkeypatch):
    monkeypatch.setenv("SYNTH_MODEL", "m")
    monkeypatch.delenv("SYNTH_TEMP", raising=False)
    monkeypatch.delenv("SYNTH_BACKEND", raising=False)
    assert from_env().temperature >= 0.8
    monkeypatch.setenv("SYNTH_TEMP", "0.3")
    assert from_env().temperature == 0.3
