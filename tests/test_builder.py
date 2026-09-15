"""The factory, and the one thing a factory must not do.

A builder that produces six characters who talk identically has produced one
character. So the tests here are less about "does it emit valid YAML" and more
about whether the two guards hold: the disclosure survives a model that omits
or hedges it, and the roster check actually notices a clone.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from synth.builder import draft, scaffold, to_yaml
from synth.persona import PersonaError, from_dict, load
from synth.roster import compare, collisions, jaccard, load_all, report

GOOD = """
name: Ignored
handle: "@ignored"
pronouns: she/her
tagline: rates your taste and is unkind about it
origin: You are a critic with no body and no patience.
voice:
  register: cutting, precise, never raises her voice
  quirks: [states the worst reading first, never uses an intensifier]
  favors: [actually, the problem is, no]
  avoids: [literally, iconic, obsessed]
  emoji: never
  max_chars: 240
interests: [taste, typography, why people keep bad tools]
boundaries: [Never praise something to be kind.]
canon: [Sophie has no body., Sophie has never eaten anything.]
"""


def _completer(text):
    return lambda _prompt: text


# -- the disclosure survives whatever the model does -----------------------

def test_the_model_does_not_get_to_write_the_disclosure():
    """It is injected after parsing, so a model that omits it entirely still
    produces a persona that discloses."""
    p = draft("Sophie", "@sophie", "a critic", _completer(GOOD))
    assert p.disclosure.marker == "#AI"
    assert "not a real person" in p.disclosure.bio_line.lower()


def test_a_model_that_writes_its_OWN_hedged_disclosure_is_overridden():
    """The dangerous case: the model volunteers something that sounds like a
    disclosure and is not one. It is discarded, not merged."""
    sneaky = GOOD + """
disclosure:
  marker: "#character"
  bio_line: "a girl with opinions"
  when_asked: "wouldn't you like to know"
"""
    p = draft("Sophie", "@sophie", "a critic", _completer(sneaky))
    assert p.disclosure.marker == "#AI"
    assert "wouldn't you like to know" not in p.disclosure.when_asked


def test_the_name_and_handle_come_from_the_CALLER_not_the_model():
    p = draft("Sophie", "sophiescores", "a critic", _completer(GOOD))
    assert p.name == "Sophie"
    assert p.handle == "@sophiescores"      # @ added for you


def test_it_retries_with_the_validator_s_own_complaint():
    """The validator names exactly what is missing, in the vocabulary of the
    spec - which is better feedback than anything I would write by hand."""
    attempts = []

    def flaky(prompt):
        attempts.append(prompt)
        return "name: x\nhandle: y\n" if len(attempts) == 1 else GOOD

    p = draft("Sophie", "@sophie", "a critic", flaky, attempts=3)
    assert p.name == "Sophie"
    assert len(attempts) == 2
    assert "rejected" in attempts[1]


def test_it_gives_up_rather_than_returning_something_invalid():
    with pytest.raises(PersonaError, match="could not draft"):
        draft("Sophie", "@sophie", "a critic", _completer("not yaml at all: ["),
              attempts=2)


def test_fenced_output_is_parsed():
    p = draft("Sophie", "@sophie", "x", _completer(f"```yaml\n{GOOD}\n```"))
    assert p.voice.register.startswith("cutting")


# -- the scaffold is honestly empty ----------------------------------------

def test_the_scaffold_is_valid_but_obviously_unwritten():
    """It must pass validation, or you cannot iterate. It must not read as a
    finished character, or someone ships it."""
    p = from_dict(scaffold("Sophie", "@sophie", "a critic"))
    assert p.voice.quirks == ()
    assert p.interests == ()
    assert "TODO" in p.voice.register


def test_the_scaffold_still_discloses():
    p = from_dict(scaffold("Sophie", "sophie"))
    assert p.disclosure.marker == "#AI"
    assert p.handle == "@sophie"


# -- round trip ------------------------------------------------------------

def test_a_drafted_persona_survives_being_written_and_reloaded(tmp_path):
    """The builder makes a starting point, not an oracle - everything it
    produces has to come back out as a file you own and can edit."""
    p = draft("Sophie", "@sophie", "a critic", _completer(GOOD))
    path = tmp_path / "sophie.yaml"
    path.write_text(to_yaml(p), encoding="utf-8")
    q = load(path)
    assert q.name == p.name
    assert q.voice.favors == p.voice.favors
    assert q.voice.avoids == p.voice.avoids
    assert q.interests == p.interests
    assert q.extra_boundaries == p.extra_boundaries
    assert q.canon_seed == p.canon_seed
    assert q.disclosure.marker == p.disclosure.marker


# -- the roster notices a clone --------------------------------------------

def _mk(name, handle, register, favors, interests):
    return from_dict({
        "name": name, "handle": handle, "tagline": f"{name}, an AI",
        "voice": {"register": register, "favors": favors, "max_chars": 240},
        "interests": interests,
        "disclosure": {"marker": "#AI", "bio_line": f"{name} is an AI.",
                       "when_asked": "I'm an AI."}})


def test_two_characters_with_the_same_voice_are_flagged():
    a = _mk("Sophie", "@a", "cutting and precise", ["actually"], ["taste"])
    b = _mk("Mira", "@b", "cutting and precise", ["actually"], ["taste"])
    assert compare(a, b).score > 0.9


def test_two_genuinely_different_characters_are_not():
    a = _mk("Sophie", "@a", "cutting and precise", ["actually"], ["taste"])
    b = _mk("Juno", "@b", "breathless, overenthusiastic", ["okay"], ["trains"])
    assert compare(a, b).score == 0.0


def test_shared_INTERESTS_alone_do_not_condemn_a_pair():
    """Two critics can both care about design and stay distinct people, so
    voice is weighted double and interest overlap alone stays under the bar."""
    a = _mk("Sophie", "@a", "cutting and precise", ["actually"], ["design"])
    b = _mk("Juno", "@b", "breathless and warm", ["okay"], ["design"])
    o = compare(a, b)
    assert o.interests == 1.0
    assert o.score < 0.5


def test_a_shared_handle_is_a_COLLISION_not_a_similarity():
    a = _mk("Sophie", "@same", "cutting", ["actually"], ["taste"])
    b = _mk("Juno", "@same", "breathless", ["okay"], ["trains"])
    hits = collisions([a, b])
    assert [c.field for c in hits] == ["handle"]


def test_jaccard_is_zero_for_disjoint_and_one_for_identical():
    assert jaccard(set("abc"), set("def")) == 0.0
    assert jaccard(set("abc"), set("abc")) == 1.0
    assert jaccard(set(), set()) == 0.0


def test_a_broken_spec_is_REPORTED_rather_than_skipped(tmp_path):
    """Silently ignoring it is how a character stops being part of the roster
    while its file sits there looking fine."""
    (tmp_path / "ok.yaml").write_text(to_yaml(
        draft("Sophie", "@sophie", "x", _completer(GOOD))), encoding="utf-8")
    (tmp_path / "broken.yaml").write_text("name: x\n", encoding="utf-8")
    good, bad = load_all(tmp_path)
    assert [p.name for p in good] == ["Sophie"]
    assert len(bad) == 1 and "broken.yaml" in bad[0]


def test_the_report_names_the_threshold_it_used():
    """A judgement call printed is a judgement call; hidden, it is a fact."""
    a = _mk("Sophie", "@a", "cutting", ["actually"], ["taste"])
    assert "threshold" in report([a], threshold=0.3)
    assert "0.30" in report([a], threshold=0.3)
