"""Canon: the thing that stops a character contradicting herself into nothing.

The bug these caught on the way in is the one worth keeping in mind. The first
`conflicts()` filtered candidate facts through keyword relevance before running
the contradiction check - which silently dropped exactly the contradictions it
existed to find, because semantic opposites do not share words. "I am
vegetarian" and "best steak of my life" overlap on nothing.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from synth.canon import Canon


def _c(*facts):
    c = Canon()
    for f in facts:
        c.add(f)
    return c


def test_it_catches_a_contradiction_that_SHARES_words():
    c = _c("She has never been to Lisbon.")
    assert [f.text for f in c.conflicts("that year I lived in Lisbon")]


@pytest.mark.parametrize("fact,draft", [
    ("She is vegetarian.", "best steak of my life tonight"),
    ("She is an only child.", "my sister sent me this"),
    ("She is vegan.", "too much cheese, no regrets"),
])
def test_it_catches_a_contradiction_that_shares_NO_words(fact, draft):
    """The regression. These pairs are semantically opposite and lexically
    disjoint, so any relevance filter in front of the check hides them."""
    assert [f.text for f in _c(fact).conflicts(draft)], (fact, draft)


def test_an_unrelated_draft_is_not_flagged():
    c = _c("She has never been to Lisbon.", "She is vegetarian.")
    assert c.conflicts("the new album is very good") == []


def test_a_retcon_KEEPS_the_original():
    """The audience saw it. Deleting it is how you end up gaslighting the
    people who were paying closest attention."""
    c = _c("She lives alone.")
    assert c.retcon("She lives alone.", by="got a flatmate in March")
    assert len(c.facts) == 1
    assert c.facts[0].retconned_by == "got a flatmate in March"
    assert not c.facts[0].live
    assert c.live_facts() == []


def test_a_retconned_fact_stops_causing_conflicts():
    c = _c("She has never been to Lisbon.")
    c.retcon("She has never been to Lisbon.", by="went in April")
    assert c.conflicts("that year I lived in Lisbon") == []


def test_retconning_something_that_was_never_said_reports_failure():
    assert _c("She lives alone.").retcon("She owns a boat.", by="x") is False


def test_relevance_returns_the_related_facts_not_everything():
    c = _c("She has never been to Lisbon.", "She is vegetarian.",
           "She likes rain.", "Her favourite colour is green.")
    rel = [f.text for f in c.relevant("thinking about Lisbon")]
    assert "She has never been to Lisbon." in rel
    assert "Her favourite colour is green." not in rel


def test_the_brief_is_empty_when_nothing_is_relevant():
    """An empty canon block keeps unrelated facts out of the prompt. Pasting
    everything ever recorded drowns the instruction that matters."""
    assert _c("She likes rain.").brief("quantum chromodynamics") == ""


def test_it_round_trips_through_disk(tmp_path):
    path = tmp_path / "vee.canon.jsonl"
    c = Canon(path)
    c.add("She likes rain.", topic="weather", source="post/1")
    c.add("She has never been to Lisbon.", topic="travel")
    c.retcon("She likes rain.", by="changed her mind")

    again = Canon(path)
    assert len(again.facts) == 2
    assert len(again.live_facts()) == 1
    assert again.live_facts()[0].topic == "travel"
    assert [f for f in again.facts if not f.live][0].retconned_by == \
        "changed her mind"


def test_a_partial_write_never_replaces_a_good_file(tmp_path):
    """Written to a .partial and renamed, so an interrupted save cannot leave
    a truncated canon where the real one was."""
    path = tmp_path / "vee.canon.jsonl"
    c = Canon(path)
    c.add("She likes rain.")
    assert path.exists()
    assert not list(tmp_path.glob("*.partial"))
