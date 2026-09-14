"""Fundamentals store and the news probe, which had no tests at all.

The module is reachable from the CLI (`bipbip fundamentals`) and nothing
exercised it. A static scan for silent exception handlers found one here, and
reading it turned up a real defect: the `break` that ends the key search sat
OUTSIDE the try, so the first non-empty field won even when it failed to parse.
One malformed `pubDate` meant `providerPublishTime` was never consulted and the
item counted as undated.

`probe_news` exists to report what news is actually retrievable rather than
pretending, so silently under-counting dated items is the one failure it must
not have.
"""
import json

import pandas as pd
import pytest

from bipbip.data import fundamentals as F


class _Ticker:
    """Stands in for yfinance.Ticker. `news` is whatever the test plants."""

    registry = {}

    def __init__(self, symbol):
        self.symbol = symbol

    @property
    def news(self):
        out = self.registry[self.symbol]
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def fake_yf(monkeypatch):
    import types
    mod = types.ModuleType("yfinance")
    mod.Ticker = _Ticker
    _Ticker.registry = {}
    monkeypatch.setitem(__import__("sys").modules, "yfinance", mod)
    return _Ticker.registry


def test_a_malformed_first_field_falls_through_to_the_next_one(fake_yf):
    """The defect, stated directly. `pubDate` is unparseable and
    `providerPublishTime` is fine, so the item IS dated."""
    fake_yf["AAA"] = [{"content": {"pubDate": "not-a-date",
                                   "providerPublishTime": 1_700_000_000}}]
    rep = F.probe_news(["AAA"])
    assert rep["checked"][0]["items"] == 1
    assert rep["checked"][0]["dated"] == 1


def test_an_item_with_no_parseable_field_is_counted_but_not_dated():
    """Under-reporting is the failure mode; over-reporting is the other one.
    An item nothing can date must still be counted as an item."""
    class _T:
        def __init__(self, s): pass
        news = [{"content": {"pubDate": "nonsense", "displayTime": "also-bad"}}]
    import sys, types
    mod = types.ModuleType("yfinance"); mod.Ticker = _T
    sys.modules["yfinance"] = mod
    try:
        rep = F.probe_news(["AAA"])
        assert rep["checked"][0]["items"] == 1
        assert rep["checked"][0]["dated"] == 0
    finally:
        sys.modules.pop("yfinance", None)


def test_an_epoch_integer_is_read_as_seconds_not_nanoseconds(fake_yf):
    """`unit="s"`. Read as nanoseconds, 1.7e9 lands in 1970 and every news
    item looks decades old - which would change the verdict the probe exists
    to deliver."""
    fake_yf["AAA"] = [{"content": {"providerPublishTime": 1_700_000_000}}]
    rep = F.probe_news(["AAA"])
    assert rep["checked"][0]["dated"] == 1
    assert pd.Timestamp(rep["oldest"]).year == 2023


def test_one_symbol_failing_does_not_lose_the_others(fake_yf):
    """A fetch error is recorded against its own symbol and the sweep goes on."""
    fake_yf["BAD"] = RuntimeError("connection reset")
    fake_yf["GOOD"] = [{"content": {"providerPublishTime": 1_700_000_000}}]
    rep = F.probe_news(["BAD", "GOOD"])
    by = {c["symbol"]: c for c in rep["checked"]}
    assert "error" in by["BAD"] and "connection reset" in by["BAD"]["error"]
    assert by["GOOD"]["dated"] == 1


def test_the_sample_cap_limits_how_many_symbols_are_probed(fake_yf):
    for s in ("A", "B", "C"):
        fake_yf[s] = [{"content": {"providerPublishTime": 1_700_000_000}}]
    assert len(F.probe_news(["A", "B", "C"], sample=2)["checked"]) == 2


def test_an_item_without_a_content_wrapper_is_still_read(fake_yf):
    """yfinance has shipped both shapes; the flat one must not read as empty."""
    fake_yf["AAA"] = [{"providerPublishTime": 1_700_000_000}]
    assert F.probe_news(["AAA"])["checked"][0]["dated"] == 1


# --------------------------------------------------------------------------
# FundamentalsStore
# --------------------------------------------------------------------------

def test_a_saved_frame_round_trips(tmp_path):
    st = F.FundamentalsStore(tmp_path)
    df = pd.DataFrame({"symbol": ["AAA", "BBB"], "n": [1, 2]})
    st.save_frame("earnings", df)
    pd.testing.assert_frame_equal(st.load_frame("earnings"), df)


def test_loading_a_frame_that_was_never_saved_is_empty_not_an_error(tmp_path):
    """A missing file must read as "nothing collected yet", because that is the
    normal state before the first fetch."""
    assert F.FundamentalsStore(tmp_path).load_frame("absent").empty


def test_a_saved_json_round_trips_and_a_missing_one_takes_the_default(tmp_path):
    st = F.FundamentalsStore(tmp_path)
    st.save_json("sectors", {"AAA": "Tech"})
    assert st.load_json("sectors") == {"AAA": "Tech"}
    assert st.load_json("absent", default={"x": 1}) == {"x": 1}


def test_the_store_creates_its_directory_rather_than_failing(tmp_path):
    st = F.FundamentalsStore(tmp_path / "deep" / "nested")
    st.save_json("x", {"a": 1})
    assert st.load_json("x") == {"a": 1}
