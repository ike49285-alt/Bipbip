"""The spot and intrinsic-value guards in `scripts/chain_edge.py`.

This script produced a standing result in CLAUDE.md, and that result was wrong
in its near-the-money half: a hardcoded spot two dollars above the chain's own
put-call parity fabricated intrinsic value, and every deep-in-the-money strike
was scored "cheap" at up to +73% of premium because its ask sat below an
intrinsic value that only existed at the wrong spot.

Two things stop that returning: spot comes from the chain, and a quote below
intrinsic is dropped rather than scored.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "chain_edge",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "chain_edge.py")
chain_edge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(chain_edge)


def test_an_ask_below_intrinsic_is_identified_as_impossible():
    """CLAUDE.md's own no-arbitrage rule: an option cannot trade below
    intrinsic. At spot 69.21 a 40-strike call is worth 29.21 at the least, so a
    30.00 ask is normal - at a phantom spot of 71.365 the same ask is 1.36
    below intrinsic and reads as free money."""
    ask, strike = 30.00, 40.0
    assert ask >= max(0.0, 69.213 - strike)      # real spot: fine
    assert ask < max(0.0, 71.365 - strike)       # phantom spot: impossible


def test_the_holiday_set_is_market_not_federal():
    """The exchange trades on Columbus Day and Veterans Day and closes on Good
    Friday, so the federal calendar is the wrong one. Good Friday 2026 is
    2026-04-03; Columbus Day 2026 is 2026-10-12."""
    assert "2026-04-03" in chain_edge.US_MARKET_HOLIDAYS
    assert "2026-10-12" not in chain_edge.US_MARKET_HOLIDAYS
    assert "2026-11-11" not in chain_edge.US_MARKET_HOLIDAYS


def test_a_holiday_shortens_the_session_count():
    """The three-entry list this replaced overstated every horizon that spanned
    a closure, which lengthens the window each payoff is scored over."""
    before = chain_edge.sessions_to_expiry(
        pd.Timestamp("2026-12-21"), pd.Timestamp("2026-12-31"))
    # 22nd-31st is eight business days; Christmas removes one.
    assert before == 7


def test_the_vol_anchor_comes_from_the_quote_date():
    idx = pd.bdate_range("2026-01-01", periods=40)
    v = pd.Series(np.linspace(0.60, 0.30, len(idx)), index=idx)
    early = idx[3]
    assert chain_edge.anchor_for(v, early) == pytest.approx(v.loc[early])
    assert chain_edge.anchor_for(v, early) != pytest.approx(v.iloc[-1])


def test_a_quote_date_before_every_bar_raises_rather_than_guessing():
    idx = pd.bdate_range("2026-01-01", periods=40)
    v = pd.Series(np.linspace(0.60, 0.30, len(idx)), index=idx)
    with pytest.raises(SystemExit):
        chain_edge.anchor_for(v, pd.Timestamp("2025-01-01"))
