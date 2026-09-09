"""Option-chain snapshots, and the line between derivation and invention."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bipbip.data.options_chain import (ChainStore, add_greeks, empirical_delta,
                                       iv_vs_realised)


def _chain(spot=72.0, stamp="t0", bids=(3.0, 1.0), asks=(3.1, 1.1)):
    return pd.DataFrame({
        "contractSymbol": ["TQQQ260919C70", "TQQQ260919C75"],
        "strike": [70.0, 75.0], "side": ["call", "call"],
        "impliedVolatility": [0.65, 0.68],
        "bid": list(bids), "ask": list(asks),
        "expiry": ["2026-09-19"] * 2, "spot": [spot, spot],
        "fetched_at": [stamp] * 2,
    })


def test_four_snapshots_a_day_do_not_overwrite_each_other(tmp_path):
    """Regression: a date-stamped filename keeps one chain a day.

    The point of snapshotting repeatedly is the sequence. Keying the file on
    the DATE means each run silently replaces the last and the panel never
    grows past one row per day - the feature deleting itself while appearing
    to work.
    """
    store = ChainStore(tmp_path)
    for hour in (14, 16, 18, 20):
        store.save("TQQQ", _chain(),
                   dt.datetime(2026, 9, 9, hour, 30, tzinfo=dt.timezone.utc))
    assert len(store.dates("TQQQ")) == 4
    assert len(list(tmp_path.glob("TQQQ_*.parquet"))) == 4


def test_the_underlying_price_is_stored_with_the_chain():
    """Without it a snapshot cannot be compared to the next one at all."""
    c = _chain(spot=72.5)
    assert "spot" in c and float(c["spot"].iloc[0]) == 72.5


def test_greeks_are_derived_from_the_market_s_own_implied_vol():
    """The legitimate extrapolation: greeks are functions of IV, not new facts."""
    g = add_greeks(_chain(), pd.Timestamp("2026-09-09", tz="UTC"))
    for col in ("delta", "gamma", "theta_per_day", "vega"):
        assert col in g
    # A call closer to the money carries the larger delta.
    assert g["delta"].iloc[0] > g["delta"].iloc[1]
    assert (g["delta"].between(0, 1)).all()
    assert (g["theta_per_day"] <= 0).all(), "a long option decays"
    assert (g["vega"] > 0).all()


def test_greeks_use_a_trading_clock_not_a_calendar_one():
    """A weekend contributes no decay, so calendar time overstates theta."""
    g = add_greeks(_chain(), pd.Timestamp("2026-09-09", tz="UTC"))
    calendar_days = (pd.Timestamp("2026-09-19", tz="UTC")
                     - pd.Timestamp("2026-09-09", tz="UTC")).days
    implied_days = float(g["minutes_to_expiry"].iloc[0]) / 390.0
    assert implied_days < calendar_days, "trading time should be shorter"


def test_an_expired_contract_has_no_time_left_rather_than_negative():
    past = _chain()
    past["expiry"] = "2026-09-01"
    g = add_greeks(past, pd.Timestamp("2026-09-09", tz="UTC"))
    assert (g["minutes_to_expiry"] >= 0).all()


def test_delta_can_be_measured_from_consecutive_snapshots():
    """The reason for storing spot: delta becomes an observation.

    If the measured value disagrees with Black-Scholes, that disagreement is
    information about the quote rather than an error to reconcile away.
    """
    asof = pd.Timestamp("2026-09-09", tz="UTC")
    before = add_greeks(_chain(spot=72.0, stamp="t0"), asof)
    after = add_greeks(_chain(spot=73.0, stamp="t1",
                              bids=(3.6, 1.35), asks=(3.7, 1.45)), asof)
    d = empirical_delta([before, after])
    assert len(d) == 2
    assert d["d_spot"].iloc[0] == pytest.approx(1.0)
    # (3.65 - 3.05) / 1.0 = 0.60
    assert d["empirical_delta"].iloc[0] == pytest.approx(0.60, abs=1e-6)
    assert "modelled_delta" in d


def test_a_flat_underlying_yields_no_delta_measurement():
    """Dividing by a zero move would manufacture an enormous number."""
    asof = pd.Timestamp("2026-09-09", tz="UTC")
    a = add_greeks(_chain(spot=72.0, stamp="t0"), asof)
    b = add_greeks(_chain(spot=72.0, stamp="t1"), asof)
    assert empirical_delta([a, b]).empty


def test_a_single_snapshot_measures_nothing():
    assert empirical_delta([_chain()]).empty
    assert empirical_delta([]).empty


def test_the_module_offers_no_way_to_invent_prices_between_snapshots():
    """The boundary that keeps this honest.

    Greeks from IV are derivation - the market supplied the volatility and the
    rest follows. Option prices between snapshots, or before collection began,
    are invention, and a backtest run over them tests the interpolator rather
    than the market. Snapshots are observations; the gaps stay empty.
    """
    import bipbip.data.options_chain as mod

    names = [n for n in dir(mod) if not n.startswith("_")]
    for banned in ("interpolate", "resample", "backfill", "reconstruct",
                   "synthesize", "synthesise"):
        assert not any(banned in n.lower() for n in names), (
            f"a {banned!r} helper would let fabricated prices reach a backtest")


def test_iv_below_realised_is_flagged_as_unusual():
    """Options normally trade above realised; the reverse wants checking."""
    c = _chain()
    low = iv_vs_realised(c, 72.0, 0.90)
    high = iv_vs_realised(c, 72.0, 0.30)
    assert "BELOW" in low["verdict"]
    assert low["ratio"] < 1
    assert "above" in high["verdict"]
