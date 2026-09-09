"""Survivorship, and the constraints that surfaced measuring it.

The gap between a clean ETF universe and a 2026 membership list is the largest
single effect in this repository, and it runs in the direction that flatters a
strategy. These pin the claim and the machinery it rests on.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.costs import CostModel
from bipbip.core.panel import build_panel, load_panel
from bipbip.core.portfolio import PortfolioEngine
from bipbip.data.universe import UNIVERSES, get_universe
from bipbip.strategies.cross_sectional import EqualWeightBuyHold


def _series(n, start, seed, first=0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n)))
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-04", periods=n))
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                       "close": close, "volume": 1e7}, index=idx)
    if first:
        df.iloc[:first] = np.nan
    return df


def test_stock_universes_are_labelled_as_severely_biased():
    """The label is load-bearing: it is why results from these are not quoted."""
    for name in ("megacap", "everything", "largecap250", "full"):
        assert UNIVERSES[name]["survivorship"] == "SEVERE"
    for name in ("etf_core", "etf_wide", "etf_all"):
        assert UNIVERSES[name]["survivorship"] == "mild"


def test_spy_is_absent_from_the_stock_universes():
    """Regression: the benchmark silently never traded.

    buy-and-hold SPY reported $50, 0.00% CAGR and zero trades on largecap250,
    having never found SPY tradeable. A benchmark that cannot be bought is not
    a benchmark, and the comparison it anchored was meaningless.
    """
    assert "SPY" not in get_universe("largecap250")
    assert "SPY" in get_universe("etf_wide")


def test_slice_from_aligns_two_panels_to_a_common_start():
    """Comparing universes over different windows measures the window.

    largecap250 reaches back to 1962 and the ETF list to 1993, so an unaligned
    comparison credits one side with thirty extra years of compounding and
    reads the difference as survivorship.
    """
    early = build_panel({"OLD": _series(900, 100, 1)})
    late = build_panel({"NEW": _series(900, 100, 2, first=400)})

    cut = early.dates[400]
    a, b = early.slice_from(cut), late.slice_from(cut)
    assert a.dates[0] == b.dates[0] == cut
    assert len(a) == len(b)


def test_slice_from_rejects_a_start_beyond_the_data():
    panel = build_panel({"AAA": _series(200, 100, 3)})
    with pytest.raises(ValueError):
        panel.slice_from("2099-01-01")


def test_slice_from_handles_a_timezone_mismatch():
    """The store keeps a timezone and build_panel drops it."""
    panel = build_panel({"AAA": _series(300, 100, 4)})
    assert panel.dates.tz is None
    out = panel.slice_from(pd.Timestamp("2010-06-01"))
    assert len(out) < len(panel) and len(out) > 0


def test_equal_weighting_a_wide_universe_is_infeasible_on_fifty_dollars():
    """A real constraint of the account, not a strategy result.

    266 names on $50 is 19 cents each, under the engine's dust threshold, so
    every buy is refused and the curve sits flat at the starting balance. Read
    as a backtest that would say 'equal weight returns 0%'.
    """
    n_symbols = 120
    panel = build_panel({f"S{i:03d}": _series(300, 100 + i, i) for i in range(n_symbols)})
    res = PortfolioEngine(CostModel(), starting_equity=50.0, settle_days=0,
                          min_trade_value=0.50).run(panel, EqualWeightBuyHold())
    slice_value = 50.0 / n_symbols
    assert slice_value < 0.50
    assert res.trades == [], "expected every dust-sized buy to be refused"
    assert res.final_equity == pytest.approx(50.0)


def test_the_same_account_can_hold_a_narrow_universe():
    """The threshold is about position size, not about breadth being banned."""
    panel = build_panel({f"S{i:02d}": _series(300, 100 + i, i) for i in range(10)})
    res = PortfolioEngine(CostModel(), starting_equity=50.0, settle_days=0,
                          min_trade_value=0.50).run(panel, EqualWeightBuyHold())
    assert res.trades, "10 names at ~$4.90 each should be tradeable"


def test_an_equal_weight_book_at_exactly_the_band_never_initiates():
    """A boundary worth knowing about, found by a test that expected to pass.

    The engine trades when drift is strictly GREATER than the rebalance band.
    An equal-weight book of exactly 1/band names starts with a per-name drift
    of exactly the band - twenty names against a 0.05 band - so the opening
    purchase is never made and the curve sits flat, looking like a strategy
    that chose to hold cash.
    """
    panel = build_panel({f"S{i:02d}": _series(300, 100 + i, i) for i in range(20)})
    at_band = PortfolioEngine(CostModel(), starting_equity=50.0, settle_days=0,
                              rebalance_band=0.05).run(panel, EqualWeightBuyHold())
    below = PortfolioEngine(CostModel(), starting_equity=50.0, settle_days=0,
                            rebalance_band=0.04).run(panel, EqualWeightBuyHold())
    assert at_band.trades == []
    assert below.trades, "a band under the per-name weight should initiate"


def test_readme_survivorship_table_keeps_its_control_row():
    """The control is what makes the rest of the table readable."""
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    body = readme.read_text().split("**Survivorship, measured.**")
    assert len(body) == 2, "survivorship section missing from README"
    flat = " ".join(body[1].split())

    rows = re.findall(
        r"\| ([^|]+?) \| ([\d.]+)% \| ([\d.]+)% \| \+([\d.]+)pp \|", flat)
    assert len(rows) >= 4, f"expected the survivorship table, parsed {len(rows)}"

    control = rows[0]
    assert "buy and hold" in control[0].lower(), "control row is not first"
    assert float(control[1]) == float(control[2]), \
        "the benchmark must read identically on both universes"
    assert float(control[3]) == 0.0

    gaps = [float(r[3]) for r in rows[1:]]
    assert all(g > 10.0 for g in gaps), f"gaps shrank below the claim: {gaps}"
    # Each strategy row must be recomputable from its own two columns.
    for name, clean, dirty, gap in rows[1:]:
        assert float(dirty) - float(clean) == pytest.approx(float(gap), abs=0.02), name


def test_readme_clean_universe_table_keeps_holding_on_top():
    """The section's claim is that nothing beats holding on total return."""
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    body = readme.read_text().split("**The clean universe, in full.**")
    assert len(body) == 2, "clean-universe section missing from README"
    flat = " ".join(body[1].split())

    rows = re.findall(
        r"\| ([^|]+?) \| \$([\d,]+) \| ([\d.]+)% \| ([\d.]+) \| ([\d.]+)% \| (\d+) \|",
        flat)
    assert len(rows) == 6, f"expected 6 rows, parsed {len(rows)}"

    finals = [float(r[1].replace(",", "")) for r in rows]
    assert "buy and hold" in rows[0][0].lower()
    assert finals[0] == max(finals), "something now beats holding on dollars"
    assert finals == sorted(finals, reverse=True), "table is not ordered by result"

    # Two rows beat holding on Sharpe, and both give up return for it.
    hold_sharpe = float(rows[0][3])
    better = [r for r in rows[1:] if float(r[3]) > hold_sharpe]
    assert len(better) == 2, f"expected 2 rows above {hold_sharpe} Sharpe"
    for r in better:
        assert float(r[2]) < float(rows[0][2]), "a Sharpe win should cost return"
