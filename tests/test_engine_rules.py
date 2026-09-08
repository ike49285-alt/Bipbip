"""Engine invariants: things that must hold for every strategy, always."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bipbip.core import BacktestEngine, CashAccount, CostModel
from bipbip.core.strategy import Strategy
from bipbip.core.types import HOLD, Intent
from bipbip.data import make_intraday_bars
from bipbip.data.sessions import EXCHANGE_TZ
from bipbip.strategies import get_strategy

STRATEGIES = ["buy_hold", "orb", "vwap_reversion"]


def _engine(**kw):
    return BacktestEngine(CashAccount(starting_equity=10_000.0), CostModel(), **kw)


def _session(rows, day="2025-03-03"):
    """Build a single RTH session from (open, high, low, close) tuples."""
    idx = pd.date_range(
        start=pd.Timestamp(f"{day} 09:30", tz=EXCHANGE_TZ), periods=len(rows), freq="1min"
    )
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype="float64")
    df["volume"] = 100_000.0
    df.index.name = "timestamp"
    return df


class EnterAtBar(Strategy):
    """Enters once, at a chosen bar, with an explicit stop and target."""

    name = "enter_at_bar"

    def __init__(self, bar=0, stop=None, target=None):
        self.bar, self.stop, self.target = bar, stop, target

    def on_bar(self, ctx):
        if ctx.i == self.bar and not ctx.in_position:
            return Intent(action="enter", reason="test", stop_price=self.stop, target_price=self.target)
        return HOLD


@pytest.mark.parametrize("name", STRATEGIES)
def test_never_holds_a_position_overnight(name):
    """An intraday system that carries risk overnight is not an intraday system."""
    bars = make_intraday_bars(n_sessions=15, seed=21)
    res = _engine().run("SPY", bars, get_strategy(name))
    for t in res.trades:
        assert t.entry_time.date() == t.exit_time.date(), f"{t.entry_time} -> {t.exit_time}"


@pytest.mark.parametrize("name", STRATEGIES)
def test_cash_account_takes_at_most_one_round_trip_per_session(name):
    bars = make_intraday_bars(n_sessions=15, seed=22)
    res = _engine().run("SPY", bars, get_strategy(name))
    per_day = {}
    for t in res.trades:
        per_day[t.entry_time.date()] = per_day.get(t.entry_time.date(), 0) + 1
    assert per_day, "no trades taken"
    assert max(per_day.values()) <= 1, per_day


@pytest.mark.parametrize("name", STRATEGIES)
def test_never_shorts_and_never_spends_unsettled_cash(name):
    bars = make_intraday_bars(n_sessions=15, seed=23)
    acct = CashAccount(starting_equity=10_000.0)
    BacktestEngine(acct, CostModel()).run("SPY", bars, get_strategy(name))
    assert acct.position.shares >= 0
    assert acct.settled_cash >= -1e-9


@pytest.mark.parametrize("name", STRATEGIES)
def test_equity_curve_covers_every_bar(name):
    bars = make_intraday_bars(n_sessions=8, seed=24)
    res = _engine().run("SPY", bars, get_strategy(name))
    assert len(res.equity_curve) == len(bars)
    assert res.equity_curve.index.equals(bars.index)
    assert (res.equity_curve > 0).all()


def test_gap_through_a_stop_fills_at_the_open_not_the_stop():
    """A stop is not a guarantee. If the market gaps past it you get the open.

    An engine that fills gaps at the stop price invents money that a live
    account would never see.
    """
    rows = [(100, 100.5, 99.5, 100)] * 3
    rows.append((90, 90.5, 89.0, 90))  # violent gap down, straight through the stop
    rows += [(90, 90.5, 89.5, 90)] * 6

    costs = CostModel()
    eng = BacktestEngine(CashAccount(starting_equity=10_000.0), costs)
    res = eng.run("SPY", _session(rows), EnterAtBar(bar=0, stop=99.0))

    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.exit_reason == "stop"
    # Filled at the gapped-down open (90), NOT at the 99 stop.
    assert trade.exit_price == pytest.approx(costs.fill_price("sell", 90.0, "SPY"))
    assert trade.exit_price < 95.0


def test_stop_is_checked_before_target_when_a_bar_touches_both():
    """Ambiguous bars resolve against the trader, never in their favour."""
    rows = [(100, 100.2, 99.9, 100)] * 2
    rows.append((100, 105.0, 95.0, 100))  # range spans both stop and target
    rows += [(100, 100.2, 99.9, 100)] * 6

    res = _engine().run("SPY", _session(rows), EnterAtBar(bar=0, stop=98.0, target=104.0))
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "stop"


def test_force_flat_closes_before_the_bell():
    bars = make_intraday_bars(n_sessions=3, seed=25)
    res = BacktestEngine(
        CashAccount(starting_equity=10_000.0), CostModel(), force_flat_at="15:30"
    ).run("SPY", bars, get_strategy("buy_hold"))
    assert res.trades
    for t in res.trades:
        assert t.exit_time.time() <= dt.time(15, 30), t.exit_time
        assert t.exit_reason == "force_flat"


def test_late_entries_are_refused():
    """No new risk once the day's remaining time cannot reach the target."""
    bars = make_intraday_bars(n_sessions=2, seed=26)
    eng = BacktestEngine(
        CashAccount(starting_equity=10_000.0), CostModel(), no_new_entries_after="10:00"
    )
    res = eng.run("SPY", bars, EnterAtBar(bar=120))  # 11:30, well past the cutoff
    assert res.trades == []
    assert res.blocked, "a refused entry should be recorded for diagnostics"


def test_misaligned_indicator_frame_is_rejected():
    """A silent index mismatch would scramble every signal. Fail loudly."""

    class Broken(Strategy):
        name = "broken"

        def prepare(self, bars):
            return pd.DataFrame(index=bars.index[:-5])

        def on_bar(self, ctx):
            return HOLD

    with pytest.raises(ValueError, match="misaligned"):
        _engine().run("SPY", make_intraday_bars(n_sessions=2), Broken())


def test_random_entry_null_respects_the_same_account_rules():
    """The null must be subject to identical constraints, or the comparison
    measures rule differences rather than timing."""
    from bipbip.strategies.random_entry import RandomEntry

    bars = make_intraday_bars(n_sessions=15, seed=61)
    acct = CashAccount(starting_equity=10_000.0)
    res = BacktestEngine(acct, CostModel()).run("SPY", bars, RandomEntry(seed=3))

    per_day = {}
    for t in res.trades:
        assert t.entry_time.date() == t.exit_time.date()
        per_day[t.entry_time.date()] = per_day.get(t.entry_time.date(), 0) + 1
    assert not per_day or max(per_day.values()) <= 1
    assert acct.position.shares >= 0


def test_random_entry_is_reproducible_and_varies_by_seed():
    """Seeds must be stable, or a significance run is not reproducible."""
    from bipbip.strategies.random_entry import RandomEntry

    bars = make_intraday_bars(n_sessions=10, seed=62)

    def run(seed):
        return BacktestEngine(CashAccount(10_000.0), CostModel()).run(
            "SPY", bars, RandomEntry(seed=seed)).equity_curve.iloc[-1]

    assert run(1) == run(1)
    assert len({run(s) for s in range(6)}) > 1


def test_context_lazy_slice_matches_eager_slice():
    """The lazy `bars` property must expose exactly the visible window - no
    more. This is the guarantee that made the optimisation safe."""
    seen = []

    class Recorder(Strategy):
        name = "recorder"

        def on_bar(self, ctx):
            seen.append((len(ctx.bars), ctx.i, ctx.bars.index[-1] == ctx.session_bars.index[ctx.i]))
            return HOLD

    bars = make_intraday_bars(n_sessions=2, seed=63)
    _engine().run("SPY", bars, Recorder())
    assert seen
    for n, i, last_is_current in seen:
        assert n == i + 1, "visible window must end at the current bar"
        assert last_is_current
