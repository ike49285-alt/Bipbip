"""Event-driven intraday backtester.

Bar ordering is the whole correctness story, so it is stated explicitly. At
each bar `i` the engine does, in this order:

  1. Fill any order queued at bar i-1, at bar i's OPEN plus slippage.
  2. Check stop / target against bar i's LOW and HIGH.
  3. Force-flat if the session is closing.
  4. Ask the strategy for an intent, showing it bars 0..i inclusive.
  5. Queue the resulting order for bar i+1's open.

A strategy therefore decides on information it could actually have had, and
pays the next open for it. Deciding and filling on the same bar's close would
inflate every result and is not offered.

Intrabar exits are modelled pessimistically. If a bar gaps through a stop, the
fill is the open, not the stop price - a stop does not protect you from a gap.
When a bar's range touches both the stop and the target, the stop is assumed
to have been hit first.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..data.sessions import iter_sessions, next_session_date, session_dates
from .account import CashAccount, RuleViolation
from .costs import BUY, SELL, CostModel
from .strategy import Context, Strategy
from .types import BacktestResult, Trade


def _parse_time(value: str) -> dt.time:
    hh, mm = str(value).split(":")
    return dt.time(int(hh), int(mm))


class BacktestEngine:
    def __init__(
        self,
        account: CashAccount,
        costs: CostModel,
        no_new_entries_after: str = "15:00",
        force_flat_at: str = "15:55",
        intraday: bool = True,
    ):
        """`intraday=False` selects SWING mode: positions are held across
        sessions instead of being force-flat at the bell.

        The distinction is not cosmetic. An intraday run groups bars by session
        so a strategy sees only today, and closes everything before the close.
        A swing run treats the series as one continuous timeline, which is what
        daily bars require - and what makes 33 years of history usable, where
        the minute archive holds 21 sessions.

        Everything else is shared: fills at the next bar's open, pessimistic
        intrabar stops, the cost model, and the cash account's settlement rules.
        """
        self.account = account
        self.costs = costs
        self.no_new_entries_after = _parse_time(no_new_entries_after)
        self.force_flat_at = _parse_time(force_flat_at)
        self.intraday = intraday

    def run(self, symbol: str, bars: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        if bars.empty:
            raise ValueError("no bars supplied; fetch data first")
        if not self.intraday:
            return self._run_swing(symbol, bars, strategy)

        # Give the strategy the instrument's round-trip cost so it can floor
        # its risk unit. A stop tighter than the round trip is a guaranteed
        # loss when hit, and on real SPY data a one-minute ATR is below that
        # threshold on 37% of bars.
        strategy.cost_hurdle_bps = self.costs.round_trip_cost_bps(
            symbol, float(bars["close"].iloc[-1])
        )
        indicators = strategy.prepare(bars)
        if not indicators.index.equals(bars.index):
            raise ValueError(f"{strategy.name}.prepare() returned a misaligned index")

        calendar = session_dates(bars)
        equity_times: list = []
        equity_values: list = []
        trades: list = []
        blocked: list = []

        # Open position bookkeeping, reset on every exit.
        entry_price = 0.0
        entry_time = None
        entry_reason = ""
        entry_fees = 0.0
        bars_held = 0
        stop_price = None
        target_price = None
        pending = None  # intent queued for the next bar's open
        prior_close = float("nan")

        for session_date, session in iter_sessions(bars):
            self.account.start_session(session_date)
            strategy.on_session_start(session_date)
            logged_block = False
            settle_on = next_session_date(calendar, session_date)

            sess_ind = indicators.loc[session.index]
            n = len(session)

            for i in range(n):
                ts = session.index[i]
                bar = session.iloc[i]

                # -- 1. fill the order queued at the previous bar -----------
                if pending is not None:
                    action = pending.action
                    if action == "enter" and not self.account.position.is_open:
                        fill = self.costs.fill_price(BUY, float(bar["open"]), symbol)
                        shares = self.account.affordable_shares(fill)
                        shares = shares * max(0.0, min(1.0, pending.size_pct))
                        if not self.account.allow_fractional:
                            shares = float(int(shares))
                        if shares > 0:
                            fees = self.costs.fees(BUY, shares, fill)
                            try:
                                self.account.buy(shares, fill, fees)
                            except RuleViolation as exc:
                                blocked.append({"time": ts, "reason": str(exc)})
                            else:
                                entry_price, entry_time = fill, ts
                                entry_reason, entry_fees = pending.reason, fees
                                bars_held = 0
                                stop_price, target_price = pending.stop_price, pending.target_price
                    elif action == "exit" and self.account.position.is_open:
                        fill = self.costs.fill_price(SELL, float(bar["open"]), symbol)
                        trades.append(
                            self._close(symbol, fill, ts, entry_price, entry_time,
                                        entry_reason, pending.reason, entry_fees, settle_on)
                        )
                        stop_price = target_price = None
                    pending = None

                # -- 2. intrabar risk exits --------------------------------
                if self.account.position.is_open:
                    bars_held += 1
                    hit, why = None, ""
                    if stop_price is not None and float(bar["low"]) <= stop_price:
                        # A gap through the stop fills at the open, not the stop.
                        hit, why = min(stop_price, float(bar["open"])), "stop"
                    elif target_price is not None and float(bar["high"]) >= target_price:
                        hit, why = max(target_price, float(bar["open"])), "target"
                    if hit is not None:
                        fill = self.costs.fill_price(SELL, hit, symbol)
                        trades.append(
                            self._close(symbol, fill, ts, entry_price, entry_time,
                                        entry_reason, why, entry_fees, settle_on)
                        )
                        stop_price = target_price = None

                # -- 3. force flat before the close ------------------------
                is_last_bar = i == n - 1
                if self.account.position.is_open and (ts.time() >= self.force_flat_at or is_last_bar):
                    fill = self.costs.fill_price(SELL, float(bar["close"]), symbol)
                    trades.append(
                        self._close(symbol, fill, ts, entry_price, entry_time,
                                    entry_reason, "force_flat", entry_fees, settle_on)
                    )
                    stop_price = target_price = None
                    pending = None

                # -- 4. ask the strategy -----------------------------------
                if not is_last_bar and i >= strategy.warmup_bars:
                    can_open = self.account.can_open() and ts.time() < self.no_new_entries_after
                    ctx = Context(
                        symbol=symbol,
                        session_bars=session,
                        session_indicators=sess_ind,
                        i=i,
                        in_position=self.account.position.is_open,
                        entry_price=entry_price if self.account.position.is_open else float("nan"),
                        bars_held=bars_held,
                        can_open=can_open,
                        session_date=session_date,
                        minutes_to_close=n - 1 - i,
                        prior_session_close=prior_close,
                    )
                    intent = strategy.on_bar(ctx)

                    if intent.action == "enter":
                        if self.account.position.is_open:
                            pass  # already long; nothing to add in a one-position account
                        elif not can_open:
                            # One record per session: enough to quantify what the
                            # account rules cost, without a row for every bar.
                            if not logged_block:
                                reason = (
                                    self.account.blocked_reason()
                                    or "entries closed for the session"
                                )
                                blocked.append(
                                    {"time": ts, "reason": reason, "wanted": intent.reason}
                                )
                                logged_block = True
                        else:
                            pending = intent
                    elif intent.action == "exit" and self.account.position.is_open:
                        pending = intent

                # -- 5. mark to market -------------------------------------
                equity_times.append(ts)
                equity_values.append(self.account.equity(float(bar["close"])))

            prior_close = float(session["close"].iloc[-1])

        curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_times), name="equity")
        return BacktestResult(
            symbol=symbol,
            equity_curve=curve,
            trades=trades,
            blocked=blocked,
            starting_equity=self.account.starting_equity,
        )

    def _close(self, symbol, fill, ts, entry_price, entry_time, entry_reason,
               exit_reason, entry_fees, settle_on) -> Trade:
        shares = self.account.position.shares
        fees = self.costs.fees(SELL, shares, fill)
        self.account.sell(fill, fees, settle_date=settle_on)
        gross = (fill - entry_price) * shares
        return Trade(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=ts,
            exit_price=fill,
            shares=shares,
            pnl=gross - fees - entry_fees,
            fees=fees + entry_fees,
            entry_reason=entry_reason,
            exit_reason=exit_reason,
        )


    def _run_swing(self, symbol: str, bars: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """Continuous timeline: positions survive the close.

        Deliberately mirrors the intraday loop rather than sharing it, because
        the two differ in exactly the places where a subtle mistake is
        expensive - session grouping, the force-flat, and what the strategy is
        allowed to see. Duplication here is cheaper than a clever abstraction
        that quietly force-flats a swing position or lets an intraday one run
        overnight.
        """
        strategy.cost_hurdle_bps = self.costs.round_trip_cost_bps(
            symbol, float(bars["close"].iloc[-1])
        )
        indicators = strategy.prepare(bars)
        if not indicators.index.equals(bars.index):
            raise ValueError(f"{strategy.name}.prepare() returned a misaligned index")

        calendar = session_dates(bars)
        equity_times: list = []
        equity_values: list = []
        trades: list = []
        blocked: list = []

        entry_price = 0.0
        entry_time = None
        entry_reason = ""
        entry_fees = 0.0
        bars_held = 0
        stop_price = None
        target_price = None
        pending = None
        prev_date = None
        n = len(bars)

        for i in range(n):
            ts = bars.index[i]
            bar = bars.iloc[i]
            date = ts.date()

            # A new calendar day settles matured proceeds and refreshes the
            # day's turnover budget, exactly as in an intraday run.
            if date != prev_date:
                self.account.start_session(date)
                strategy.on_session_start(date)
                prev_date = date
            settle_on = next_session_date(calendar, date)

            if pending is not None:
                action = pending.action
                if action == "enter" and not self.account.position.is_open:
                    fill = self.costs.fill_price(BUY, float(bar["open"]), symbol)
                    shares = self.account.affordable_shares(fill)
                    shares = shares * max(0.0, min(1.0, pending.size_pct))
                    if not self.account.allow_fractional:
                        shares = float(int(shares))
                    if shares > 0:
                        fees = self.costs.fees(BUY, shares, fill)
                        try:
                            self.account.buy(shares, fill, fees)
                        except RuleViolation as exc:
                            blocked.append({"time": ts, "reason": str(exc)})
                        else:
                            entry_price, entry_time = fill, ts
                            entry_reason, entry_fees = pending.reason, fees
                            bars_held = 0
                            stop_price, target_price = pending.stop_price, pending.target_price
                elif action == "exit" and self.account.position.is_open:
                    fill = self.costs.fill_price(SELL, float(bar["open"]), symbol)
                    trades.append(
                        self._close(symbol, fill, ts, entry_price, entry_time,
                                    entry_reason, pending.reason, entry_fees, settle_on)
                    )
                    stop_price = target_price = None
                pending = None

            if self.account.position.is_open:
                bars_held += 1
                hit, why = None, ""
                if stop_price is not None and float(bar["low"]) <= stop_price:
                    hit, why = min(stop_price, float(bar["open"])), "stop"
                elif target_price is not None and float(bar["high"]) >= target_price:
                    hit, why = max(target_price, float(bar["open"])), "target"
                if hit is not None:
                    fill = self.costs.fill_price(SELL, hit, symbol)
                    trades.append(
                        self._close(symbol, fill, ts, entry_price, entry_time,
                                    entry_reason, why, entry_fees, settle_on)
                    )
                    stop_price = target_price = None

            is_last_bar = i == n - 1
            # Only the END OF THE DATA closes a swing position, never the bell.
            if self.account.position.is_open and is_last_bar:
                fill = self.costs.fill_price(SELL, float(bar["close"]), symbol)
                trades.append(
                    self._close(symbol, fill, ts, entry_price, entry_time,
                                entry_reason, "end_of_data", entry_fees, settle_on)
                )
                stop_price = target_price = None
                pending = None

            if not is_last_bar and i >= strategy.warmup_bars:
                can_open = self.account.can_open()
                ctx = Context(
                    symbol=symbol,
                    session_bars=bars,
                    session_indicators=indicators,
                    i=i,
                    in_position=self.account.position.is_open,
                    entry_price=entry_price if self.account.position.is_open else float("nan"),
                    bars_held=bars_held,
                    can_open=can_open,
                    session_date=date,
                    minutes_to_close=n - 1 - i,
                    prior_session_close=float(bars["close"].iloc[i - 1]) if i else float("nan"),
                )
                intent = strategy.on_bar(ctx)

                if intent.action == "enter":
                    if self.account.position.is_open:
                        pass
                    elif not can_open:
                        reason = self.account.blocked_reason() or "entry refused"
                        blocked.append({"time": ts, "reason": reason, "wanted": intent.reason})
                    else:
                        pending = intent
                elif intent.action == "exit" and self.account.position.is_open:
                    pending = intent

            equity_times.append(ts)
            equity_values.append(self.account.equity(float(bar["close"])))

        curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_times), name="equity")
        return BacktestResult(
            symbol=symbol,
            equity_curve=curve,
            trades=trades,
            blocked=blocked,
            starting_equity=self.account.starting_equity,
        )
