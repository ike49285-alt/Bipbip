"""Cross-sectional portfolio backtester.

Holds many symbols at once and rebalances toward weights a strategy chooses,
which is what "rank the universe and trade the best" requires and the
single-symbol engine cannot express.

The cash-account rules carry over and bite harder here. Sale proceeds settle
T+1, so a portfolio cannot rotate out of one name and into another on the same
day: the sell frees cash that is not spendable until tomorrow. That friction is
modelled rather than waved away, because it is the difference between a
rebalance that works on paper and one that a broker rejects. Unfunded buys are
recorded instead of being silently filled.

Ordering matches the single-symbol engine: a strategy decides on bar i's close
and the resulting trades fill at bar i+1's open.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import BUY, SELL, CostModel
from .panel import Panel


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    trades: list = field(default_factory=list)
    weights: pd.DataFrame | None = None
    unfunded: list = field(default_factory=list)
    stops: list = field(default_factory=list)
    starting_equity: float = 0.0
    universe: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.starting_equity


@dataclass
class StopPolicy:
    """A protective stop on each position, checked once per bar.

    Three details decide whether a stop backtest is honest, and all three are
    modelled here rather than assumed away:

    FILL PRICE. A stop is not a promise to sell at the stop price. It becomes a
    market order once touched, so if the bar OPENS below the level the fill is
    the open, not the level. Backtests that always fill at the stop price
    manufacture free downside protection - exactly on the gap days the stop is
    supposed to earn its keep.

    THE TRAILING REFERENCE. For a trailing stop the high-water mark is taken
    through the PREVIOUS bar. Using the current bar's high would let a high
    that may have occurred after the low raise the level that the low is then
    checked against, which is lookahead dressed as risk management.

    RE-ENTRY. A stop that sells is worthless if the next rebalance buys the
    same thing straight back; that is a spread-paying round trip and no risk
    reduction. `lockout_days` is how long the symbol stays untouchable, and it
    is the parameter that decides whether stops help or merely churn.
    """

    kind: str = "trailing"          # "trailing" (from the high) or "fixed" (from entry)
    pct: float = 0.10               # distance below the reference
    lockout_days: int = 21          # bars before the symbol may be bought again
    slippage_bps: float = 0.0       # extra cost on a stop fill, beyond the spread

    def __post_init__(self):
        if self.kind not in ("trailing", "fixed"):
            raise ValueError(f"kind must be 'trailing' or 'fixed', got {self.kind!r}")
        if not 0.0 < self.pct < 1.0:
            raise ValueError(f"pct must be a fraction between 0 and 1, got {self.pct}")


@dataclass
class PortfolioContext:
    """What a cross-sectional strategy may see: everything up to bar i."""

    panel: Panel
    indicators: dict
    i: int
    tradeable: pd.Index
    current_weights: dict
    equity: float
    date: dt.date

    @property
    def closes(self) -> pd.DataFrame:
        return self.panel.closes.iloc[: self.i + 1]

    def ind(self, name: str) -> pd.Series:
        """Row `i` of a precomputed indicator frame."""
        return self.indicators[name].iloc[self.i]


class PortfolioStrategy:
    """Base class. Subclasses return target weights summing to <= 1."""

    name = "portfolio_base"
    warmup_bars = 0

    def prepare(self, panel: Panel) -> dict:
        """Precompute causal indicator frames keyed by name."""
        return {}

    def target_weights(self, ctx: PortfolioContext) -> dict | None:
        """Weights to hold, summing to <= 1.

        Return `None` to express no opinion, leaving the book as it is - what a
        monthly strategy does on the other twenty days. Return an empty dict to
        ask for cash. The distinction matters: `{}` is a decision to be flat and
        is executed, `None` is an abstention.
        """
        raise NotImplementedError


class PortfolioEngine:
    def __init__(self, costs: CostModel, starting_equity: float = 50.0,
                 max_invested: float = 0.98, settle_days: int = 1,
                 allow_fractional: bool = True, rebalance_band: float = 0.05,
                 min_trade_value: float = 0.50, stop: "StopPolicy | None" = None):
        """`rebalance_band` is the weight drift tolerated before trading.

        Without it a strategy that names the same target every day still trades
        every day, because prices drift the actual weights off target
        overnight. On a 34-symbol universe that produced 24,001 trades across
        8,459 sessions - one round trip per symbol per fortnight, all of it
        paying spread to correct rounding noise. The band converts continuous
        drift into occasional, deliberate rebalances.

        `min_trade_value` suppresses dust: on a $50 account a 1% weight change
        is 50 cents, and trading it costs more than the drift it corrects.
        """
        self.costs = costs
        self.starting_equity = starting_equity
        self.max_invested = max_invested
        self.settle_days = settle_days
        self.allow_fractional = allow_fractional
        self.rebalance_band = rebalance_band
        self.min_trade_value = min_trade_value
        # None means no stop at all, which is the default: a stop is a strategy
        # decision, not a free safety feature, and it has to earn its place.
        self.stop = stop

    def run(self, panel: Panel, strategy: PortfolioStrategy,
            universe: str = "") -> PortfolioResult:
        indicators = strategy.prepare(panel)
        n = len(panel)
        dates = panel.dates

        cash = self.starting_equity
        settled = self.starting_equity
        pending_settlement: list = []
        shares: dict = {}
        pending_target: dict | None = None
        # Last price at which each symbol was seen trading. A symbol with a
        # data gap must still be VALUED - otherwise one missing bar turns the
        # entire equity curve into NaN, which is what happened before this
        # existed.
        last_px: dict = {}

        equity_values, weight_rows, trades, unfunded = [], [], [], []
        # Stop bookkeeping. `ref` is the price the stop is measured from -
        # entry for a fixed stop, the high-water mark for a trailing one - and
        # `blocked` is the bar index each stopped-out symbol becomes buyable
        # again.
        stop_ref: dict = {}
        blocked: dict = {}
        stops_fired: list = []

        for i in range(n):
            date = dates[i].date()

            matured = [amt for d, amt in pending_settlement if d <= i]
            pending_settlement = [(d, a) for d, a in pending_settlement if d > i]
            settled += sum(matured)

            opens = panel.opens.iloc[i]
            closes = panel.closes.iloc[i]
            tradeable = panel.tradeable(i)

            for sym in panel.symbols:
                px = float(closes.get(sym, np.nan))
                if np.isfinite(px) and px > 0:
                    last_px[sym] = px

            if pending_target is not None:
                equity_now = cash + sum(
                    sh * self._price(sym, opens, closes, last_px)
                    for sym, sh in shares.items()
                )
                cash, settled, shares, filled, unf = self._rebalance(
                    pending_target, shares, cash, settled, opens, tradeable,
                    equity_now, i, pending_settlement, trades, dates[i],
                )
                if unf:
                    unfunded.append({"date": dates[i], "shortfall": unf})
                pending_target = None

            # A position bought at this bar's open can be stopped out on this
            # bar's low, so the check comes after the rebalance.
            if self.stop is not None:
                for sym in list(shares):
                    entry = self._price(sym, opens, closes, last_px)
                    if sym not in stop_ref and np.isfinite(entry) and entry > 0:
                        stop_ref[sym] = entry

                lows = panel.lows.iloc[i]
                highs = panel.highs.iloc[i]
                for sym in list(shares):
                    ref = stop_ref.get(sym)
                    if ref is None or not np.isfinite(ref) or ref <= 0:
                        continue
                    level = ref * (1.0 - self.stop.pct)
                    low = float(lows.get(sym, np.nan))
                    if not np.isfinite(low) or low > level:
                        continue

                    # Touched. The fill is the stop level, or the open if the
                    # bar gapped straight through it.
                    op = float(opens.get(sym, np.nan))
                    ref_fill = min(level, op) if np.isfinite(op) and op > 0 else level
                    ref_fill *= (1.0 - self.stop.slippage_bps / 10_000.0)
                    qty = shares[sym]
                    fill = self.costs.fill_price(SELL, ref_fill, sym)
                    fees = self.costs.fees(SELL, qty, fill)
                    proceeds = qty * fill - fees
                    cash += proceeds
                    if self.settle_days <= 0:
                        settled += proceeds
                    else:
                        pending_settlement.append((i + self.settle_days, proceeds))
                    del shares[sym]
                    stop_ref.pop(sym, None)
                    blocked[sym] = i + self.stop.lockout_days
                    trades.append({"date": dates[i], "symbol": sym, "side": "sell",
                                   "shares": qty, "price": fill, "fees": fees,
                                   "reason": "stop"})
                    stops_fired.append({"date": dates[i], "symbol": sym,
                                        "level": level, "fill": fill,
                                        "gapped": bool(np.isfinite(op) and op < level)})

                # Trail the reference on what is still held, using THIS bar's
                # high - it applies from the next bar onward, never to the low
                # just tested.
                if self.stop.kind == "trailing":
                    for sym in shares:
                        hi = float(highs.get(sym, np.nan))
                        if np.isfinite(hi) and hi > stop_ref.get(sym, 0.0):
                            stop_ref[sym] = hi

            for sym in list(stop_ref):
                if sym not in shares:
                    del stop_ref[sym]

            mark = {s: self._price(s, closes, closes, last_px) for s in shares}
            held_value = sum(sh * mark[s] for s, sh in shares.items())
            equity = cash + held_value
            if not np.isfinite(equity):
                raise RuntimeError(f"equity became non-finite at {dates[i]}")
            equity_values.append(equity)
            weight_rows.append({s: (sh * mark[s] / equity if equity > 0 else 0.0)
                                for s, sh in shares.items()})

            if i < n - 1 and i >= strategy.warmup_bars and len(tradeable):
                cur_w = weight_rows[-1]
                ctx = PortfolioContext(
                    panel=panel, indicators=indicators, i=i, tradeable=tradeable,
                    current_weights=cur_w, equity=equity, date=date,
                )
                target = strategy.target_weights(ctx)
                # None means "no opinion, keep holding". An EMPTY DICT means
                # "hold nothing" and is acted on. Conflating the two made every
                # go-to-cash rule inoperative: a strategy that returned {} in a
                # downtrend stayed fully invested through it, so momentum's
                # absolute filter and the trend filter below it never once
                # fired. Both then scored identically to buy-and-hold in 2008,
                # which is what exposed this.
                if target is not None:
                    if blocked:
                        # A stopped-out symbol stays out until its lockout
                        # expires. The freed weight is held as CASH rather than
                        # spread over the survivors: redistributing it would
                        # quietly turn a stop into a concentration rule.
                        target = {sym: w for sym, w in target.items()
                                  if blocked.get(sym, -1) <= i}
                        blocked = {sym: until for sym, until in blocked.items()
                                   if until > i}
                    total = sum(max(0.0, w) for w in target.values())
                    if total > self.max_invested:
                        target = {s: max(0.0, w) * self.max_invested / total
                                  for s, w in target.items()}
                    # Trade only when the portfolio has drifted materially, or
                    # the strategy has actually changed its mind about what to
                    # hold. Correcting a fraction of a percent costs more in
                    # spread than the drift is worth.
                    names = set(target) | {s for s, w in cur_w.items() if w > 1e-6}
                    drift = max((abs(target.get(s, 0.0) - cur_w.get(s, 0.0))
                                 for s in names), default=0.0)
                    if drift > self.rebalance_band:
                        pending_target = target

        curve = pd.Series(equity_values, index=dates, name="equity")
        weights = pd.DataFrame(weight_rows, index=dates).fillna(0.0)
        return PortfolioResult(equity_curve=curve, trades=trades, weights=weights,
                               unfunded=unfunded, starting_equity=self.starting_equity,
                               universe=universe, stops=stops_fired)

    @staticmethod
    def _price(sym, primary, fallback, last_px) -> float:
        """Best available price: today's, else today's close, else last seen."""
        px = float(primary.get(sym, np.nan))
        if np.isfinite(px) and px > 0:
            return px
        px = float(fallback.get(sym, np.nan))
        if np.isfinite(px) and px > 0:
            return px
        return float(last_px.get(sym, 0.0))

    def _rebalance(self, target, shares, cash, settled, opens, tradeable,
                   equity, i, pending_settlement, trades, ts):
        """Returns updated cash, settled cash, positions, and any shortfall."""
        """Sells first, then buys limited by SETTLED cash.

        The ordering is not a convenience: in a cash account a sale's proceeds
        are unsettled until T+1, so selling A cannot fund buying B today. Buys
        are therefore capped by settled cash and any shortfall is recorded.
        """
        target = {s: w for s, w in target.items() if s in tradeable}

        # --- sells -------------------------------------------------------
        for sym in list(shares):
            px = float(opens.get(sym, np.nan))
            want = target.get(sym, 0.0) * equity
            have = shares[sym] * px if np.isfinite(px) else 0.0
            if not np.isfinite(px) or px <= 0:
                continue
            if want >= have - 1e-9:
                continue
            sell_shares = shares[sym] - (want / px)
            if sell_shares <= 0 or sell_shares * px < self.min_trade_value:
                continue
            fill = self.costs.fill_price(SELL, px, sym)
            fees = self.costs.fees(SELL, sell_shares, fill)
            proceeds = sell_shares * fill - fees
            cash += proceeds
            if self.settle_days <= 0:
                # Margin settles instantly, so proceeds fund a purchase in the
                # SAME rebalance. Routing them through the pending queue meant
                # they matured only at the next bar's start, which silently
                # imposed T+1 even when the caller asked for none - and made a
                # rotation strategy permanently unable to fund its own switch.
                settled += proceeds
            else:
                pending_settlement.append((i + self.settle_days, proceeds))
            shares[sym] -= sell_shares
            trades.append({"date": ts, "symbol": sym, "side": "sell",
                           "shares": sell_shares, "price": fill, "fees": fees})
            if shares[sym] <= 1e-9:
                del shares[sym]

        # --- buys, funded only by settled cash ---------------------------
        wants = []
        for sym, w in sorted(target.items(), key=lambda kv: -kv[1]):
            px = float(opens.get(sym, np.nan))
            if not np.isfinite(px) or px <= 0 or w <= 0:
                continue
            have = shares.get(sym, 0.0) * px
            need = w * equity - have
            if need > max(1e-9, self.min_trade_value):
                wants.append((sym, px, need))

        shortfall = 0.0
        for sym, px, need in wants:
            fill = self.costs.fill_price(BUY, px, sym)
            spend = min(need, settled)
            if spend <= 1e-9:
                shortfall += need
                continue
            qty = spend / fill
            if not self.allow_fractional:
                qty = float(int(qty))
            if qty <= 0:
                shortfall += need
                continue
            fees = self.costs.fees(BUY, qty, fill)
            cost = qty * fill + fees
            if cost > settled + 1e-9:
                qty = max((settled - fees) / fill, 0.0)
                cost = qty * fill + fees
            if qty <= 0:
                shortfall += need
                continue
            cash -= cost
            settled -= cost
            shares[sym] = shares.get(sym, 0.0) + qty
            trades.append({"date": ts, "symbol": sym, "side": "buy",
                           "shares": qty, "price": fill, "fees": fees})
            if need - spend > 1e-9:
                shortfall += need - spend

        return cash, settled, shares, True, shortfall
