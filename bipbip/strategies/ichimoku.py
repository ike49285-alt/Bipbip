"""Ichimoku Kinko Hyo, with a full stochastic as the timing filter.

The textbook long condition stacks four confirmations:

  1. price above the cloud       - the trend filter
  2. Tenkan above Kijun          - the short average above the long one
  3. the cloud itself bullish    - Senkou A leading Senkou B
  4. the lagging span above price 26 bars back

Each is a trend condition, and they overlap heavily, which is the thing worth
knowing before running it: four confirmations that mostly measure the same
quantity are not four independent votes, they are one vote counted four times.
That is testable, and `confirmations` exposes how many are required so the
question "does stacking them add anything?" can be answered rather than
assumed.

The stochastic is included because it is the natural counterpart: it is the one
component here that is NOT a trend measure. Whether an oscillator improves a
trend system or just delays it is exactly the sort of claim this project
exists to check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import indicators as ind
from ..core.panel import Panel
from ..core.portfolio import PortfolioContext, PortfolioStrategy


def _per_symbol(panel: Panel, fn) -> dict:
    """Run a per-symbol indicator across a panel, returning aligned frames."""
    out: dict = {}
    for sym in panel.symbols:
        bars = pd.DataFrame({
            "open": panel.opens[sym], "high": panel.highs[sym],
            "low": panel.lows[sym], "close": panel.closes[sym],
            "volume": panel.volumes[sym],
        }).dropna(subset=["close"])
        if bars.empty:
            continue
        res = fn(bars)
        for col in res.columns:
            out.setdefault(col, {})[sym] = res[col]
    return {col: pd.DataFrame(cols).reindex(index=panel.dates,
                                            columns=panel.symbols)
            for col, cols in out.items()}


class IchimokuCloud(PortfolioStrategy):
    """Long while the Ichimoku conditions hold; cash otherwise.

    `confirmations` is how many of the four must agree. One is the bare cloud
    filter, four is the textbook system.
    """

    name = "ichimoku"

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
                 displacement: int = 26, confirmations: int = 4,
                 symbols=None, monthly: bool = False):
        self.tenkan, self.kijun = tenkan, kijun
        self.senkou_b, self.displacement = senkou_b, displacement
        if not 1 <= confirmations <= 4:
            raise ValueError("confirmations must be between 1 and 4")
        self.confirmations = confirmations
        self.symbols = list(symbols) if symbols else None
        self.monthly = monthly
        self.warmup_bars = senkou_b + displacement + 5

    def prepare(self, panel: Panel) -> dict:
        f = _per_symbol(panel, lambda b: ind.ichimoku(
            b, self.tenkan, self.kijun, self.senkou_b, self.displacement))

        # Nullable booleans go missing during warmup on purpose. Summing votes
        # needs a number, and an undefined condition is not a vote FOR - so it
        # is counted as absent, which is what fillna(False) means HERE and is
        # stated rather than left to a default.
        def votes(name):
            return f[name].astype("boolean").fillna(False).astype(float)

        score = (votes("above_cloud") + votes("cloud_bull")
                 + votes("chikou_above")
                 + (f["tenkan"] > f["kijun"]).astype(float))

        m = panel.dates.to_period("M")
        return {"score": score,
                "defined": f["cloud_top"].notna() & f["senkou_b"].notna(),
                "ms": pd.Series(m != np.roll(m, 1), index=panel.dates)}

    def _universe(self, ctx):
        if self.symbols is None:
            return list(ctx.tradeable)
        return [s for s in self.symbols if s in ctx.tradeable]

    def target_weights(self, ctx: PortfolioContext) -> dict | None:
        if self.monthly and not bool(ctx.indicators["ms"].iloc[ctx.i]):
            return None
        score = ctx.ind("score")
        defined = ctx.ind("defined")
        picks = [s for s in self._universe(ctx)
                 if bool(defined.get(s, False))
                 and float(score.get(s, 0.0)) >= self.confirmations]
        if not picks:
            return {}          # nothing qualifies: hold cash
        return {s: 1.0 / len(picks) for s in picks}


class IchimokuStochastic(IchimokuCloud):
    """Ichimoku for direction, full stochastic for entry timing.

    The stochastic adds the one thing the cloud does not measure: where price
    sits inside its own recent range. `entry_below` refuses to buy a market
    already at the top of its range, which is the standard argument for pairing
    an oscillator with a trend system.
    """

    name = "ichimoku_stoch"

    def __init__(self, k_window: int = 14, k_smooth: int = 3, d_smooth: int = 3,
                 entry_below: float = 80.0, exit_above: float | None = None,
                 **kw):
        super().__init__(**kw)
        self.k_window, self.k_smooth, self.d_smooth = k_window, k_smooth, d_smooth
        self.entry_below = entry_below
        self.exit_above = exit_above
        self.warmup_bars = max(self.warmup_bars,
                               k_window + k_smooth + d_smooth + 5)

    def prepare(self, panel: Panel) -> dict:
        base = super().prepare(panel)
        st = _per_symbol(panel, lambda b: ind.full_stochastic(
            b, self.k_window, self.k_smooth, self.d_smooth))
        base["stoch_k"] = st["stoch_k"]
        return base

    def target_weights(self, ctx: PortfolioContext) -> dict | None:
        if self.monthly and not bool(ctx.indicators["ms"].iloc[ctx.i]):
            return None
        score, defined = ctx.ind("score"), ctx.ind("defined")
        k = ctx.ind("stoch_k")
        held = {s for s, w in ctx.current_weights.items() if w > 1e-6}

        picks = []
        for s in self._universe(ctx):
            if not bool(defined.get(s, False)):
                continue
            kv = float(k.get(s, np.nan))
            trending = float(score.get(s, 0.0)) >= self.confirmations
            if not trending:
                continue
            if s in held:
                # An overbought reading only closes a position when the caller
                # asked for that; otherwise the trend filter alone holds it.
                if self.exit_above is not None and np.isfinite(kv) and kv > self.exit_above:
                    continue
                picks.append(s)
            elif np.isfinite(kv) and kv < self.entry_below:
                picks.append(s)

        if not picks:
            return {}
        return {s: 1.0 / len(picks) for s in picks}
