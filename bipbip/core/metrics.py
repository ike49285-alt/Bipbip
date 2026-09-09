"""Performance statistics.

Two rules govern this module. First, ratios are computed from SESSION-END
equity, not from bar-level equity: 390 marks a day inflates the observation
count and makes a Sharpe ratio look far more certain than it is. Second,
anything annualised from a short sample is labelled as such - projecting a
year from twenty sessions is arithmetic, not evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_equity(curve: pd.Series) -> pd.Series:
    """Collapse a bar-level equity curve to one observation per session."""
    if curve.empty:
        return curve
    return curve.groupby([ts.date() for ts in curve.index]).last()


def max_drawdown(curve: pd.Series) -> tuple:
    """Return ``(depth_as_fraction, peak_time, trough_time)``."""
    if curve.empty:
        return 0.0, None, None
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    trough = dd.idxmin()
    depth = float(dd.min())
    peak = curve.loc[:trough].idxmax() if depth < 0 else None
    return depth, peak, trough


def compute(result, bars: pd.DataFrame | None = None) -> dict:
    """Build the metric dictionary for a `BacktestResult`."""
    curve = result.equity_curve
    start = result.starting_equity
    out: dict = {}

    if curve.empty or start <= 0:
        return {"sessions": 0, "trades": 0, "note": "no data"}

    daily = daily_equity(curve)
    rets = daily.pct_change().dropna()
    sessions = len(daily)
    final = float(daily.iloc[-1])

    out["starting_equity"] = start
    out["final_equity"] = final
    out["total_return_pct"] = (final / start - 1.0) * 100.0
    out["sessions"] = sessions

    years = sessions / TRADING_DAYS
    out["annualised_return_pct"] = (
        ((final / start) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and final > 0 else float("nan")
    )
    out["annualised_vol_pct"] = float(rets.std() * np.sqrt(TRADING_DAYS) * 100.0) if len(rets) > 1 else float("nan")

    if len(rets) > 1 and rets.std() > 0:
        out["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS))
        # Downside deviation is the root-mean-square of returns BELOW the
        # target, measured about the target - not the standard deviation of the
        # negative subset. The latter is a common substitution and it flatters:
        # it measures spread about the mean of the losses rather than their
        # size, so a run of uniformly bad days looks almost riskless. On a
        # sample here it overstated Sortino by 50%.
        downside = np.minimum(rets.to_numpy(dtype="float64"), 0.0)
        dd = float(np.sqrt(np.mean(downside ** 2)))
        out["sortino"] = (
            float(rets.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else float("nan")
        )
    else:
        out["sharpe"] = out["sortino"] = float("nan")

    depth, _, trough = max_drawdown(curve)
    out["max_drawdown_pct"] = depth * 100.0
    out["max_drawdown_at"] = str(trough) if trough is not None else None
    out["calmar"] = (
        out["annualised_return_pct"] / abs(out["max_drawdown_pct"]) if out["max_drawdown_pct"] else float("nan")
    )

    trades = result.trades
    out["trades"] = len(trades)
    out["blocked_entries"] = len(result.blocked)

    if trades:
        pnls = np.array([t.pnl for t in trades], dtype="float64")
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        out["win_rate_pct"] = len(wins) / len(pnls) * 100.0
        out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
        out["expectancy"] = float(pnls.mean())
        gross_win, gross_loss = float(wins.sum()), abs(float(losses.sum()))
        out["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else float("inf")
        out["avg_hold_min"] = float(np.mean([t.hold_minutes for t in trades]))
        out["total_fees"] = float(sum(t.fees for t in trades))
        out["gross_pnl"] = float(pnls.sum() + out["total_fees"])
        out["cost_drag_pct_of_gross"] = (
            out["total_fees"] / abs(out["gross_pnl"]) * 100.0 if out["gross_pnl"] else float("nan")
        )
        out["trades_per_session"] = len(trades) / sessions
        exits: dict = {}
        for t in trades:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
        out["exit_breakdown"] = exits
    else:
        out["win_rate_pct"] = out["expectancy"] = 0.0
        out["profit_factor"] = float("nan")

    if bars is not None and not bars.empty:
        first, last = float(bars["close"].iloc[0]), float(bars["close"].iloc[-1])
        out["buy_hold_return_pct"] = (last / first - 1.0) * 100.0

    return out


def confidence_note(metrics: dict) -> str:
    """State plainly whether the sample supports any conclusion at all."""
    n = metrics.get("trades", 0)
    if n == 0:
        return "No trades. Nothing to conclude."
    if n < 30:
        return (
            f"{n} trades is far too few to distinguish edge from luck. Treat every "
            "ratio here as noise; the standard error on the win rate alone is about "
            f"{50 / max(n, 1) ** 0.5:.0f} percentage points."
        )
    if n < 100:
        return f"{n} trades is a weak sample. Directionally suggestive at best."
    return f"{n} trades. Enough to start taking the numbers seriously, not enough to bet the account on."


def format_report(result, metrics: dict, strategy_name: str = "") -> str:
    """Render a human-readable summary."""
    m = metrics

    def g(key, fmt="{:.2f}", default="n/a"):
        v = m.get(key)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return default
        return fmt.format(v)

    lines = [
        "=" * 62,
        f"  {strategy_name or 'strategy'}  |  {result.symbol}",
        "=" * 62,
        f"  Sessions            {m.get('sessions', 0)}",
        f"  Trades              {m.get('trades', 0)}   ({g('trades_per_session', '{:.2f}')}/session)",
        f"  Blocked entries     {m.get('blocked_entries', 0)}",
        "",
        f"  Start equity        ${g('starting_equity', '{:,.2f}')}",
        f"  Final equity        ${g('final_equity', '{:,.2f}')}",
        f"  Total return        {g('total_return_pct')}%",
        f"  Buy & hold          {g('buy_hold_return_pct')}%",
        "",
        f"  Annualised return   {g('annualised_return_pct')}%   <- extrapolated, see note",
        f"  Annualised vol      {g('annualised_vol_pct')}%",
        f"  Sharpe              {g('sharpe')}",
        f"  Sortino             {g('sortino')}",
        f"  Max drawdown        {g('max_drawdown_pct')}%",
        f"  Calmar              {g('calmar')}",
        "",
        f"  Win rate            {g('win_rate_pct')}%",
        f"  Profit factor       {g('profit_factor')}",
        f"  Expectancy/trade    ${g('expectancy')}",
        f"  Avg win / loss      ${g('avg_win')} / ${g('avg_loss')}",
        f"  Avg hold            {g('avg_hold_min', '{:.0f}')} min",
        "",
        f"  Gross P&L           ${g('gross_pnl')}",
        f"  Fees + slippage     ${g('total_fees')}   ({g('cost_drag_pct_of_gross', '{:.1f}')}% of gross)",
    ]
    if m.get("exit_breakdown"):
        parts = ", ".join(f"{k}={v}" for k, v in sorted(m["exit_breakdown"].items()))
        lines.append(f"  Exits               {parts}")
    if m.get("regime_note"):
        lines += ["", "  " + m["regime_note"]]
    lines += ["", "  " + confidence_note(m), "=" * 62]
    return "\n".join(lines)


def regime_note(daily, sessions_tested: int, window: int = 21) -> str:
    """One line describing the market the result was measured in.

    A backtest is a statement about the tape it ran on. Attaching the regime to
    the report keeps that caveat from being separated from the number it
    qualifies.
    """
    if daily is None or daily.empty:
        return ""
    close = daily["close"]
    rets = np.log(close / close.shift(1)).dropna()
    rv = (rets.rolling(window).std() * np.sqrt(TRADING_DAYS)).dropna()
    if rv.empty:
        return ""
    current, pct = float(rv.iloc[-1]), float((rv < rv.iloc[-1]).mean() * 100)
    note = (f"Regime: {current:.1%} vol, {pct:.0f}th percentile of "
            f"{len(close) / 252:.0f}y history")
    if pct < 25:
        note += f" - unusually CALM; {100 - pct:.0f}% of history was wilder."
    elif pct > 75:
        note += f" - unusually VOLATILE; {pct:.0f}% of history was calmer."
    return note
