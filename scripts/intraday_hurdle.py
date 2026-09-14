"""What an intraday trade has to clear, measured against what it can win.

This is the question that decides whether an intraday system is possible at
all, and unlike a strategy backtest it does not need years of data: it needs
only the distribution of N-minute moves and the cost of capturing one. A short
sample estimates a spread distribution far better than it estimates an edge.

The framing is the break-even hit rate. If a trade wins +m and loses -m with
probability p, and each round trip costs c, the expected value is
p*m - (1-p)*m - c, which is zero at p = 0.5 + c/(2m). That number is the
honest bar: it says what fraction of trades a system must get right BEFORE it
has made a cent, purely to pay the spread.
"""
import sys, pathlib
# Repo ROOT first so `bipbip` resolves, then scripts/ for sibling
# imports. Only the second was here, so this ran only when something
# else had already fixed the path - an accident of import order.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np, pandas as pd

from bipbip.core.costs import CostModel
from bipbip.data.sessions import restrict_to_rth
from bipbip.data.store import BarStore

HORIZONS = [1, 2, 5, 15, 30, 60, 120, 390]


def analyse(symbol: str, costs: CostModel) -> None:
    bars = restrict_to_rth(BarStore("data/bars").load(symbol, "1m")).dropna()
    if bars.empty:
        print(f"{symbol}: no minute bars")
        return

    px = bars["close"]
    day = bars.index.normalize()
    sessions = day.nunique()
    cost_bps = costs.round_trip_cost_bps(symbol, price=float(px.iloc[-1]),
                                         shares=50.0 / float(px.iloc[-1]))

    print(f"\n=== {symbol} ===")
    print(f"{len(bars):,} minute bars over {sessions} sessions "
          f"({bars.index[0]:%Y-%m-%d} to {bars.index[-1]:%Y-%m-%d})")
    print(f"round-trip cost: {cost_bps:.2f} bps\n")
    print(f"{'hold':>6} {'median |move|':>14} {'mean |move|':>12} "
          f"{'P(|move|>cost)':>15} {'breakeven win rate':>20}")

    for h in HORIZONS:
        # Forward return over h minutes, never crossing a session boundary.
        fwd = px.shift(-h) / px - 1.0
        same_day = pd.Series(day, index=bars.index).shift(-h) == pd.Series(day, index=bars.index)
        r = fwd[same_day].dropna()
        if len(r) < 100:
            continue
        a = r.abs() * 1e4
        med, mean = float(a.median()), float(a.mean())
        beat = float((a > cost_bps).mean())
        # p = 0.5 + c/(2m), using the MEDIAN move as the payoff a trade earns.
        need = 0.5 + cost_bps / (2.0 * med) if med > 0 else float("nan")
        need_txt = f"{need*100:.1f}%" if need <= 1.0 else "impossible"
        print(f"{h:>5}m {med:>13.2f} {mean:>12.2f} {beat:>14.0%} {need_txt:>20}")

    # Where the cost sits relative to the whole distribution of 1-minute moves.
    one = (px.shift(-1) / px - 1.0)
    one = one[pd.Series(day, index=bars.index).shift(-1) == pd.Series(day, index=bars.index)]
    one = (one.abs() * 1e4).dropna()
    pct = float((one < cost_bps).mean())
    print(f"\n{pct:.0%} of one-minute moves in {symbol} are smaller than the "
          f"cost of trading one.")


def sensitivity(symbol: str) -> None:
    """The answer depends on execution quality, so vary it rather than pick.

    The default model charges 1.0 bps of slippage per side on SPY, which is
    market impact and latency rather than spread: SPY quotes a penny wide on a
    $766 share, so the spread alone is 0.13 bps and a $50 order moves nothing.
    A retail marketable order routed to a wholesaler often fills at or inside
    the quote. The truth sits somewhere in the range, and the range is wide
    enough to change the conclusion, so both ends are shown.
    """
    bars = restrict_to_rth(BarStore("data/bars").load(symbol, "1m")).dropna()
    if bars.empty:
        return
    px = bars["close"]
    day = pd.Series(bars.index.normalize(), index=bars.index)

    # The middle scenario is this symbol's OWN modelled slippage, not a shared
    # constant: TQQQ is charged 2.5 bps a side against SPY's 1.0, and reusing
    # one number across both would understate the leveraged fund's hurdle.
    default_side = CostModel().slippage_for(symbol)
    quote_side = {"SPY": 0.065, "TQQQ": 0.35}.get(symbol, 0.5)
    scenarios = [
        ("quote-only", quote_side),
        (f"model ({default_side:.1f}/side)", default_side),
        (f"poor ({default_side * 3:.1f}/side)", default_side * 3.0),
    ]
    sec_fee_bps = 0.278          # SEC fee, charged on the sell leg only

    print(f"\n--- {symbol}: break-even win rate under different execution ---")
    print(f"{'holding period':>16}" + "".join(f"{n:>22}" for n, _ in scenarios))
    for h in (1, 5, 15, 30, 60, 120):
        fwd = px.shift(-h) / px - 1.0
        r = fwd[day.shift(-h) == day].dropna()
        if len(r) < 100:
            continue
        med = float((r.abs() * 1e4).median())
        cells = ""
        for _, per_side in scenarios:
            cost = 2.0 * per_side + sec_fee_bps
            need = 0.5 + cost / (2.0 * med)
            cells += f"{(f'{need*100:.1f}%' if need <= 1.0 else 'impossible'):>22}"
        print(f"{str(h) + 'm':>16}{cells}")

    print(f"{'round trip (bps)':>16}" +
          "".join(f"{2.0 * ps + sec_fee_bps:>22.2f}" for _, ps in scenarios))


def sub_minute_note(symbol: str) -> None:
    """The 'seconds' end of the question, which no data here can reach.

    Yahoo's finest interval is one minute, so there are no sub-minute bars to
    test and no free source for them. The scaling is not in doubt, though:
    diffusive price moves grow with the SQUARE ROOT of time, so a 1-second move
    is roughly 1/sqrt(60) of a 1-minute one while the cost of trading it is
    unchanged.
    """
    bars = restrict_to_rth(BarStore("data/bars").load(symbol, "1m")).dropna()
    if bars.empty:
        return
    px = bars["close"]
    day = pd.Series(bars.index.normalize(), index=bars.index)
    one = (px.shift(-1) / px - 1.0)[day.shift(-1) == day]
    med_1m = float((one.abs() * 1e4).median())

    print(f"\n--- {symbol}: extrapolating below one minute ---")
    print("No sub-minute data exists here; this is sqrt-of-time scaling from")
    print(f"the measured 1-minute median of {med_1m:.2f} bps.\n")
    best_case = 2.0 * 0.065 + 0.278      # quote-only round trip on SPY
    print(f"{'horizon':>10} {'implied median |move|':>24} "
          f"{f'vs {best_case:.2f} bps best-case cost':>30}")
    for secs, label in ((1, "1 second"), (5, "5 seconds"), (15, "15 seconds"),
                        (30, "30 seconds"), (60, "1 minute")):
        m = med_1m * np.sqrt(secs / 60.0)
        print(f"{label:>10} {m:>23.3f} bps {m / best_case:>23.2f}x")


def asymmetric(symbol: str) -> None:
    """The symmetric formula is a simplification, so state the general one.

    p = 0.5 + c/(2m) assumes a win and a loss are the same size. A real system
    with a target and a stop is deliberately asymmetric, and a 2:1 winner needs
    a much lower hit rate. The general break-even is

        p = (risk + cost) / (reward + risk)

    which is the fair bar for a system that lets winners run. It does not
    rescue the fastest horizons - cost still dominates when the whole move is
    a basis point - but it does change what "achievable" means at 15 minutes
    and beyond.
    """
    bars = restrict_to_rth(BarStore("data/bars").load(symbol, "1m")).dropna()
    if bars.empty:
        return
    px = bars["close"]
    day = pd.Series(bars.index.normalize(), index=bars.index)

    print(f"\n--- {symbol}: break-even hit rate by reward-to-risk ---")
    print("Risk is set to the median move over the holding period; reward is a")
    print("multiple of it. Modelled execution.\n")
    cost = 2.0 * CostModel().slippage_for(symbol) + 0.278
    print(f"{'hold':>6} {'risk (bps)':>12}" +
          "".join(f"{f'{r:.1f}:1':>10}" for r in (1.0, 1.5, 2.0, 3.0)))
    for h in (5, 15, 30, 60, 120):
        fwd = px.shift(-h) / px - 1.0
        r = fwd[day.shift(-h) == day].dropna()
        if len(r) < 100:
            continue
        risk = float((r.abs() * 1e4).median())
        cells = ""
        for mult in (1.0, 1.5, 2.0, 3.0):
            p = (risk + cost) / (mult * risk + risk)
            cells += f"{(f'{p*100:.1f}%' if p <= 1.0 else 'imposs.'):>10}"
        print(f"{str(h) + 'm':>6} {risk:>12.2f}{cells}")


def main():
    costs = CostModel()
    print("Break-even win rate is the fraction of trades a system must get")
    print("right just to pay the spread, before earning anything. A coin flip")
    print("is 50%; the best systematic equity strategies run 52-55%.")
    for sym in ("SPY", "TQQQ"):
        analyse(sym, costs)
    for sym in ("SPY", "TQQQ"):
        sensitivity(sym)
    sub_minute_note("SPY")
    for sym in ("SPY", "TQQQ"):
        asymmetric(sym)


if __name__ == "__main__":
    main()
