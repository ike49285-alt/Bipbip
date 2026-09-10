"""What $50 can actually express in SPY options, and the delta/theta balance.

The arithmetic that makes options attractive is real: a contract is 100
shares, so a $0.02 move in the premium is $2.00 gross, about $1.90 after the
regulatory pass-through. Two dollars on a fifty dollar account is 4% in one
trade, which nothing in the equity book can touch.

Two things sit between that and the money. A contract is priced per share but
sold in lots of 100, so $50 buys only contracts quoted under $0.50 - and the
things quoted that cheap are cheap for a reason. And the bid-ask spread on an
option is quoted in the same pennies as the move: a $0.02 gain is exactly the
width of a typical quote, so whether it is profit or the spread depends
entirely on which contract you are in.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.options import pricing as bs
from bipbip.options.overlay import FEE_PER_CONTRACT, MULTIPLIER
from bipbip.options.synth import quote_spread
from bipbip.data.store import BarStore

BUDGET = 50.0
MIN_PER_SESSION = 390


def spy_spot_and_vol():
    """Spot, and the IV a buyer actually pays.

    Options do not trade at realised volatility. They trade above it - the
    variance risk premium - which is the single largest documented cost of
    being long options and the reason selling them is the crowded side. Pricing
    this analysis at realised vol would quietly hand the buyer that premium and
    understate every drag below.
    """
    from bipbip.options.iv import DEFAULT_VARIANCE_RISK_PREMIUM, iv_floor_for

    b = BarStore("data/bars").load("SPY", "1d").dropna()
    S = float(b["close"].iloc[-1])
    r = np.log(b["close"] / b["close"].shift(1)).dropna()
    realised = max(float(r.tail(60).std() * np.sqrt(252)), iv_floor_for("SPY"))
    iv = max(realised * DEFAULT_VARIANCE_RISK_PREMIUM, iv_floor_for("SPY"))
    return S, iv


def affordability(S, sigma):
    """Which contracts a $50 account can even buy."""
    print(f"SPY = ${S:.2f}, 60-day realised vol = {sigma:.1%}\n")
    print("What $50 buys. A contract costs 100x the quoted premium, and there")
    print("are no fractional options, so anything over $0.50 is out of reach.\n")
    print(f"{'expiry':>8} {'strike':>9} {'moneyness':>10} {'premium':>9} "
          f"{'cost':>9} {'delta':>7} {'affordable':>11}")

    for days, dlabel in ((0, "0DTE"), (1, "1DTE"), (7, "7DTE"), (30, "30DTE")):
        minutes = max(days * MIN_PER_SESSION, 60)   # 0DTE = one hour left
        T = bs.minutes_to_years(minutes)
        for otm in (0.0, 0.005, 0.01, 0.02):
            K = round(S * (1 - otm))                 # puts: strike below spot
            prem = float(bs.price(S, K, T, 0.04, sigma, bs.PUT))
            cost = prem * MULTIPLIER + FEE_PER_CONTRACT
            d = float(bs.delta(S, K, T, 0.04, sigma, bs.PUT))
            ok = "yes" if cost <= BUDGET else "no"
            print(f"{dlabel:>8} {K:>9.0f} {otm*100:>9.1f}% {prem:>9.2f} "
                  f"{cost:>9.2f} {d:>7.3f} {ok:>11}")


def spread_reality(S, sigma):
    """The $0.02 move, against the spread you pay to capture it."""
    print("\n\nA $0.02 move is $2.00 a contract. Here is what the round trip")
    print("costs on the same contract, so the two can be compared directly.\n")
    print(f"{'expiry':>8} {'premium':>9} {'spread':>8} {'spread $':>10} "
          f"{'fees $':>8} {'cost to trade':>14} {'net on $0.02':>13}")

    for days, dlabel in ((0, "0DTE"), (1, "1DTE"), (7, "7DTE"), (30, "30DTE")):
        minutes = max(days * MIN_PER_SESSION, 60)
        T = bs.minutes_to_years(minutes)
        K = round(S)
        prem = float(bs.price(S, K, T, 0.04, sigma, bs.PUT))
        spread = float(quote_spread(np.array([prem]))[0])
        spread_dollars = spread * MULTIPLIER          # paid once, round trip
        fees = 2 * FEE_PER_CONTRACT
        total = spread_dollars + fees
        net = 2.00 - total
        print(f"{dlabel:>8} {prem:>9.2f} {spread:>8.3f} {spread_dollars:>10.2f} "
              f"{fees:>8.2f} {total:>14.2f} {net:>+13.2f}")


def spy_move_distribution():
    """Empirical |move| over a holding period, from the minute archive."""
    from bipbip.data.sessions import restrict_to_rth

    bars = restrict_to_rth(BarStore("data/bars").load("SPY", "1m")).dropna()
    px, day = bars["close"], pd.Series(bars.index.normalize(), index=bars.index)
    out = {}
    for h in (30, 120):
        fwd = px.shift(-h) / px - 1.0
        r = fwd[day.shift(-h) == day].dropna()
        out[h] = float(r.abs().median())
    return out


def breakeven_hit_rate(S, sigma):
    """The honest bar, with the option repriced rather than approximated.

    Delta against theta is the right frame but the wrong arithmetic on its own,
    because it is linear and a long option is not: gamma means a favourable
    move gains more than an equal adverse move loses, and that convexity is
    exactly what you are paying theta for. So both legs are FULLY REPRICED at
    the exit clock rather than estimated from delta.

    What comes out is the same bar used for the equity horizons: the hit rate
    required to break even. If a win pays G and a loss costs L, the break-even
    is L / (G + L). A coin flip is 50%.
    """
    moves = spy_move_distribution()
    print("\n\nBreak-even hit rate, with both legs fully repriced so gamma is")
    print("counted. The move is SPY's own median over the holding period, so")
    print("this asks: on a typical move, how often must you be right?\n")
    print(f"{'expiry':>8} {'hold':>6} {'premium':>8} {'cost $':>8} "
          f"{'win $':>8} {'lose $':>8} {'breakeven':>10} {'affordable':>11}")

    for days, dlabel in ((0, "0DTE"), (1, "1DTE"), (7, "7DTE"), (30, "30DTE")):
        # Minutes remaining at entry: the whole expiry session, plus a full
        # session for each day before it. An earlier version collapsed 0DTE and
        # 1DTE onto the same clock and priced them identically.
        minutes_left = (days + 1) * MIN_PER_SESSION
        K = round(S)
        for hold, mv in moves.items():
            if hold >= minutes_left:
                continue
            T_in = bs.minutes_to_years(minutes_left)
            T_out = bs.minutes_to_years(minutes_left - hold)
            prem = float(bs.price(S, K, T_in, 0.04, sigma, bs.PUT))
            spread = float(quote_spread(np.array([prem]))[0])
            cost = spread * MULTIPLIER + 2 * FEE_PER_CONTRACT

            # A put gains when SPY falls, so "right" is a move DOWN.
            up = float(bs.price(S * (1 + mv), K, T_out, 0.04, sigma, bs.PUT))
            down = float(bs.price(S * (1 - mv), K, T_out, 0.04, sigma, bs.PUT))
            win = (down - prem) * MULTIPLIER - cost
            lose = (prem - up) * MULTIPLIER + cost
            be = lose / (win + lose) if (win + lose) > 0 else float("nan")
            afford = "yes" if prem * MULTIPLIER + FEE_PER_CONTRACT <= BUDGET else "no"
            be_txt = f"{be*100:.1f}%" if 0 < be <= 1 else "impossible"
            print(f"{dlabel:>8} {hold:>5}m {prem:>8.2f} {cost:>8.2f} "
                  f"{win:>8.2f} {lose:>8.2f} {be_txt:>10} {afford:>11}")

    print(f"\n(SPY median move: {moves[30]*1e4:.1f} bp over 30 min, "
          f"{moves[120]*1e4:.1f} bp over 120 min.)")


def random_entry_drag(S, sigma):
    """What buying costs with NO signal, over the full move distribution.

    The median-move test above is unfair to options: they are convex, so their
    payoff lives in the tail rather than the median, and a test built on the
    typical move charges them theta without crediting the gamma they bought it
    for. This reprices against EVERY move in the archive, which credits the
    tail in full. Whatever remains is the drag a directional edge has to beat.
    """
    from bipbip.data.sessions import restrict_to_rth

    m = restrict_to_rth(BarStore("data/bars").load("SPY", "1m")).dropna()
    px, day = m["close"], pd.Series(m.index.normalize(), index=m.index)

    print("\n\nDrag on a randomly-timed put, full distribution, gamma credited.")
    print("This is the hurdle a signal must clear, not a verdict on options.\n")
    print(f"{'expiry':>8} {'hold':>6} {'premium':>9} {'mean P&L':>10} "
          f"{'P(win)':>8} {'% of premium':>13} {'affordable':>11}")

    for days, dlabel in ((0, "0DTE"), (1, "1DTE"), (7, "7DTE"), (30, "30DTE")):
        minutes_left = (days + 1) * MIN_PER_SESSION
        K = round(S)
        for hold in (30, 120):
            if hold >= minutes_left:
                continue
            fwd = px.shift(-hold) / px - 1.0
            r = fwd[day.shift(-hold) == day].dropna().values
            T_in = bs.minutes_to_years(minutes_left)
            T_out = bs.minutes_to_years(minutes_left - hold)
            prem = float(bs.price(S, K, T_in, 0.04, sigma, bs.PUT))
            cost = (float(quote_spread(np.array([prem]))[0]) * MULTIPLIER
                    + 2 * FEE_PER_CONTRACT)
            pnl = (bs.price(S * (1 + r), K, T_out, 0.04, sigma, bs.PUT)
                   - prem) * MULTIPLIER - cost
            aff = "yes" if prem * MULTIPLIER + FEE_PER_CONTRACT <= BUDGET else "no"
            print(f"{dlabel:>8} {hold:>5}m {prem:>9.2f} {pnl.mean():>10.2f} "
                  f"{(pnl > 0).mean():>7.0%} "
                  f"{pnl.mean() / (prem * MULTIPLIER) * 100:>12.1f}% {aff:>11}")


def within_budget(S, sigma):
    """Every contract a $50 account can buy, and what it does."""
    from bipbip.data.sessions import restrict_to_rth

    m = restrict_to_rth(BarStore("data/bars").load("SPY", "1m")).dropna()
    px, day = m["close"], pd.Series(m.index.normalize(), index=m.index)
    fwd = px.shift(-30) / px - 1.0
    r = fwd[day.shift(-30) == day].dropna().values

    print("\n\nEverything a $50 account can actually buy, held 30 minutes.")
    print("Cheap options are cheap because they are unlikely to pay, and the")
    print("spread is quoted in the same pennies whatever the premium.\n")
    print(f"{'contract':>30} {'premium':>9} {'cost':>8} {'spread/prem':>12} "
          f"{'mean P&L':>10} {'P(win)':>8} {'per trade':>11}")

    any_found = False
    for days, dlabel in ((0, "0DTE"), (1, "1DTE"), (2, "2DTE")):
        minutes_left = (days + 1) * MIN_PER_SESSION
        for otm in (0.002, 0.003, 0.005, 0.0075, 0.01, 0.015):
            K = round(S * (1 - otm))
            T_in = bs.minutes_to_years(minutes_left)
            prem = float(bs.price(S, K, T_in, 0.04, sigma, bs.PUT))
            cost_to_own = prem * MULTIPLIER + FEE_PER_CONTRACT
            if cost_to_own > BUDGET or prem < 0.02:
                continue
            any_found = True
            spread = float(quote_spread(np.array([prem]))[0])
            rt = spread * MULTIPLIER + 2 * FEE_PER_CONTRACT
            pnl = (bs.price(S * (1 + r), K,
                            bs.minutes_to_years(minutes_left - 30),
                            0.04, sigma, bs.PUT) - prem) * MULTIPLIER - rt
            print(f"{dlabel + f' {otm*100:.2f}% OTM K={K:.0f}':>30} {prem:>9.2f} "
                  f"{cost_to_own:>8.2f} {spread/prem*100:>11.0f}% "
                  f"{pnl.mean():>10.2f} {(pnl > 0).mean():>7.0%} "
                  f"{pnl.mean()/cost_to_own*100:>10.1f}%")
    if not any_found:
        print("   nothing within budget that is not already worthless")


def main():
    S, sigma = spy_spot_and_vol()
    print(f"Priced at {sigma:.1%} implied vol.\n")
    affordability(S, sigma)
    spread_reality(S, sigma)
    breakeven_hit_rate(S, sigma)
    random_entry_drag(S, sigma)
    within_budget(S, sigma)


if __name__ == "__main__":
    main()
