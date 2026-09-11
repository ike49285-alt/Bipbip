"""A disciplined search for a 15-minute-to-2-hour signal.

The horizon was chosen because the cost arithmetic allows it: at 15 minutes
with a 2:1 payoff the break-even hit rate is 52.2%, which is ordinary. What
remains is whether a signal exists, and that question is easy to answer wrongly
by running enough tests until one looks good.

Three guards, applied to every hypothesis rather than to the survivors:

MULTIPLE TESTING. The battery is written down in full BEFORE any of it runs,
and the significance threshold is Bonferroni-corrected for the whole battery.
Running fifteen tests at the 5% level produces one false positive by
construction; the corrected threshold is what "significant" has to mean here.

SPLIT SAMPLE. Every result is recomputed on the first and second halves
separately. An effect present in the full sample but only one half is a
subsample masquerading as a period.

TRADEABILITY. Every signal is decided at the close of one bar and earns the
NEXT bar's open-to-close return, so nothing is measured that could not be
traded, and every edge is quoted against the round-trip cost of capturing it.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore

COST_BPS = 2.28          # SPY modelled round trip


def load(symbol="SPY"):
    b = BarStore("data/bars").load(symbol, "1h").dropna()
    b = b[b.groupby(b.index.normalize())["close"].transform("size") == 7]
    day = b.index.normalize()

    f = pd.DataFrame(index=b.index)
    f["day"] = day
    f["slot"] = b.groupby(day).cumcount()          # 0 = 09:30 ... 6 = 15:30
    f["ret"] = b["close"] / b["open"] - 1.0        # this bar, open to close

    # EVERY predictor below is stated as "known at the close of bar i", and the
    # target is bar i+1. An earlier version shifted the predictors one bar
    # further back while leaving the target where it was, so each lagged
    # hypothesis silently tested bar i-1 against bar i+1 - skipping the bar
    # that had just closed, which is the only one a trader would be reacting
    # to. A planted reversal edge went undetected until that was fixed.
    f["prev_ret"] = f["ret"]
    f["range_pos"] = ((b["close"] - b["low"])
                      / (b["high"] - b["low"]).replace(0, np.nan))
    f["prev_range_pos"] = f["range_pos"]
    f["prev_absret"] = f["prev_ret"].abs()
    # Overnight gap, known at the open of slot 0 and therefore all session.
    prev_close = b["close"].shift(1)
    f["gap"] = (b["open"] / prev_close - 1.0).where(f["slot"] == 0)
    f["gap"] = f["gap"].groupby(day).transform("first")
    # First-hour return, known once slot 0 has closed - which includes row 0.
    f["first_hour"] = f["ret"].where(f["slot"] == 0).groupby(day).transform("first")
    # Trailing volatility through bar i inclusive.
    f["vol20"] = f["ret"].rolling(20, min_periods=20).std()

    # THE TARGET: enter at the next bar's open, exit at its close.
    f["fwd"] = f["ret"].shift(-1)
    f["fwd_slot"] = f["slot"].shift(-1)
    # Never hold across a session boundary.
    f.loc[f["fwd_slot"] == 0, "fwd"] = np.nan
    return f.dropna(subset=["fwd"])


def tstat(x: pd.Series) -> tuple:
    x = x.dropna()
    if len(x) < 30:
        return np.nan, np.nan, len(x)
    t = x.mean() / x.std() * np.sqrt(len(x))
    return x.mean() * 1e4, t, len(x)


class Battery:
    """Every hypothesis, declared up front."""

    def __init__(self, f):
        self.f = f
        self.tests = []

    def add(self, name, signal, note=""):
        """`signal` is +1 long / -1 short / 0 flat, known before the bar opens."""
        self.tests.append((name, signal, note))

    def build(self):
        f = self.f
        # --- time of day: is any hour directional on its own? -------------
        for s in range(1, 7):
            self.add(f"long slot {s} ({['','10:30','11:30','12:30','13:30','14:30','15:30'][s]})",
                     (f["fwd_slot"] == s).astype(float),
                     "intraday U-shape / close auction imbalance")

        # --- hourly autocorrelation --------------------------------------
        self.add("momentum: follow last hour", np.sign(f["prev_ret"]),
                 "continuation of intraday order flow")
        self.add("reversal: fade last hour", -np.sign(f["prev_ret"]),
                 "liquidity provision to impatient flow")
        self.add("fade last hour, only if it was big",
                 -np.sign(f["prev_ret"]) * (f["prev_absret"] > f["vol20"]).astype(float),
                 "reversal should be strongest after outsized moves")

        # --- where the bar closed in its own range ------------------------
        self.add("fade a close at the high", -(f["prev_range_pos"] > 0.8).astype(float),
                 "exhaustion")
        self.add("buy a close at the low", (f["prev_range_pos"] < 0.2).astype(float),
                 "exhaustion, long side")

        # --- overnight gap ------------------------------------------------
        self.add("fade the overnight gap", -np.sign(f["gap"]),
                 "gaps overshoot and retrace")
        self.add("follow the overnight gap", np.sign(f["gap"]),
                 "gaps carry information")

        # --- first hour as a day signal -----------------------------------
        self.add("follow the first hour", np.sign(f["first_hour"]),
                 "opening range breakout")
        self.add("fade the first hour", -np.sign(f["first_hour"]),
                 "opening range fade")

        # --- volatility conditioning --------------------------------------
        self.add("long when quiet", (f["prev_absret"] < f["vol20"] * 0.5).astype(float),
                 "risk premium accrues in calm tape")
        self.add("long when wild", (f["prev_absret"] > f["vol20"] * 2.0).astype(float),
                 "compensation for bearing volatility")
        return self.tests


def run(symbol="SPY"):
    f = load(symbol)
    tests = Battery(f).build()
    n_tests = len(tests)
    # Bonferroni: the threshold for the whole battery, not for one test.
    crit = 2.807 if n_tests <= 10 else 3.02        # ~ two-sided 0.05 / n
    half = f.index[len(f) // 2]

    print(f"=== {symbol}, 1-hour bars ===")
    print(f"{len(f):,} tradeable observations, "
          f"{f['day'].nunique()} sessions "
          f"({f.index[0]:%Y-%m-%d} to {f.index[-1]:%Y-%m-%d})")
    print(f"{n_tests} hypotheses declared in advance. Bonferroni threshold for "
          f"the battery: |t| > {crit:.2f}")
    print(f"Round-trip cost to beat: {COST_BPS:.2f} bps\n")
    print(f"{'hypothesis':<42} {'bps':>7} {'t':>6} {'n':>6} "
          f"{'t 1st half':>11} {'t 2nd half':>11}")

    rows = []
    for name, sig, note in tests:
        pnl = (sig.reindex(f.index).fillna(0.0) * f["fwd"])
        active = pnl[sig.reindex(f.index).fillna(0.0) != 0]
        bps, t, n = tstat(active)
        if np.isnan(t):
            continue
        _, t1, _ = tstat(active[active.index < half])
        _, t2, _ = tstat(active[active.index >= half])
        rows.append((name, bps, t, n, t1, t2, note))
        flag = "  <<<" if abs(t) > crit else ""
        print(f"{name:<42} {bps:>7.2f} {t:>6.2f} {n:>6} "
              f"{t1:>11.2f} {t2:>11.2f}{flag}")

    survivors = [r for r in rows if abs(r[2]) > crit]
    print()
    if not survivors:
        best = max(rows, key=lambda r: abs(r[2]))
        print(f"Nothing clears the corrected threshold. Strongest was "
              f"{best[0]!r} at t={best[2]:.2f}, which needs |t|>{crit:.2f}.")
        uncorrected = [r for r in rows if abs(r[2]) > 1.96]
        print(f"{len(uncorrected)} of {len(rows)} would pass an UNCORRECTED 5% "
              f"test; {len(rows) * 0.05:.1f} are expected by chance alone.")
    else:
        for r in survivors:
            name, bps, t, n, t1, t2, note = r
            consistent = np.sign(t1) == np.sign(t2) and min(abs(t1), abs(t2)) > 1.0
            print(f"SURVIVES: {name}  {bps:+.2f} bps  t={t:.2f}")
            print(f"    halves: t={t1:.2f} / t={t2:.2f}  "
                  f"{'consistent' if consistent else 'NOT consistent - likely a subsample'}")
            print(f"    net of {COST_BPS:.2f} bps cost: {bps - COST_BPS:+.2f} bps per trade")
    return rows


if __name__ == "__main__":
    for sym in (sys.argv[1:] or ["SPY", "TQQQ"]):
        run(sym)
        print()
