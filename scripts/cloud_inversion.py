"""Is resistance the cloud colour DISAGREEING between timeframes?

The observation this comes from: "the top of today never made it above the
30-minute cloud - as above, so below." The claim is not that a cloud is
resistance, which mtf_cloud.py already tested by measuring approaches from
below. It is that the two timeframes DISAGREEING is the signal - a rally with a
bullish cloud underneath it running into a bearish cloud above.

Nothing in this repo could have detected that. Every Ichimoku test here has
been single-timeframe, and mtf_cloud tested a cruder proxy: proximity to the
higher cloud, ignoring colour entirely.

THE CONFOUND THAT DECIDES THE TEST. "Higher-timeframe cloud is bearish, so
returns are worse" is not an inversion effect - it is a trend signal, and a
well-known one. The same goes for the lower timeframe on its own. Only the
INTERACTION is the claim, so this is a 2x2 on (higher colour, lower colour)
with both main effects estimated alongside it. With +/-1 coding the product
term is +1 when the timeframes AGREE and -1 when they are INVERTED, so its
coefficient is exactly the effect of agreement net of both trends. If the
inversion story is right that coefficient is negative and significant while
the main effects do not explain it away. This is the same shape as the break
timing test, which showed +0.10 AUC until the clock was added and then +0.002.

The user's version is asymmetric - a RALLY capped from above - so the specific
cell (lower bullish, higher bearish) is also reported against what the two main
effects alone predict for it.

THREE THINGS THIS REPO HAS LEARNED, APPLIED HERE.

Overlapping windows inflate t by sqrt(horizon), so the sample steps by the
forward horizon rather than using every bar. Bars within a session are
correlated, so standard errors are clustered by date - the first script here to
do it through a shared helper rather than a hand-rolled two-sample t. And a
single arbitrary placebo is one draw: mtf_cloud displaced its cloud by 997
bars and compared against that one number. Here the higher-timeframe colour is
ROTATED by 1,000 random offsets with wraparound, which preserves the serial
structure of both series and destroys only their alignment, giving a
permutation p-value instead of a single comparison.

RESULT: the inversion is not there. The disagreement carries nothing once the
two trends are accounted for.

    pair             interaction   t      rotation p
    30m vs 2-hour       +20.2    1.46        0.122
    30m vs daily         +8.1    0.56        0.505
    30m vs weekly        -2.4   -0.14        0.884

Best pair p=0.122, and 0.366 after Bonferroni over the three. The sign is not
even stable: agreement looks worth +20 bps against the 2-hour cloud and -2
against the weekly. The asymmetric version fares no better - a rally under a
bearish higher cloud returns -6.1 bps where the two trends alone predict +4.0,
a gap far inside the rotation null's 12.7 bps spread.

WHAT THE 2x2 DID SURFACE, AND WHY IT IS NOT A NEW FINDING. The LOWER
timeframe's own colour carries a large contrarian effect: a bullish 30-minute
cloud is followed by returns about 43 bps WORSE over the next session than a
bearish one, rotation p=0.004 against the 2-hour pair and 0.001 against the
daily. Bearish-cloud sessions return +36.5 bps and bullish-cloud sessions
-1.6.

Three reasons that is not an edge, in descending order of importance.

It is already in this repo. ichimoku_sweep.py found the textbook four-
confirmation system returns 3.33% against 10.75% for holding, with exposure and
return falling monotonically as each bullish confirmation is added, because the
days it selects are worse. This is the same effect measured as a coefficient
rather than as a strategy, which is a better measurement of a known thing, not
a discovery.

It is decaying. Split into thirds against the daily cloud: -70.7 bps (t=-3.14),
-61.3 (t=-2.16), -27.2 (t=-1.09). The most recent five years are not
significant in any of the three pairings. That is the shape of something being
arbitraged away, and it is the same failure that disqualified levered risk
parity - an average carried by the older half of the sample.

It was chosen by reading the output. The interaction was specified in advance;
this was not. Nine coefficients were on screen, and picking the largest after
the fact is the best-of-N problem in its purest form - though the floor is
mild here (t=2.10 at N=9) and the real objection is the period decay.

The honest summary: the observation that started this - a rally failing at a
higher-timeframe cloud - does not survive being separated from the trend that
comes with it. What is left is the already-known fact that Ichimoku's bullish
readings pick bad days, which is a reason not to use the indicator long rather
than a reason to trade it short.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bipbip.core.indicators import ichimoku
from bipbip.core.stats import ols_cluster
from bipbip.data.store import BarStore

FWD = 13        # one session at 30-minute bars
ROTATIONS = 1000
AGG = {"open": "first", "high": "max", "low": "min",
       "close": "last", "volume": "sum"}


def colour_pair(base, higher_rule):
    """Lower- and higher-timeframe cloud colour on the base grid, causally."""
    lo_ich = ichimoku(base)
    hi = base.resample(higher_rule).agg(AGG).dropna()
    if len(hi) < 200:
        return None
    hi_ich = ichimoku(hi)
    # A higher-timeframe value is usable only once its bar has CLOSED: shift
    # one higher bar, THEN forward-fill onto the base grid. Ichimoku's senkou
    # spans are already displaced forward, so no further lag is needed - row i
    # is computed from bars at or before i.
    H = hi_ich["cloud_bull"].shift(1).reindex(base.index, method="ffill")
    return lo_ich["cloud_bull"], H


def sample(base, L, H):
    """Non-overlapping rows carrying both colours and a forward return."""
    c = base["close"]
    fwd = np.log(c.shift(-FWD) / c) * 1e4
    # A nullable boolean is NA where the cloud is undefined - 78 bars of
    # warm-up. float() turns that into NaN rather than silently choosing False.
    l = L.astype("float64").to_numpy()
    h = H.astype("float64").to_numpy()
    y = fwd.to_numpy()
    ok = np.isfinite(l) & np.isfinite(h) & np.isfinite(y)
    idx = np.flatnonzero(ok)[::FWD]     # one observation per holding period
    dates = np.array([d.date() for d in base.index[idx]])
    return y[idx], h[idx] * 2 - 1, l[idx] * 2 - 1, dates


def fit(y, h, l, dates):
    X = np.column_stack([np.ones(len(y)), h, l, h * l])
    return ols_cluster(y, X, dates)


def rotation_null(y, h, l, dates, which, coef, seed=0):
    """Rotate one colour series past the rest, with wraparound.

    A rotation keeps every serial property of both series - the runs, the
    persistence, the base rates - and changes only which bar each colour lands
    on. mtf_cloud's displaced placebo is one draw from exactly this
    distribution; this takes a thousand. Rotate the series whose coefficient
    is under test: the higher colour for the trend and interaction terms, the
    lower colour for its own.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    lo, hi = max(FWD * 4, n // 50), n - max(FWD * 4, n // 50)
    out = np.empty(ROTATIONS)
    for i, k in enumerate(rng.integers(lo, hi, ROTATIONS)):
        hh, ll = (np.roll(h, k), l) if which == "h" else (h, np.roll(l, k))
        out[i] = fit(y, hh, ll, dates)["beta"][coef]
    return out


def rot_p(y, h, l, dates, which, coef):
    nulls = rotation_null(y, h, l, dates, which, coef)
    real = fit(y, h, l, dates)["beta"][coef]
    ge = int((np.abs(nulls) >= abs(real)).sum())
    return (ge + 1) / (len(nulls) + 1), nulls.std(ddof=1) * 2


def report(base, label, higher_rule):
    got = colour_pair(base, higher_rule)
    if got is None:
        print(f"  {label:<26} too little history at {higher_rule}")
        return None
    y, h, l, dates = sample(base, *got)
    if len(y) < 300:
        print(f"  {label:<26} only {len(y)} non-overlapping observations")
        return None

    r = fit(y, h, l, dates)
    b, t = r["beta"], r["t"]

    print(f"\n  {label}   {len(y):,} obs, {r['clusters']:,} dates, "
          f"forward {FWD} bars")
    print(f"    {'':<22}{'mean bps':>10}{'n':>9}")
    for hv, hn in ((1, "higher bull"), (-1, "higher bear")):
        for lv, ln in ((1, "lower bull"), (-1, "lower bear")):
            m = (h == hv) & (l == lv)
            tag = "agree" if hv == lv else "INVERTED"
            print(f"    {hn} / {ln:<11}{y[m].mean():>+9.1f}{m.sum():>9,}  {tag}")
    print(f"    higher-TF trend effect  {2*b[1]:>+8.1f} bps  t={t[1]:>5.2f}")
    print(f"    lower-TF trend effect   {2*b[2]:>+8.1f} bps  t={t[2]:>5.2f}")
    print(f"    AGREEMENT (interaction) {2*b[3]:>+8.1f} bps  t={t[3]:>5.2f}"
          f"   <- the claim")

    # The user's asymmetric version: a rally under a bearish higher cloud.
    cell = (h == -1) & (l == 1)
    # The two trends ALONE, so the interaction term is deliberately dropped -
    # keeping it would just reproduce the cell mean and compare it to itself.
    additive = b[0] - b[1] + b[2]
    print(f"    lower bull / higher bear: actual {y[cell].mean():>+7.1f} vs "
          f"{additive:>+7.1f} predicted by the two trends alone")

    p_int, sd_int = rot_p(y, h, l, dates, "h", 3)
    p_low, sd_low = rot_p(y, h, l, dates, "l", 2)
    print(f"    rotation null, interaction  p={p_int:.3f}  (null sd {sd_int:.1f} bps)")
    print(f"    rotation null, lower-TF     p={p_low:.3f}  (null sd {sd_low:.1f} bps)")

    # Period stability. Levered risk parity died exactly here: a real-looking
    # average carried entirely by one stretch of history.
    thirds = np.array_split(np.arange(len(y)), 3)
    parts = []
    for part in thirds:
        rp = fit(y[part], h[part], l[part], dates[part])
        parts.append((str(dates[part][0])[:7], str(dates[part][-1])[:7],
                      2 * rp["beta"][2], rp["t"][2]))
    print("    lower-TF effect by period:")
    for a, bb, eff, tt in parts:
        print(f"      {a} -> {bb}   {eff:>+8.1f} bps  t={tt:>5.2f}")
    return p_int


def main():
    t0 = time.time()
    store = BarStore("data/bars")
    base = store.load("TQQQ", "30m").dropna()
    print(f"TQQQ 30m, {len(base):,} bars "
          f"{base.index[0].date()} -> {base.index[-1].date()}")
    print("cloud colour on two timeframes; does DISAGREEMENT predict returns "
          "beyond either trend?")

    ps = []
    for rule, label in (("2h", "30m vs 2-hour"), ("1D", "30m vs daily"),
                        ("1W", "30m vs weekly")):
        p = report(base, label, rule)
        if p is not None:
            ps.append((label, p))

    print(f"\n{'-'*64}")
    if ps:
        best = min(ps, key=lambda x: x[1])
        # Three pairs tested, so the smallest p needs a multiplicity haircut.
        print(f"best pair {best[0]} at p={best[1]:.3f}; "
              f"Bonferroni over {len(ps)} pairs -> p={min(1.0, best[1]*len(ps)):.3f}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
