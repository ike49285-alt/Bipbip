"""A model that chooses its own trades.

Everything else in this project hands the model a rule and asks it to grade the
rule's proposals. The primary rule decides direction, the asset and the moment;
the model only says take it or leave it. That caps what can ever be found at
whatever the hand-written rule is worth, and the rule this project has been
filtering turned out to be worth +0.1 bps at t=0.06.

Here there is no rule. Every (symbol, bar) in the panel is a candidate, and the
model answers one question:

    will this symbol beat the CROSS-SECTIONAL MEDIAN over the next N bars?

That framing does two things no per-trade metric here has managed. It removes
market drift by construction - if everything rises, the median rises with it,
and beating the median still means something - which is the exact contamination
that made shuffled labels "earn" +31.8 bps in the permutation test. And it
makes the prediction relative, so the model has to rank rather than forecast,
which is a far better-posed problem than "what will price do".

Features are per-symbol history plus the cross-sectional RANK of each, because
a rank is what a relative question needs: a 2% gain means something different
when the median is +3% than when it is -1%.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.panel import Panel


@dataclass
class CrossSectionalDataset:
    X: pd.DataFrame
    y: np.ndarray                 # 1 if the symbol beat the median
    excess: np.ndarray            # forward return minus the median, in return units
    fwd: np.ndarray               # raw forward return
    median_fwd: np.ndarray        # the median it was measured against
    symbols: np.ndarray
    dates: pd.DatetimeIndex
    bar_index: np.ndarray         # position in the panel, for benchmarking
    feature_names: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)

    def summary(self) -> str:
        return (f"{len(self):,} rows, {len(set(self.symbols))} symbols, "
                f"{self.dates.min().date()} to {self.dates.max().date()} | "
                f"{self.y.mean():.1%} above median | "
                f"mean |excess| {np.nanmean(np.abs(self.excess)) * 1e4:.0f} bps")


#: Lookbacks in BARS. On hourly data these are roughly one bar to four sessions.
RETURN_LAGS = (1, 2, 3, 5, 7, 14, 28)
VOL_WINDOWS = (14, 56)
SMA_WINDOWS = (14, 56)


def _causal_features(panel: Panel) -> dict:
    """Per-symbol features, every one using only bars at or before i."""
    c, o, h, l, v = (panel.closes, panel.opens, panel.highs,
                     panel.lows, panel.volumes)
    logret = np.log(c / c.shift(1))
    f: dict = {}

    for k in RETURN_LAGS:
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    for w in VOL_WINDOWS:
        f[f"vol_{w}"] = logret.rolling(w, min_periods=w).std()
    for w in SMA_WINDOWS:
        f[f"dist_sma_{w}"] = c / c.rolling(w, min_periods=w).mean() - 1.0

    # Where the bar closed inside its own range: 1 is on the high, 0 the low.
    span = (h - l).replace(0, np.nan)
    f["range_pos"] = (c - l) / span
    f["range_pct"] = span / c
    # Overnight or inter-bar gap, known at the open.
    f["gap"] = o / c.shift(1) - 1.0
    f["vol_ratio"] = (f["vol_14"] / f["vol_56"]).replace([np.inf, -np.inf], np.nan)
    f["volume_ratio"] = v / v.rolling(56, min_periods=56).mean().replace(0, np.nan)
    # Distance below the trailing high: a drawdown measure that needs no label.
    f["dist_high_56"] = c / c.rolling(56, min_periods=56).max() - 1.0

    f.update(_discretionary_features(panel))
    return f


def _discretionary_features(panel: Panel) -> dict:
    """Heikin-Ashi, fast stochastic and Ichimoku, computed per symbol.

    These are in because they are what a discretionary trader actually reads,
    and the model should get the same view rather than a private set of
    academic features. Whether they carry information is the question; leaving
    them out answers it by assumption.

    Heikin-Ashi contributes only its DERIVED readings - direction, body
    fraction, run length - never its prices. An HA close is the average of four
    numbers and an HA open the average of two earlier averages; neither ever
    traded, and because the technique smooths, the fabricated price is
    systematically kinder than the real one exactly when a signal fires.
    """
    from ..core import indicators as ind

    out: dict = {}
    per_sym: dict = {}
    for sym in panel.symbols:
        bars = pd.DataFrame({
            "open": panel.opens[sym], "high": panel.highs[sym],
            "low": panel.lows[sym], "close": panel.closes[sym],
            "volume": panel.volumes[sym],
        }).dropna(subset=["close"])
        if len(bars) < 120:
            continue
        ha = ind.heikin_ashi(bars)
        st = ind.full_stochastic(bars, k_window=14, k_smooth=1, d_smooth=3)
        ich = ind.ichimoku(bars, tenkan=9, kijun=26, senkou_b=52, displacement=26)

        cols = {
            "ha_trend": ha["ha_trend"],
            "ha_run": ha["ha_run"],
            "ha_body": ha["ha_body_frac"],
            # Fast stochastic: no smoothing on %K, which is what "fast" means.
            "stoch_k": st["stoch_k"],
            "stoch_kd": st["stoch_k"] - st["stoch_d"],
            # Ichimoku, as distances rather than booleans so the model can see
            # HOW far above the cloud price is, not merely that it is.
            "ich_cloud_dist": (bars["close"] - ich["cloud_top"]) / bars["close"],
            "ich_tk": (ich["tenkan"] - ich["kijun"]) / bars["close"],
            "ich_chikou": ich["chikou_above"].astype("float64"),
            "ich_cloud_thick": ((ich["cloud_top"] - ich["cloud_bottom"])
                                / bars["close"]),
        }
        for name, series in cols.items():
            per_sym.setdefault(name, {})[sym] = series

    for name, by_sym in per_sym.items():
        out[name] = pd.DataFrame(by_sym).reindex(index=panel.dates,
                                                 columns=panel.symbols)
    return out


def _add_cross_sectional_ranks(f: dict) -> dict:
    """Rank each feature across symbols at every timestamp.

    The question is relative, so the inputs should be too. A 2% gain is a
    different fact when the median is +3% than when it is -1%, and only the
    rank carries that. Ranks are also robust to the regime shifts that make raw
    levels incomparable across a thirty-year sample.
    """
    out = dict(f)
    for name, frame in f.items():
        out[f"rk_{name}"] = frame.rank(axis=1, pct=True)
    return out


def build_cross_sectional(panel: Panel, horizon: int = 7,
                          min_symbols: int = 20) -> CrossSectionalDataset:
    """Every (symbol, bar) as a candidate, labelled against its own cross-section.

    `horizon` is in bars. Entry is the NEXT bar's open, exit is the close
    `horizon` bars later, matching how the engine fills - so nothing is
    measured that could not be traded.
    """
    feats = _add_cross_sectional_ranks(_causal_features(panel))
    names = sorted(feats)

    o, c = panel.opens, panel.closes
    # Enter at the next open, exit `horizon` bars later at the close.
    entry = o.shift(-1)
    exit_ = c.shift(-horizon)
    fwd = exit_ / entry - 1.0

    # The cross-section is only meaningful with enough names quoting.
    valid = fwd.notna() & entry.notna()
    enough = valid.sum(axis=1) >= min_symbols
    fwd = fwd.where(valid)

    median = fwd.median(axis=1)
    excess = fwd.sub(median, axis=0)

    stacked = {n: feats[n].where(valid) for n in names}
    rows, ys, exs, fws, meds, syms, whens, bars = [], [], [], [], [], [], [], []

    dates = panel.dates
    keep_rows = np.flatnonzero(enough.to_numpy())
    sym_list = list(panel.symbols)
    feat_arrays = {n: stacked[n].to_numpy(dtype="float32") for n in names}
    fwd_a, exc_a, med_a = (fwd.to_numpy(dtype="float64"),
                           excess.to_numpy(dtype="float64"),
                           median.to_numpy(dtype="float64"))

    for i in keep_rows:
        col_ok = np.isfinite(fwd_a[i])
        if col_ok.sum() < min_symbols:
            continue
        idx = np.flatnonzero(col_ok)
        block = np.column_stack([feat_arrays[n][i, idx] for n in names])
        # A row with no usable feature at all is not a sample.
        usable = np.isfinite(block).any(axis=1)
        if not usable.any():
            continue
        idx, block = idx[usable], block[usable]
        rows.append(block)
        exs.append(exc_a[i, idx]); fws.append(fwd_a[i, idx])
        meds.append(np.full(len(idx), med_a[i]))
        syms.append(np.array([sym_list[j] for j in idx]))
        whens.append(np.full(len(idx), dates[i]))
        bars.append(np.full(len(idx), i))

    if not rows:
        raise ValueError("no usable cross-sections; is the panel wide enough?")

    X = pd.DataFrame(np.vstack(rows), columns=names)
    excess_v = np.concatenate(exs)
    return CrossSectionalDataset(
        X=X,
        y=(excess_v > 0).astype("float64"),
        excess=excess_v,
        fwd=np.concatenate(fws),
        median_fwd=np.concatenate(meds),
        symbols=np.concatenate(syms),
        dates=pd.DatetimeIndex(np.concatenate(whens)),
        bar_index=np.concatenate(bars),
        feature_names=names,
    )


# --------------------------------------------------------------------------
# The model, and the only evaluation that decides anything.
# --------------------------------------------------------------------------

def make_gbm(seed: int = 0, **kw):
    """Gradient-boosted trees, which suit this problem far better than an MLP.

    Tabular financial features are non-monotonic, interact, arrive on wildly
    different scales and are riddled with missing values during warmup. Trees
    handle all four natively; a small dense net handles none of them well and
    needs the imputation and scaling that a tree does not.

    The regularisation is still deliberate. Depth and leaf count are the knobs
    that turn a ranker into a memoriser, and 1.5 million rows is enough data to
    memorise convincingly.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    params = dict(max_depth=4, max_leaf_nodes=15, learning_rate=0.05,
                  max_iter=250, min_samples_leaf=200, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15,
                  random_state=seed)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def evaluate_ranker(ds: CrossSectionalDataset, make_model=make_gbm,
                    n_splits: int = 5, embargo_days: int = 5,
                    top_k: int = 5, cost_bps: float = 0.0) -> dict:
    """Walk forward, then trade the model's ranking and measure the excess.

    The score is NOT accuracy. At each timestamp the model ranks the available
    symbols and the top `top_k` are held; what is reported is the mean EXCESS
    return of that basket - its return minus the cross-section's median over
    the same bars. Excess is market-neutral by construction, so unlike every
    per-trade metric elsewhere in this project a positive number here cannot be
    market drift wearing a disguise.

    Folds are chronological and purged with a CALENDAR embargo, because samples
    pile up per timestamp: 305 symbols means 305 rows share a bar, and an
    embargo counted in rows would be no embargo at all.
    """
    dates = ds.dates
    uniq = np.array(sorted(set(dates)))
    n_ts = len(uniq)
    if n_ts < n_splits * 20:
        return {"note": f"only {n_ts} timestamps; not enough to validate"}

    Xv = ds.X.to_numpy(dtype="float32")
    fold_size = n_ts // (n_splits + 1)
    rows = []
    all_sel_excess, all_sel_dates = [], []

    for k in range(n_splits):
        train_end_ts = uniq[fold_size * (k + 1)]
        val_start_ts = train_end_ts + pd.Timedelta(days=embargo_days)
        val_end_ts = uniq[min(fold_size * (k + 2), n_ts - 1)]
        if val_start_ts >= val_end_ts:
            continue

        tr = np.flatnonzero(dates <= train_end_ts)
        va = np.flatnonzero((dates > val_start_ts) & (dates <= val_end_ts))
        if len(tr) < 2000 or len(va) < 200:
            continue
        if len(np.unique(ds.y[tr])) < 2:
            continue

        model = make_model().fit(Xv[tr], ds.y[tr])
        proba = model.predict_proba(Xv[va])[:, 1]

        # Trade it: at each timestamp hold the top_k by predicted probability.
        vdates, vexcess = dates[va], ds.excess[va]
        picked = []
        for ts in np.unique(vdates):
            m = vdates == ts
            if m.sum() < top_k * 2:
                continue
            p = proba[m]
            top = np.argsort(-p)[:top_k]
            picked.append(np.nanmean(vexcess[m][top]))
            all_sel_dates.append(ts)
        if not picked:
            continue
        picked = np.asarray(picked, dtype="float64") - cost_bps / 1e4
        all_sel_excess.extend(picked)

        rows.append({
            "fold": k, "n_train": len(tr), "n_val": len(va),
            "rebalances": len(picked),
            "excess_bps": float(np.nanmean(picked) * 1e4),
            "auc": _safe_auc(ds.y[va], proba),
        })

    if not rows:
        return {"note": "no usable folds"}

    sel = np.asarray(all_sel_excess, dtype="float64")
    sel = sel[np.isfinite(sel)]
    t = (sel.mean() / sel.std() * np.sqrt(len(sel))) if sel.std() > 0 else float("nan")
    return {
        "folds": rows,
        "rebalances": len(sel),
        "excess_bps": float(sel.mean() * 1e4),
        "t_stat": float(t),
        "hit_rate": float((sel > 0).mean()),
        "auc": float(np.nanmean([r["auc"] for r in rows])),
        "selected_excess": sel,
    }


def _safe_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        return float("nan")


def permutation_baseline(ds: CrossSectionalDataset, n: int = 20, seed: int = 0,
                         **kw) -> dict:
    """What the same procedure scores when the labels carry no information.

    Shuffling is done WITHIN each timestamp, not across the whole sample. A
    global shuffle would break the cross-sectional structure as well as the
    signal and make the null far too easy to beat - the model would be scored
    against a world where the median itself is meaningless.
    """
    rng = np.random.default_rng(seed)
    real = evaluate_ranker(ds, **kw)
    if "excess_bps" not in real:
        return real

    null = []
    dates = ds.dates
    for _ in range(n):
        shuffled = ds.excess.copy()
        for ts in np.unique(dates):
            m = np.flatnonzero(dates == ts)
            shuffled[m] = shuffled[rng.permutation(m)]
        fake = CrossSectionalDataset(
            X=ds.X, y=(shuffled > 0).astype("float64"), excess=shuffled,
            fwd=ds.fwd, median_fwd=ds.median_fwd, symbols=ds.symbols,
            dates=ds.dates, bar_index=ds.bar_index,
            feature_names=ds.feature_names)
        r = evaluate_ranker(fake, **kw)
        if "excess_bps" in r:
            null.append(r["excess_bps"])

    null = np.asarray(null, dtype="float64")
    obs = real["excess_bps"]
    return {
        "observed_bps": obs, "t_stat": real["t_stat"], "auc": real["auc"],
        "null_mean_bps": float(null.mean()) if len(null) else float("nan"),
        "null_std_bps": float(null.std()) if len(null) else float("nan"),
        "null_best_bps": float(null.max()) if len(null) else float("nan"),
        "p_value": float((np.sum(null >= obs) + 1) / (len(null) + 1)) if len(null) else float("nan"),
        "n_permutations": len(null),
    }
