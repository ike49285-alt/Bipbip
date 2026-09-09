"""Merge Webull bar dumps into the project's store, on the archive's own units.

Large MCP results are spilled to files rather than returned inline, which is
what makes a fifteen-year pull affordable: the bars are parsed off disk instead
of being copied through the conversation.

The one thing this script must not do is trust the prices as they arrive.
Webull's intraday feed is raw while the project's daily archive is adjusted, so
every bar is rescaled onto the adjusted series before it is stored, and the
rescaling is reported rather than assumed - see bipbip.data.webull.
"""
import sys, pathlib, json, glob, os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from bipbip.data.store import BarStore
from bipbip.data.webull import (rescale_to_adjusted, robust_factors,
                                verify_factors, reconcile_sessions, drop_leaked_closes)

RESULTS = os.path.expanduser(
    "~/.claude/projects/-home-user-Bipbip/"
    "67fd2659-0adf-539d-b6d6-3b1c7d213011/tool-results")
TZ = "America/New_York"


def collect(symbol: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(f"{RESULTS}/mcp-Webull-get_stock_bars-*.txt")):
        try:
            blob = json.load(open(f))
        except Exception:
            continue
        for entry in blob.get("result", []):
            if entry.get("symbol") != symbol:
                continue
            rows = entry.get("result") or []
            if rows:
                frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    idx = pd.to_datetime(df["time"], utc=True, format="mixed").dt.tz_convert(TZ)
    out = df[["open", "high", "low", "close", "volume"]].astype("float64")
    out.index = pd.DatetimeIndex(idx, name="time")
    return out[~out.index.duplicated(keep="last")].sort_index()


def main(symbol: str, timeframe: str) -> None:
    raw = collect(symbol)
    if raw.empty:
        print(f"no dumped bars found for {symbol}")
        return
    store = BarStore("data/bars")
    daily = store.load(symbol, "1d")
    factors = robust_factors(raw, daily["close"])
    steps = verify_factors(factors)
    print(f"{symbol} {timeframe}: {len(raw):,} raw bars  "
          f"{raw.index[0].date()} -> {raw.index[-1].date()}")
    print(f"  adjustment {factors.iloc[0]:.5f} -> {factors.iloc[-1]:.5f}, "
          f"{len(steps)} corporate actions")

    # Find sessions the independent daily record contradicts, drop the bar that
    # straddles the closing bell, then check again: a session still wrong after
    # that is not a leaked close but a different problem, and is dropped whole.
    report = reconcile_sessions(raw, daily, factors)
    n_bad = int(report["bad"].sum())
    print(f"  sessions contradicted by the daily bar: {n_bad:,} of {len(report):,} "
          f"({n_bad/len(report):.1%})")
    if n_bad:
        worst = report[report["bad"]].sort_values("close_err", ascending=False)
        print(f"  worst close errors: "
              + ", ".join(f"{d.date()} {e:.1%}"
                          for d, e in worst["close_err"].head(5).items()))
    cleaned = drop_leaked_closes(raw, report)
    again = reconcile_sessions(cleaned, daily, factors, check_close=False)
    still = set(again.index[again["bad"]])
    if still:
        day = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in cleaned.index])
        cleaned = cleaned.loc[~pd.Series(day, index=cleaned.index).isin(still).to_numpy()]
        print(f"  dropped {len(still):,} sessions still disagreeing after the fix")
    print(f"  removed {len(raw) - len(cleaned):,} contaminated bars")

    adj = rescale_to_adjusted_with(cleaned, factors)
    info = store.append(symbol, adj, bar_size=timeframe)
    print(f"  stored {info['rows_after']:,} bars  "
          f"{info['start'].date()} -> {info['end'].date()}")


def rescale_to_adjusted_with(raw, factors):
    day = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in raw.index])
    import numpy as np
    f = factors.reindex(day).to_numpy()
    keep = np.isfinite(f)
    out = raw.loc[keep].copy()
    f = f[keep]
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].to_numpy() * f
    out["volume"] = out["volume"].to_numpy() / f
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "TQQQ",
         sys.argv[2] if len(sys.argv) > 2 else "30m")
