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

import numpy as np
import pandas as pd

from bipbip.data.store import BarStore
from bipbip.data.webull import (robust_factors,
                                verify_factors, reconcile_sessions, drop_leaked_closes)

#: Where the harness spills large MCP results. The session id is part of the
#: path, so pinning one - as this did - makes the ingest runnable in exactly the
#: session that wrote it and nowhere else; the id it carried had already been
#: reaped. Resolution order is explicit argument, then $BIPBIP_TOOL_RESULTS,
#: then every session directory present, newest last so later dumps win.
#: Two wildcards, not one: the layout is projects/<project>/<session>/tool-results.
RESULTS_GLOB = "~/.claude/projects/*/*/tool-results"
TZ = "America/New_York"


def results_dirs(explicit: str | None = None) -> list:
    """Directories to read dumped Webull responses from."""
    if explicit:
        return [os.path.expanduser(explicit)]
    env = os.environ.get("BIPBIP_TOOL_RESULTS")
    if env:
        return [os.path.expanduser(p) for p in env.split(os.pathsep) if p]
    found = sorted(d for d in glob.glob(os.path.expanduser(RESULTS_GLOB))
                   if os.path.isdir(d))
    if not found:
        raise SystemExit(
            f"no tool-results directory under {RESULTS_GLOB}; pass one "
            f"explicitly or set BIPBIP_TOOL_RESULTS")
    return found


#: Minutes between consecutive bars, per timeframe we ingest.
SPACING = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
           "60m": 60, "1d": 1440}


def _spacing_minutes(rows: list) -> float:
    """Modal gap between consecutive bars, in minutes.

    The dumped responses record the bars but not the granularity that was
    requested, and every dump for a symbol lands in one directory. Without this
    the ingest globs all of them together: a 1m load happily absorbed the 30m
    chunks and stored 69,121 "minute" bars reaching back to 2011, reporting a
    clean run the whole way. The gap between timestamps is the only thing in the
    file that says what the bars actually are.
    """
    ts = pd.to_datetime([r["time"] for r in rows], utc=True, format="mixed")
    if len(ts) < 3:
        return float("nan")
    gaps = pd.Series(ts).sort_values().diff().dt.total_seconds().div(60).dropna()
    gaps = gaps[gaps > 0]
    return float(gaps.mode().iloc[0]) if len(gaps) else float("nan")


def collect(symbol: str, timeframe: str, results: str | None = None) -> pd.DataFrame:
    want = SPACING.get(timeframe)
    if want is None:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    dumps = [f for d in results_dirs(results)
             for f in glob.glob(f"{d}/mcp-Webull-get_stock_bars-*.txt")]
    frames, skipped = [], 0
    for f in sorted(dumps):
        try:
            blob = json.load(open(f))
        except Exception:
            continue
        for entry in blob.get("result", []):
            if entry.get("symbol") != symbol:
                continue
            rows = entry.get("result") or []
            if not rows:
                continue
            got = _spacing_minutes(rows)
            if not np.isfinite(got) or abs(got - want) > 1e-9:
                skipped += 1
                continue
            frames.append(pd.DataFrame(rows))
    if skipped:
        print(f"  ignored {skipped} dumps at other granularities")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    idx = pd.to_datetime(df["time"], utc=True, format="mixed").dt.tz_convert(TZ)
    out = df[["open", "high", "low", "close", "volume"]].astype("float64")
    out.index = pd.DatetimeIndex(idx, name="time")
    return out[~out.index.duplicated(keep="last")].sort_index()


def main(symbol: str, timeframe: str, results: str | None = None) -> None:
    raw = collect(symbol, timeframe, results)
    if raw.empty:
        print(f"no dumped bars found for {symbol}")
        return
    store = BarStore("data/bars")
    if timeframe == "1d":
        # Webull's DAILY bars are already split- and dividend-adjusted, so there
        # is nothing to rescale and nothing to anchor against - the anchor is
        # what this branch is creating. Rescaling here would be circular.
        info = store.append(symbol, raw, bar_size="1d")
        print(f"{symbol} 1d: stored {info['rows_after']:,} adjusted bars  "
              f"{info['start'].date()} -> {info['end'].date()}")
        return
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
        print("  worst close errors: "
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

    # Report short sessions rather than storing them quietly. Each request
    # returns a fixed bar COUNT ending at a timestamp, so stepping back by a
    # slightly wider span than the count covers leaves a sliver of every chunk
    # unfetched - which shows up as sessions holding thirty minutes instead of
    # a full day. Averaged into a bar count they vanish; listed here they are
    # obviously fetch gaps to backfill rather than days the market closed early.
    expected = {"1m": 390, "5m": 78, "15m": 26, "30m": 13, "1h": 7, "60m": 7}
    want = expected.get(timeframe)
    if want:
        per = cleaned.groupby(
            pd.DatetimeIndex([pd.Timestamp(d.date()) for d in cleaned.index])).size()
        short = per[per < want * 0.9]
        print(f"  sessions: {len(per):,} total, {int((per >= want * 0.9).sum()):,} "
              f"complete ({want} bars expected)")
        if len(short):
            print(f"  INCOMPLETE ({len(short)}): "
                  + ", ".join(f"{d.date()}={n}" for d, n in short.head(8).items())
                  + (" ..." if len(short) > 8 else ""))

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
         sys.argv[2] if len(sys.argv) > 2 else "30m",
         sys.argv[3] if len(sys.argv) > 3 else None)
