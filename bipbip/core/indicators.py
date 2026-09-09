"""Intraday indicators.

Every function here is causal: the value at bar `i` uses only bars `<= i`.
That property is what keeps lookahead bias out of the strategies, so any new
indicator added to this module must preserve it. Anything using `.shift(-n)`,
a centred window, or a whole-session aggregate computed up front does not
belong here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _session_key(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray([ts.date() for ts in index])


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, re-anchored at each session open."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    key = _session_key(df.index)
    pv = (typical * df["volume"]).groupby(key).cumsum()
    vol = df["volume"].groupby(key).cumsum()
    return (pv / vol.replace(0, np.nan)).rename("vwap")


def session_vwap_bands(df: pd.DataFrame, min_bars: int = 15) -> pd.DataFrame:
    """Session VWAP with its volume-weighted standard deviation.

    This exists because normalising the distance from VWAP by a one-minute ATR
    is dimensionally wrong. Displacement from session VWAP accumulates over the
    whole session and grows roughly with the square root of elapsed time, while
    ATR is a per-minute quantity. Dividing one by the other produced a median
    "stretch" of 3.7 ATR on real TQQQ data - so a 1.5-ATR threshold fired
    almost constantly, and readings of 11 ATR were ordinary rather than
    extreme.

    Sigma here is the dispersion of price around VWAP measured on the same
    clock as VWAP itself, so ``(price - vwap) / sigma`` is a genuine z-score
    that means the same thing at 09:45 and at 15:30.

    Computed from cumulative volume-weighted moments, so it stays causal.
    Sigma is NaN until `min_bars` have accumulated, because a standard
    deviation over three bars is noise pretending to be a statistic.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    key = _session_key(df.index)
    vol = df["volume"]

    cum_v = vol.groupby(key).cumsum()
    cum_pv = (typical * vol).groupby(key).cumsum()
    cum_p2v = (typical * typical * vol).groupby(key).cumsum()

    safe_v = cum_v.replace(0, np.nan)
    vwap = cum_pv / safe_v
    # Var(X) = E[X^2] - E[X]^2, volume-weighted. Clipped at zero: floating point
    # can make this marginally negative when price barely moves.
    var = (cum_p2v / safe_v) - vwap**2
    sigma = np.sqrt(var.clip(lower=0.0))

    n = pd.Series(1, index=df.index).groupby(key).cumsum()
    sigma = sigma.where(n >= min_bars)

    out = pd.DataFrame(index=df.index)
    out["vwap"] = vwap
    out["vwap_sigma"] = sigma
    return out


def zscore_from_bands(close: pd.Series, bands: pd.DataFrame,
                      min_sigma_bps: float = 1.0) -> pd.Series:
    """Z-score from bands already computed, so callers holding them do not
    recompute - and cannot drift from the flooring rule below.

    `min_sigma_bps` floors sigma so a near-motionless stretch of tape cannot
    blow the ratio up; a dispersion below a basis point of price is not a
    tradeable dislocation whatever the arithmetic says.
    """
    sigma = bands["vwap_sigma"].clip(lower=close * (min_sigma_bps / 10_000.0))
    return ((close - bands["vwap"]) / sigma).rename("vwap_z")


def vwap_zscore(df: pd.DataFrame, min_bars: int = 15, min_sigma_bps: float = 1.0) -> pd.Series:
    """Signed z-score of price relative to session VWAP."""
    return zscore_from_bands(df["close"], session_vwap_bands(df, min_bars=min_bars),
                             min_sigma_bps=min_sigma_bps)


def cost_floored_risk(close: pd.Series, atr_series: pd.Series, hurdle_bps: float,
                      min_multiple: float = 2.0) -> pd.Series:
    """Risk unit for stops and targets, floored at a multiple of trading cost.

    A stop closer than the round trip is a guaranteed loss when hit. On real
    SPY data 37% of bars had a one-minute ATR smaller than the 2.3bp round-trip
    cost, so an unfloored ATR stop was inside the noise a third of the time -
    which is what produced trades that entered and stopped out on the same bar.
    """
    floor = close * (hurdle_bps * min_multiple / 10_000.0)
    return atr_series.clip(lower=floor)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range - the unit position sizing and stops are quoted in."""
    return true_range(df).ewm(alpha=1.0 / window, adjust=False).mean().rename("atr")


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index, with the undefined cases stated honestly.

    The obvious implementation divides average gain by average loss and fills
    whatever comes back NaN with 50. That is wrong in an ASYMMETRIC way, which
    is worse than being wrong evenly: a window containing no losses at all -
    every bar up, the most overbought a tape can be - divides by zero, and the
    fill reports 50, dead neutral. The same window with no GAINS correctly
    reports 0. So the indicator was accurate at the oversold extreme and
    inverted at the overbought one, and RSI(2) meets that case constantly
    because two up bars in a row are ordinary.

    That fed a mean-reversion rule holding positions while RSI stayed under 60,
    which therefore never sold into the strongest advances, and an ML feature
    in which every maximally-overbought sample was stamped with the same value
    as every genuinely neutral one.

    Now: no losses and some gains is 100, no gains and some losses is 0, a tape
    that did not move at all is NaN because there is no reading to give, and
    the first bar is NaN because a difference needs two prices.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()

    out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.where(avg_loss > 0))
    # avg_loss == 0: fully overbought if anything was gained, undefined if not.
    no_loss = (avg_loss <= 0) & avg_loss.notna()
    out = out.mask(no_loss & (avg_gain > 0), 100.0)
    out = out.mask(no_loss & (avg_gain <= 0), np.nan)
    # A difference needs two prices, so the first bar has no reading.
    out.iloc[:1] = np.nan
    return out.rename("rsi")


def opening_range(df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    """Running high/low of the first `minutes` of each session.

    Before the range completes, the values are the running extremes so far and
    must not be traded on; `or_complete` marks when the window has closed.
    """
    key = _session_key(df.index)
    elapsed = df.groupby(key).cumcount()
    in_window = elapsed < minutes

    high = df["high"].where(in_window)
    low = df["low"].where(in_window)
    out = pd.DataFrame(index=df.index)
    out["or_high"] = high.groupby(key).cummax().groupby(key).ffill()
    out["or_low"] = low.groupby(key).cummin().groupby(key).ffill()
    out["or_complete"] = elapsed >= minutes
    return out


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume against its own trailing average - a proxy for participation."""
    avg = df["volume"].rolling(window, min_periods=window).mean()
    return (df["volume"] / avg.replace(0, np.nan)).rename("rvol")


def realised_vol(series: pd.Series, window: int = 30, bars_per_year: int = 252 * 390) -> pd.Series:
    """Annualised realised volatility from log returns."""
    r = np.log(series / series.shift(1))
    return (r.rolling(window, min_periods=window).std() * np.sqrt(bars_per_year)).rename("rvol_ann")


def minutes_since_open(index: pd.DatetimeIndex) -> pd.Series:
    """Bars elapsed in the session - the natural intraday clock."""
    key = _session_key(index)
    return pd.Series(index, index=index).groupby(key).cumcount().rename("bar_of_day")


def full_stochastic(df: pd.DataFrame, k_window: int = 14, k_smooth: int = 3,
                    d_smooth: int = 3) -> pd.DataFrame:
    """Full stochastic oscillator: %K and %D.

    Where the raw oscillator asks "where in its recent range is price?", the
    FULL version smooths that answer twice - once into %K, again into %D - with
    both periods chosen by the caller. Fast, slow and full are the same formula
    at different smoothing: fast is (n, 1, 3), slow is (n, 3, 3).

    Returns NaN, not 50, when the lookback window is flat. A motionless range
    makes the ratio 0/0, and filling it with the midpoint invents a reading of
    "perfectly neutral" out of an absence of information - which downstream
    becomes a tradeable signal the tape never gave.
    """
    low = df["low"].rolling(k_window, min_periods=k_window).min()
    high = df["high"].rolling(k_window, min_periods=k_window).max()
    span = high - low

    raw_k = 100.0 * (df["close"] - low) / span.where(span > 0)
    k = raw_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d, "stoch_raw_k": raw_k},
                        index=df.index)


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
             senkou_b: int = 52, displacement: int = 26) -> pd.DataFrame:
    """Ichimoku Kinko Hyo, with every line aligned to the bar that may USE it.

    Ichimoku is drawn with two time shifts, and both are traps in a backtest.

    THE CLOUD IS SHIFTED FORWARD. Senkou A and B are plotted `displacement`
    bars into the future, so the cloud sitting above bar i was computed from
    data at bar i - 26. Shifting forward is therefore safe: the value returned
    on row i is derived entirely from history. This is the one place a
    `.shift(+n)` is doing what the chart shows.

    THE CHIKOU SPAN IS SHIFTED BACKWARD, AND THAT IS WHERE BACKTESTS BREAK. On
    a chart the lagging line is today's close drawn 26 bars to the LEFT, so
    "chikou is above price" compares close(i) against close(i - 26). Written as
    `close.shift(-displacement)` and read at row i, the same line yields
    close(i + 26) - a price 26 bars in the future - and a strategy consuming it
    trades on tomorrow's news. That comparison is precomputed here as
    `chikou_above`, causally, so nothing downstream has to get the sign right.

    Every column on row i uses only bars at or before i.
    """
    high, low = df["high"], df["low"]

    def mid(window: int) -> pd.Series:
        return (high.rolling(window, min_periods=window).max()
                + low.rolling(window, min_periods=window).min()) / 2.0

    tenkan_sen = mid(tenkan)
    kijun_sen = mid(kijun)

    # Forward displacement: row i carries the value computed `displacement`
    # bars ago, which is exactly what the chart draws over bar i.
    span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(displacement)
    span_b = mid(senkou_b).shift(displacement)

    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

    close = df["close"]

    def flag(cond: pd.Series, *inputs: pd.Series) -> pd.Series:
        """A boolean that is MISSING where its inputs are, not False.

        A plain `a > b` with a NaN operand returns False, which asserts the
        condition is definitely not met on bars where it is simply unknown.
        This project has already shipped that bug once, in a market filter that
        read as "below its average" for thirty years of history it did not
        have. Senkou B needs 52 bars plus a 26-bar displacement, so the cloud
        is undefined for 78 bars that price data alone covers - long enough for
        a caller to mistake the gap for a signal.

        Nullable booleans make the gap explicit: `bool(pd.NA)` raises rather
        than quietly choosing a side, so a strategy has to say what it wants.
        """
        missing = pd.concat([s.isna() for s in inputs], axis=1).any(axis=1)
        return cond.astype("boolean").mask(missing, pd.NA)

    return pd.DataFrame({
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "senkou_a": span_a,
        "senkou_b": span_b,
        "cloud_top": cloud_top,
        "cloud_bottom": cloud_bottom,
        "above_cloud": flag(close > cloud_top, close, cloud_top),
        "below_cloud": flag(close < cloud_bottom, close, cloud_bottom),
        # The lagging-line comparison, stated causally. NOT close.shift(-n).
        "chikou_above": flag(close > close.shift(displacement),
                             close, close.shift(displacement)),
        # Positive when Senkou A leads B: the "bullish cloud" colour.
        "cloud_bull": flag(span_a > span_b, span_a, span_b),
    }, index=df.index)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi candles: a smoothed re-drawing of the same price series.

    HA_close is the bar's own average of open, high, low and close. HA_open is
    the running average of the PREVIOUS HA candle, which makes the series
    recursive and is what produces the long unbroken runs of one colour that
    make trends easy to read.

    THESE ARE NOT PRICES YOU CAN TRADE. An HA close is an average of four
    numbers and an HA open is an average of two earlier averages; neither ever
    appeared on an exchange. Filling an order at one is not optimistic, it is
    fictional - and because HA smooths, the fabricated price is systematically
    better than the real one at exactly the moments a signal fires. Use these to
    DECIDE and the real OHLC to FILL, which is what `ha_trend` is for.

    Causal: row i uses bars up to and including i, and nothing after.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ha_close = (o + h + l + c) / 4.0

    # HA_open is recursive: seed on the first bar, then average forward.
    ha_open = np.empty(len(df), dtype="float64")
    ha_open[:] = np.nan
    co = ha_close.to_numpy(dtype="float64")
    oo = o.to_numpy(dtype="float64")
    prev = (oo[0] + co[0]) / 2.0 if len(df) else np.nan
    for i in range(len(df)):
        ha_open[i] = prev
        if np.isfinite(prev) and np.isfinite(co[i]):
            prev = (prev + co[i]) / 2.0
        elif np.isfinite(co[i]):
            prev = co[i]
    ha_open = pd.Series(ha_open, index=df.index)

    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)
    body = ha_close - ha_open

    return pd.DataFrame({
        "ha_open": ha_open, "ha_high": ha_high,
        "ha_low": ha_low, "ha_close": ha_close,
        # The colour, as a number: +1 rising, -1 falling.
        "ha_trend": np.sign(body),
        # Body size relative to the whole candle - a doji is near zero, and a
        # run of full-bodied candles is the pattern the technique is read for.
        "ha_body_frac": (body.abs() / (ha_high - ha_low).replace(0, np.nan)),
        # An unbroken run of one colour, which is the actual signal people use.
        "ha_run": _signed_run(np.sign(body).to_numpy(dtype="float64"), df.index),
    }, index=df.index)


def _signed_run(sign: np.ndarray, index=None) -> pd.Series:
    """Length of the current unbroken run of one sign, signed by direction.

    The index is passed explicitly: returning a default RangeIndex and letting
    pandas align it into a DatetimeIndex frame silently produced a column of
    NaN, which looked like a feature that simply had no readings.
    """
    out = np.zeros(len(sign), dtype="float64")
    run = 0.0
    for i, s in enumerate(sign):
        if not np.isfinite(s) or s == 0:
            run = 0.0
        elif i > 0 and s == sign[i - 1]:
            run += s
        else:
            run = s
        out[i] = run
    return pd.Series(out, index=index)
