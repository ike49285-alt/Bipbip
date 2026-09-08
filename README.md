# Bipbip

An intraday trading research engine for SPY / TQQQ, built for a **Reg T cash
account** on Webull.

Status: **research engine complete and tested. No edge has been demonstrated
yet, and nothing here is ready to trade.**

## Why it is shaped this way

The account type drove every design decision.

US equities settle **T+1**, and a cash account may only buy with *settled*
funds. So:

```
09:35  buy $10k SPY        settled cash -> $0
09:40  sell                 $10k proceeds, unsettled until tomorrow
09:45  buy again            allowed, using unsettled proceeds
09:50  sell that position   GOOD FAITH VIOLATION
```

Three good-faith violations in twelve months and the broker restricts the
account to settled-cash-only for ninety days. The practical limit is therefore
**one round trip per session** — which happens to suit a "one good setup, hold
30 minutes to a few hours, flat by the close" system exactly. `CashAccount`
enforces this structurally: the violation above cannot be expressed.

## Design guarantees

These are enforced by tests, not by convention:

- **No lookahead.** A strategy decides at the close of bar `i` and fills at the
  open of bar `i+1`. `tests/test_lookahead.py` corrupts every bar after a cut
  point and asserts the equity curve before it does not move.
- **Causal indicators.** Every indicator computed on truncated history matches
  the full-history value at overlapping timestamps.
- **Pessimistic fills.** Slippage always moves against you. A bar gapping
  through a stop fills at the open, not the stop. A bar touching both stop and
  target resolves as the stop.
- **Never overnight.** Every position is force-flat before the close.
- **Never short, never unsettled.** Cash-account rules hold for all strategies.

## Costs

Modelled per round trip, since this is the hurdle any edge must clear:

| Symbol | Round-trip cost |
|--------|-----------------|
| SPY    | ~2.3 bps        |
| TQQQ   | ~5.3 bps        |

Webull charges no commission on US stocks/ETFs; SEC and FINRA fees apply to
sells only. **The SEC fee rate resets annually — verify it before trusting live
P&L.**

## The data problem, and the fix

Free intraday data (yfinance) serves only a trailing ~30 days of one-minute
bars. At one trade per day that is ~21 trades: far too few to distinguish edge
from luck.

So `BarStore` is **append-only**. Every fetch merges into the parquet archive,
deduplicating on timestamp and preferring the freshest copy of a revised bar.
History therefore accumulates well past any single provider window — and the
archive is committed to the repo on purpose, because that accumulation is the
only route this project has to a usable sample. Parquet keeps it cheap, around
20KB per symbol per session.

**The collector is the long pole.** Everything downstream is gated on calendar
time, so it is worth starting before writing another line of strategy code.

`.github/workflows/collect-bars.yml` runs at 22:30 UTC on weekdays — after the
close in both EST and EDT — and commits new bars back to the branch. Prefer it
over a laptop crontab, which only fires when the laptop is awake and is exactly
how these archives end up full of holes. Each run requests the whole 30-day
window rather than just the previous session, so a skipped, delayed or failed
run leaves no permanent gap.

Two caveats worth knowing: GitHub disables scheduled workflows after 60 days of
repository inactivity, and scheduled runs can be delayed under load. Check
`coverage` occasionally rather than assuming.

`scripts/fetch_daily.sh` is the local fallback, with crontab instructions in
its header.

## Layout

```
bipbip/
  data/     bar archive, providers, RTH sessions, synthetic generator
  core/     account rules, cost model, indicators, engine, metrics
  strategies/
            buy_hold      buy the open, sell the close - the benchmark to beat
            orb           opening-range breakout, risk in range-width units
            vwap_reversion  stretch below session VWAP, confirmed turning up
```

## Use

```bash
pip install -r requirements.txt
python -m pytest tests/ -q

python -m bipbip.cli fetch --symbols SPY TQQQ   # merge new bars into the archive
python -m bipbip.cli coverage                   # how much history exists
python -m bipbip.cli compare --symbol SPY       # every strategy, side by side
python -m bipbip.cli backtest --symbol SPY --strategy orb --trades
```

With no archive, `backtest` exits non-zero rather than inventing data. Passing
`--synthetic` allows a mechanics-only run, labelled loudly so its numbers can
never be mistaken for evidence.

Settings live in `config/default.yaml`. Copy it to `config/local.yaml` for
machine-specific overrides; that file is gitignored.

## Two scaling bugs found on real data

Worth recording, because the first diagnosis was wrong and the real cause was
more interesting.

A trade log showed VWAP-reversion entering at "11.6 ATR below VWAP". The
obvious explanation is a collapsed ATR denominator. Measuring it said
otherwise: TQQQ's one-minute ATR never fell below the round-trip cost at all,
yet the median stretch was **3.7 ATR**. Extreme readings were not outliers -
the metric was simply wrong.

**The stretch metric mixed timescales.** Distance from session VWAP accumulates
all session and grows roughly with the square root of elapsed time; a
one-minute ATR does not. Dividing one by the other drifts upward through the
day, so a threshold of 1.5 fired almost every bar by the afternoon and meant
something different at 10:00 than at 15:00. `vwap_zscore` divides instead by
the volume-weighted dispersion of price around VWAP, measured on the same
clock. On real data the median went from 3.70 to 0.98 and the maximum from
31.4 to 3.3, and a threshold of 2 now fires on about 9% of bars. The same
error affected the distance-from-session-high and -low features, which are
rebased on the same dispersion.

**Stops could sit inside the cost of the trade.** On real SPY data **37% of
bars** had a one-minute ATR smaller than the 2.3bp round-trip cost, so an
unfloored 1-ATR stop was inside the noise a third of the time - which is what
produced trades that entered and stopped out on the same bar.
`cost_floored_risk` floors the risk unit at a multiple of the round trip, and
the engine injects each instrument's hurdle before `prepare` so no strategy has
to be told about costs by hand.

Correcting the metric removed the project's most promising-looking result,
which is the point. VWAP-reversion on TQQQ had returned +1.53% at the 100th
percentile against matched random entries, p=0.000 - but a single trade was
105% of that P&L, and with a correct threshold the strategy fires 3 times
instead of 12 and returns -0.58%.

## Hand-chosen parameters

Every threshold here was chosen by judgement, not fitted. That is the correct
state for a 20-session archive - fitting them now would produce a beautiful
backtest and lose money live - but it does mean they carry different weights of
justification, so the table says which is which.

| Parameter | Value | Basis |
|-----------|-------|-------|
| `stretch_z` | 2.0 | Principled. Two standard deviations, firing on ~9% of bars. |
| `min_risk_multiple` | 2.0 | Principled. A stop must clear the round trip to be meaningful. |
| `target_r` | 2.0 | Conventional. Standard 2:1 reward-to-risk. |
| `or_minutes` | 30 | Conventional. The usual opening range. |
| `stop_frac` | 0.5 | Weak. Half the range is tidy, not derived. |
| `max_rsi` | 35 | **Arbitrary.** Inherited from convention, untested. |
| `confirm_bars` | 2 | **Arbitrary.** |
| `min_rvol` | 1.2 | **Arbitrary.** |
| `min_range_bps` | 15 | **Weak.** Roughly 7x the SPY cost hurdle, not derived. |

`sensitivity` sweeps any of them and reports the SHAPE of the result rather
than the best cell:

```bash
python -m bipbip.cli sensitivity --symbol SPY --strategy vwap_reversion --param stretch_z
```

A real effect degrades gently either side of the chosen value. An artefact is
an isolated spike whose neighbours lose money. Deliberately **not a tuner** -
adopting the best cell on a short sample is exactly the overfitting the rest of
the project exists to prevent, so the peak is reported as a warning rather than
a recommendation, and any sweep whose busiest setting takes fewer than 30
trades is labelled underpowered.

Swept across both symbols at 20 sessions, every setting of every parameter
takes at most 11 trades, so no shape in the current data is interpretable. That
is the honest answer, and it will stay the answer for some months.

## The ML layer

Hand-crafted signals are **features**, not labels. Training a net to imitate
ORB or VWAP-reversion is behavioural cloning: it can only reproduce a strategy
that already exists, and on synthetic data both of those lose money. There is
no edge to distil. So the model instead sees "1.4 ATR below VWAP, RSI 28,
opening range broke twenty minutes ago on 1.8x volume, 11:15am" and learns
*when those setups pay*.

Labels are **cost-aware** and use triple barriers. A model trained on "did
price rise" learns to predict moves smaller than the spread and then loses
money being technically correct, so the target barrier must clear the
round-trip hurdle. Barriers match how the engine actually exits — stop, target,
or the closing bell, whichever comes first — because training on fixed-horizon
returns while deploying stops answers a question nobody asked.

### Why the validation machinery is the real deliverable

The net is twenty lines. Knowing whether to believe it is the hard part.

Ordinary k-fold cross-validation is catastrophically wrong here: it shuffles,
so the model trains on the future. Even chronological splits leak, because a
label at bar `i` is resolved by bars out to `event_end[i]`. So splits are
purged — training samples whose label window reaches into the validation fold
are dropped, plus an embargo for serial correlation.

`permutation_test` is the other half: it shuffles the labels and re-runs the
entire pipeline, showing what the *procedure* scores on data with no signal.

This is not theoretical. On synthetic random-walk bars, which contain no
exploitable structure by construction, the first run reported:

```
model               AUC  trades    hit%  mean bps
logistic          0.512      13    61.5     +5.52
mlp               0.487      25    52.0     +2.25
always_enter      0.500   17799    39.6     -2.46
```

Both models look profitable on data known to have none. The tells are AUC at
0.50 — no discrimination whatsoever — and trade counts of 13 out of 17,799:
the models were not predicting but *selecting* a few lucky samples. The
permutation test found that shuffled labels reached +20.72 bps by luck alone,
comfortably beating the observed +5.52.

The verdict logic now refuses this in three ways, each added because its
absence produced that false positive: results are withheld below 30
out-of-sample trades, an observed score at or under the null's best is called
noise outright, and p-values from too few permutations are reported as
unresolvable rather than significant.

## 0DTE options

`bipbip/options/` prices 0DTE contracts and re-expresses any strategy's signals
as long calls or puts:

```bash
python -m bipbip.cli options --symbol SPY --strategy orb
```

**Predict the underlying, not the option.** An option's price is a
deterministic function of spot, time to expiry and implied volatility, and
Black-Scholes returns it exactly. A model trained on option prices spends its
sample size rediscovering that formula badly.

**The clock is the thing to get right.** Volatility estimated from one-minute
bars is annualised over TRADING minutes, so time to expiry must use the same
clock. An ATM 0DTE SPY call at $640 and 13% vol prices at $0.91 on a calendar
clock and $2.14 on a trading clock; real contracts trade near the latter,
because markets accumulate variance while open and little overnight. An earlier
version of this file used the calendar clock and consequently quoted 353x
leverage, which was wrong - the true figure is around 200x.

**Implied vol is modelled, not observed**, and is the largest source of error
here. It is realised vol times a risk premium, floored per instrument, because
implied vol does not follow realised vol down: measured on this archive SPY
realised 6.2% annualised while its options would never have been quoted near
that. Without the floor the model prices an ATM 0DTE call at $0.86 against a
real $2-4 and manufactures free money. There is no smile and no intraday term
structure; both omissions flatter the results.

### The hurdle grows with holding time

Breakeven underlying move for an ATM 0DTE SPY call entered at the open, after
theta and both spreads:

| Hold | 0DTE | Stock |
|------|------|-------|
| 15 min | 1.3 bps | 2.28 bps |
| 30 min | 2.3 bps | 2.28 bps |
| 60 min | 4.3 bps | 2.28 bps |
| 240 min | 17.1 bps | 2.28 bps |

The stock's cost is flat in holding time; theta is not. This is deterministic
arithmetic, not an observation about any trade.

### Risk has to be managed in the option's own terms

Keeping the stock's stops destroyed the results. Option winners averaged +27.6%
while losers averaged -53.7% - a win/loss magnitude ratio of **0.51**, which
cannot be profitable at any hit rate near 50%. Three causes:

**Holding time.** One trade held 152 minutes on a +33bp underlying move returned
+11.7%; another held 21 minutes on a *smaller* +24bp move returned +54.4%.

**Losses running.** Underlying stops sat 13-21bp away, already 60-70% of premium
by the time they trigger at this leverage. The cap belongs in premium terms.

**Strike choice.** An ATM 0DTE contract is 100% extrinsic value - a pure bet on
theta not happening. At delta 0.87, decay over 45 minutes falls from 7.3% of
premium to 1.4% and the payoff turns near-symmetric.

| Strategy | Stock | ATM + stock exits | Native option risk |
|----------|-------|-------------------|--------------------|
| SPY / orb | +0.04% | -9.50% (W/L 0.53) | +4.65% (W/L 1.86) |
| SPY / vwap_reversion | -0.11% | -3.39% | -1.36% |
| TQQQ / orb | -0.92% | -33.91% (W/L 0.77) | -7.14% (W/L 2.71) |
| TQQQ / vwap_reversion | -0.58% | -8.14% | -1.98% |

### What is fitted, and what is not

The risk parameters were chosen after examining these failures, so they are
in-sample by construction. Sweeping them separates mechanism from coordinate:

- `target_delta` is positive at 6 of 6 settings (sd 1.28) - a broad plateau.
- `premium_stop_pct` (sd 0.06) and `premium_target_pct` (sd 0.19) are nearly
  **inert**; they rarely trigger. Two of the four fixes do almost nothing here.
- `max_hold_minutes` is an **inverted U, not monotonic decay**:

  | hold | SPY/orb | SPY/vwap | TQQQ/orb | TQQQ/vwap |
  |------|---------|----------|----------|-----------|
  | 5m   | -0.26%  | -2.64%   | -0.27%   | **-1.34%** |
  | 10m  | +1.00%  | -2.85%   | **+0.22%** | -1.68%  |
  | 30m  | **+5.33%** | -1.02% | -4.31%  | -1.98%    |
  | 40m  | +5.06%  | **-1.02%** | -6.54% | -1.98%    |
  | 240m | -0.68%  | -2.21%   | -10.09%  | -1.98%    |

  Very short holds do *worse*, because they cut winners before the move
  happens. Theta grows with holding time while a signal needs time to work, and
  the optimum sits between them - but it lands at 5, 10, 30 and 40 minutes
  across the four combinations, so its LOCATION is not identifiable here.
  SPY/orb's 30-minute peak is a fitted coordinate.

**None of this demonstrates an edge.** SPY/orb returns +0.04% on stock,
indistinguishable from zero, so the option result is leverage applied to noise
across 9 trades. Managing options natively stops the machinery destroying
value; it cannot manufacture an edge, which is why TQQQ/orb still loses after
the fix. The gate stands: a signal must clear 2.28bp on stock before options
are worth considering.

## What is not done

- No broker connection. Nothing places an order.
- No demonstrated edge. On synthetic random-walk bars every strategy loses,
  which is the correct result and confirms the engine invents nothing — but it
  says nothing about real markets.
- Real-data validation is blocked until the archive has accumulated enough
  sessions. This is calendar time, not work.
- No ML forecaster yet, and none should be fitted until the archive is deep
  enough to support walk-forward validation with purged splits. Fitting a model
  to ~21 sessions produces a beautiful equity curve and loses money live.
- No options pricing module yet.
