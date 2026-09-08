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

## On 0DTE options

The intended direction is an ML price target for 0DTE SPY options. Three
findings from costing that out, recorded here so they are not rediscovered:

**Predict the underlying, not the option.** An option's price is a
deterministic function of spot, time to expiry and implied volatility;
Black-Scholes returns it exactly. A model trained on option prices spends its
sample size rediscovering that formula badly. The forecasting problem is SPY
over the hold horizon, and the option target is arithmetic on top of it.

**The hurdle is about 8bp of SPY movement per two-hour hold.** At $640 spot and
13% IV, an ATM 0DTE call runs ~353x leverage, so a 10bp move in SPY is a +23%
move in the option — but the same position sheds 16.9% to theta over two hours
with spot unchanged, and the spread costs ~1.1% round trip. Leverage offsets
most of the spread; theta is the real cost, and it is certain while the edge is
not. Compare 2.3bp for the stock.

**0DTE cannot be backtested on free data.** Historical intraday option chains
are not freely available; yfinance exposes only a current snapshot. The chosen
route is synthesising option prices from SPY bars via Black-Scholes with a
calibrated IV assumption — defensible for ATM SPY, and to be labelled as an
approximation wherever it appears in results.

The consequent gate: a directional signal must clear 2.3bp on SPY *stock*
before it is worth expressing in options, since 353x leverage amplifies errors
as readily as edges.

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
