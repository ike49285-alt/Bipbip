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

## Effects tested, and why none of them pay

Every result below is measured on this repository's own data, on
dividend-adjusted daily bars. That qualifier is load-bearing: the archive held
price-only history until it was caught, under which SHY compounded at 0.03% a
year and SPY's benchmark was understated by its whole yield, and every figure
in this section moved when it was fixed.

The pattern is consistent: the effects are real historically, and gone by the
time you could have traded them.

**Overnight versus intraday.** SPY has returned 10.03% annually overnight
against 0.73% intraday over 34 years - 91% of the equity premium arrives while
the market is closed, and QQQ's intraday component is negative. The effect is
large and undisputed. It is not harvestable: isolating it needs 252 round trips
a year, costing 5.7% annually, which exceeds the entire gap being captured.

**Turn of month.** The best-evidenced finding here. SPY returns 8.20 bps on the
sessions around each month boundary against 2.5 bps elsewhere, and unlike
everything else it has NOT decayed - 9.36, 7.24, 8.52 and 7.98 bps across four
decades. Eight assets not used to find it all confirm it. But rotating into
short Treasuries between windows returns 6.33% at a Sharpe of 0.53 against
buying and holding at 10.84% and 0.65: being out of the market two thirds of
the time costs more compounding than the concentrated days provide.

**Post-earnings drift.** The classic long-short version is dead, with t-stats
near zero. Long-only winners looked alive at +1.17% over 40 days, t=5.2 - until
split by era: +3.88%, +1.82%, +0.57%, +1.12%, and -0.25% in 2020-2027. The
losing decile drifts up as well in every pre-2020 era, which is the signature
of survivorship rather than signal, since stocks that crashed and never
recovered are absent from a 2026 membership list.

**Low-priced shares.** With fractional trading, share price is cosmetic: $50
buys 0.064 shares of a $777 stock or 10 shares of a $5 one, and the percentage
return is identical. What is not cosmetic is the spread. A one-cent spread
costs 0.3 bps round trip on SPY and 40 bps on a $5 stock - over a hundred times
more - and sub-$10 names rarely quote a penny wide. Cheap stocks in this
universe do show higher forward returns, 1.93% against 0.79% over 20 days, but
that is survivorship in its purest form: a large cap trading under $10 in the
2026 index is one that fell a long way and CAME BACK, and the ones that did not
are unobtainable.

**Leveraged trend following.** The one construction that could plausibly turn
$50 into real money. Holding a simulated 3x SPY while the index is above its
200-day average beats buying and holding on total return - and loses on every
risk measure, at every financing cost. A 3x fund borrows two dollars for each
dollar of equity, so the borrowing rate is a first-order input:

| borrowing rate | $50 becomes | CAGR | Sharpe | max drawdown |
|---|---|---|---|---|
| 0.0% | $16,707 | 18.88% | 0.67 | 67.1% |
| 2.0% | $6,154 | 15.40% | 0.58 | 68.6% |
| 3.5% | $2,911 | 12.85% | 0.52 | 69.7% |
| 5.0% | $1,377 | 10.37% | 0.46 | 72.3% |
| *buy and hold SPY* | *$1,547* | *10.75%* | *0.65* | *54.9%* |

At 3.5% - roughly the average short rate over the period - it turns $50 into
$2,911 against $1,547, which is a real difference in dollars. It gets there by
taking more risk rather than by being better at anything: Sharpe falls to 0.52
against 0.65, and the worst drawdown is 70% against 55%. Only the free-money
row clears buy-and-hold on Sharpe, and it clears it by 0.02. Leverage is not an
edge, it is a dial, and this is what the dial costs.

Parameter sensitivity is the disqualifying part. Reading the same signal daily,
monthly, or with three days of confirmation returned -0.6%, +13.5% and -1.0%
during 2000-2009. Fourteen points of spread across a choice that should not
matter is noise, and the monthly variant that produced +13.5% is the one a
search over variants would have selected.

**Moving-average timing.** SMA(200) on SPY, evaluated monthly, has the best
risk-adjusted record of any single-asset rule here and is still not a way to
make more money. It returns 8.80% against 10.75% for buying and holding, at a
Sharpe of 0.70 against 0.65 and a maximum drawdown of 35.6% against 54.9%. The
decade split shows exactly what it buys: +7.0% during 2000-2009 when holding
lost 0.9%, and less than buy-and-hold in every other decade. It is insurance.
The premium is paid in the good decades, the payout arrives in the bad one, and
over 34 years the two very nearly cancel.

**Levered risk parity.** The only idea tested here with a structural argument
behind it rather than a pattern spotted in a backtest: investors who want
return but will not borrow bid up risky assets instead, so a stock/bond blend
carries a better Sharpe ratio than stocks alone, and borrowing converts the
better ratio into more money. It produced the first result in this project to
beat buying and holding on return, Sharpe and drawdown at once - and it does
not survive being split by period.

Measured from 2004, when all three sleeves exist, weighting SPY, TLT and GLD by
inverse volatility and levering 2.09x to match SPY's volatility returns 13.44%
against 10.91%, at a Sharpe of 0.77 against 0.65 and a 45% drawdown against
54%. Split into five-year blocks, financed at 3.5%:

| | 2005-2010 | 2011-2015 | 2016-2020 | 2021-2026 |
|---|---|---|---|---|
| buy & hold SPY | 2.7% | 12.1% | 15.3% | 15.2% |
| risk parity + gold, 2.09x | 16.3% | 8.2% | 20.3% | **9.7%** |
| risk parity, no gold, 2.03x | 5.5% | 19.4% | 20.0% | **1.2%** |
| *GLD alone* | *21.6%* | *-6.0%* | *11.7%* | *14.8%* |

Two things kill it. The advantage over the whole sample is carried by
2005-2010, and the sleeve responsible is gold, which returned 21.6% a year
through the financial crisis - so the strategy is a bet on one asset's one
historic move, chosen after seeing it. Without gold the same construction
returns 1.2% a year since 2021 against 15.2% for holding SPY, and draws down
52% where SPY draws down 24%.

The mechanism is not mysterious, which is what makes it disqualifying rather
than unlucky. The premise is that bonds diversify stocks. When rates rise both
fall together - TLT returned -7.8% a year over 2021-2026 - and leverage
applied to a blend that has stopped diversifying doubles the loss instead of
smoothing it. That is the regime the money would be going in today.

Starting the same test in 1993 instead reports a Sharpe of 0.74 and looks far
better. That number is an artefact: bond ETFs do not exist in the archive
before 2002 or gold before 2004, so for the first nine years - the strongest
bull run in the sample - the strategy is 92% SPY and simply inherits its
results. The harness now begins where every sleeve exists.


**Protective stops.** Added to the engine and swept across width, type and
re-entry delay, on plain SPY, on the SMA(200) rule, and on the 3x book where a
70% drawdown gives them the most to work with. They damage every one of them,
and the way they fail is more interesting than the fact that they do.

On SPY, a 5% trailing stop with immediate re-entry produced a **73.0% maximum
drawdown against 54.9% for holding, and 55.2% for the index itself**. The stop
did not fail to prevent a drawdown; it manufactured eighteen points of drawdown
that the market never had. Each stop realises a loss and re-enters higher, and
in a market that spends most of its declines recovering, doing that repeatedly
ratchets the account below anything the price did.

| stop on SPY | $50 becomes | CAGR | Sharpe | max drawdown | times fired |
|---|---|---|---|---|---|
| none | $1,547 | 10.75% | 0.65 | 54.9% | 0 |
| trailing 5%, re-enter next bar | $434 | 6.64% | 0.46 | 73.0% | 314 |
| trailing 5%, 21-day lockout | $307 | 5.55% | 0.54 | 47.7% | 126 |
| trailing 10%, 21-day lockout | $861 | 8.84% | 0.66 | 50.3% | 43 |
| trailing 20%, 21-day lockout | $825 | 8.70% | 0.60 | 54.6% | 13 |

The same holds where stops should help most. On the 3x book every width from
10% to 20% cuts the result to a fraction - $318, $720, $827 against $2,911
unstopped - and two of those settings end with a *worse* drawdown than no stop
at all. Only a 30% trailing stop roughly matches leaving it alone, and it fires
seven times in thirty-four years.

That is the pattern everywhere: the only settings that do not hurt are the ones
too wide to act. The best-looking row in the whole sweep is a 20% trailing stop
on the SMA(200) rule, which lifts Sharpe from 0.70 to 0.74 and cuts drawdown
from 35.6% to 26.0% - on **two** fills in thirty-four years, 1998-08-31 and
2020-03-11. An improvement resting on two events is two data points, not a
property, and choosing 20% because those two landed well is choosing a
coordinate.

None of this says stops are useless in general. It says they are a bet that
declines continue rather than revert, and a long-only equity book is the wrong
place to make that bet, because the asset's own tendency is to recover. The
engine now supports them so the claim is testable rather than assumed - and it
models the two details that would otherwise make them look free: a bar that
gaps through the level fills at the open rather than the level, and a trailing
reference is taken through the previous bar so this bar's high cannot raise the
level that this bar's low is tested against.

**Ichimoku cloud, and the full stochastic.** Both implemented, swept, and
conditioned on. The textbook system stacks four confirmations - price above the
cloud, Tenkan above Kijun, a bullish cloud, and the lagging span above price -
and every one of them is a trend measure, so they do not vote independently.
Requiring more of them narrows the window without gathering more evidence:

| on SPY | $50 becomes | CAGR | Sharpe | max drawdown | in market |
|---|---|---|---|---|---|
| buy and hold | $1,547 | 10.75% | 0.65 | 54.9% | 100% |
| cloud filter alone | $1,219 | 9.97% | 0.68 | 42.1% | 92% |
| 2 confirmations | $367 | 6.11% | 0.57 | 44.8% | 74% |
| 3 confirmations | $274 | 5.19% | 0.55 | 29.7% | 63% |
| 4 confirmations, the textbook system | $150 | 3.33% | 0.47 | 19.8% | 40% |

Monotonic in both directions, and the same ordering holds in all four decades.
Only the bare cloud filter is competitive, and it lands where every other trend
filter here lands - a shade better on Sharpe, materially worse on total return,
which is the signature of insurance rather than an edge.

Conditioning on next-day returns says why, and it is worse than dilution. The
days the system SITS OUT beat the days it holds: 5.70 bp against 3.42 bp at
four confirmations, against 4.78 bp for all days unconditionally. The filter
does not merely reduce exposure, it selects the below-average half. Each extra
confirmation moves the mean by -1.57, +0.30 and -0.21 bp, which is noise around
no improvement at all. That is what short-horizon mean reversion does to any
rule whose entry condition is "price has already gone up".

The stochastic is the one component that is not a trend measure, and it is the
only thing here that showed a real gradient. Within four-confirmation days,
next-day returns run 8.86 bp when %K is below 60, 4.43 bp between 60 and 80,
and 0.97 bp above 80, with t = 2.50 on the low bucket. Split by era it is the
familiar corpse:

| low-minus-high %K spread | 1993-1999 | 2000-2009 | 2010-2019 | 2020-2026 |
|---|---|---|---|---|
| basis points per day | +14.41 | +9.74 | +3.22 | +1.33 |
| t-statistic | 2.65 | 1.64 | 0.74 | 0.18 |

Strong when it was worth publishing, halved each decade since, and at +1.33 bp
with t = 0.18 it is now indistinguishable from zero and below the round-trip
cost of acting on it. Bolted onto the backtest it makes every configuration
worse, because it cuts exposure from 40% to 21% in exchange for an edge that
stopped existing around 2010.

Both indicators carry the time shifts that make them easy to get wrong, so both
are pinned by tests. The cloud is displaced FORWARD, which is safe: the cloud
drawn over a bar was computed 26 bars earlier. The lagging span is displaced
BACKWARD, and written the obvious way - `close.shift(-26)`, read at row i - it
hands the strategy a price 26 bars in the future. It is stored here as the
causal comparison it actually represents, close(i) against close(i-26). The
cloud flags are also nullable rather than boolean, because Senkou B needs 78
bars to exist and `NaN > x` is False rather than unknown - the same bug this
project already shipped once in a market filter that read as "below its
average" for thirty years it had no data for.

**Intraday, and the cost hurdle that defines it.** The minute archive holds 21
sessions and grows by one a day, so no intraday strategy here can be validated
for roughly a year. What CAN be settled now is whether one is possible at all,
because that turns on the size of a move against the cost of capturing it, and
a short sample estimates a move distribution far better than it estimates an
edge.

The honest bar is the break-even hit rate: the fraction of trades a system must
get right purely to pay the spread, before earning anything. A coin flip is
50%; the best systematic equity strategies run 52-55%. Measured on SPY, against
three assumptions about execution quality:

| holding period | quote-only (0.41 bps) | modelled (2.28 bps) | poor fills (6.28 bps) |
|---|---|---|---|
| 1 minute | 69.6% | impossible | impossible |
| 5 minutes | 58.7% | 98.3% | impossible |
| 15 minutes | 55.1% | 78.4% | impossible |
| 60 minutes | 52.6% | 64.8% | 90.7% |

"Impossible" is literal: the round trip costs more than the entire median move,
so no hit rate pays. **81% of one-minute SPY moves are smaller than the cost of
trading one.**

That table assumes wins and losses are the same size, which understates a real
system with a target and a stop. The general bar is (risk + cost) / (reward +
risk), and letting winners run changes the verdict at the slower end:

| SPY, modelled execution | 1:1 | 1.5:1 | 2:1 | 3:1 |
|---|---|---|---|---|
| 5 minutes | 98.3% | 78.7% | 65.6% | 49.2% |
| 15 minutes | 78.4% | 62.7% | 52.2% | 39.2% |
| 60 minutes | 64.8% | 51.8% | 43.2% | 32.4% |

So the conclusion splits. Below about five minutes the cost hurdle rules the
horizon out arithmetically, whatever the signal. From fifteen minutes upward
with a 2:1 payoff the bar is 52%, which is ordinary for a working strategy -
cost is no longer what stands in the way, and the open question becomes whether
a signal exists, which 21 sessions cannot answer.

Seconds are settled, and settled against. Yahoo's finest interval is one
minute, so no sub-minute data exists here and there is no free source for it.
The scaling is not in doubt: diffusive moves grow with the square root of time
while the cost of trading does not, so from a measured 1-minute median of 1.04
bps, a 1-second SPY move is 0.13 bps against a best-case round trip of 0.41 -
**a third of the cost of capturing it.** The crossover sits near 15 seconds,
and that is against a quote-only cost that assumes every fill lands at the
quote with no adverse selection. There is no version of this arithmetic in
which a retail account trades SPY profitably on a one-second horizon.

**The 15-minute-to-2-hour search.** The cost arithmetic left this horizon open,
so it was searched properly: seventeen hypotheses written down in full before
any of them ran, on 2,976 tradeable hourly observations across 496 sessions of
SPY, each signal decided at one bar's close and earning the next bar's
open-to-close return.

Nothing survives. The strongest result is fading the overnight gap at t = 1.95
against a Bonferroni threshold of 3.02 for a battery this size, and it is worth
+0.99 bps against a 2.28 bps round trip, so it would lose money even if it were
real. **Zero of the seventeen clear an uncorrected 5% test, where chance alone
predicts 0.9.** The families covered are time-of-day, hourly momentum and
reversal, reversal conditioned on an outsized prior bar, closing position within
the bar's range, overnight gap continuation and fade, first-hour continuation
and fade, and volatility conditioning.

This is a real negative rather than a data shortage, which is the part worth
being careful about. With 2,976 observations and a 28 bps hourly standard
deviation, the smallest edge reaching the corrected threshold is 1.53 bps -
below the 2.28 bps it costs to trade. **Any edge large enough to be worth
having would have been detected.** The harness is checked against a planted
reversal edge of exactly 2.28 bps and finds it, so a null result from it means
something. TQQQ is the marginal case: its minimum detectable edge is 5.86 bps
against a 5.28 bps cost, so a barely-tradeable edge there could hide, and it
needs about 513 sessions to resolve against the 496 available.

The gap fade was then taken to the daily archive, where it has an exact analog
with 8,458 observations over 34 years rather than 496 sessions. It returns
+1.06 bps at t = 1.02, below cost. By era it is +1.96, +2.60, -1.08 and +1.02
bps, and by gap size it is positive for small and medium gaps and negative for
large and extreme ones - the reverse of what a gap-overshoot mechanism predicts.
Sign flips across the buckets a mechanism would order are noise wearing a
result's clothing.

What this does NOT rule out is worth stating plainly, because the search was
narrow by design. It covers time-series signals on one instrument. It says
nothing about cross-sectional signals at this horizon, which need intraday
history for many symbols that this archive does not have; nothing about
order-book or quote-level signals, since only OHLCV is stored and no free
source provides depth; and nothing about a model over richer features, which
would need far more history before it could be validated rather than fitted.

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
