# Bipbip — baseline

Read this before proposing or evaluating a strategy. It is the domain
grounding the searches in this repo assume; without it the same wrong turns
get made again, and several of them were made here already.

## What we are actually trading

**TQQQ is 3x QQQ's DAILY return, rebalanced every day. It is not "QQQ times
three."** The leverage resets at each close, so the payoff is path-dependent
and compounding works against it: a symmetric round trip in the underlying
leaves the levered fund lower. QQQ down 10% then up 11.1% is flat; TQQQ down
30% then up 33.3% is down 6.7%. That gap is volatility decay, and it grows
with realised variance.

Consequences that matter here:

- Long-horizon expected return is **not** 3x the index. It is roughly
  `3·μ − (9−3)/2·σ²` less financing, so at 25% index vol the drag is around
  19 percentage points a year before costs. The +42%/yr this archive shows for
  2011–2026 is *realised* history from an unusually strong, unusually calm
  decade for the Nasdaq. Do not project it forward.
- Choppy markets bleed the fund even when the index ends flat. Any strategy
  that is passively long TQQQ is short realised variance whether or not it
  meant to be.
- This makes TQQQ a poor buy-and-hold and a reasonable *short-horizon*
  instrument, which is roughly the opposite of most equities.

## Where returns come from

Three sources, and confusing them is the most common error in this file's
subject:

1. **Risk premia** — compensation for holding something others want to shed.
   The equity premium; the variance risk premium (why selling options has
   positive expectancy); term and credit premia. These are *real, durable, and
   require no skill* — only the willingness and the capital to bear the risk.
   You do not have to be right about anything.
2. **Alpha** — being right when the market is wrong. Requires an
   informational, structural, or behavioural reason the mispricing survives.
   Absent such a reason, assume it does not exist.
3. **Noise** — the overwhelming majority of what a search finds.

Before believing any result, name which of the three it is. If the answer is
"alpha", state the reason it has not been arbitraged away. "The model found
it" is not a reason.

## The prior is no edge

Markets are close enough to efficient that **the default hypothesis is zero**,
and evidence has to clear that bar rather than merely be positive. A search
flexible enough to find an edge is flexible enough to manufacture one, so
every claim needs a null: the identical procedure run on shuffled labels, or a
placebo condition structurally matched to the real one.

The best-of-N floor: searching N candidates and keeping the winner yields
`sqrt(2·ln N)` even when nothing is there. At N=3,600 that is t=4.05. An
in-sample t below the floor is worse than a coin flip you got lucky on.

## What is already priced

These are true, well documented, and therefore **not edges**. Rediscovering
one and reporting it as a finding has happened repeatedly in this project.

- **Volatility clusters and mean-reverts.** High vol is followed by high vol,
  and drifts back toward its average. Every option chain prices this — it is
  why the VIX term structure slopes.
- **The leverage effect.** Volatility rises after declines more than after
  rallies.
- **Intraday volatility is U-shaped**, high at the open and close. A model
  given time-of-day will "discover" this; it is not an edge.
- **Implied exceeds realised on average** (the variance risk premium), around
  4–5 points on index products. Harvestable as a *premium*, not as a
  prediction.
- **Volume is U-shaped intraday** and clusters at the close.
- **Prices cluster at round numbers** — limit orders genuinely pile there.
  Note this repo tested it and found the effect is not about roundness: a
  $X.50 placebo produced the same result. Re-tested on the full 534-symbol
  archive with date clustering and a fixed 3-cent band: real minus placebo is
  +0.1 bps at t=0.02 daily and +3.8 at t=0.94 intraday. Nothing.
  **Any roundness test needs UNADJUSTED prices.** This archive is
  split-adjusted, and the inverse levered funds have had large reverse splits —
  SOXS's adjusted 2015 price is $11 million. A price that never traded cannot
  be near a round number in any sense a human order ticket recognises.

## No-arbitrage relations

These hold by construction and are the most reliable tool available, because
they need no model:

- **Put-call parity**: `C − P = S − K·e^(−rT)`. Over a few days rates are
  negligible, so `S ≈ C − P + K`. This recovers the underlying's price *at the
  instant a chain was quoted*, which is strictly better than a spot fetched
  separately — a live quote taken 49 minutes after a snapshot here was 20
  cents off, enough to make puts look 19 vol points dearer than calls.
- A call and a put on the same strike and expiry imply **one** volatility. If
  they disagree, the spot or the quotes are wrong, not the market.
- An option cannot trade below intrinsic value. A price at or under intrinsic
  has no time value and no volatility to invert.

## Options, concretely

You are buying **extrinsic value**, which decays to zero at expiry. Break it
into what actually moves the position:

- **Delta** — sensitivity to the underlying. A 0.20-delta option needs the
  underlying to move ~5x the option's price change to matter.
- **Theta** — decay per day. Accelerates as expiry approaches; brutal on
  short-dated out-of-the-money contracts.
- **Vega** — sensitivity to implied volatility.
- **Break-even** is strike + premium (for a call), and the underlying must
  reach it *by expiry*, not at some point on the way.

The cheap ones are cheap because they are unlikely. This repo measured the
gradient across an entire chain: expectation falls monotonically with
moneyness, from indistinguishable-from-zero near the money to −100% far out.
**The contracts a small account can afford are the ones priced worst.**

Time to expiry runs on a **trading** clock, not a calendar one. Weekends carry
no variance. Pricing two calendar days as 2/365 rather than 2 sessions/252
overstated short-dated implied vol here by twelve points.

## Costs, and who collects them

- The **spread** is the dominant cost at small size. Crossing it pays the
  market maker; posting a resting order and getting filled earns it. Round
  trip on TQQQ: ~2.82 bps crossing, plus a 0.278 bps SEC fee on sales.
- **Options spreads are enormous** relative to premium — 14.5% of mid on a
  short-dated TQQQ call, about a thousand times the stock's.
- The catch on posting is **adverse selection**: a resting bid fills precisely
  when the price is about to fall. The spread earned is only worth having if
  it exceeds the post-fill drift. *This is the open question in this repo.*
- Market makers are paid to quote, must show two sides continuously, and hedge
  continuously. You are none of those things — which is a disadvantage on
  cost and an advantage on patience.

## What a small account actually has

Not information, and not speed. Two real structural advantages:

- **It can hold.** Market makers are flat and hedged; you can sit for months.
  Anything that pays for patience is available to you and not to them.
- **It never has to quote.** You can decline 95% of sessions at no cost.

And one structural disadvantage that dominates everything: **you cross the
spread and they earn it.**

## Sizing and ruin

- Correlated positions are **one bet**. Five call spreads on the same
  underlying, same expiry, adjacent strikes is not diversification — they lose
  together. This repo found $460 of a $500 account in "five spreads" was 92%
  on a single position.
- With no measured edge, Kelly sizing says bet **zero**. Fractional Kelly on a
  positive edge is small; on a zero edge it is nothing.
- A strategy that loses a fixed fraction per trade compounds to ruin
  regardless of starting capital. $500 with no edge and $50,000 with no edge
  fail identically, just at different speeds.

## Statistical traps this repo has actually hit

Each of these produced a fake positive here. They are cheap to check and
expensive to miss.

- **Overlapping observations** inflate significance by `sqrt(horizon)`. A
  t-statistic walked 1.60 → 5.55 → 10.78 on no new information. Sample one
  observation per holding period.
- **Pooled statistics across symbols** mostly measure which ticker it is.
  Within-symbol figures collapsed a HAR R² from 0.829 to 0.329, a correlation
  from 0.541 to 0.283, and a persistence R² from 0.356 to −0.015. Always
  report within-symbol.
- **Clustered dates.** A market-wide move puts every symbol in the same bucket
  on one date; that is one event, not one per symbol. Cluster standard errors
  by date — it roughly doubled them here.
- **Survivorship.** The archive is today's universe; everything that fell and
  delisted is missing. Measured at 15–21 percentage points a year in this
  repo, which is *larger than most effects being tested*, and it biases toward
  fake mean reversion.
- **A hit rate is not a P&L.** 53.8% directional accuracy still lost 3.38 bps
  a trade, because it was right on small moves and wrong on big ones.
- **Cost must be charged after the direction is chosen.** Netting it into the
  label with the sign inverted paid shorts 4.34 bps instead of charging them.
- **Report gross and cost separately.** Folding a 3.10 bps charge into the
  headline buried a real +1.01 bps signal for most of a session.
- **Benchmark against the right thing.** For a price forecast that is the
  random walk, not zero. For volatility it is HAR, not persistence. For a
  strategy it is buy-and-hold, not cash.
- **The scoring rule can be the edge.** Taking `max(long_score, short_score)`
  and comparing it to long-only is biased by construction: the max of two
  noisy numbers beats one of them. Shuffled labels paid **+0.33 bps** through
  this alone. A null run on the *whole procedure* catches this; a null run on
  the model does not.
- **Five nulls is not a null.** The number that decides significance is the
  null's *spread*, and five draws cannot measure it — here they put it at
  sd 0.415 when the truth was 0.829, and the null's centre anywhere in
  [−0.37, +1.04]. Worse, five draws return "beats every null" **72% of the
  time** on data whose true p is 0.069, so the control reads as confirmation
  when it is nearly no evidence. Use ~100, and report a permutation p-value
  with the Phipson–Smyth +1 (the real run is itself a draw, so p=0 is not
  attainable).
- **An arbitrary seed is a free parameter.** Changing only the model seed moved
  a headline margin by sd 0.204 bps, against a claimed effect of ~1 bps. If a
  result moves when you reseed, that movement belongs in the error bar.
- **Check that the event fires on what you think it does.** A round-number
  condition compared a distance capped at 0.50 against a threshold of
  `0.0015 × close`, so above $333 it matched EVERY bar: 2.7% of sub-$20 bars,
  100% above $333. It was selecting expensive stocks, not round prices, and it
  sat inside a standing result for months. Print the event rate, and print it
  sliced by whatever the condition could be secretly keying on.

## Standing results

Direction is dead — tested six ways including a genetic search that invented
its own indicators. Price targets are worse than the random walk. Volatility is
predictable but the predictable part is priced. Round-number and extreme-level
resistance did not survive matched controls — re-tested on all 534 symbols with
date clustering, roundness is +0.1 bps at t=0.02, and the one surviving effect
(stalling at a 20-bar high, −13.7 bps at t=−2.95) is carried entirely by
1990–2004, the worst survivorship period; every decade since is insignificant,
and on the twenty levered ETFs actually traded it is t=−1.70 daily and t=−0.24
intraday. Trend maturity is confounded by survivorship. Break *size and timing* looked promising at +0.10 AUC over a
vol-only baseline and collapsed to +0.002 once time-of-day was added — it was
the intraday U-shape, which is already priced. The levered-ETF rebalancing
mechanism is falsified: it predicts the edge concentrates in the last half
hour, and the edge is flat across the session. **Cloud colour inverted between
timeframes** — a rally with a bullish cloud under it running into a bearish one
above — carries nothing once both trends are held out: the 2×2 interaction is
p=0.12 at best and 0.37 after Bonferroni, and its sign flips across pairs.

One thing that keeps replicating, and is a reason *not* to trade rather than a
trade: Ichimoku's bullish readings select bad days. The four-confirmation
system returns 3.33% against 10.75% for holding, and measured directly a
bullish 30-minute cloud is followed by ~43 bps worse returns than a bearish
one. It decays hard — −70.7 bps in the first third of the archive, −27.2 and
insignificant in the last five years.

**There is no live result.** The one that stood longest — a barrier-labelled
GBM on a 20-ETF levered panel, 696,521 bars, 66,256 held-out trades over 4.5
years, +1.63 bps over always-long at paired t=2.22 — did not survive a
properly powered null. Against 100 shuffles: null mean +0.33, **null sd 0.83**,
and the real run at +1.63 sits **below the null's own 95th percentile of
+1.71**. Six shuffles beat it outright; the best reached +2.77. Permutation
**p = 0.069**.

It had passed seven controls, including always-long, coin-flip, inverse-fund
(ruling out drift), robustness across 32 barrier cells, and a five-shuffle
null. The five-shuffle null was the weak link, and it was weak in the
direction of agreeing (see the traps above).

This is not a refutation — the point estimate is positive and p=0.069 is not
nothing. It is a finding that does not clear the bar, in a repo whose stated
prior is that the default hypothesis is zero. Discount it further for the
32-cell grid it was selected from (best-of-N floor t=2.63) and for model-seed
noise of sd 0.204.

The execution question — whether a resting order keeps its half spread after
adverse selection — is therefore no longer decisive for any particular
strategy. It is still worth measuring, because it sets what ANY intraday idea
here has to clear: crossing costs ~1.35 bps round trip, which is larger than
every gross effect this repo has ever measured.
