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
  $X.50 placebo produced the same result.

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

## Standing results

Direction is dead — tested six ways including a genetic search that invented
its own indicators. Price targets are worse than the random walk. Volatility is
predictable but the predictable part is priced. Round-number and extreme-level
resistance did not survive matched controls. Trend maturity is confounded by
survivorship.

One live result: at 30 minutes with ~90 features, gross **+1.01 bps at t=2.49**,
beating all five nulls. It is −2.09 bps crossing the spread and +3.55 earning
it, so it rests entirely on the adverse-selection measurement.
