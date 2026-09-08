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
