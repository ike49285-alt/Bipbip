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

So `BarStore` is **append-only**. Every fetch merges into a local parquet
archive, deduplicating on timestamp and preferring the freshest copy of a
revised bar. Run `fetch` on a schedule and history accumulates well past any
single provider window. Starting that clock early is the cheapest thing you can
do for this project.

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
```

Note that `data/bars/` is gitignored: the archive is local and rebuildable.

## What is not done

- No broker connection. Nothing places an order.
- No demonstrated edge. On synthetic random-walk bars every strategy loses,
  which is the correct result and confirms the engine invents nothing — but it
  says nothing about real markets.
- Real-data validation is blocked until the archive has accumulated enough
  sessions.
