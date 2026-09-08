"""Account models, including the cash-account settlement rules.

The cash account is the interesting one. Under Reg T a purchase must be paid
for with SETTLED funds, and US equities settle T+1. The failure mode this
class exists to make impossible:

    09:35  buy $10k SPY          settled cash -> $0
    09:40  sell                   $10k proceeds, unsettled until tomorrow
    09:45  buy again              allowed, using unsettled proceeds
    09:50  sell that position     GOOD FAITH VIOLATION

Three good-faith violations in twelve months and the broker restricts the
account to settled-cash-only for ninety days. So the account permits exactly
one round trip per session, and settles proceeds against the real trading
calendar rather than calendar days - a Friday sale settles Monday, not
Saturday.

Cash accounts also cannot short, so positions are long-or-flat.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


class RuleViolation(RuntimeError):
    """Raised when an order would break an account rule. Never caught silently."""


@dataclass
class Position:
    shares: float = 0.0
    avg_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.shares > 0

    def market_value(self, price: float) -> float:
        return self.shares * price


@dataclass
class CashAccount:
    """A Reg T cash account: settled-funds-only, long/flat, one round trip a day."""

    starting_equity: float = 10_000.0
    max_position_pct: float = 0.95
    #: Webull supports fractional shares on most US stocks and ETFs. Whole-share
    #: sizing strands the remainder: at $777 a share, a $9,500 budget buys 12
    #: shares and leaves $176 (1.85%) uninvested on every trade.
    allow_fractional: bool = True

    cash: float = field(init=False)
    settled_cash: float = field(init=False)
    position: Position = field(default_factory=Position)
    # (settlement_date, amount) for proceeds not yet available to trade.
    pending: list = field(default_factory=list)
    # Sessions in which a position has already been opened and closed.
    round_trips_today: int = 0
    current_session: dt.date | None = None
    max_round_trips_per_session: int = 1

    def __post_init__(self):
        self.cash = float(self.starting_equity)
        self.settled_cash = float(self.starting_equity)

    # -- session lifecycle -------------------------------------------------

    def start_session(self, date: dt.date) -> None:
        """Settle matured proceeds and reset the day's turnover budget."""
        matured = [amt for settle_date, amt in self.pending if settle_date <= date]
        self.pending = [(d, a) for d, a in self.pending if d > date]
        self.settled_cash += sum(matured)
        self.round_trips_today = 0
        self.current_session = date

    # -- queries -----------------------------------------------------------

    def equity(self, price: float) -> float:
        return self.cash + self.position.market_value(price)

    def can_open(self) -> bool:
        """True when a new long may be opened under the account's rules."""
        if self.position.is_open:
            return False
        if self.round_trips_today >= self.max_round_trips_per_session:
            return False
        return self.settled_cash > 0

    def affordable_shares(self, price: float) -> float:
        """Shares purchasable with settled cash, after the position cap.

        Fractional quantities are rounded DOWN to five decimals rather than
        nearest, so rounding can never manufacture buying power the account
        does not have.
        """
        if price <= 0:
            return 0.0
        budget = self.settled_cash * self.max_position_pct
        if not self.allow_fractional:
            return float(int(budget // price))
        import math
        return math.floor((budget / price) * 1e5) / 1e5

    def blocked_reason(self) -> str | None:
        """Why an entry is currently disallowed, for logging and diagnostics."""
        if self.position.is_open:
            return "already holding a position"
        if self.round_trips_today >= self.max_round_trips_per_session:
            return (
                f"round-trip budget spent ({self.round_trips_today}/"
                f"{self.max_round_trips_per_session}); further trading today would "
                "use unsettled proceeds and risk a good-faith violation"
            )
        if self.settled_cash <= 0:
            return "no settled cash available"
        return None

    # -- mutations ---------------------------------------------------------

    def buy(self, shares: float, price: float, fees: float = 0.0) -> None:
        cost = shares * price + fees
        if shares <= 0:
            raise RuleViolation("buy requires a positive share count")
        if self.position.is_open:
            raise RuleViolation("cash account holds one position at a time")
        if self.round_trips_today >= self.max_round_trips_per_session:
            raise RuleViolation(self.blocked_reason() or "round-trip budget spent")
        if cost > self.settled_cash + 1e-9:
            raise RuleViolation(
                f"purchase of {cost:,.2f} exceeds settled cash {self.settled_cash:,.2f}; "
                "buying with unsettled funds risks a good-faith violation"
            )
        self.cash -= cost
        self.settled_cash -= cost
        self.position = Position(shares=shares, avg_price=price)

    def sell(self, price: float, fees: float = 0.0, settle_date: dt.date | None = None) -> float:
        """Close the position. Proceeds are unsettled until `settle_date`."""
        if not self.position.is_open:
            raise RuleViolation("no position to sell")
        shares = self.position.shares
        proceeds = shares * price - fees
        realised = proceeds - shares * self.position.avg_price

        self.cash += proceeds
        # Deliberately NOT added to settled_cash: that is the whole point.
        if settle_date is not None:
            self.pending.append((settle_date, proceeds))
        else:
            self.settled_cash += proceeds

        self.position = Position()
        self.round_trips_today += 1
        return realised


@dataclass
class MarginAccount(CashAccount):
    """Margin account. Funds settle immediately; PDT may cap day trades.

    Provided so the engine is not welded to one account type. Set
    `max_round_trips_per_session` to 3 per five sessions' worth of budget for a
    sub-$25k account, or leave it unbounded above the PDT threshold.
    """

    max_round_trips_per_session: int = 10_000

    def sell(self, price: float, fees: float = 0.0, settle_date: dt.date | None = None) -> float:
        # Margin buying power is available immediately; ignore the settle date.
        return super().sell(price, fees, settle_date=None)
