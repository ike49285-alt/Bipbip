"""Cash-account regulation. These rules are law, not preference.

Breaking them in live trading costs a 90-day account restriction, so they are
tested harder than the strategies are.
"""
import datetime as dt

import pytest

from bipbip.core.account import CashAccount, MarginAccount, RuleViolation

MON, TUE = dt.date(2025, 1, 6), dt.date(2025, 1, 7)
FRI, NEXT_MON = dt.date(2025, 1, 10), dt.date(2025, 1, 13)


def _opened(equity=10_000.0):
    acct = CashAccount(starting_equity=equity, max_position_pct=1.0)
    acct.start_session(MON)
    return acct


def test_sale_proceeds_are_not_spendable_same_day():
    acct = _opened()
    acct.buy(100, 100.0)
    assert acct.settled_cash == pytest.approx(0.0)

    acct.sell(101.0, settle_date=TUE)
    # Cash exists for valuation, but none of it is settled and therefore
    # none of it may fund another purchase today.
    assert acct.cash == pytest.approx(10_100.0)
    assert acct.settled_cash == pytest.approx(0.0)
    assert acct.can_open() is False


def test_proceeds_settle_on_the_next_session():
    acct = _opened()
    acct.buy(100, 100.0)
    acct.sell(101.0, settle_date=TUE)

    acct.start_session(TUE)
    assert acct.settled_cash == pytest.approx(10_100.0)
    assert acct.can_open() is True


def test_settlement_follows_the_trading_calendar_across_a_weekend():
    """A Friday sale settles Monday. Calendar arithmetic would say Saturday."""
    acct = CashAccount(starting_equity=10_000.0, max_position_pct=1.0)
    acct.start_session(FRI)
    acct.buy(100, 100.0)
    acct.sell(101.0, settle_date=NEXT_MON)

    acct.start_session(dt.date(2025, 1, 11))  # Saturday: nothing settles
    assert acct.settled_cash == pytest.approx(0.0)

    acct.start_session(NEXT_MON)
    assert acct.settled_cash == pytest.approx(10_100.0)


def test_second_round_trip_in_a_session_is_refused():
    """The good-faith-violation guard. One round trip per day, hard stop."""
    acct = _opened()
    acct.buy(100, 100.0)
    acct.sell(101.0, settle_date=TUE)

    assert acct.can_open() is False
    assert "good-faith" in acct.blocked_reason()
    with pytest.raises(RuleViolation):
        acct.buy(1, 100.0)


def test_cannot_buy_beyond_settled_cash():
    acct = _opened(1_000.0)
    with pytest.raises(RuleViolation, match="exceeds settled cash"):
        acct.buy(100, 100.0)  # $10k of stock against $1k settled


def test_position_cap_limits_size():
    acct = CashAccount(starting_equity=10_000.0, max_position_pct=0.95)
    acct.start_session(MON)
    assert acct.affordable_shares(100.0) == 95


def test_no_second_position_while_one_is_open():
    acct = _opened()
    acct.buy(50, 100.0)
    with pytest.raises(RuleViolation, match="one position at a time"):
        acct.buy(10, 100.0)


def test_selling_nothing_raises():
    with pytest.raises(RuleViolation, match="no position"):
        _opened().sell(100.0)


def test_realised_pnl_is_net_of_fees():
    acct = _opened()
    acct.buy(100, 100.0)
    realised = acct.sell(101.0, fees=5.0, settle_date=TUE)
    assert realised == pytest.approx(100 * 1.0 - 5.0)


def test_margin_account_settles_immediately():
    """The contrast case: margin has no settlement constraint."""
    acct = MarginAccount(starting_equity=10_000.0, max_position_pct=1.0)
    acct.start_session(MON)
    acct.buy(100, 100.0)
    acct.sell(101.0, settle_date=TUE)
    assert acct.settled_cash == pytest.approx(10_100.0)
    assert acct.can_open() is True


# ---------------------------------------------------------------------------
# The rules that keep a cash account LEGAL, at their boundaries.
#
# Mutation testing left 19 of 34 alive here. The survivors were not obscure:
# the T+1 settlement comparison, the position-size cap (multiply could become
# divide), the round-DOWN that the docstring promises "can never manufacture
# buying power", and the affordability check in buy(). Those are the rules that
# stand between this account and a good-faith violation, and none of them was
# pinned at the point where it decides.
# ---------------------------------------------------------------------------

def test_proceeds_settle_on_the_settle_date_and_not_a_session_early():
    """T+1 means available ON the settle date. Off by one in either direction
    is a real-money error: early makes the backtest trade with money it does
    not have, late makes it sit out sessions it could have traded."""
    a = CashAccount(starting_equity=1000.0)
    a.pending = [(dt.date(2026, 3, 11), 500.0)]
    before = a.settled_cash

    a.start_session(dt.date(2026, 3, 10))          # day before
    assert a.settled_cash == before
    assert a.pending, "the proceeds must still be pending"

    a.start_session(dt.date(2026, 3, 11))          # the settle date itself
    assert a.settled_cash == before + 500.0
    assert not a.pending


def test_proceeds_already_overdue_settle_rather_than_being_stranded():
    """A gap in the calendar - a holiday, a missing bar - must not leave money
    permanently unsettled."""
    a = CashAccount(starting_equity=0.0)
    a.pending = [(dt.date(2026, 3, 2), 100.0), (dt.date(2026, 3, 3), 250.0)]
    a.start_session(dt.date(2026, 3, 20))
    assert a.settled_cash == pytest.approx(350.0)
    assert not a.pending


def test_the_position_cap_scales_the_budget_down_not_up():
    """`settled_cash * max_position_pct`. Dividing instead would turn a 50%
    cap into a 2x one, which is leverage in a cash account."""
    full = CashAccount(starting_equity=1000.0, max_position_pct=1.0)
    half = CashAccount(starting_equity=1000.0, max_position_pct=0.5)
    assert full.affordable_shares(10.0) == pytest.approx(100.0)
    assert half.affordable_shares(10.0) == pytest.approx(50.0)
    assert half.affordable_shares(10.0) < full.affordable_shares(10.0)


def test_fractional_quantities_round_down_and_never_up():
    """The docstring's promise: rounding must never manufacture buying power.

    $100 at $3 is 33.333... shares. Rounded to five decimals DOWN that is
    33.33333, and 33.33333 * 3 must not exceed the budget. Rounding to nearest
    would give 33.33333 here but 33.33334 at other prices, and the account
    would be a fraction of a cent short on the fill.
    """
    a = CashAccount(starting_equity=100.0, max_position_pct=1.0,
                    allow_fractional=True)
    for price in (3.0, 7.0, 11.0, 13.0, 6.28, 99.97):
        qty = a.affordable_shares(price)
        assert qty * price <= a.settled_cash + 1e-9, (price, qty)


def test_a_whole_share_account_never_returns_a_fraction():
    a = CashAccount(starting_equity=100.0, max_position_pct=1.0,
                    allow_fractional=False)
    for price in (3.0, 7.0, 33.33, 99.97):
        qty = a.affordable_shares(price)
        assert float(qty).is_integer(), (price, qty)
        assert qty * price <= a.settled_cash + 1e-9


def test_a_nonpositive_price_buys_nothing_rather_than_dividing_by_it():
    a = CashAccount(starting_equity=100.0)
    assert a.affordable_shares(0.0) == 0.0
    assert a.affordable_shares(-5.0) == 0.0


def test_a_buy_may_spend_exactly_the_settled_cash_but_not_a_cent_more():
    """The boundary the whole cash-account rule turns on."""
    a = CashAccount(starting_equity=100.0)
    a.buy(shares=10.0, price=10.0)                 # exactly 100.00
    assert a.position.is_open

    b = CashAccount(starting_equity=100.0)
    with pytest.raises(RuleViolation, match="exceeds settled cash"):
        b.buy(shares=10.0, price=10.01)            # 100.10
    assert not b.position.is_open


def test_fees_count_against_settled_cash():
    """A fill that fits on notional but not once fees are added must be
    refused, or the account overdraws by exactly the commission."""
    a = CashAccount(starting_equity=100.0)
    with pytest.raises(RuleViolation, match="exceeds settled cash"):
        a.buy(shares=10.0, price=10.0, fees=0.50)


def test_a_second_position_is_refused_while_one_is_open():
    a = CashAccount(starting_equity=1000.0)
    a.buy(shares=1.0, price=10.0)
    with pytest.raises(RuleViolation, match="one position at a time"):
        a.buy(shares=1.0, price=10.0)


def test_a_nonpositive_share_count_is_refused():
    a = CashAccount(starting_equity=1000.0)
    for bad in (0.0, -1.0):
        with pytest.raises(RuleViolation, match="positive share count"):
            a.buy(shares=bad, price=10.0)


def test_can_open_and_blocked_reason_never_disagree():
    """Two code paths answering the same question. They are used in different
    places - one gates, one explains - so a divergence would show as a trade
    that is refused without a reason, or logged as allowed and then refused."""
    cases = [
        CashAccount(starting_equity=1000.0),
        CashAccount(starting_equity=0.0),
    ]
    held = CashAccount(starting_equity=1000.0)
    held.buy(shares=1.0, price=10.0)
    cases.append(held)

    spent = CashAccount(starting_equity=1000.0)
    spent.round_trips_today = spent.max_round_trips_per_session
    cases.append(spent)

    for a in cases:
        assert a.can_open() == (a.blocked_reason() is None)


def test_the_round_trip_budget_blocks_a_further_entry_the_same_session():
    """One round trip a session is what keeps a cash account clear of a
    good-faith violation, so the boundary is the rule, not a nicety."""
    a = CashAccount(starting_equity=1000.0, max_round_trips_per_session=1)
    assert a.can_open()
    a.round_trips_today = 1
    assert not a.can_open()
    assert "round-trip budget spent" in a.blocked_reason()
    with pytest.raises(RuleViolation):
        a.buy(shares=1.0, price=10.0)


def test_starting_a_session_resets_the_round_trip_budget():
    a = CashAccount(starting_equity=1000.0)
    a.round_trips_today = 1
    a.start_session(dt.date(2026, 3, 11))
    assert a.round_trips_today == 0
    assert a.current_session == dt.date(2026, 3, 11)


def test_an_account_with_no_settled_cash_cannot_open():
    a = CashAccount(starting_equity=0.0)
    assert not a.can_open()
    assert a.blocked_reason() == "no settled cash available"


def test_a_position_is_worth_shares_times_price():
    """`Position.market_value` had nothing pinning it: shares * price could
    become shares / price with the whole suite green. Everything downstream -
    equity, position sizing, every reported return - is built on this."""
    from bipbip.core.account import Position
    p = Position(shares=10.0, avg_price=5.0)
    assert p.market_value(7.0) == pytest.approx(70.0)
    assert p.market_value(14.0) == pytest.approx(140.0)      # scales in price
    assert Position(shares=20.0, avg_price=5.0).market_value(7.0) \
        == pytest.approx(140.0)                              # and in shares


def test_a_flat_position_is_worth_nothing_at_any_price():
    from bipbip.core.account import Position
    assert Position().market_value(1234.0) == pytest.approx(0.0)


def test_equity_adds_the_position_to_cash_rather_than_subtracting_it():
    """`cash + market_value`. The sign here decides whether holding a position
    looks like an asset or a liability, and a backtest would still produce a
    plausible-looking curve with it inverted."""
    a = CashAccount(starting_equity=1000.0)
    a.buy(shares=10.0, price=50.0)                 # 500 spent, 10 shares held
    assert a.cash == pytest.approx(500.0)
    assert a.equity(50.0) == pytest.approx(1000.0)   # unchanged at entry price
    assert a.equity(60.0) == pytest.approx(1100.0)   # +10 per share
    assert a.equity(40.0) == pytest.approx(900.0)


def test_equity_with_no_position_is_just_cash():
    a = CashAccount(starting_equity=250.0)
    assert a.equity(99.0) == pytest.approx(250.0)


def test_sub_dollar_settled_cash_still_counts_as_having_cash():
    """`settled_cash > 0`, not `> 1`. The account this project is sized for
    holds $2.10, so the difference between "some cash" and "a dollar of cash"
    is the difference between trading and not."""
    a = CashAccount(starting_equity=0.50)
    assert a.can_open()
    assert a.blocked_reason() is None


def test_a_sub_dollar_price_is_a_real_price_not_a_rejected_one():
    """`price <= 0`, not `<= 1`. Sub-dollar names are exactly the universe a
    small account reaches - the odd-lot tender work puts the $500 universe at
    "stocks under about $5" - so rejecting them would silently empty it."""
    a = CashAccount(starting_equity=100.0, max_position_pct=1.0,
                    allow_fractional=False)
    assert a.affordable_shares(0.50) == pytest.approx(200.0)
    assert a.affordable_shares(1.00) == pytest.approx(100.0)


def test_the_blocked_reason_is_reported_not_replaced_by_a_generic_message():
    """buy() raises with `blocked_reason() or <fallback>`. Swapping that for
    `and` would discard the specific diagnosis every time there was one, which
    is precisely when it is needed."""
    a = CashAccount(starting_equity=1000.0, max_round_trips_per_session=1)
    a.round_trips_today = 1
    with pytest.raises(RuleViolation, match="good-faith violation"):
        a.buy(shares=1.0, price=10.0)


def test_a_round_trip_counts_as_one_not_two():
    """`round_trips_today += 1`. At the default cap of one, incrementing by two
    is indistinguishable - both exceed it - so the counter's STEP is only
    observable on an account allowed more than one round trip a session. A
    step of two would silently halve the permitted turnover.
    """
    a = CashAccount(starting_equity=1000.0, max_round_trips_per_session=2)
    a.buy(shares=1.0, price=10.0)
    a.sell(price=11.0)
    assert a.round_trips_today == 1
    assert a.can_open(), "a second round trip is still permitted at a cap of 2"

    a.buy(shares=1.0, price=10.0)
    a.sell(price=11.0)
    assert a.round_trips_today == 2
    assert not a.can_open()


# The single mutation that survives this file is `cost > settled_cash + 1e-9`
# becoming `>=`. The two differ only when the purchase cost equals the settled
# cash to within one part in a billion, which floating point does not produce
# from any realistic price and quantity; a test constructing it would pin the
# epsilon rather than the rule. Recorded as an EQUIVALENT MUTANT so it is not
# chased again.
