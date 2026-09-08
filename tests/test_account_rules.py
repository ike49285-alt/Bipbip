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
