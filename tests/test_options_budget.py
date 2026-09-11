"""The $50 options constraint, and the asymmetry of theta.

A contract is 100 shares, so a two-cent move in the premium is two dollars -
real money on a fifty dollar account. The arithmetic is right; what it omits is
that the contracts where it applies cannot be bought with fifty dollars, and
that theta does not merely shrink a win, it enlarges the matching loss.
"""
import numpy as np
import pytest

from bipbip.options import pricing as bs
from bipbip.options.iv import DEFAULT_VARIANCE_RISK_PREMIUM, iv_floor_for
from bipbip.options.overlay import FEE_PER_CONTRACT, MULTIPLIER
from bipbip.options.synth import quote_spread

S, IV, R = 766.0, 0.141, 0.04
SESSION = 390


def _prem(days, K=None, kind=bs.PUT, minutes=None):
    K = round(S) if K is None else K
    m = minutes if minutes is not None else (days + 1) * SESSION
    return float(bs.price(S, K, bs.minutes_to_years(m), R, IV, kind))


def test_a_two_cent_move_is_two_dollars_a_contract():
    """The premise, which is arithmetically correct."""
    gross = 0.02 * MULTIPLIER
    net = gross - 2 * FEE_PER_CONTRACT
    assert gross == pytest.approx(2.00)
    assert net == pytest.approx(1.90, abs=0.01)


def test_no_at_the_money_contract_is_affordable_on_fifty_dollars():
    """Options are quoted per share and sold in hundreds, with no fractions."""
    for days in (0, 1, 7, 30):
        cost = _prem(days) * MULTIPLIER + FEE_PER_CONTRACT
        assert cost > 50.0, f"{days}DTE ATM cost ${cost:.2f}, expected over $50"


def test_the_spread_eats_most_or_all_of_a_two_cent_move():
    """A two-cent gain is the same order as the quote it must cross."""
    gain = 0.02 * MULTIPLIER
    for days, expect_loss in ((0, False), (1, False), (7, True), (30, True)):
        prem = _prem(days)
        rt = float(quote_spread(np.array([prem]))[0]) * MULTIPLIER + 2 * FEE_PER_CONTRACT
        net = gain - rt
        if expect_loss:
            assert net < 0, f"{days}DTE: expected the spread to exceed the move"
        else:
            assert 0 < net < gain * 0.95, f"{days}DTE: spread should take a real bite"


def test_theta_enlarges_the_loss_as_well_as_shrinking_the_win():
    """The answer to 'just balance delta and theta'.

    Theta is not a tax on winners. It is a fixed drag that applies whichever way
    the underlying goes, so an equal-sized move up and down does not produce
    equal and opposite P&L - the loss is strictly the larger.
    """
    minutes_left, hold, move = SESSION, 30, 0.00059      # SPY median 30-min move
    K = round(S)
    T_in = bs.minutes_to_years(minutes_left)
    T_out = bs.minutes_to_years(minutes_left - hold)
    prem = float(bs.price(S, K, T_in, R, IV, bs.PUT))
    cost = float(quote_spread(np.array([prem]))[0]) * MULTIPLIER + 2 * FEE_PER_CONTRACT

    win = (float(bs.price(S * (1 - move), K, T_out, R, IV, bs.PUT)) - prem) * MULTIPLIER - cost
    lose = (prem - float(bs.price(S * (1 + move), K, T_out, R, IV, bs.PUT))) * MULTIPLIER + cost

    assert lose > win, "theta should make the adverse leg the bigger one"
    breakeven = lose / (win + lose)
    assert breakeven > 0.60, f"break-even hit rate {breakeven:.1%} looks too kind"


def test_holding_a_short_dated_option_longer_is_strictly_worse():
    """Theta accelerates into expiry, so the same option decays faster later."""
    K = round(S)
    prem = float(bs.price(S, K, bs.minutes_to_years(SESSION), R, IV, bs.PUT))
    after_30 = float(bs.price(S, K, bs.minutes_to_years(SESSION - 30), R, IV, bs.PUT))
    after_120 = float(bs.price(S, K, bs.minutes_to_years(SESSION - 120), R, IV, bs.PUT))
    decay_first = prem - after_30
    decay_next = after_30 - after_120
    assert decay_next > decay_first, "later decay should outpace earlier decay"


def test_a_longer_dated_option_decays_less_over_the_same_hold():
    """Why the delta/theta balance points at longer expiries held briefly."""
    K = round(S)
    frac = {}
    for days in (0, 30):
        m = (days + 1) * SESSION
        p0 = float(bs.price(S, K, bs.minutes_to_years(m), R, IV, bs.PUT))
        p1 = float(bs.price(S, K, bs.minutes_to_years(m - 120), R, IV, bs.PUT))
        frac[days] = (p0 - p1) / p0
    assert frac[30] < frac[0] / 5, (
        f"30DTE lost {frac[30]:.1%} of premium vs 0DTE {frac[0]:.1%} over the "
        "same two hours; the gap should be large")


def test_the_cheap_contracts_within_budget_have_punishing_spreads():
    """What $50 can reach, and why reaching it does not help.

    A one-cent quote is the floor whatever the premium, so it is 3% of a $0.38
    option and 10% of a $0.10 one - a hurdle before the underlying moves at all.
    """
    affordable = []
    for otm in (0.005, 0.01, 0.015):
        K = round(S * (1 - otm))
        prem = _prem(0, K)
        if prem < 0.02 or prem * MULTIPLIER + FEE_PER_CONTRACT > 50.0:
            continue
        spread = float(quote_spread(np.array([prem]))[0])
        affordable.append((prem, spread / prem))

    assert affordable, "expected at least one sub-$50 contract to exist"
    # Every one of them pays a spread that is a meaningful fraction of premium.
    assert all(frac > 0.02 for _, frac in affordable)
    # And the cheaper it is, the worse that fraction gets.
    affordable.sort()
    assert affordable[0][1] > affordable[-1][1]


def test_pricing_uses_implied_not_realised_volatility():
    """Realised vol would hand the buyer the variance risk premium for free."""
    assert DEFAULT_VARIANCE_RISK_PREMIUM > 1.0
    realised = 0.123
    iv = max(realised * DEFAULT_VARIANCE_RISK_PREMIUM, iv_floor_for("SPY"))
    assert iv > realised
    # A buyer pays more for the same contract at the higher vol.
    K = round(S)
    T = bs.minutes_to_years(SESSION)
    assert (float(bs.price(S, K, T, R, iv, bs.PUT))
            > float(bs.price(S, K, T, R, realised, bs.PUT)))


def test_readme_options_tables_are_internally_consistent():
    """The section's argument is an ordering; check the ordering survives edits."""
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    body = readme.read_text().split("**Options on $50, and the delta/theta balance.**")
    assert len(body) == 2, "options-on-$50 section missing from README"
    # Bound at the next section, or later tables are read as this one's.
    flat = " ".join(body[1].split("\n\n**")[0].split())

    # The first row spells out "of premium" and the rest do not, so the
    # pattern has to tolerate the trailing words rather than require them.
    drag = re.findall(
        r"\| (\d+DTE) \| (-[\d.]+)%[^|]*\| (-[\d.]+)%[^|]*\| \$([\d,]+) \|", flat)
    assert len(drag) == 4, f"expected 4 drag rows, parsed {len(drag)}"

    two_hour = [float(d[2]) for d in drag]
    costs = [float(d[3].replace(",", "")) for d in drag]
    # The whole claim: longer expiry drags less over the same hold, and costs more.
    assert two_hour == sorted(two_hour), f"2-hour drag no longer improves with expiry: {two_hour}"
    assert costs == sorted(costs), "cost should rise with expiry"
    # And the affordable corner is the worst corner.
    assert costs[0] > 50.0, "the section claims no ATM contract is affordable"


# ---------------------------------------------------------------------------
# bipbip/options/pricing.py against scipy. Mutation testing put this module at
# 18%: eight of forty-five deliberate defects detected. It is separate code
# from the Black-Scholes in chain_features, and it is what prices the greeks.
# scipy is the oracle because it is already a declared dependency and already
# tested - a hand-written reference risks pinning a wrong constant.
# ---------------------------------------------------------------------------

def _bs_reference(S, K, T, r, sigma, kind):
    """Textbook Black-Scholes, written here independently of the module."""
    from scipy.stats import norm
    import math
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


@pytest.mark.parametrize("S,K,T,r,sigma", [
    (100.0, 100.0, 1.0, 0.04, 0.20),
    (100.0, 90.0, 0.5, 0.04, 0.30),
    (100.0, 110.0, 0.25, 0.00, 0.60),
    (640.0, 640.0, 1.0 / 252, 0.04, 0.13),      # the docstring's 0DTE case
    (50.0, 55.0, 2.0, 0.05, 0.15),
])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_the_price_matches_scipy_across_the_surface(S, K, T, r, sigma, kind):
    from bipbip.options import pricing
    k = pricing.CALL if kind == "call" else pricing.PUT
    got = float(pricing.price(S, K, T, r, sigma, k))
    assert got == pytest.approx(_bs_reference(S, K, T, r, sigma, kind), rel=1e-9)


@pytest.mark.parametrize("S,K,T,r,sigma", [
    (100.0, 100.0, 1.0, 0.04, 0.20),
    (100.0, 90.0, 0.5, 0.04, 0.30),
    (120.0, 100.0, 0.25, 0.02, 0.45),
])
def test_put_call_parity_holds_between_the_two_legs(S, K, T, r, sigma):
    """C - P = S - K*exp(-rT), by construction and with no model assumption.
    CLAUDE.md calls this the most reliable tool available; it is also the one
    relation a sign error in either leg cannot survive."""
    import math
    from bipbip.options import pricing
    c = float(pricing.price(S, K, T, r, sigma, pricing.CALL))
    p = float(pricing.price(S, K, T, r, sigma, pricing.PUT))
    assert c - p == pytest.approx(S - K * math.exp(-r * T), rel=1e-9)


def test_an_expired_option_is_worth_exactly_intrinsic():
    """The docstring's promise. Extrinsic value is zero at expiry, so anything
    above intrinsic there is money invented by the model."""
    from bipbip.options import pricing
    assert float(pricing.price(110.0, 100.0, 0.0, 0.04, 0.2, pricing.CALL)) \
        == pytest.approx(10.0)
    assert float(pricing.price(90.0, 100.0, 0.0, 0.04, 0.2, pricing.CALL)) \
        == pytest.approx(0.0)
    assert float(pricing.price(90.0, 100.0, 0.0, 0.04, 0.2, pricing.PUT)) \
        == pytest.approx(10.0)


def test_the_clock_is_trading_time_not_calendar_time():
    """The module's central claim, checked as arithmetic: one session is
    1/252 of a year, not 1/365, and a full year of sessions is exactly 1.0."""
    from bipbip.options.pricing import (TRADING_MINUTES_PER_YEAR,
                                        minutes_to_years)
    assert TRADING_MINUTES_PER_YEAR == pytest.approx(252 * 390)
    assert float(minutes_to_years(390)) == pytest.approx(1.0 / 252)
    assert float(minutes_to_years(TRADING_MINUTES_PER_YEAR)) == pytest.approx(1.0)


def test_the_trading_clock_prices_a_zero_dte_call_near_its_traded_level():
    """The docstring's worked example: SPY at $640, 13% vol, one session to
    expiry is about $2.10 on a trading clock and about $0.91 on a calendar one.
    Getting this backwards under-prices every short-dated option and makes an
    options backtest look far too profitable."""
    from bipbip.options import pricing
    trading = float(pricing.price(640.0, 640.0, 1.0 / 252, 0.04, 0.13,
                                  pricing.CALL))
    calendar = float(pricing.price(640.0, 640.0, 1.0 / 365, 0.04, 0.13,
                                   pricing.CALL))
    assert trading > calendar
    assert 1.5 < trading < 3.0


def test_delta_matches_the_numerical_derivative_of_the_price():
    """Each greek is checked against a bump of the price, so a wrong greek
    cannot agree with a right price."""
    from bipbip.options import pricing
    S, K, T, r, v = 100.0, 100.0, 0.5, 0.04, 0.25
    h = 1e-4
    up = float(pricing.price(S + h, K, T, r, v, pricing.CALL))
    dn = float(pricing.price(S - h, K, T, r, v, pricing.CALL))
    assert float(pricing.delta(S, K, T, r, v, pricing.CALL)) \
        == pytest.approx((up - dn) / (2 * h), rel=1e-5)


def test_gamma_matches_the_numerical_second_derivative():
    from bipbip.options import pricing
    S, K, T, r, v = 100.0, 100.0, 0.5, 0.04, 0.25
    h = 1e-2
    up = float(pricing.price(S + h, K, T, r, v, pricing.CALL))
    mid = float(pricing.price(S, K, T, r, v, pricing.CALL))
    dn = float(pricing.price(S - h, K, T, r, v, pricing.CALL))
    assert float(pricing.gamma(S, K, T, r, v)) \
        == pytest.approx((up - 2 * mid + dn) / h ** 2, rel=1e-3)


def test_vega_matches_the_numerical_derivative_in_volatility():
    from bipbip.options import pricing
    S, K, T, r, v = 100.0, 100.0, 0.5, 0.04, 0.25
    h = 1e-6
    up = float(pricing.price(S, K, T, r, v + h, pricing.CALL))
    dn = float(pricing.price(S, K, T, r, v - h, pricing.CALL))
    assert float(pricing.vega(S, K, T, r, v)) \
        == pytest.approx((up - dn) / (2 * h), rel=1e-4)


def test_theta_is_negative_for_a_long_option_and_grows_toward_expiry():
    """Decay is a cost to the holder, and it accelerates - the property that
    makes short-dated out-of-the-money contracts brutal. A positive theta here
    would pay a buyer for waiting."""
    from bipbip.options import pricing
    far = float(pricing.theta_per_minute(100.0, 100.0, 0.5, 0.04, 0.25,
                                         pricing.CALL))
    near = float(pricing.theta_per_minute(100.0, 100.0, 1.0 / 252, 0.04, 0.25,
                                          pricing.CALL))
    assert far < 0 and near < 0
    assert near < far


def test_a_call_is_monotone_in_spot_and_in_volatility():
    """Two properties no correct pricer can violate, and which a sign error in
    d1 or d2 breaks while still matching at a single point."""
    from bipbip.options import pricing
    px = [float(pricing.price(s, 100.0, 0.5, 0.04, 0.25, pricing.CALL))
          for s in (80.0, 95.0, 100.0, 105.0, 120.0)]
    assert px == sorted(px)
    vol = [float(pricing.price(100.0, 100.0, 0.5, 0.04, v, pricing.CALL))
           for v in (0.05, 0.15, 0.30, 0.60)]
    assert vol == sorted(vol)


def test_a_call_never_exceeds_the_underlying_nor_falls_below_intrinsic():
    """The two no-arbitrage bounds. CLAUDE.md: an option cannot trade below
    intrinsic value."""
    from bipbip.options import pricing
    for S in (60.0, 100.0, 150.0):
        c = float(pricing.price(S, 100.0, 0.75, 0.04, 0.3, pricing.CALL))
        assert c <= S + 1e-9
        assert c >= max(0.0, S - 100.0) - 1e-9
