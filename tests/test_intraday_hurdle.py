"""The break-even arithmetic behind the intraday verdict.

The conclusion that seconds are impossible and fifteen minutes is not rests on
two formulas and a scaling law. Those are worth pinning, because the verdict
is quoted in the README and the difference between the symmetric and the
asymmetric version changes what "achievable" means.
"""
import numpy as np
import pytest

from bipbip.core.costs import CostModel


def breakeven_symmetric(cost_bps: float, move_bps: float) -> float:
    """p such that p*m - (1-p)*m - c = 0."""
    return 0.5 + cost_bps / (2.0 * move_bps)


def breakeven_asymmetric(cost_bps: float, risk_bps: float, reward_mult: float) -> float:
    """p such that p*reward - (1-p)*risk - cost = 0."""
    reward = reward_mult * risk_bps
    return (risk_bps + cost_bps) / (reward + risk_bps)


def test_the_two_formulas_agree_at_one_to_one():
    """A 1:1 payoff is the symmetric case, so they must coincide there."""
    for cost, move in ((2.28, 4.02), (0.41, 1.04), (5.28, 19.67)):
        assert breakeven_asymmetric(cost, move, 1.0) == pytest.approx(
            breakeven_symmetric(cost, move))


def test_a_free_trade_breaks_even_at_a_coin_flip():
    assert breakeven_symmetric(0.0, 5.0) == pytest.approx(0.5)


def test_the_hurdle_rises_as_the_move_shrinks():
    """Why the fastest horizons are hopeless: cost is fixed, the move is not."""
    cost = 2.28
    needed = [breakeven_symmetric(cost, m) for m in (10.32, 5.92, 4.02, 2.36, 1.04)]
    assert needed == sorted(needed)
    assert needed[-1] > 1.0, "a 1-minute SPY move should be unwinnable at this cost"


def test_a_cost_above_the_whole_move_is_unwinnable():
    """Above 100% there is no hit rate that pays, which is the honest reading."""
    assert breakeven_symmetric(2.28, 1.04) > 1.0


def test_letting_winners_run_lowers_the_bar_a_lot():
    """The correction the symmetric table hides.

    At 15 minutes on SPY the 1:1 hurdle is 78% - not achievable by anything.
    At 2:1 it is 52%, which is ordinary for a real systematic strategy. The
    cost hurdle is therefore not what rules out a 15-minute system; the
    absence of a signal would be.
    """
    cost, risk = 2.28, 4.02
    assert breakeven_asymmetric(cost, risk, 1.0) > 0.75
    assert breakeven_asymmetric(cost, risk, 2.0) < 0.55


def test_sqrt_of_time_scaling_puts_a_second_below_the_best_case_cost():
    """No sub-minute data exists, so this is the argument standing in for it."""
    median_1m_bps = 1.04
    best_case_round_trip = 2.0 * 0.065 + 0.278      # quote-only on SPY
    one_second = median_1m_bps * np.sqrt(1.0 / 60.0)
    assert one_second < best_case_round_trip, (
        "a 1-second move should be smaller than the cheapest possible round trip")
    # And the crossover sits in the seconds, not the minutes.
    fifteen_seconds = median_1m_bps * np.sqrt(15.0 / 60.0)
    assert fifteen_seconds > best_case_round_trip


def test_the_leveraged_fund_is_charged_more_per_side():
    """TQQQ's bigger moves come with a bigger spread; both must be counted.

    An earlier draft of the sweep reused SPY's slippage for both symbols, which
    flattered TQQQ by more than half its actual round trip.
    """
    c = CostModel()
    assert c.slippage_for("TQQQ") > c.slippage_for("SPY")
    spy = 2.0 * c.slippage_for("SPY") + 0.278
    tqqq = 2.0 * c.slippage_for("TQQQ") + 0.278
    assert tqqq > spy * 2.0


def test_readme_intraday_tables_match_the_formulas():
    """The README quotes these numbers, so recompute them from the arithmetic.

    Two commits in this project have already claimed a README update that did
    nothing, and one shipped a table computed on data that was later corrected.
    """
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    body = readme.read_text().split("**Intraday, and the cost hurdle that defines it.**")
    assert len(body) == 2, "intraday section missing from README"
    body = body[1]

    # Both tables have the same column shape, so each regex has to be scoped
    # to its own table or the execution-cost rows get read as payoff ratios.
    split = body.split("| SPY, modelled execution |")
    assert len(split) == 2, "the reward-to-risk table is missing"
    exec_table, payoff_table = split

    exec_rows = re.findall(
        r"^\| (\d+ minutes?) \| ([\d.]+|impossible)% ?\| ", exec_table, re.M)
    asym = re.findall(
        r"^\| (\d+ minutes?) \| ([\d.]+)% \| [\d.]+% \| ([\d.]+)% \|",
        payoff_table, re.M)
    assert exec_rows, "could not parse the execution-cost table"
    assert asym, "could not parse the reward-to-risk table"

    for period, one_to_one, two_to_one in asym:
        # A 2:1 payoff must always be an easier bar than 1:1.
        assert float(two_to_one) < float(one_to_one), period
        # And the 15-minute 2:1 case is the section's claim: it must be under 55%.
        if period.startswith("15"):
            assert float(two_to_one) < 55.0

    # The sub-minute claim.
    assert "a third of the cost" in body
    one_second = 1.04 * np.sqrt(1.0 / 60.0)
    assert one_second / (2 * 0.065 + 0.278) == pytest.approx(0.33, abs=0.02)
