import pandas as pd
import pytest

from bipbip.core import BacktestEngine, CashAccount, CostModel
from bipbip.data import make_intraday_bars


@pytest.fixture
def bars():
    return make_intraday_bars(n_sessions=12, seed=3)


@pytest.fixture
def costs():
    return CostModel()


@pytest.fixture
def engine(costs):
    return BacktestEngine(CashAccount(starting_equity=10_000.0), costs)
