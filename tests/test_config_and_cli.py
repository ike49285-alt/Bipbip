"""Config loading and CLI wiring."""
import pandas as pd
import pytest
import yaml

from bipbip.cli import main
from bipbip.config import _deep_merge, build_account, load_config
from bipbip.core.account import CashAccount, MarginAccount
from bipbip.data import BarStore, make_intraday_bars


def test_local_overrides_are_merged_not_replaced():
    base = {"account": {"type": "cash", "starting_equity": 10_000},
            "costs": {"slippage_bps": {"SPY": 1.0}}}
    override = {"account": {"starting_equity": 5_000}}
    merged = _deep_merge(base, override)
    assert merged["account"]["starting_equity"] == 5_000
    assert merged["account"]["type"] == "cash"       # untouched key survives
    assert merged["costs"]["slippage_bps"]["SPY"] == 1.0


def test_shipped_default_config_loads_and_is_a_cash_account():
    cfg = load_config("config/default.yaml")
    acct = build_account(cfg)
    assert isinstance(acct, CashAccount)
    assert acct.max_round_trips_per_session == 1


def test_account_type_selects_the_right_rules():
    assert isinstance(build_account({"account": {"type": "margin_full"}}), MarginAccount)
    pdt = build_account({"account": {"type": "margin_pdt"}})
    assert pdt.max_round_trips_per_session == 1
    with pytest.raises(ValueError, match="unknown account type"):
        build_account({"account": {"type": "nonsense"}})


def _config_file(tmp_path, store_dir):
    cfg = load_config("config/default.yaml")
    cfg["data"]["store_dir"] = str(store_dir)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_backtest_command_runs_against_the_archive(tmp_path, capsys):
    store_dir = tmp_path / "bars"
    BarStore(store_dir).append("SPY", make_intraday_bars(n_sessions=10, seed=44))
    cfg = _config_file(tmp_path, store_dir)

    assert main(["--config", cfg, "backtest", "--symbol", "SPY", "--strategy", "orb"]) == 0
    out = capsys.readouterr().out
    assert "opening_range_breakout" in out
    assert "SYNTHETIC" not in out  # real archive was used


def test_backtest_refuses_to_invent_data_when_the_archive_is_empty(tmp_path):
    cfg = _config_file(tmp_path, tmp_path / "empty")
    with pytest.raises(SystemExit) as exc:
        main(["--config", cfg, "backtest", "--symbol", "SPY"])
    assert exc.value.code == 2


def test_synthetic_results_are_loudly_labelled(tmp_path, capsys):
    """Synthetic numbers must never be mistakable for evidence."""
    cfg = _config_file(tmp_path, tmp_path / "empty")
    assert main(["--config", cfg, "backtest", "--symbol", "SPY", "--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "MEANINGLESS" in out


def test_coverage_flags_an_inadequate_sample(tmp_path, capsys):
    store_dir = tmp_path / "bars"
    BarStore(store_dir).append("SPY", make_intraday_bars(n_sessions=5, seed=45))
    cfg = _config_file(tmp_path, store_dir)

    assert main(["--config", cfg, "coverage", "--symbols", "SPY"]) == 0
    assert "too few to validate" in capsys.readouterr().out


def test_compare_runs_every_registered_strategy(tmp_path, capsys):
    store_dir = tmp_path / "bars"
    BarStore(store_dir).append("SPY", make_intraday_bars(n_sessions=10, seed=46))
    cfg = _config_file(tmp_path, store_dir)

    assert main(["--config", cfg, "compare", "--symbol", "SPY"]) == 0
    out = capsys.readouterr().out
    for name in ["session_buy_hold", "opening_range_breakout", "vwap_reversion"]:
        assert name in out


def test_coverage_can_fail_loudly_on_an_empty_archive(tmp_path, capsys):
    """A scheduled collector whose provider is blocked would otherwise report
    'no new bars' and exit green. Silent breakage is worse than loud failure."""
    cfg = _config_file(tmp_path, tmp_path / "empty")
    assert main(["--config", cfg, "coverage", "--symbols", "SPY"]) == 0
    assert main(["--config", cfg, "coverage", "--symbols", "SPY", "--require-data"]) == 1
    assert "no archived bars" in capsys.readouterr().err


def test_require_data_passes_when_the_archive_has_bars(tmp_path):
    store_dir = tmp_path / "bars"
    BarStore(store_dir).append("SPY", make_intraday_bars(n_sessions=3, seed=47))
    cfg = _config_file(tmp_path, store_dir)
    assert main(["--config", cfg, "coverage", "--symbols", "SPY", "--require-data"]) == 0


def test_fetch_reports_failure_so_a_scheduled_job_cannot_fail_silently(tmp_path, capsys):
    """A collector that exits 0 while fetching nothing is worse than useless."""
    cfg = _config_file(tmp_path, tmp_path / "bars")
    rc = main(["--config", cfg, "fetch", "--symbols", "SPY", "--provider", "alpaca"])
    assert rc == 1
    assert "FAILED" in capsys.readouterr().err
