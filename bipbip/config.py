"""Configuration loading.

`config/default.yaml` is the committed baseline; `config/local.yaml` overrides
it and is gitignored, so machine-specific settings and anything sensitive stay
out of version control.
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path("config/default.yaml")
LOCAL_PATH = Path("config/local.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load defaults, then apply local overrides if present."""
    base_path = Path(path) if path else DEFAULT_PATH
    if not base_path.exists():
        raise FileNotFoundError(f"config not found: {base_path}")

    cfg = yaml.safe_load(base_path.read_text()) or {}
    if path is None and LOCAL_PATH.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(LOCAL_PATH.read_text()) or {})
    return cfg


def build_account(cfg: dict):
    """Construct the account described by the config."""
    from .core.account import CashAccount, MarginAccount

    acct = cfg.get("account", {})
    kind = str(acct.get("type", "cash")).lower()
    equity = float(acct.get("starting_equity", 10_000.0))
    cap = float(acct.get("max_position_pct", 0.95))
    frac = bool(acct.get("allow_fractional", True))

    if kind == "cash":
        return CashAccount(starting_equity=equity, max_position_pct=cap,
                           allow_fractional=frac)
    if kind == "margin_pdt":
        # PDT allows three day trades per rolling five sessions. Modelling that
        # as a per-session budget is a simplification, and a conservative one.
        return MarginAccount(starting_equity=equity, max_position_pct=cap,
                             allow_fractional=frac, max_round_trips_per_session=1)
    if kind == "margin_full":
        return MarginAccount(starting_equity=equity, max_position_pct=cap,
                             allow_fractional=frac)
    raise ValueError(f"unknown account type {kind!r}")
