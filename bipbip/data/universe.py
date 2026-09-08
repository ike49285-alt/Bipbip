"""Tradeable universes.

Handing a system "the whole market" is the right instinct and the fastest way
to fool yourself, because of SURVIVORSHIP BIAS. A list of today's listings
contains only the companies that made it. Backtesting it over decades never
lets the system buy Enron, Lehman, Bear Stearns, Sears or Kodak, so it measures
a market where nothing ever went to zero. Published estimates put the inflation
at roughly 1-4% annually - easily enough to manufacture an edge out of nothing.

Free data cannot fix this: delisted tickers are simply not served. So the
universes below are labelled by how badly they suffer, and results carry the
label.

ETFs are the honest choice here. Funds close far less often than companies
fail, the survivors are not selected for performance in the same way, and a
few dozen of them span sectors, bonds, commodities and international markets -
genuine diversification rather than fifty correlated bets on one economy.
"""
from __future__ import annotations

#: Sector SPDRs. Listed 1998, still trading, and they partition the S&P.
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

#: Broad equity indices.
EQUITY = ["SPY", "QQQ", "IWM", "DIA", "MDY", "RSP"]

#: Non-US equity, which decorrelates from the domestic sectors above.
INTERNATIONAL = ["EFA", "EEM", "EWJ", "EWZ", "EWG", "EWU", "FXI"]

#: Bonds, the main diversifier when equities fall together.
BONDS = ["TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "TIP"]

#: Commodities and real assets.
REAL_ASSETS = ["GLD", "SLV", "USO", "DBC", "VNQ"]

#: Leveraged funds. High cost of carry and severe decay; included because the
#: project began with TQQQ, and excluded from the default universe.
LEVERAGED = ["TQQQ", "SQQQ", "UPRO", "SPXL", "TNA"]

#: Large, long-listed US names. SURVIVORSHIP BIASED: these are companies that
#: survived, and the ones that did not are absent and unobtainable.
MEGACAP = [
    "AAPL", "MSFT", "JNJ", "PG", "KO", "PEP", "WMT", "XOM", "CVX", "JPM",
    "BAC", "WFC", "T", "VZ", "MRK", "PFE", "ABT", "MCD", "HD", "LOW",
    "CAT", "BA", "GE", "MMM", "IBM", "INTC", "CSCO", "ORCL", "TXN", "QCOM",
    "AMGN", "GILD", "BMY", "LLY", "UNH", "CVS", "COST", "TGT", "NKE", "SBUX",
    "DIS", "CMCSA", "F", "GM", "UPS", "FDX", "HON", "LMT", "RTX", "DE",
]

UNIVERSES = {
    "etf_core": {
        "symbols": sorted(set(SECTORS + EQUITY + BONDS + REAL_ASSETS)),
        "survivorship": "mild",
        "note": "Sector, index, bond and real-asset ETFs. Funds close far less "
                "often than companies fail, so the bias is small but not zero.",
    },
    "etf_wide": {
        "symbols": sorted(set(SECTORS + EQUITY + INTERNATIONAL + BONDS + REAL_ASSETS)),
        "survivorship": "mild",
        "note": "As etf_core plus international equity, for decorrelation.",
    },
    "etf_all": {
        "symbols": sorted(set(SECTORS + EQUITY + INTERNATIONAL + BONDS
                              + REAL_ASSETS + LEVERAGED)),
        "survivorship": "mild",
        "note": "Everything, including leveraged funds that decay badly.",
    },
    "megacap": {
        "symbols": sorted(set(MEGACAP)),
        "survivorship": "SEVERE",
        "note": "Survivors only. Every company that failed is missing and cannot "
                "be obtained from free data. Treat returns as an upper bound, "
                "not an estimate.",
    },
    "everything": {
        "symbols": sorted(set(SECTORS + EQUITY + INTERNATIONAL + BONDS
                              + REAL_ASSETS + LEVERAGED + MEGACAP)),
        "survivorship": "SEVERE",
        "note": "The widest list here. Inherits the stock universe's bias.",
    },
}


def get_universe(name: str = "etf_wide") -> list:
    if name not in UNIVERSES:
        raise ValueError(f"unknown universe {name!r}; choose from {sorted(UNIVERSES)}")
    return list(UNIVERSES[name]["symbols"])


def survivorship_warning(name: str) -> str:
    """A sentence to attach to any result computed on this universe."""
    u = UNIVERSES.get(name)
    if not u:
        return ""
    if u["survivorship"] == "SEVERE":
        return (f"SURVIVORSHIP BIAS ({name}): this universe contains only "
                "companies that survived. Failures are absent and unobtainable "
                "from free data, so returns here are an upper bound, inflated by "
                "roughly 1-4% annually on published estimates.")
    return (f"Survivorship bias ({name}): mild. Funds that closed are absent, "
            "but fund closure is far less performance-selected than company failure.")
