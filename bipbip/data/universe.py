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


#: A broader large-cap list, roughly the larger half of the S&P by long-listed
#: names. SURVIVORSHIP BIASED for the same reason as MEGACAP, and more so the
#: further back a test runs: this is the 2026 membership, and the index has
#: replaced dozens of constituents per decade. Companies that were large enough
#: to be here and then failed - Lehman, Enron, Sears, Bear Stearns, Countrywide,
#: Washington Mutual, GM's old entity - are structurally absent.
LARGECAP_EXTRA = [
    "ABBV", "ACN", "ADBE", "ADI", "ADP", "AEP", "AFL", "AIG", "AJG", "ALL",
    "AMAT", "AMD", "AME", "AMT", "AON", "APD", "APH", "AVGO", "AXP", "AZO",
    "BAX", "BDX", "BEN", "BIIB", "BK", "BKNG", "BLK", "BSX", "C", "CAH",
    "CB", "CCI", "CCL", "CI", "CINF", "CL", "CLX", "CMA", "CME", "CMI",
    "CNP", "COF", "COP", "CPB", "CRM", "CSX", "CTAS", "CTSH", "D", "DAL",
    "DD", "DG", "DHI", "DHR", "DLTR", "DOV", "DTE", "DUK", "EA", "EBAY",
    "ECL", "ED", "EFX", "EIX", "EL", "EMR", "EOG", "EQR", "ES", "ETN",
    "ETR", "EW", "EXC", "EXPD", "F", "FAST", "FDX", "FIS", "FITB", "GD",
    "GIS", "GLW", "GPC", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HES",
    "HIG", "HOLX", "HPQ", "HRL", "HSY", "HST", "HUM", "IDXX", "IEX", "IFF",
    "INCY", "INTU", "IP", "IPG", "IRM", "ISRG", "ITW", "IVZ", "JBHT", "JCI",
    "JKHY", "K", "KEY", "KIM", "KLAC", "KMB", "KR", "L", "LEN", "LH",
    "LIN", "LNC", "LNT", "LRCX", "LUV", "MAR", "MAS", "MCHP", "MCK", "MCO",
    "MDT", "MET", "MKC", "MLM", "MNST", "MO", "MOS", "MS", "MSI", "MTB",
    "MU", "NEE", "NEM", "NKE", "NOC", "NSC", "NTAP", "NTRS", "NUE", "NVDA",
    "NWL", "O", "ODFL", "OKE", "OMC", "ORLY", "OXY", "PAYX", "PCAR", "PEG",
    "PGR", "PH", "PHM", "PKG", "PLD", "PNC", "PNW", "PPG", "PPL", "PRU",
    "PSA", "PWR", "RCL", "REG", "REGN", "RF", "RJF", "ROK", "ROP", "ROST",
    "RSG", "SBAC", "SCHW", "SHW", "SJM", "SLB", "SNA", "SO", "SPG", "SPGI",
    "SRE", "STT", "STZ", "SWK", "SWKS", "SYK", "SYY", "TAP", "TFC", "TJX",
    "TMO", "TROW", "TRV", "TSCO", "TSN", "TT", "TXT", "UNP", "URI", "USB",
    "VFC", "VLO", "VMC", "VRSN", "VRTX", "VTR", "WAT", "WBA", "WEC", "WM",
    "WMB", "WY", "XEL", "YUM", "ZBH", "ZION",
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
    "largecap250": {
        "symbols": sorted(set(MEGACAP + LARGECAP_EXTRA)),
        "survivorship": "SEVERE",
        "note": "Roughly 250 long-listed large caps. Many more training samples "
                "for a meta-labelling model, and the same survivorship problem "
                "as MEGACAP - worse the further back a test runs, since this is "
                "the 2026 membership.",
    },
    "full": {
        "symbols": sorted(set(SECTORS + EQUITY + INTERNATIONAL + BONDS
                              + REAL_ASSETS + LEVERAGED + MEGACAP + LARGECAP_EXTRA)),
        "survivorship": "SEVERE",
        "note": "Everything: ETFs plus roughly 250 large caps.",
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
