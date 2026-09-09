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

#: A wider net of currently-listed, generally liquid US equities, added to make
#: the CROSS-SECTION deep rather than the history long.
#:
#: The distinction matters and cuts against how the rest of this file is
#: framed. Measured on the minute archive, raw returns across 225 symbols carry
#: a mean pairwise correlation of 0.122 - they behave like eight independent
#: series, because everything rides the market. After the cross-sectional
#: median is removed the correlation falls to 0.002 and the same 225 symbols
#: behave like 145 nearly independent ones. So for a model that ranks names
#: against each other, adding symbols adds information; for one forecasting a
#: single series, it does not.
#:
#: SURVIVORSHIP: these are today's listings, so over decades this list is the
#: worst offender in the file - it is 2026 membership and nothing that failed
#: is in it. Over a 29-day minute window that bias is close to absent, because
#: every name in it demonstrably traded last month. Use for intraday
#: cross-sections; do NOT use for multi-year backtests.
LIQUID_EXTRA = [
    "ABNB", "ADSK", "AEE", "AES", "AFRM", "AKAM", "ALB", "ALGN", "ALL", "ALLE",
    "AMCR", "AMD", "AME", "AMP", "ANET", "ANSS", "AON", "AOS", "APA", "APH",
    "APTV", "ARE", "ATO", "AVB", "AVY", "AWK", "AXON", "AZO", "BALL", "BAX",
    "BBWI", "BBY", "BEN", "BF-B", "BG", "BIIB", "BIO", "BKR", "BLDR", "BMRN",
    "BR", "BRO", "BSX", "BWA", "BXP", "CAG", "CAH", "CBOE", "CBRE", "CCL",
    "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CINF",
    "CLX", "CMA", "CMS", "CNC", "CNP", "COO", "COIN", "CPB", "CPRT", "CPT",
    "CRL", "CSGP", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CZR", "DAL", "DAY",
    "DD", "DECK", "DFS", "DG", "DGX", "DHI", "DLR", "DLTR", "DOC", "DOV",
    "DOW", "DPZ", "DRI", "DTE", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL",
    "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "ENPH", "EPAM", "EQIX",
    "EQR", "EQT", "ES", "ESS", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD",
    "EXPE", "EXR", "F", "FANG", "FAST", "FDS", "FDX", "FE", "FFIV", "FICO",
    "FIS", "FITB", "FLT", "FMC", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GDDY",
    "GEHC", "GEN", "GILD", "GIS", "GL", "GLW", "GNRC", "GPC", "GPN", "GRMN",
    "GWW", "HAL", "HAS", "HBAN", "HCA", "HES", "HIG", "HII", "HOLX", "HOOD",
    "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUBS", "HUM", "HWM",
    "IDXX", "IEX", "IFF", "INCY", "INVH", "IP", "IPG", "IQV", "IR", "IRM",
    "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNPR", "K",
    "KDP", "KEY", "KEYS", "KHC", "KIM", "KMB", "KMI", "KMX", "KR", "KVUE",
    "L", "LDOS", "LEN", "LH", "LHX", "LII", "LKQ", "LNT", "LULU", "LUV",
    "LVS", "LW", "LYB", "LYV", "MAA", "MAR", "MAS", "MCHP", "MCK", "MCO",
    "MDLZ", "MGM", "MHK", "MKC", "MKTX", "MLM", "MNST", "MOH", "MOS", "MPC",
    "MPWR", "MRNA", "MRO", "MSCI", "MSI", "MTB", "MTCH", "MTD", "NCLH", "NDAQ",
    "NDSN", "NET", "NI", "NRG", "NTAP", "NTRS", "NUE", "NVR", "NWS", "NWSA",
    "O", "ODFL", "OKE", "OKTA", "OMC", "ON", "ORLY", "OTIS", "OXY", "PANW",
    "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PFG", "PH", "PHM", "PKG",
    "PLD", "PLTR", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA",
    "PSX", "PTC", "PWR", "PYPL", "QRVO", "RCL", "REG", "REGN", "RF", "RJF",
    "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RVTY", "SBAC", "SHW",
    "SJM", "SMCI", "SNA", "SNAP", "SNPS", "SOFI", "SOLV", "SPG", "SQ", "SRE",
    "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYY",
    "TAP", "TDG", "TDY", "TECH", "TER", "TFC", "TFX", "TPR", "TRGP", "TRMB",
    "TROW", "TRV", "TSCO", "TSN", "TT", "TTWO", "TWLO", "TXT", "TYL", "UAL",
    "UBER", "UDR", "UHS", "ULTA", "URI", "USB", "VICI", "VLO", "VMC", "VRSK",
    "VRSN", "VRTX", "VST", "VTR", "VTRS", "WAB", "WAT", "WBA", "WBD", "WDAY",
    "WDC", "WEC", "WELL", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XYL",
    "YUM", "ZBH", "ZBRA", "ZION", "ZM", "ZS", "ZTS",
    # Liquid ETFs beyond the core set, useful as cross-sectional anchors.
    "VOO", "VTI", "IVV", "QQQM", "SCHD", "VYM", "ARKK", "SMH", "SOXX", "XBI",
    "IBB", "KRE", "XOP", "XME", "ITB", "IYR", "XLC", "XLRE", "GDX", "GDXJ",
    "SLV", "UNG", "TLH", "SHV", "BIL", "JNK", "EMB", "VEA", "VWO", "IEMG",
]

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
    "liquid": {
        "symbols": sorted(set(SECTORS + EQUITY + INTERNATIONAL + BONDS
                              + REAL_ASSETS + LEVERAGED + MEGACAP
                              + LARGECAP_EXTRA + LIQUID_EXTRA)),
        "survivorship": "SEVERE",
        "note": "The widest net here, built to make the CROSS-SECTION deep "
                "rather than the history long. Measured on minute bars, "
                "de-marketed returns across symbols are almost uncorrelated "
                "(0.002), so each name added is close to an independent "
                "observation - unlike raw returns, where 225 symbols behave "
                "like 8. Intended for intraday cross-sectional work; over "
                "years it is the most survivorship-contaminated list in this "
                "file.",
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


def liquid_top(n: int = 1000, bar_size: str = "1m", root: str = "data/bars") -> list:
    """The `n` most liquid symbols, by MEASURED spread and dollar volume.

    "Liquid" is not a judgement to make from a ticker list. A name is tradeable
    or not according to what its own bars say, so this ranks on two things that
    can be measured: median dollar volume per bar, and the effective spread
    implied by bid-ask bounce. Both come from the archive, so a symbol that
    turned out to be thin is dropped on evidence rather than on a guess about
    which index it belongs to.

    Falls back to the declared universe when the archive has nothing to measure.
    """
    import glob
    import os

    import numpy as np
    import pandas as pd

    from .sessions import restrict_to_rth
    from .store import BarStore

    store = BarStore(root)
    rows = []
    for path in sorted(glob.glob(f"{root}/*_{bar_size}.parquet")):
        sym = os.path.basename(path).split("_")[0]
        try:
            bars = store.load(sym, bar_size)
            if bar_size in ("1m", "1h"):
                bars = restrict_to_rth(bars)
            bars = bars.dropna()
        except Exception:
            continue
        if len(bars) < 500:
            continue
        close = bars["close"]
        dollar_vol = float((close * bars["volume"]).median())
        d = close.diff().dropna()
        cov = float(np.cov(d.values[1:], d.values[:-1])[0, 1]) if len(d) > 100 else 0.0
        spread = (2.0 * np.sqrt(-cov) / float(close.mean()) * 1e4
                  if cov < 0 else np.nan)
        spread = max(spread if np.isfinite(spread) else 50.0,
                     0.01 / float(close.iloc[-1]) * 1e4)
        rows.append({"symbol": sym, "dollar_vol": dollar_vol, "spread": spread})

    if not rows:
        return get_universe("liquid")

    df = pd.DataFrame(rows)
    # Rank on both, then combine: cheap to trade AND actually trades.
    df["score"] = (df["dollar_vol"].rank(pct=True)
                   + (-df["spread"]).rank(pct=True))
    return sorted(df.nlargest(min(n, len(df)), "score")["symbol"].tolist())
