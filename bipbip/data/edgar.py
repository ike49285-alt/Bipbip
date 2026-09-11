"""Read tender offers out of EDGAR, and find the ones that favour small holders.

WHY THIS FILE EXISTS. Every result in this project so far came from searching
price for a pattern, and all fourteen came back empty. This is the other kind of
source: a structural obligation. In a US issuer tender offer the company files
an SC TO-I, and the offer very often carries an ODD LOT PROVISION - holders of
fewer than 100 shares are exempt from proration. When an offer is oversubscribed
the institutions get cut back and the odd-lot holder is filled in full.

That is an edge which exists BECAUSE the account is small and disappears as it
grows, which is the exact opposite of everything else here. CLAUDE.md lists two
structural advantages of a small account - it can hold, and it never has to
quote. This is a third one that file does not know about.

The capital arithmetic gates it and is worth stating before any code runs. The
position is 99 shares, not a dollar figure you choose: at $500 the tradeable
universe is stocks under about $5, at $2,000 it reaches $20, at $10,000 about
$100. Thousands, not millions - but not $2.10 either.

ARCHITECTURE, AND WHY IT IS SPLIT THIS WAY. EDGAR is unreachable from the
research session (every sec.gov endpoint returns a connection failure through
the proxy) and reachable from the CI runner, which is how every bar and chain in
this archive arrived. So FETCHING lives behind two thin functions and every
judgement - is there an odd-lot provision, what is the price, when does it
expire - is a PURE function of text that tests exercise on fixtures. Code that
cannot be run where it is written must at least be code whose decisions can be.

Nothing here guesses. A field that cannot be parsed comes back None rather than
a plausible-looking default, because this repo has already been bitten once by a
silent fallback that quietly contaminated 23.4% of a null.
"""
from __future__ import annotations

import re
import datetime as dt

#: EDGAR full-text search. Covers 2001 onward, which is what makes a historical
#: study possible at all - tender offers cannot be recovered from price data.
FTS_URL = "https://efts.sec.gov/LATEST/search-index"

#: The SEC blocks requests without a real contact address, and rate-limits at
#: ten a second. Both are conditions of use rather than suggestions.
SEC_RATE_LIMIT_PER_SEC = 10

#: Issuer tender offer, and its amendments.
TENDER_FORMS = ("SC TO-I", "SC TO-I/A")

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[\s ]+")

# "fewer than 100 shares" is the operative phrase; "odd lot" alone is not
# enough, because a filing can mention odd lots without granting them anything.
_ODD_LOT_PHRASE = re.compile(
    r"(odd[\s-]?lot|fewer than (?:one hundred|100) shares"
    r"|less than (?:one hundred|100) shares)", re.I)
# The provision only matters if it exempts them from being cut back.
_PRORATION_PHRASE = re.compile(
    r"(proration|prorat\w*|not be subject to|priority|preferential)", re.I)

_MONEY = r"\$\s?([0-9]+(?:\.[0-9]{1,2})?)"
_RANGE = re.compile(
    rf"(?:between|from)\s+{_MONEY}\s+(?:and|to)\s+{_MONEY}", re.I)
_SINGLE = re.compile(rf"purchase price of\s+{_MONEY}", re.I)

_MONTHS = ("january february march april may june july august september "
           "october november december").split()
# The gap must allow PERIODS. A tender expiry sentence almost always reads
# "will expire at 5:00 p.m., New York City time, on November 14, 2025", and a
# character class excluding "." cannot cross "p.m." - which would have returned
# None for the expiry of nearly every real filing while looking like it worked.
# The distance bound is what keeps it local instead; DOTALL so it survives the
# newlines that stripping leaves behind.
_EXPIRY = re.compile(
    r"expir\w*.{0,160}?(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),\s*(\d{4})",
    re.I | re.S)


def strip_html(raw: str) -> str:
    """Filing text with markup removed and whitespace normalised.

    Filings are hand-built HTML with tags mid-sentence, so a phrase can be
    split by markup that means nothing. Collapsing first is what makes the
    patterns below match the sentence a human reads rather than the bytes.
    """
    text = _TAG.sub(" ", raw)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#8217;", "'").replace("&#8220;", '"')
                .replace("&#8221;", '"').replace("&#151;", "-"))
    return _WS.sub(" ", text).strip()


def has_odd_lot_provision(text: str, window: int = 400) -> bool:
    """True when the filing exempts sub-100-share holders from proration.

    Both halves are required and they must be NEAR each other. A tender
    document mentions odd lots and proration in many places; only a passage
    that says both together is granting the priority this is looking for, and
    matching them anywhere in a fifty-page filing would fire on almost all of
    them.
    """
    for m in _ODD_LOT_PHRASE.finditer(text):
        lo = max(0, m.start() - window)
        if _PRORATION_PHRASE.search(text[lo:m.end() + window]):
            return True
    return False


def tender_price_range(text: str) -> tuple[float, float] | None:
    """(low, high) offer price. Equal for a fixed-price offer, None if unclear.

    Modified Dutch auctions quote a range and the issuer settles at the lowest
    clearing price, so the LOW end is what a tendering holder should assume.
    Returning a single number for those would overstate the offer.
    """
    m = _RANGE.search(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi) if lo <= hi else (hi, lo)
    m = _SINGLE.search(text)
    if m:
        p = float(m.group(1))
        return (p, p)
    return None


def expiration_date(text: str) -> dt.date | None:
    """When the offer expires, or None when no unambiguous date is stated."""
    m = _EXPIRY.search(text)
    if not m:
        return None
    month = _MONTHS.index(m.group(1).lower()) + 1
    try:
        return dt.date(int(m.group(3)), month, int(m.group(2)))
    except ValueError:
        return None


def parse_tender(text: str) -> dict:
    """Everything decidable from the filing body alone.

    `odd_lot` is the field that matters; the rest is context for sizing. Each
    is independently None-able so a partially parseable filing is still worth
    keeping rather than being dropped whole.
    """
    clean = strip_html(text)
    price = tender_price_range(clean)
    return {
        "odd_lot": has_odd_lot_provision(clean),
        "price_low": price[0] if price else None,
        "price_high": price[1] if price else None,
        "dutch_auction": bool(price and price[0] != price[1]),
        "expires": expiration_date(clean),
        "chars": len(clean),
    }


# --------------------------------------------------------------------------
# Fetching. Only these two touch the network, and only CI can run them.
# --------------------------------------------------------------------------

def _require_user_agent(user_agent: str | None) -> str:
    """The contact address the SEC requires, normalised to one header line.

    Whitespace is COLLAPSED rather than merely stripped, and the reason is not
    cosmetic. A value pasted into a CI variable box across two lines arrives as
    "Bipbip Research \\r\\nike49285@gmail.com", and `requests` refuses to send a
    header containing CR or LF at all - so the collector died before its first
    request with a traceback that named the header rather than the cause. A
    newline in a header value is also how header injection is spelled, so
    removing it is the correct handling in both directions; the address itself
    is what the SEC actually asks for and it survives intact.
    """
    ua = _WS.sub(" ", user_agent or "").strip()
    if not ua or "@" not in ua:
        raise ValueError(
            "SEC requires a User-Agent carrying a real contact address, e.g. "
            "'Bipbip Research you@example.com'. Set SEC_USER_AGENT.")
    return ua


def full_text_search(query: str, forms=TENDER_FORMS, date_from: str | None = None,
                     date_to: str | None = None, user_agent: str | None = None,
                     session=None) -> list[dict]:
    """Hits from EDGAR full-text search. Raises rather than returning nothing.

    A silent empty list here would look identical to "no tender offers this
    week", which is a normal and expected result - so a transport failure has
    to be loud or the archive quietly stops growing and nobody notices.
    """
    import requests

    ua = _require_user_agent(user_agent)
    params = {"q": query, "forms": ",".join(forms)}
    if date_from:
        params["dateRange"] = "custom"
        params["startdt"] = date_from
        params["enddt"] = date_to or date_from
    get = (session or requests).get
    r = get(FTS_URL, params=params, headers={"User-Agent": ua}, timeout=30)
    r.raise_for_status()
    return r.json().get("hits", {}).get("hits", [])


def fetch_document(url: str, user_agent: str | None = None, session=None) -> str:
    """Raw filing text. Raises on any non-200 rather than returning a stub."""
    import requests

    ua = _require_user_agent(user_agent)
    get = (session or requests).get
    r = get(url, headers={"User-Agent": ua}, timeout=60)
    r.raise_for_status()
    return r.text


def hit_to_row(hit: dict) -> dict:
    """Flatten one search hit into the columns the archive stores.

    EDGAR's `_id` is "<accession>:<document>", and the accession with its
    dashes removed is the directory the document actually lives in - which is
    the only way to build a fetchable URL from a search result.
    """
    src = hit.get("_source", {})
    ident = hit.get("_id", "")
    accession, _, document = ident.partition(":")
    cik = (src.get("ciks") or [""])[0]
    bare = accession.replace("-", "")
    url = (f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/"
           f"{bare}/{document}") if cik and bare and document else None
    return {
        "accession": accession,
        "cik": cik,
        "company": (src.get("display_names") or [""])[0],
        "form": src.get("root_form") or src.get("file_type"),
        "filed": src.get("file_date"),
        "url": url,
    }
