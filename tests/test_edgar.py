"""Tender-offer parsing, on text shaped like the filings it will actually meet.

The fetching half of `bipbip/data/edgar.py` cannot run here - every sec.gov
endpoint is unreachable through this session's proxy, which is why the collector
lives in CI. That makes these tests the only place the module's JUDGEMENTS get
checked, so they carry more weight than usual: each one is a decision the
collector will make unsupervised, on filings nobody reads.

The fixtures are written the way real SC TO-I documents are written - tags in
the middle of sentences, &nbsp; instead of spaces, the operative phrase split
across markup - because that is exactly what defeats a naive regex.
"""
import datetime as dt

import pytest

from bipbip.data import edgar


ODD_LOT_REAL = """
<P>7. <B>Odd&nbsp;Lots.</B> The Company, upon the terms and subject to the
conditions of the Offer, will accept for purchase <I>all</I> Shares properly
tendered by any stockholder who <B>owns, beneficially or of record, an
aggregate of fewer than 100&nbsp;Shares</B> and who tenders all of such Shares.
Such Shares <U>will not be subject to proration</U>.</P>
"""

NO_ODD_LOT = """
<P>If more Shares are properly tendered than the Company is willing to purchase,
the Company will purchase Shares on a pro rata basis, with appropriate
adjustments to avoid the purchase of fractional Shares.</P>
"""

# Mentions both ideas, but pages apart and with no priority granted. This is
# the case a match-anywhere implementation gets wrong.
DISTANT_MENTION = ("<P>Holders of odd lots may contact the Information Agent "
                   "with questions about the mechanics of tendering.</P>"
                   + "<P>Filler paragraph.</P>" * 200
                   + "<P>Shares will be purchased subject to proration as "
                     "described in Section 1.</P>")

DUTCH = """
<P>The Company is offering to purchase for cash up to 10,000,000 Shares at a
price <B>between&nbsp;$41.00 and $47.00</B> per Share, net to the seller in
cash. The Offer will <B>expire</B> at 5:00 p.m., New York City time, on
<B>November&nbsp;14, 2025</B>, unless extended.</P>
"""

FIXED = """
<P>The Company hereby offers to purchase Shares at a purchase price of
<B>$18.50</B> per Share. The Offer expires on March 3, 2026.</P>
"""


def test_html_is_stripped_so_split_phrases_still_match():
    """The operative phrase is routinely broken by markup mid-sentence."""
    clean = edgar.strip_html(ODD_LOT_REAL)
    assert "<" not in clean and "&nbsp;" not in clean
    assert "fewer than 100 Shares" in clean
    # And the collapse is idempotent - no double spaces left to defeat a regex.
    assert "  " not in clean


def test_a_real_odd_lot_provision_is_found():
    assert edgar.parse_tender(ODD_LOT_REAL)["odd_lot"] is True


def test_a_plain_proration_clause_is_not_mistaken_for_one():
    assert edgar.parse_tender(NO_ODD_LOT)["odd_lot"] is False


def test_odd_lots_mentioned_far_from_any_priority_do_not_count():
    """The failure mode of matching both phrases anywhere in a 50-page filing.

    Nearly every tender document says "odd lot" somewhere and "proration"
    somewhere. Only a passage saying both TOGETHER is granting the exemption,
    so the two have to be near each other or the collector flags everything.
    """
    assert edgar.parse_tender(DISTANT_MENTION)["odd_lot"] is False


def test_a_dutch_auction_returns_the_range_not_a_point():
    """The issuer settles at the lowest clearing price, so a tendering holder
    should assume the LOW end. Collapsing the range to one number would
    overstate every Dutch auction in the archive."""
    r = edgar.parse_tender(DUTCH)
    assert (r["price_low"], r["price_high"]) == (41.00, 47.00)
    assert r["dutch_auction"] is True
    assert r["expires"] == dt.date(2025, 11, 14)


def test_a_fixed_price_offer_has_an_equal_range():
    r = edgar.parse_tender(FIXED)
    assert (r["price_low"], r["price_high"]) == (18.50, 18.50)
    assert r["dutch_auction"] is False
    assert r["expires"] == dt.date(2026, 3, 3)


def test_unparseable_fields_come_back_none_rather_than_guessed():
    """This repo has already been bitten by a silent fallback that handed a
    null run 23.4% real signal. A missing price must read as missing."""
    r = edgar.parse_tender("<P>The Offer is described in the Offer to "
                           "Purchase.</P>")
    assert r["price_low"] is None and r["price_high"] is None
    assert r["expires"] is None
    assert r["odd_lot"] is False


def test_a_reversed_range_is_ordered():
    r = edgar.parse_tender("<P>between $30.00 and $25.00 per Share</P>")
    assert (r["price_low"], r["price_high"]) == (25.00, 30.00)


def test_an_impossible_date_is_rejected_not_clamped():
    assert edgar.expiration_date("the Offer will expire on February 31, 2026") is None


@pytest.mark.parametrize("ua", [None, "", "Bipbip Research", "no-at-sign",
                                "   ", "\n\n"])
def test_fetching_without_a_contact_address_is_refused(ua):
    """The SEC blocks anonymous automated access, so failing here with a clear
    message beats being silently throttled or banned in CI."""
    with pytest.raises(ValueError, match="contact address"):
        edgar.full_text_search("odd lot", user_agent=ua)
    with pytest.raises(ValueError, match="contact address"):
        edgar.fetch_document("https://example.com/x.htm", user_agent=ua)


def test_a_user_agent_pasted_across_two_lines_still_sends():
    """The collector's real first failure, pinned.

    A contact address entered into a CI variable box over two lines arrives
    carrying CR and LF. `requests` refuses to send a header containing either,
    so the run died on its FIRST call with an InvalidHeader traceback - the
    variable was set correctly by any reasonable reading and the collector
    still could not start. Collapsing the whitespace is also what stops a
    newline in a header value being an injection, so it is right either way.
    """
    ua = edgar._require_user_agent("Bipbip Research \r\nike49285@gmail.com")
    assert ua == "Bipbip Research ike49285@gmail.com"
    assert "\r" not in ua and "\n" not in ua


def test_the_normalised_agent_is_what_actually_goes_on_the_wire():
    """Normalising and then sending the ORIGINAL would fix nothing."""
    sent = {}

    class _Capture:
        def get(self, url, params=None, headers=None, timeout=None):
            sent.update(headers)
            raise RuntimeError("stop here; the header is what is under test")

    with pytest.raises(RuntimeError):
        edgar.full_text_search("odd lot", user_agent="Bipbip\r\n t@e.com",
                               session=_Capture())
    assert sent["User-Agent"] == "Bipbip t@e.com"


def test_a_search_hit_becomes_a_fetchable_url():
    """EDGAR's accession number carries dashes and the directory does not, so
    the URL cannot be built by concatenating the id as it arrives."""
    row = edgar.hit_to_row({
        "_id": "0000320193-25-000079:aapl-20250927.htm",
        "_source": {"ciks": ["0000320193"], "display_names": ["Apple Inc."],
                    "root_form": "SC TO-I", "file_date": "2025-10-31"},
    })
    assert row["cik"] == "0000320193"
    assert row["company"] == "Apple Inc."
    assert row["url"] == ("https://www.sec.gov/Archives/edgar/data/320193/"
                          "000032019325000079/aapl-20250927.htm")


def test_a_hit_missing_its_document_yields_no_url_rather_than_a_broken_one():
    row = edgar.hit_to_row({"_id": "", "_source": {}})
    assert row["url"] is None


def test_search_failure_raises_instead_of_reading_as_a_quiet_week():
    """An empty list is a NORMAL result - most weeks have few tender offers -
    so a transport failure that returned one would stop the archive growing
    without anyone noticing."""
    class _Boom:
        def get(self, *a, **k):
            raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError):
        edgar.full_text_search("odd lot", user_agent="Test t@e.com",
                               session=_Boom())


def test_the_expiry_gap_tolerates_abbreviations_but_stays_local():
    """Two failure modes at once, in opposite directions.

    Excluding periods from the gap cannot cross "5:00 p.m.", which appears in
    almost every tender expiry sentence - so the parser would have returned
    None on nearly every real filing while its tests passed. Allowing anything
    unbounded goes the other way and picks up a date from an unrelated
    paragraph. The distance bound is what separates them.
    """
    near = ("The Offer will expire at 5:00 p.m., New York City time, on "
            "November 14, 2025, unless extended.")
    assert edgar.expiration_date(near) == dt.date(2025, 11, 14)

    far = "The Offer will expire as described. " + ("Filler. " * 60) + \
          "The annual meeting is on June 3, 2027."
    assert edgar.expiration_date(far) is None


def test_an_unrecognised_response_shape_is_loud_not_a_quiet_week():
    """The one thing no test here can verify is EDGAR's live response shape,
    because sec.gov is unreachable from this session. If it differs from what
    hit_to_row assumes, every row loses its URL and the collector writes
    nothing - printing "0 new", which is indistinguishable from a normal quiet
    week for SC TO-I. This pins the guard that separates them.
    """
    unusable = [{"id": "wrong-key", "source": {"cik": "320193"}}]
    assert all(edgar.hit_to_row(h)["url"] is None for h in unusable)
