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
import pathlib

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


@pytest.mark.parametrize("display,expected", [
    ("Arbutus Biopharma Corp  (ABUS)  (CIK 0001447380)", "ABUS"),
    ("Ares Private Markets Fund  (CIK 0001876006)", ""),
    ("Virtus Dividend, Interest & Premium Strategy Fund  (NFJ)  (CIK 1268884)",
     "NFJ"),
    ("Berkshire Hathaway Inc  (BRK.A)  (CIK 0001067983)", "BRK.A"),
    ("", ""),
])
def test_a_ticker_is_read_from_the_display_name_and_cik_is_not_one(display,
                                                                   expected):
    """The CIK parenthetical sits in the same position and is all capitals, so
    a naive parenthesis grab returns "CIK" for every unlisted issuer - which
    would mark the entire non-traded population as tradeable."""
    assert edgar.ticker_from_display(display) == expected


def test_listed_is_the_gate_and_non_traded_funds_do_not_pass():
    """Not obvious until real filings arrived: six of the first eight SC TO-I
    hits were non-traded closed-end funds repurchasing AT NAV. Those are a
    redemption feature, not the odd-lot opportunity - there is no market price
    to buy below and the shares are not exchange-listed."""
    assert edgar.is_listed({"ticker": "ABUS"}) is True
    assert edgar.is_listed({"ticker": ""}) is False
    assert edgar.is_listed({}) is False


def test_the_filing_form_falls_back_to_the_plural_key():
    """root_form is NOT always present. A run that assumed it was left the
    column empty for all eight rows - the previous code only looked right
    because it was silently falling through to the EXHIBIT type."""
    row = edgar.hit_to_row({"_id": "a-b:c.htm",
                            "_source": {"ciks": ["1"], "root_forms": ["SC TO-I"]}})
    assert row["form"] == "SC TO-I"
    blank = edgar.hit_to_row({"_id": "a-b:c.htm", "_source": {"ciks": ["1"],
                                                              "file_type": "EX-99"}})
    assert blank["form"] == ""          # not the exhibit type
    assert blank["doc_type"] == "EX-99"


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


def test_the_form_filter_is_omitted_entirely_rather_than_sent_empty():
    """The canary depends on this. A control query has to be genuinely
    unfiltered; sending `forms=` would restrict it to nothing and the canary
    would confirm a breakage that was its own doing."""
    assert "forms" not in edgar.search_params("odd lot", forms=None)
    assert edgar.search_params("odd lot", forms=("SC TO-I",))["forms"] == "SC TO-I"


def test_dates_are_only_sent_when_asked_for():
    bare = edgar.search_params("odd lot")
    assert "startdt" not in bare and "dateRange" not in bare
    dated = edgar.search_params("odd lot", date_from="2026-08-12",
                                date_to="2026-09-11")
    assert dated["dateRange"] == "custom"
    assert (dated["startdt"], dated["enddt"]) == ("2026-08-12", "2026-09-11")


def test_a_missing_end_date_means_a_single_day_not_an_open_range():
    p = edgar.search_params("odd lot", date_from="2026-08-12")
    assert p["startdt"] == p["enddt"] == "2026-08-12"


@pytest.mark.parametrize("payload,expected", [
    ({"hits": {"total": {"value": 1234}}}, 1234),   # elasticsearch object form
    ({"hits": {"total": 7}}, 7),                    # bare int form
    ({"hits": {}}, 0),
    ({}, 0),
])
def test_total_hits_reads_both_shapes_elasticsearch_uses(payload, expected):
    """`len(hits)` caps at one page, so it cannot tell ten matches from ten
    thousand - the canary needs the TOTAL. Elasticsearch reports it as a bare
    int in older versions and as {"value": n} in newer ones, and reading only
    one shape would silently return 0 and fire the canary on a healthy feed."""
    class _Resp:
        def raise_for_status(self): pass
        def json(self): return payload

    class _Sess:
        def get(self, *a, **k): return _Resp()

    assert edgar.total_hits("odd lot", user_agent="T t@e.com",
                            session=_Sess()) == expected


def test_the_form_filter_names_the_root_form_only():
    """Measured live, and it is the opposite of what it looks like.

    Naming the amendment alongside the root form does not widen the search -
    EDGAR's filter matches on root_form, which already covers amendments, so
    listing "SC TO-I/A" INTERSECTS instead. The probe run recorded 4,409
    documents for the root form undated against 992 for the pair, and with a
    date range the pair returned ZERO against the root form's 26. That zero was
    reported by the first real collection run as a quiet month.
    """
    assert edgar.TENDER_FORMS == ("SC TO-I",)
    assert edgar.TENDER_AMENDMENT_FORM not in edgar.TENDER_FORMS
    assert edgar.search_params("odd lot")["forms"] == "SC TO-I"


class _Pager:
    """EDGAR-shaped paging: ten documents a page, `from` selects the offset."""

    def __init__(self, n_docs):
        self.n = n_docs
        self.offsets = []

    def get(self, url, params=None, headers=None, timeout=None):
        start = int(params.get("from", 0))
        self.offsets.append(start)
        page = [{"_id": f"000-{i}:doc{i}.htm", "_source": {"ciks": ["1"]}}
                for i in range(start, min(start + edgar.PAGE_SIZE, self.n))]

        class _R:
            def raise_for_status(self): pass
            def json(self): return {"hits": {"hits": page}}
        return _R()


def test_search_pages_past_the_first_ten_results():
    """The shortfall that was silent. The first real run wrote 8 rows from a
    window the probe said held 26 matches, because EDGAR returns ten documents
    a page and a FULL page looks exactly like a complete result."""
    pager = _Pager(26)
    hits = edgar.full_text_search("odd lot", user_agent="T t@e.com",
                                  session=pager)
    assert len(hits) == 26
    assert pager.offsets[:3] == [0, 10, 20]


def test_paging_stops_on_a_short_page_rather_than_asking_forever():
    pager = _Pager(14)
    assert len(edgar.full_text_search("odd lot", user_agent="T t@e.com",
                                      session=pager)) == 14
    assert pager.offsets == [0, 10]        # stopped after the short page


def test_an_exactly_full_final_page_still_terminates():
    """20 documents is two full pages; the third comes back empty and ends it.
    Stopping only on a short page would otherwise loop to the cap."""
    pager = _Pager(20)
    assert len(edgar.full_text_search("odd lot", user_agent="T t@e.com",
                                      session=pager)) == 20
    assert pager.offsets == [0, 10, 20]


def test_paging_is_capped_so_a_broad_query_cannot_run_away():
    pager = _Pager(10_000)
    hits = edgar.full_text_search("odd lot", user_agent="T t@e.com",
                                  session=pager, max_pages=3)
    assert len(hits) == 30


def test_the_filing_form_is_not_overwritten_by_the_exhibit_type():
    """Full-text search indexes DOCUMENTS, so a hit is usually an exhibit and
    file_type reads "EX-99.(A)(1)". The first real run wrote that into a column
    named `form`, where it would later read as the filing's type."""
    row = edgar.hit_to_row({
        "_id": "0001104659-26-104051:tm_ex99-a1d.htm",
        "_source": {"ciks": ["0001661458"], "display_names": ["Highlands REIT"],
                    "root_form": "SC TO-I", "file_type": "EX-99.(A)(1)",
                    "file_date": "2026-09-01"},
    })
    assert row["form"] == "SC TO-I"
    assert row["doc_type"] == "EX-99.(A)(1)"


def test_appending_to_an_archive_with_a_different_schema_is_refused(tmp_path):
    """A misaligned archive is worse than no archive because it looks fine.

    The schema changed once already - doc_type was added when the first real
    run revealed `form` was being filled with the exhibit type - so a file
    written under the old header is a real case, not a hypothetical.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "collect_tenders",
        pathlib.Path(__file__).resolve().parents[1] / "scripts"
        / "collect_tenders.py")
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)

    # Columns ADDED since the file was written: lossless, so it migrates.
    stale = tmp_path / "tenders.csv"
    stale.write_text("accession,cik,company,form,filed,url\n"
                     "A,1,Widget Co,SC TO-I,2026-09-01,http://x\n")
    assert ct.existing(stale) == {"A"}
    migrated = stale.read_text().splitlines()
    assert migrated[0].split(",") == ct.FIELDS
    # and the data that WAS there survives the rewrite
    assert "Widget Co" in migrated[1] and "SC TO-I" in migrated[1]
    # idempotent: a second pass is a no-op rather than a second migration
    assert ct.existing(stale) == {"A"}

    # A column that no longer exists cannot be carried forward without
    # deciding what it meant, so that one is refused rather than guessed.
    weird = tmp_path / "weird.csv"
    weird.write_text("accession,mystery_column\nB,?\n")
    with pytest.raises(SystemExit, match="does not know"):
        ct.existing(weird)

    good = tmp_path / "ok.csv"
    good.write_text(",".join(ct.FIELDS) + "\n")
    assert ct.existing(good) == set()
