"""Archive issuer tender offers, and flag the ones that favour small holders.

Runs in CI, not in a research session: every sec.gov endpoint is unreachable
through the session proxy, exactly as Yahoo is, which is why every bar and chain
in this archive was also fetched by a runner.

WHAT IT IS FOR. An SC TO-I with an ODD LOT PROVISION exempts holders of fewer
than 100 shares from proration. When the offer is oversubscribed the
institutions are cut back and the odd-lot holder is filled in full - an
advantage that exists BECAUSE the account is small. Every other result in this
project came from searching price for a pattern and came back empty; this is a
structural obligation instead, and the first thing here that a small account is
positioned to be paid for rather than smart enough to find.

WHY COLLECT BEFORE THE CAPITAL EXISTS. The position is 99 shares, so at $500 the
universe is stocks under about $5 and at $2,000 it reaches $20. That is a real
gate. But tender offers cannot be reconstructed from a price archive at any
later date - only from filings, as they are made - so an archive that starts
today is worth more than one that starts when the money arrives. EDGAR full-text
search reaches back to 2001, so the same collector can also backfill a history
to measure what the provision has actually paid.

The archive stores one row per filing and never overwrites: a filing already
present is skipped by accession, so re-running is safe and a failed run costs
nothing but the window it missed.
"""
import os
import sys
import csv
import time
import pathlib
import argparse
import datetime as dt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bipbip.data import edgar

OUT = pathlib.Path("data/tenders.csv")
FIELDS = ["accession", "cik", "company", "form", "filed", "url",
          "odd_lot", "price_low", "price_high", "dutch_auction", "expires",
          "chars", "collected_at"]


def existing(path: pathlib.Path) -> set:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {r["accession"] for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7,
                    help="look back this many days from --end")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD, default today")
    ap.add_argument("--query", default='"odd lot"',
                    help="full-text query; the default finds the provision "
                         "directly, but the FILING is what gets parsed")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit(
            "SEC_USER_AGENT is unset. The SEC blocks automated access without "
            "a contact address; set it to something like "
            "'Bipbip Research you@example.com'.")

    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today()
    start = end - dt.timedelta(days=a.days)
    print(f"EDGAR {edgar.TENDER_FORMS} matching {a.query} "
          f"{start} -> {end}", flush=True)

    hits = edgar.full_text_search(a.query, date_from=start.isoformat(),
                                  date_to=end.isoformat(), user_agent=ua)
    print(f"  {len(hits)} hits", flush=True)

    # The response SHAPE is the one thing that cannot be verified from a
    # research session, because sec.gov is unreachable there. If EDGAR returns
    # hits in a form hit_to_row does not understand, every row loses its
    # accession and url, nothing is written, and the run prints "0 new" - which
    # reads exactly like a quiet week. That is the silent failure this repo has
    # already been burned by once, so it is made loud here instead.
    usable = sum(1 for h in hits if edgar.hit_to_row(h)["url"])
    if hits and usable == 0:
        raise SystemExit(
            f"{len(hits)} hits returned but NONE could be turned into a "
            f"document URL. EDGAR's response shape has changed or was never "
            f"what hit_to_row assumed; first hit keys: "
            f"{sorted(hits[0].keys())}, _source keys: "
            f"{sorted(hits[0].get('_source', {}).keys())}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = existing(out)
    new = 0
    write_header = not out.exists()

    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for hit in hits:
            row = edgar.hit_to_row(hit)
            if not row["accession"] or row["accession"] in seen:
                continue
            if not row["url"]:
                print(f"  skip {row['accession']}: no document url", flush=True)
                continue
            # The SEC asks for no more than ten requests a second. This is a
            # condition of access, not a suggestion - exceeding it gets the
            # runner's address blocked, which would be a silent outage.
            time.sleep(1.0 / edgar.SEC_RATE_LIMIT_PER_SEC)
            try:
                body = edgar.fetch_document(row["url"], user_agent=ua)
            except Exception as exc:
                # One unreachable document must not lose the rest of the batch,
                # but it is reported rather than swallowed.
                print(f"  FETCH FAILED {row['accession']}: "
                      f"{type(exc).__name__} {str(exc)[:100]}", flush=True)
                continue
            row.update(edgar.parse_tender(body))
            row["collected_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            w.writerow(row)
            seen.add(row["accession"])
            new += 1
            flag = "ODD-LOT" if row["odd_lot"] else "       "
            px = (f"${row['price_low']:.2f}" if row["price_low"] else "  ?  ")
            if row["dutch_auction"]:
                px += f"-${row['price_high']:.2f}"
            print(f"  {flag} {row['filed']} {row['company'][:38]:<38} {px}",
                  flush=True)

    total = len(seen)
    print(f"\n{new} new, {total} in archive -> {out}")
    if new == 0:
        # Not an error. Most weeks have few issuer tender offers, and a quiet
        # week must not look like a broken collector - the fetch layer raises
        # on transport failure precisely so this line stays meaningful.
        print("(no new filings; a quiet week is normal for SC TO-I)")


if __name__ == "__main__":
    main()
