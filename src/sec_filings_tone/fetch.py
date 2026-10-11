"""Phase 1: download 10-K Item 1A and Item 7 text into data/cache/.

Usage:
    uv run python -m sec_filings_tone.fetch NVDA AMD    # specific tickers
    uv run python -m sec_filings_tone.fetch             # every company in data/companies.csv
"""

import os
import re
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from edgar import Company, set_identity

DATA = Path("data")
CACHE = DATA / "cache"
COMPANIES = DATA / "companies.csv"
MANIFEST = DATA / "filings.csv"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FIRST_FY = 2019
MIN_CHARS = 5_000  # shorter than this means extraction failed
ITEMS = {"1A": "risk_factors", "7": "management_discussion"}


def load_companies() -> pd.DataFrame:
    """S&P 500 tickers and GICS sectors, saved once so the list stays fixed between runs."""
    if not COMPANIES.exists():
        # Wikipedia returns 403 to urllib's default User-Agent
        req = urllib.request.Request(SP500_URL, headers={"User-Agent": os.environ["SEC_USER_AGENT"]})
        html = urllib.request.urlopen(req).read().decode()
        df = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
        df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector",
                                "GICS Sub-Industry": "sub_industry", "CIK": "cik"})
        DATA.mkdir(exist_ok=True)
        df[["ticker", "name", "sector", "sub_industry", "cik"]].to_csv(COMPANIES, index=False)
    # GOOG/GOOGL, FOX/FOXA, NWS/NWSA are one company with two share classes
    return pd.read_csv(COMPANIES).drop_duplicates("cik")


# Fallback section boundaries, matched against whole heading lines (see heading())
STARTS = {
    "1A": r"(item 1\W*a\W*)?risk factors( \(continued\))?",  # "Item 1(a)" (HAL), page headers (BNY)
    # "Management Discussion" (IBM), "...Management's Financial Discussion" (ETR), "Financial Review" (WFC),
    # "s Discussion and Analysis" when the heading breaks at the apostrophe (JBHT)
    "7": r"(item 7\W*)?(.{0,60}\b)?(management'?s? (financial )?discussion( and analysis)?|'?s discussion and analysis)"
         r"( of .{0,80})?|financial review",
}
ENDS = {
    "1A": r"(item 1\W*b|item 1\W*c|item 2)\b.*|unresolved staff comments|cybersecurity|properties"
          r"|controls and procedures|report of management|report of independent registered public accounting firm",
    "7": r"(item 7\W*a|item 8|item 9)\b.*|quantitative and qualitative disclosures? about market risk"
         r"|financial statements and supplementary data|report of independent registered public accounting firm"
         r"|management's report on internal control over financial reporting|report of management"
         r"|risk factors( and other key information)?",
}


def heading(line: str) -> str | None:
    """Normalized text if the line could be a section title, else None. Table rows count
    only if no cell is empty or a page number, which drops table-of-contents rows."""
    s = line.strip().replace("’", "'")
    if s.startswith("|"):
        cells = [c.strip() for c in s.strip("|").split("|")]
        if any(not c or re.fullmatch(r"[\d\s–-]+", c) for c in cells):
            return None
        s = " ".join(cells)
    s = re.sub(r"\s+", " ", re.sub(r"[#*\\]", "", s)).strip(" .:").lower()
    return s if 0 < len(s) < 150 else None


def find_section(text: str, item: str) -> str:
    """Longest span from a section title to the next section's title. Each end keeps its
    earliest start, so repeated page headers ("MD&A" on every page) stay in one span,
    while a short "Item 7: see MD&A on page 50" stub loses to the real section."""
    lines = text.splitlines()
    heads = [(i, h) for i, l in enumerate(lines) if (h := heading(l))]
    starts = [i for i, h in heads if re.fullmatch(STARTS[item], h)]
    ends = [i for i, h in heads if re.fullmatch(ENDS[item], h)]
    spans = {}
    for s in starts:
        # no later title means the section is the last one in the document (ETN)
        spans.setdefault(next((e for e in ends if e > s), len(lines)), s)
    if not spans:
        return ""
    e, s = max(spans.items(), key=lambda es: es[0] - es[1])
    return "\n".join(lines[s:e])


def full_text(filing) -> str:
    """Main 10-K document plus any Exhibit 13 or 99, where companies like WFC (13) and
    CLX (99) put the sections they "incorporate by reference"."""
    parts = [filing.markdown()]
    parts += [a.markdown() or "" for a in filing.attachments if (a.document_type or "").startswith(("EX-13", "EX-99"))]
    return "\n".join(parts)


def fetch_ticker(ticker: str, sector: str) -> list[dict]:
    rows, seen = [], set()
    # EDGAR uses BRK-B where Wikipedia uses BRK.B
    for filing in Company(ticker.replace(".", "-")).get_filings(form="10-K", amendments=False):
        fy = int(filing.report_date[:4])
        if fy < FIRST_FY:
            break  # filings come newest first
        if fy in seen:
            continue  # keep the latest 10-K for a fiscal year
        seen.add(fy)

        paths = {item: CACHE / f"{ticker}_{fy}_{item}.txt" for item in ITEMS}
        if not all(p.exists() for p in paths.values()):
            tenk, full = filing.obj(), None
            for item, attr in ITEMS.items():
                text = str(getattr(tenk, attr) or "")
                if len(text) < MIN_CHARS:  # parser missed it or it's incorporated by reference
                    full = full or full_text(filing)
                    text = find_section(full, item) or text
                paths[item].write_text(text)

        lengths = {f"len_{item}": len(p.read_text()) for item, p in paths.items()}
        rows.append({"ticker": ticker, "sector": sector, "fy": fy, "report_date": filing.report_date,
                     "filing_date": str(filing.filing_date), "accession": filing.accession_no, **lengths,
                     "ok": all(n >= MIN_CHARS for n in lengths.values())})
    return rows


def main() -> None:
    load_dotenv()
    set_identity(os.environ["SEC_USER_AGENT"])
    CACHE.mkdir(parents=True, exist_ok=True)

    companies = load_companies()
    if tickers := sys.argv[1:]:
        companies = companies[companies.ticker.isin(tickers)]
        if missing := set(tickers) - set(companies.ticker):
            sys.exit(f"Not in {COMPANIES}: {sorted(missing)}")

    rows = []
    for i, (ticker, sector) in enumerate(zip(companies.ticker, companies.sector), 1):
        try:
            got = fetch_ticker(ticker, sector)
            rows += got
            print(f"[{i}/{len(companies)}] {ticker}: {len(got)} filings, {sum(not r['ok'] for r in got)} short")
        except Exception as e:  # one bad company shouldn't stop a 500-company run
            print(f"[{i}/{len(companies)}] {ticker}: FAILED {type(e).__name__}: {e}")
            rows.append({"ticker": ticker, "sector": sector, "ok": False})

    new = pd.DataFrame(rows)
    if MANIFEST.exists():  # replace this run's tickers, keep the rest
        old = pd.read_csv(MANIFEST)
        new = pd.concat([old[~old.ticker.isin(new.ticker)], new])
    new.sort_values(["ticker", "fy"]).to_csv(MANIFEST, index=False)

    bad = new[~new.ok]
    print(f"\n{len(new)} rows in {MANIFEST}, {len(bad)} failed or short:")
    if len(bad):
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
