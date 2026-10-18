"""Phase 1: download 10-K Item 1A and Item 7, and 10-Q MD&A, text into data/cache/.

Usage:
    uv run python -m sec_filings_tone.fetch NVDA AMD          # 10-Ks for specific tickers
    uv run python -m sec_filings_tone.fetch                   # 10-Ks for every company in data/companies.csv
    uv run python -m sec_filings_tone.fetch --10q [TICKERS]   # 10-Q MD&A (Part I, Item 2) instead
"""

import os
import re
import sys
import urllib.request
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from edgar import Company, set_identity

DATA = Path(__file__).parents[2] / "data"
CACHE = DATA / "cache"
COMPANIES = DATA / "companies.csv"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FIRST_FY = 2019
MIN_CHARS = 5_000  # shorter than this means extraction failed
# form -> (manifest, {item: how to get it from the parsed report})
FORMS = {
    "10-K": (DATA / "filings.csv", {"1A": lambda r: r.risk_factors, "7": lambda r: r.management_discussion}),
    "10-Q": (DATA / "filings_10q.csv", {"mdna": lambda r: r["Part I, Item 2"]}),
}


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


# "Management Discussion" (IBM), "...Management's Financial Discussion" (ETR), "Financial Review" (WFC),
# "s Discussion and Analysis" / "ement's Discussion..." when the heading breaks mid-word (JBHT, ZBH)
MDNA = (r"(.{0,60}\b)?(management'?s? (financial )?discussion( and analysis)?|\w*'?s discussion and analysis)"
        r"( of\b.{0,80})?|financial review")

# Fallback section boundaries, matched against whole heading lines (see heading())
STARTS = {
    "1A": r"(item 1\W*a\W*)?risk factors( \(continued\))?",  # "Item 1(a)" (HAL), page headers (BNY)
    "7": rf"(item 7\W*)?({MDNA})",
    # 10-Q Part I, Item 2; BNY titles it "Items 2. and 3. MD&A...; Quantitative and Qualitative..."
    "mdna": rf"(items? 2\W*(and 3\W*)?)?({MDNA})(; .*)?",
}
ENDS = {
    "1A": r"(item 1\W*b|item 1\W*c|item 2)\b.*|unresolved staff comments|cybersecurity|properties"
          r"|controls and procedures|report of management|report of independent registered public accounting firm",
    "7": r"(item 7\W*a|item 8|item 9)\b.*|quantitative and qualitative disclosures? about market risk"
         r"|financial statements and supplementary data|report of independent registered public accounting firm"
         r"|management's report on internal control over financial reporting|report of management"
         r"|risk factors( and other key information)?",
    "mdna": r"(item 3|item 4|part ii)\b.*|quantitative and qualitative disclosures? about market risk"
            r"|controls and procedures|legal proceedings|risk factors",
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
    return s if 0 < len(s) < 200 else None


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
    """Main document plus any Exhibit 13 or 99, where companies like WFC (13) and
    CLX (99) put the sections they "incorporate by reference"."""
    parts = [filing.markdown()]
    parts += [a.markdown() or "" for a in filing.attachments if (a.document_type or "").startswith(("EX-13", "EX-99"))]
    return "\n".join(parts)


def fiscal_year(report_date: str) -> int:
    """A 10-K's fiscal year: the calendar year its period ends in, but a 52/53-week year that ends in
    the first week of January belongs to the year before (JNJ's fiscal 2020 ended 2021-01-03)."""
    return (date.fromisoformat(report_date) - timedelta(days=7)).year


def fetch_ticker(ticker: str, sector: str, form: str) -> list[dict]:
    items = FORMS[form][1]
    rows, seen = [], {}
    # EDGAR uses BRK-B where Wikipedia uses BRK.B
    for filing in Company(ticker.replace(".", "-")).get_filings(form=form, amendments=False):
        # 10-Qs keep the calendar year: they're named by period end, so the year never decides which is kept
        fy = fiscal_year(filing.report_date) if form == "10-K" else int(filing.report_date[:4])
        if fy < FIRST_FY:
            break  # filings come newest first
        # one 10-K per fiscal year; 10-Qs are named by period end since they have no FY label
        key = fy if form == "10-K" else filing.report_date
        if key in seen:
            if seen[key] != filing.report_date:  # two periods with one label: the rule above is wrong for them
                raise ValueError(f"{ticker}: 10-Ks for periods ending {seen[key]} and {filing.report_date} "
                                 f"both map to fiscal {fy}")
            continue  # keep the latest filing for a period
        seen[key] = filing.report_date

        paths = {item: CACHE / f"{ticker}_{key}_{item}.txt" for item in items}
        if not all(p.exists() for p in paths.values()):
            report, full = filing.obj(), None
            for item, get in items.items():
                text = str(get(report) or "")
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

    args = sys.argv[1:]
    form = "10-Q" if "--10q" in args else "10-K"
    manifest = FORMS[form][0]
    companies = load_companies()
    if tickers := [a for a in args if a != "--10q"]:
        companies = companies[companies.ticker.isin(tickers)]
        if missing := set(tickers) - set(companies.ticker):
            sys.exit(f"Not in {COMPANIES}: {sorted(missing)}")

    rows, failed = [], []
    for i, (ticker, sector) in enumerate(zip(companies.ticker, companies.sector), 1):
        try:
            got = fetch_ticker(ticker, sector, form)
            rows += got
            print(f"[{i}/{len(companies)}] {ticker}: {len(got)} filings, {sum(not r['ok'] for r in got)} short")
        except Exception as e:  # one bad company shouldn't stop a 500-company run
            print(f"[{i}/{len(companies)}] {ticker}: FAILED {type(e).__name__}: {e}")
            failed.append({"ticker": ticker, "sector": sector, "ok": False})

    new = merge_manifest(pd.read_csv(manifest) if manifest.exists() else None, rows, failed)
    new.sort_values(["ticker", "report_date"]).to_csv(manifest, index=False)

    bad = new[~new.ok]
    print(f"\n{len(new)} rows in {manifest}, {len(bad)} failed or short:")
    if len(bad):
        print(bad.to_string(index=False))
    if failed:
        print(f"failed this run, earlier rows kept: {', '.join(f['ticker'] for f in failed)}")


def merge_manifest(old: pd.DataFrame | None, rows: list[dict], failed: list[dict]) -> pd.DataFrame:
    """This run's rows replace each fetched ticker's earlier ones; other tickers keep theirs. A ticker
    that failed keeps its earlier rows, so a network error doesn't drop it from the next ingest, and
    gets a failure row only if it has none."""
    kept = pd.DataFrame(columns=["ticker"]) if old is None else old[~old.ticker.isin({r["ticker"] for r in rows})]
    failed = [f for f in failed if f["ticker"] not in set(kept.ticker)]
    return pd.concat([x for x in (kept, pd.DataFrame(rows + failed)) if len(x)], ignore_index=True)


if __name__ == "__main__":
    main()
