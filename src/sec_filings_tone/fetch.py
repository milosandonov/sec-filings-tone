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
    return pd.read_csv(COMPANIES)


def mdna_fallback(filing) -> str:
    """MD&A for 10-Ks without a standard "Item 7" heading (Intel). Their section titles
    are one-cell table rows ("| Risk Factors |") or, before FY2023, "# Risk Factors"
    headings. MD&A runs from its title to the Risk Factors title."""
    lines = filing.markdown().replace("’", "'").splitlines()
    start = next((i for i, l in enumerate(lines)
                  if re.fullmatch(r"\|\s*management's discussion and analysis\s*\|", l.strip(), re.I)), None)
    if start is None:
        return ""
    end = next((i for i, l in enumerate(lines[start:], start)
                if re.fullmatch(r"[#|\s]*risk factors( and other key information)?[\s|]*", l, re.I)), None)
    return "\n".join(lines[start:end]) if end else ""


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
            tenk = filing.obj()
            for item, attr in ITEMS.items():
                text = str(getattr(tenk, attr) or "")
                if item == "7" and len(text) < MIN_CHARS:
                    text = mdna_fallback(filing)
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
