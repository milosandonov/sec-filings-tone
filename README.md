# SEC Filings Tone

This project tracks how companies' tone on specific topics (supply chain, AI, China, interest rates, tariffs, and others) changes across their SEC filings, in every sector of the S&P 500. Retrieval (RAG) finds the passages about each topic, three scorers rate their sentiment, and a Q&A layer lets users ask their own questions about any company or sector.

## Research questions

1. How well do FinBERT, an LLM and the Loughran-McDonald dictionary score SEC filing text, both against labels and against real events?
2. Does tone react to known shocks (export controls, the 2023 bank failures, tariffs) in the sectors and topics they hit, and which scorer picks that up?
3. How much of a 10-K is boilerplate carried over from last year, and do scorers behave differently on boilerplate than on new text?
4. Does tone move together within a sector more than across sectors?

## Status

Work in progress. The filing download is done. The other stages are planned but not built yet. See [PLAN.md](PLAN.md) for the full design.

| Stage | Status |
|---|---|
| 1. Fetch and parse filings (`fetch.py`) | Done |
| 2. Chunk and embed (`ingest.py`) | Not started |
| 3. Retrieval: metadata filter, hybrid search, reranking (`retrieve.py`) | Not started |
| 4. Retrieval eval (`eval/`) | Not started |
| 5. Sentiment scorers: FinBERT, LLM, Loughran-McDonald (`sentiment.py`) | Not started |
| 6. Labeled benchmark (silver and gold sets) | Not started |
| 7. Event validation (`events.csv`) | Not started |
| 8. Boilerplate vs new text (`diff.py`) | Not started |
| 9. Topic tone over time and across sectors | Not started |
| 10. Q&A layer (`answer.py`) | Not started |

## Data

- **Companies:** current S&P 500 constituents and their GICS sectors, from the Wikipedia list, saved once to `data/companies.csv` (about 500 companies once share classes such as GOOG/GOOGL are merged).
- **10-K:** Item 1A (Risk Factors) and Item 7 (MD&A), FY2019 onward. There are 3,446 filings, and 99.4% have both sections over 5,000 characters. The manifest is `data/filings.csv`.
- **10-Q:** MD&A (Part I, Item 2), from 2019 onward. There are 11,101 quarterly filings, and 99.8% have an MD&A over 5,000 characters. The manifest is `data/filings_10q.csv`. The 10-Qs show events like the March 2023 bank failures months before the next 10-K, and they give four tone readings a year instead of one.
- Sections are extracted with [edgartools](https://github.com/dgunning/edgartools). When that misses a section, a heading-based fallback takes over, and it also reads Exhibits 13 and 99 for companies that put their MD&A there. The extracted text is cached in `data/cache/` (about 1.4 GB, gitignored). Raw HTML is not stored.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file in the project root. The SEC rejects requests without a User-Agent:

```
SEC_USER_AGENT="Your Name your.email@example.com"
```

## Usage

```bash
uv run python -m sec_filings_tone.fetch NVDA AMD          # 10-Ks for specific tickers
uv run python -m sec_filings_tone.fetch                   # 10-Ks for every company in data/companies.csv
uv run python -m sec_filings_tone.fetch --10q [TICKERS]   # 10-Q MD&A instead
uv run python tests/test_fetch.py                         # section-extraction test
```

Filings whose sections are already in `data/cache/` are skipped, so reruns only download what's missing. Each run updates the manifest CSV and lists the filings whose sections came out too short.

## Known limitations

- **Survivorship bias:** the sample uses today's S&P 500 list, so it leaves out companies that dropped out of the index or went bankrupt since 2019.
- **Fiscal years don't line up:** some companies end their fiscal year in January (NVDA) or September (AAPL). Charts and event windows use the filing date.
- **Extraction isn't perfect:** a small share of sections come out too short (a few quarters for WM, GE, HCA and INTC), and some big-bank MD&As run long, likely into the financial statements. These are flagged in the manifests and excluded from analysis rather than fixed by hand.
- Results are about how companies describe risk and performance. They make no claim that tone predicts returns.
