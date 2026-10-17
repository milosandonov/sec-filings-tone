# SEC Filings Tone

This project tracks how companies' tone on specific topics (supply chain, AI, China, interest rates, tariffs, and others) changes across their SEC filings, in every sector of the S&P 500. Retrieval (RAG) finds the passages about each topic, an LLM scores a tone profile toward each topic (polarity, magnitude, realization, uncertainty, specificity) with FinBERT and the Loughran-McDonald dictionary as baselines, and a Q&A layer lets users ask their own questions about any company or sector.

## Research questions

1. How well do FinBERT, an LLM and the Loughran-McDonald dictionary score SEC filing text, both against labels and against real events? Does a tone profile with several dimensions measure anything that polarity alone misses?
2. Does tone react to known shocks (export controls, the 2023 bank failures, tariffs) in the sectors and topics they hit, and which scorer and which dimension picks that up?
3. How much of a 10-K is boilerplate carried over from last year, and do scorers behave differently on boilerplate than on new text?
4. Does tone move together within a sector more than across sectors?

## Status

Work in progress. The filing download is done, and chunking, embedding and retrieval run on a 10-company pilot. The other stages are planned but not built yet. See [PLAN.md](PLAN.md) for the full design.

| Stage | Status |
|---|---|
| 1. Fetch and parse filings (`fetch.py`) | Done |
| 2. Chunk and embed (`ingest.py`) | Done for the pilot |
| 3. Retrieval: metadata filter, hybrid search, reranking (`retrieve.py`) | Done for the pilot |
| 4. Retrieval eval and chunking experiment (`eval/`) | Done for the pilot |
| 5. Tone profile (LLM), with FinBERT and Loughran-McDonald baselines (`sentiment.py`) | Not started |
| 6. Labeled benchmark (silver and gold sets) | Not started |
| 7. Event validation (`events.csv`) | Not started |
| 8. Boilerplate vs new text (`diff.py`) | Not started |
| 9. Topic tone over time and across sectors | Not started |
| 10. Q&A layer (`answer.py`) | Not started |

## Retrieval eval

30 test questions on the 10 pilot semiconductor companies (`eval/tests.jsonl`), each with metadata filters, keywords taken from the passages that answer it, and a reference answer. Scores use the top 10 chunks per stage; a chunk counts as relevant to a keyword if it contains the keyword as a whole word.

| Stage | MRR | nDCG@10 | Keyword coverage |
|---|---|---|---|
| Vector search only | 0.866 | 0.802 | 98.9% |
| BM25 only | 0.755 | 0.755 | 95.0% |
| Hybrid (RRF) | 0.845 | 0.793 | 98.9% |
| Hybrid + Claude rerank | 0.903 | 0.879 | 98.9% |

These scores come after the chunking and embedding experiment (`eval/experiment.py`, results in `eval/FINDINGS.md`). It tested chunk size, heading boundaries, overlap, a context header, two OpenAI embedding models and embedding sizes. The biggest finding: the context header and overlap that were prepended to each chunk before embedding made vector search worse, and removing them raised its MRR from 0.673 to 0.866. Chunks are now embedded as plain text. The LLM reranker gives the best result at every stage of the experiment. Cross-sector questions will be added once all sectors are embedded.

## Data

- **Companies:** current S&P 500 constituents and their GICS sectors, from the Wikipedia list, saved once to `data/companies.csv` (about 500 companies once share classes such as GOOG/GOOGL are merged).
- **10-K:** Item 1A (Risk Factors) and Item 7 (MD&A), FY2019 onward. There are 3,446 filings, and 99.4% have both sections over 5,000 characters. The manifest is `data/filings.csv`.
- **10-Q:** MD&A (Part I, Item 2), from 2019 onward. There are 11,101 quarterly filings, and 99.8% have an MD&A over 5,000 characters. The manifest is `data/filings_10q.csv`. The 10-Qs show events like the March 2023 bank failures months before the next 10-K, and they give four tone readings a year instead of one.
- Sections are extracted with [edgartools](https://github.com/dgunning/edgartools). When that misses a section, a heading-based fallback takes over, and it also reads Exhibits 13 and 99 for companies that put their MD&A there. The extracted text is cached in `data/cache/` (about 1.4 GB, gitignored). Raw HTML is not stored.

## Setup

Requires Python 3.13 or newer (`requires-python = ">=3.13"` in `pyproject.toml`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file in the project root. The SEC rejects requests without a User-Agent:

```
SEC_USER_AGENT="Your Name your.email@example.com"
OPENAI_API_KEY=...       # embeddings (ingest --embed, retrieve)
ANTHROPIC_API_KEY=...    # reranking (retrieve)
```

## Usage

```bash
uv run python -m sec_filings_tone.fetch NVDA AMD          # 10-Ks for specific tickers
uv run python -m sec_filings_tone.fetch                   # 10-Ks for every company in data/companies.csv
uv run python -m sec_filings_tone.fetch --10q [TICKERS]   # 10-Q MD&A instead
uv run python tests/test_fetch.py                         # section-extraction test
uv run python -m sec_filings_tone.retrieve "China export restrictions" --ticker NVDA --form 10-K --fy 2023 --item 1A
uv run python tests/test_retrieve.py                      # retrieval test (offline)
uv run python eval/eval.py                                # retrieval eval: all 30 questions, all stages
uv run python eval/eval.py 3                              # one question, with first-hit ranks per stage
```

Filings whose sections are already in `data/cache/` are skipped, so reruns only download what's missing. Each run updates the manifest CSV and lists the filings whose sections came out too short.

## Known limitations

- **Survivorship bias:** the sample uses today's S&P 500 list, so it leaves out companies that dropped out of the index or went bankrupt since 2019.
- **Fiscal years don't line up:** some companies end their fiscal year in January (NVDA) or September (AAPL). Charts are by fiscal year, labeled with the calendar year it ends in, so NVDA's year ending January 2024 is FY2024 though it mostly covers 2023. Event windows pick each company's 10-Ks by filing date and period end.
- **Extraction isn't perfect:** a small share of sections come out too short (a few quarters for WM, GE, HCA and INTC), and some big-bank MD&As run long, likely into the financial statements. These are flagged in the manifests and excluded from analysis rather than fixed by hand.
- Results are about how companies describe risk and performance. They make no claim that tone predicts returns.

## License

Code (`src/`, `tests/` and the Python scripts in `eval/`): [MIT](LICENSE). Data, labels, figures and documentation: [CC BY 4.0](LICENSE-DATA). Text quoted from SEC filings stays the filers' own; `LICENSE-DATA` lists what each license covers.
