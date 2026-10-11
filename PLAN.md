# SEC Filings Tone: Plan

Track how a company's tone on specific topics (supply chain, AI, China, guidance, ...) changes across its 10-K filings, using RAG to find the relevant passages and comparing three ways of scoring sentiment.

**The question we're answering:** how well does FinBERT work on SEC filing text, compared with an LLM and the Loughran-McDonald dictionary? And what does topic-level tone look like over time for one sector?

## Scope

- **Companies:** about 10 from one sector. Proposed: semiconductors (NVDA, AMD, INTC, QCOM, AVGO, TXN, MU, AMAT, LRCX, KLAC).
- **Filings:** 10-K only, the last 5 fiscal years, so about 50 filings.
- **Sections:** Item 1A (Risk Factors) and Item 7 (MD&A).
- **Out of scope for now:** 10-Qs, earnings call transcripts (paywalled or unclear licensing), and any claim that tone predicts returns.

## Data size and storage

- A full 10-K HTML file is roughly 2–10 MB, mostly inline XBRL markup. Item 1A plus Item 7 as plain text is roughly 100–300 KB per filing.
- 50 filings come to about 10–15 MB of extracted text. That fits in memory easily.
- **Decision:** keep the raw HTML in memory only, and save just the extracted section text to `data/cache/{ticker}_{year}_{item}.txt`. That avoids sending about 50 SEC requests on every run and keeps runs reproducible. The cache is gitignored, and `ingest.py` rebuilds it from scratch.
- Embeddings: a few thousand chunks × 3072 dimensions × 4 bytes is about 50–100 MB, stored in a local Chroma database (`vector_db/`, also gitignored).

## Phases

### 0. Setup
- `uv init`, plus dependencies: `edgartools`, `chromadb`, `openai`, `litellm`, `transformers`, `torch`, `pandas`, `matplotlib`, `python-dotenv`.
- Put `OPENAI_API_KEY` and `SEC_USER_AGENT="Name email@example.com"` in `.env`. The SEC rejects requests that don't include a User-Agent.

### 1. Fetch (`fetch.py`)
- For each ticker, use `edgartools` to get the last 5 10-Ks and extract Item 1A and Item 7 as text.
- Stay under 10 requests per second (SEC limit). Skip any filing whose sections are already in the cache.
- **Check:** every ticker/year has both sections, and each section is longer than about 5k characters. If a section is shorter, extraction failed.

### 2. Chunk and ingest (`ingest.py`)
- Split by section first, then by paragraph, into chunks of about 500–800 tokens with a small overlap.
- Start each chunk with a context header: `"{ticker} 10-K FY{year}, {item name}"`.
- Store metadata on every chunk: `ticker`, `year`, `item`, `filing_date`.
- Embed with `text-embedding-3-large` and save to Chroma.
- Reuse patterns from `llm_engineering-main/week5/implementation/ingest.py`, and from `pro_implementation/` for the LLM-assisted chunking.

### 3. Retrieval (`retrieve.py`)
- Filter on metadata first (ticker, year, item), then run vector search inside that subset. This is the most important step for financial documents.
- Add BM25 keyword search alongside vector search (hybrid search), to catch exact terms such as tickers and product names.
- Rerank the results with the LLM reranker from `pro_implementation/answer.py`, or with a cross-encoder.

### 4. Retrieval eval (`eval/`)
- Write about 30 test questions with keywords and reference answers, in the same format as `week5/evaluation/tests.jsonl`. Examples: "What supply chain risks did NVDA cite in FY2023?" and "How did AMD describe data center demand in FY2024 MD&A?"
- Score with MRR, nDCG and keyword coverage. `week5/evaluation/eval.py` can be reused almost unchanged.
- Measure a baseline (vector search only), then hybrid, then hybrid plus reranking. The before/after table goes into the README.

### 5. Sentiment scoring (`sentiment.py`)
Run three scorers on the same retrieved passages:
1. **FinBERT** (`ProsusAI/finbert`), sentence by sentence, averaged per passage.
2. **Loughran-McDonald dictionary:** (positive − negative words) / total words.
3. **LLM zero-shot:** a structured output with label, score from −1 to 1, and a one-line reason.

### 6. Labeled set and benchmark (the actual contribution)
- Hand-label about 250 sentences from Item 1A and Item 7, sampled evenly across companies and years. Labels: positive, negative, neutral.
- Report accuracy, macro-F1 and a confusion matrix for each scorer.
- Expected finding: FinBERT rates neutral risk-factor boilerplate as negative. Check whether that's true.
- Publish the labeled set on the Hugging Face Hub with a dataset card covering source, labeling rules, and how many sentences each label has.

### 7. Topic tone over time
- For each (topic, ticker, year): retrieve the top-k passages and score them with the best scorer from Phase 6.
- Output a heatmap of tone by ticker and year for each topic, plus a line chart for 2–3 topics.
- Add a "what changed" query: compare the year N risk factors with year N−1 for one ticker and summarize what was added or removed.

### 8. Publish
- README: the question, the data, the retrieval eval table, the scorer benchmark, two charts, and limitations.
- LinkedIn post: the single most surprising chart, plus links to the repo and the HF dataset.

## Repo layout (target)

```
sec-filings-tone/
  PLAN.md
  README.md
  pyproject.toml
  fetch.py
  ingest.py
  retrieve.py
  sentiment.py
  eval/
    tests.jsonl
    eval.py
  labels/
    sentences.csv
  notebooks/
    analysis.ipynb
```

## Risks

- **Section extraction breaks on some filings:** the length check in Phase 1 catches it; fix those filings by hand.
- **Fiscal years don't line up across companies:** NVDA's fiscal year ends in January. Use the filing date for charts and note this in the README.
- **One person labeling everything introduces bias:** write the labeling rules down before starting, and re-label a 50-sentence sample a week later to measure consistency.
