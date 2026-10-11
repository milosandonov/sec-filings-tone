# SEC Filings Tone: Plan

Track how companies' tone on specific topics (supply chain, AI, China, interest rates, tariffs, ...) changes across their 10-K filings, across every sector of the S&P 500. RAG finds the relevant passages, three scorers rate their sentiment, and a Q&A layer lets users ask their own questions about any company or sector.

**The questions we're answering:**
1. How well do FinBERT, an LLM and the Loughran-McDonald dictionary score SEC filing text, both against labels and against real events?
2. Does tone react to known shocks (export controls, the 2023 bank failures, tariffs) in the sectors and topics they hit, and which scorer picks that up?
3. How much of a 10-K is boilerplate carried over from last year, and do scorers behave differently on boilerplate than on new text?
4. Does tone move together within a sector more than across sectors?

## Scope

- **Companies:** current S&P 500 constituents (about 500 companies, 11 GICS sectors). Sector comes from the S&P 500 list on Wikipedia.
- **Pilot first:** build and check every phase on 10 semiconductor companies (NVDA, AMD, INTC, QCOM, AVGO, TXN, MU, AMAT, LRCX, KLAC), then scale to the full list. Nothing in the code should assume one sector.
- **Filings:** 10-K only, FY2019 onward (about 7 per company, about 3,500 in total). Starting at FY2019 gives a pre-COVID baseline for the event checks.
- **Sections:** Item 1A (Risk Factors) and Item 7 (MD&A).
- **Out of scope for now:** 10-Qs, earnings call transcripts (paywalled or unclear licensing), and any claim that tone predicts returns.

## Data size and storage

- A full 10-K HTML file is roughly 2–10 MB, mostly inline XBRL markup. Item 1A plus Item 7 as plain text is roughly 100–300 KB per filing.
- 3,500 filings come to about 350 MB–1 GB of extracted text. That's too much to hold in memory at once, so process one filing at a time.
- **Decision:** keep the raw HTML in memory only, and save just the extracted section text to `data/cache/{ticker}_{year}_{item}.txt`. That avoids re-downloading about 3,500 filings on every run and keeps runs reproducible. The cache is gitignored, and `ingest.py` rebuilds it from scratch.
- Embeddings: about 350k chunks. At the full 3,072 dimensions that's about 4 GB, so request 1,024 dimensions from `text-embedding-3-large` (about 1.4 GB). Store in a local Chroma database (`vector_db/`, also gitignored). Embedding is roughly 100–250M tokens in total, so check the current price before running the full set.
- The LLM scorer runs only on retrieved topic passages and the labeled sets, not on every sentence. Scoring every sentence would be about 5M LLM calls.

## Phases

### 0. Setup
- `uv init`, plus dependencies: `edgartools`, `chromadb`, `openai`, `litellm`, `transformers`, `torch`, `pandas`, `lxml`, `matplotlib`, `python-dotenv`.
- Put `OPENAI_API_KEY` and `SEC_USER_AGENT="Name email@example.com"` in `.env`. The SEC rejects requests that don't include a User-Agent.

### 1. Fetch (`fetch.py`)
- Load the ticker and GICS sector list with `pandas.read_html` from the Wikipedia S&P 500 page, and save a copy to `data/companies.csv` so the list stays fixed between runs.
- For each ticker, use `edgartools` to get its 10-Ks from FY2019 onward and extract Item 1A and Item 7 as text.
- Stay under 10 requests per second (SEC limit). Skip any filing whose sections are already in the cache.
- **Check:** every ticker/year has both sections, and each section is longer than about 5k characters. If a section is shorter, extraction failed. Log the failures and the share of filings that failed instead of stopping.

### 2. Chunk and ingest (`ingest.py`)
- Split by section first, then by paragraph, into chunks of about 500–800 tokens with a small overlap.
- Start each chunk with a context header: `"{ticker} ({sector}) 10-K FY{year}, {item name}"`.
- Store metadata on every chunk: `ticker`, `sector`, `year`, `item`, `filing_date`.
- Embed with `text-embedding-3-large` (1,024 dimensions) and save to Chroma.
- Reuse patterns from `llm_engineering-main/week5/implementation/ingest.py`, and from `pro_implementation/` for the LLM-assisted chunking.

### 3. Retrieval (`retrieve.py`)
- Filter on metadata first (ticker, sector, year, item), then run vector search inside that subset. This is the most important step for financial documents.
- Add BM25 keyword search alongside vector search (hybrid search), to catch exact terms such as tickers and product names.
- Rerank the results with the LLM reranker from `pro_implementation/answer.py`, or with a cross-encoder.

### 4. Retrieval eval (`eval/`)
- Write about 30 test questions with keywords and reference answers, in the same format as `week5/evaluation/tests.jsonl`, spread across sectors. Examples: "What supply chain risks did NVDA cite in FY2023?" and "How did JPM describe deposit outflows in FY2023 MD&A?"
- Score with MRR, nDCG and keyword coverage. `week5/evaluation/eval.py` can be reused almost unchanged.
- Measure a baseline (vector search only), then hybrid, then hybrid plus reranking. The before/after table goes into the README.

### 5. Sentiment scoring (`sentiment.py`)
Run three scorers on the same retrieved passages:
1. **FinBERT** (`ProsusAI/finbert`), sentence by sentence, averaged per passage.
2. **Loughran-McDonald dictionary:** (positive − negative words) / total words.
3. **LLM zero-shot:** a structured output with label, score from −1 to 1, and a one-line reason.

### 6. Labels and benchmark
The benchmark needs ground truth: without labels we can only measure whether scorers agree with each other, not which one is right. LLMs do most of the labeling, and a human checks a small set.
- **Silver set (LLM-labeled, about 3,000 sentences):** sample sentences evenly across sectors, years and items. Label each with two LLMs from different model families than the Phase 5 LLM scorer. Labels: positive, negative, neutral. If the scorer and the labeler were the same model, the LLM would win the benchmark automatically.
- **Gold set (human-verified, about 150 sentences):** every sentence where the two labeling LLMs disagree, plus a random sample of ones where they agree. A human accepts or corrects each label, which is a few hours of work instead of labeling from scratch. The gold set is the final test set.
- Write the labeling rules in `labels/guidelines.md` before starting, and use the same rules in the LLM labeling prompt.
- Report accuracy, macro-F1 and a confusion matrix for each scorer on the gold set, and how often the human agreed with the LLM labels. Use the silver set to show results by sector and item.
- Expected finding: FinBERT rates neutral risk-factor boilerplate as negative. Check whether that's true.
- Ideally a second person checks 100 gold sentences, so we can report agreement between people (Cohen's kappa). Reviewers will ask for it.

### 7. Event validation (`events.csv`)
A test that doesn't depend on our labels: tone should move where a known event hit, and stay flat where it didn't.
- List events with the date, the sectors affected, the topic, and the expected direction:
  - COVID-19 (2020): all sectors; topics: demand, supply chain.
  - US chip export controls on China (October 2022): Information Technology / semiconductors; topic: China.
  - Rate hikes (2022–23): Financials, Real Estate; topic: interest rates.
  - Regional bank failures, SVB (March 2023): Financials; topics: deposits, liquidity.
  - AI boom (2023 onward): Information Technology, Communication Services; topic: AI (expected positive in MD&A).
  - Tariffs (2025): Industrials, Consumer Discretionary, IT; topic: tariffs.
- For each event, compare the change in topic tone in the affected sectors with the change in unaffected sectors over the same years (difference-in-differences).
- For each scorer, report whether it detects each event, in the expected direction, and how strongly. A scorer that fits the labels but misses the events is the more interesting result.

### 8. Boilerplate vs new text (`diff.py`)
- For each ticker, match each year's Item 1A and Item 7 sentences against the previous year's. A sentence counts as carried over if it closely matches a sentence from last year (`difflib.SequenceMatcher` ratio above about 0.9). Everything else counts as new.
- Report how much of each filing is carried over, by sector and year.
- Score new and carried-over text separately. Check whether scorers disagree more on boilerplate, and whether tone in new text reacts more strongly to the Phase 7 events.
- Add a "what changed" query: summarize what was added or removed in one ticker's risk factors compared with the year before.

### 9. Topic tone over time and across sectors
- For each (topic, ticker, year): retrieve the top-k passages and score them with the best scorer from Phases 6–7.
- Output heatmaps of tone by ticker and year, and by sector and year, for each topic, plus line charts for 2–3 topics.
- Test whether tone moves together more within a sector than across sectors: compare the average year-over-year tone correlation for pairs of companies in the same sector with pairs from different sectors. Treat this as a hypothesis to test, not a result to expect. Shared law firms and copied boilerplate could also make tone move together, so run the test on new text only (Phase 8) as well.

### 10. Q&A layer (`answer.py`)
- A chat interface on top of the Phase 3 retrieval, so users can ask questions about any company or sector, and follow up.
- **Follow-ups:** keep the conversation history, and rewrite each new question into a standalone search query before retrieving (for example, "what about AMD?" becomes "AMD supply chain risks FY2024").
- **Multi-part questions:** split questions such as "compare NVDA and AMD's China risk in FY2022 vs FY2024" or "how did banks talk about deposits before and after SVB?" into one sub-query per ticker or sector and year, retrieve for each, then answer from the combined passages.
- **Sentiment:** attach the best scorer's tone to each answer, cite the passages it came from (ticker, year, item), and show how tone on the topic changed from the previous year.
- **Check:** add about 10 multi-part and follow-up questions to `eval/tests.jsonl`, and have an LLM judge grade the answers against reference answers for accuracy and completeness.

### 11. Publish
- **README:** the questions, the data, the retrieval eval table, the scorer benchmark, the event validation results, two charts, a Q&A demo, and limitations.
- **LinkedIn post:** the single most surprising chart, plus links to the repo, the dataset and the preprint.
- **Paper, in this order:**
  1. **Zenodo first.** Upload the gold and silver label sets, the labeling guidelines, and a preprint PDF. Connect the GitHub repo so each release is archived automatically. Everything gets a DOI and can be cited. Also put the dataset on the Hugging Face Hub with a dataset card covering source, labeling method, label counts and agreement scores.
  2. **Informal review.** Before submitting, have someone with publishing experience in finance or NLP read the paper and give feedback.
  3. **Submit.** If the review is positive, submit to the FinNLP workshop (held with ACL, EMNLP and similar conferences) or a similar venue. Optionally also post to SSRN or arXiv for visibility. Check each venue's preprint and anonymity rules before posting anywhere, so the Zenodo version doesn't break them.
- **Main claims of the paper:** the scorer benchmark on 10-K text (Phase 6), the event validation (Phase 7), and the boilerplate vs new text split (Phase 8). The released dataset supports all three.

## Repo layout (target)

```
sec-filings-tone/
  PLAN.md
  README.md
  pyproject.toml
  src/sec_filings_tone/      # run as: uv run python -m sec_filings_tone.<module>
    fetch.py
    ingest.py
    retrieve.py
    sentiment.py
    diff.py
    answer.py
  events.csv
  data/
    companies.csv            # S&P 500 list, fixed once fetched
    filings.csv              # one row per filing: dates, accession, section lengths, ok flag
  eval/
    tests.jsonl
    eval.py
  labels/
    guidelines.md
    silver.csv
    gold.csv
  notebooks/
    analysis.ipynb
```

## Risks

- **Section extraction breaks on some filings:** the Phase 1 length check catches it. Seen in the pilot: Intel's 10-Ks use their own layout with no "Item 7" heading, so its MD&A comes back empty while Item 1A is fine. Ingest should skip individual sections that are too short, not whole filings. With about 3,500 filings, fixing them by hand won't scale, so report the failure rate and leave those filings out.
- **Survivorship bias:** using today's S&P 500 list leaves out companies that dropped out or went bankrupt, which leans the sample toward companies that did well. Say so in the limitations.
- **Fiscal years don't line up across companies:** NVDA's fiscal year ends in January. Use the filing date for charts and event windows, and note this in the README.
- **The LLM labels may lean the same way as the LLM scorer:** use different model families for labeling and scoring, and treat only the human-verified gold set as the final test.
- **One person verifying the gold set introduces bias:** write the guidelines first, re-check a 50-sentence sample a week later, and get a second person for 100 sentences if possible.
- **Cost and runtime at full scale:** check embedding and LLM costs on the pilot, then estimate the full run before starting it.
