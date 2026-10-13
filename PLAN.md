# SEC Filings Tone: Plan

Track how companies' tone on specific topics (supply chain, AI, China, interest rates, tariffs, ...) changes across their 10-K filings, across every sector of the S&P 500. RAG finds the relevant passages, an LLM scores a tone profile toward each topic (polarity, magnitude, realization, uncertainty, specificity), FinBERT and the Loughran-McDonald dictionary serve as baselines, and a Q&A layer lets users ask their own questions about any company or sector.

**The questions we're answering:**
1. How well do FinBERT, an LLM and the Loughran-McDonald dictionary score SEC filing text, both against labels and against real events? Does a tone profile with several dimensions measure anything that polarity alone misses?
2. Does tone react to known shocks (export controls, the 2023 bank failures, tariffs) in the sectors and topics they hit, and which scorer and which dimension picks that up? Hypothesis: risk language is negative by design, so after a shock it changes more from hypothetical and generic to realized and specific than it turns more negative.
3. How much of a 10-K is boilerplate carried over from last year, and do scorers behave differently on boilerplate than on new text?
4. Does tone move together within a sector more than across sectors?

## Scope

- **Companies:** current S&P 500 constituents (about 500 companies, 11 GICS sectors). Sector comes from the S&P 500 list on Wikipedia.
- **Pilot first:** build and check every phase on 10 semiconductor companies (NVDA, AMD, INTC, QCOM, AVGO, TXN, MU, AMAT, LRCX, KLAC), then scale to the full list. Nothing in the code should assume one sector.
- **Filings:** FY2019 onward. Starting at FY2019 gives a pre-COVID baseline for the event checks.
  - **10-K:** Item 1A (Risk Factors) and Item 7 (MD&A). About 7 per company, about 3,450 in total.
  - **10-Q:** MD&A only (Part I, Item 2). About 23 per company, about 11,500 in total. A 10-K comes out only once a year, so an event like SVB (March 2023) wouldn't appear in most banks' 10-Ks until early 2024. The 10-Qs show it by May 2023 and give four tone readings a year instead of one.
- **Out of scope for now:** 10-Q risk factors (Part II, Item 1A, usually just "no material changes"), 8-K earnings press releases (a possible next source), earnings call transcripts (paywalled or unclear licensing), and any claim that tone predicts returns.

## Data size and storage

- A full 10-K HTML file is roughly 2–10 MB, mostly inline XBRL markup. Item 1A plus Item 7 as plain text is roughly 100–300 KB per filing.
- 3,500 filings come to about 350 MB–1 GB of extracted text. That's too much to hold in memory at once, so process one filing at a time.
- **Decision:** keep the raw HTML in memory only, and save just the extracted section text to `data/cache/{ticker}_{year}_{item}.txt`. That avoids re-downloading about 3,500 filings on every run and keeps runs reproducible. The cache is gitignored, and `ingest.py` rebuilds it from scratch.
- Embeddings: the pilot (10 semiconductor companies) gives 8,714 chunks with the fixed chunker, median 348 tokens; one company from each other sector adds 14,559 more (banks' MD&As are long). Scaling those 20: about 580k chunks and 200M tokens for all 498 companies, about $26 with `text-embedding-3-large` at $0.13 per 1M tokens (checked 2026-10-12). At 1,024 dimensions that's about 2.4 GB of vectors, about 5 GB in Chroma (`vector_db/`, gitignored). 256 dimensions scored the same on both checks and would be 4× smaller. Run with `ingest --embed --budget 30`.
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

### 2. Chunk and ingest (`ingest.py`): done for the pilot
- Cleans each cached section: drops figure rows, "Table of Contents" lines and Markdown left by the fetch fallbacks. Splits on paragraphs and packs them into chunks of up to 800 tokens, aiming for 500. A heading starts a new chunk once the current one has 100 tokens. Headings are stored in metadata, not in the chunk text.
- `data/chunks.parquet` (gitignored, rebuilt in seconds) is the source of truth, with one row per chunk. Fields: `chunk_id`, `ticker`, `sector`, `form`, `item` (`1A`, `7`, `mdna`), `fy` (int, calendar year of the period end), `report_date`, `filing_date`, `accession`, `period_quarter`, `seq`, `heading`, `n_tokens`, `text`.
- Embedded with `text-embedding-3-large` at 1,024 dimensions into Chroma `vector_db/`, collection `filings_v2`, cosine distance. The embedded string is the chunk's plain text: Phase 4 found that the context header and the overlap used in `filings_v1` made vector search worse. Chroma stores the same `text`. `embed_chunks` stops before the first request if the estimate is over `--budget`.
- Pilot: 10 semiconductor companies, 8,714 chunks, median 348 tokens. Re-running `--embed` adds nothing. `filings_v1` (9,766 chunks, old chunker, header and overlap) is kept for reference.
- Chunker fixes from the Phase 4 review (2026-10-12). Bullet points that end without punctuation ("•the demand for cryptocurrency mining; or") were read as headings, which also **deleted their text from the chunk** (0.33% of the text, 814 paragraphs), not just the `heading` field. Page footers and table rows used as headings, tiny last chunks, and the Risk Factors that QCOM, MCD and BEN put inside their 10-Q MD&A are fixed too. Details in `eval/FINDINGS.md`.

### 3. Retrieval (`retrieve.py`): done for the pilot
**What BM25 is:** keyword search with smart scoring. It doesn't understand meaning; it scores each chunk on how well its words match the query:
- **Rare words count more** (IDF). Matching "H20" or "SOFR" counts for a lot; matching "risk" counts for little.
- **Repetition helps, with diminishing returns.**
- **Length is accounted for,** so long chunks don't win just by containing more words.

Vector search finds passages with the same meaning in different words. BM25 is precise on exact terms that embeddings handle poorly: product names (H100, H20), tickers, laws and rules (CHIPS Act, Basel III), and acronyms (SOFR, CECL). Running both is "hybrid search".

**Build:**
- **Metadata filter first, for both searches** (`ticker`, `sector`, `form`, `item`, `fy`, `filing_date` range). This is the most important step for financial documents.
  - Vector: Chroma `where` on collection `filings_v2`, with the query embedded the same way as the chunks (`text-embedding-3-large`, 1,024 dimensions).
  - BM25: filter the `chunks.parquet` rows with pandas, then score inside that subset.
- **BM25 with `bm25s`** (new dependency), not `rank_bm25`. Both are fine on the pilot, but `rank_bm25` is too slow at full scale (about 800k chunks). Index `text`, lowercase, and use English stopwords.
- **Combine with Reciprocal Rank Fusion (RRF):** each search returns its top 20 or so. A chunk scores `sum(1 / (60 + rank))` across both lists. RRF uses ranks only, so BM25 scores and cosine distances don't need to be on the same scale.
- **Rerank with an LLM**, using the `rerank()` pattern from the course's `answer.py`: give it the question plus the fused top 20, get back a reordered list (`RankOrder`, structured output), and keep the top 10. Use Claude (`claude-haiku-5-5` for cost). The course file is at `/Users/milos/Desktop/My Folder/Projects/AI Engineer Udemy Course/llm_engineering-main/week5/pro_implementation/answer.py`.
  - The `.env` key is `ANTHROPIC_API_KEY` (renamed from `CLAUDE_API_KEY`).
- **One function returns each stage**, so Phase 4 can score each one separately: vector only, BM25 only, hybrid, and hybrid plus rerank. Results are rows of `chunks.parquet` (`chunk_id` plus metadata plus `text`).
- **Not in Phase 3:** query rewriting and merging two searches (also in `answer.py`). That belongs to Phase 10, the Q&A layer with follow-up questions.
- **Check:** "China export restrictions" filtered to NVDA, 10-K, FY2023, Item 1A should return the export-control risk factors at the top, for both vector and hybrid. Then try an exact-term query like "H20" and confirm BM25 finds chunks that vector search misses.
  - Result: vector, BM25, hybrid and rerank all rank the export-control chunks (`NVDA_10-K_2023_1A_0023`, `_0024`) first or second. For "H20" across NVDA, all of BM25's top 20 contain the term, against 10 of vector's top 20, and 11 BM25 hits aren't in vector's top 20 (mostly 2025–26 10-Q MD&A).
  - `uv run python tests/test_retrieve.py` checks the filters, BM25 masking, RRF and the rerank-order guard offline.

### 4. Retrieval eval (`eval/`): done for the pilot
- Write about 30 test questions with keywords and reference answers, in the same format as `week5/evaluation/tests.jsonl`, spread across sectors. Examples: "What supply chain risks did NVDA cite in FY2023?" and "How did JPM describe deposit outflows in FY2023 MD&A?"
- Score with MRR, nDCG and keyword coverage. `week5/evaluation/eval.py` can be reused almost unchanged.
- Measure a baseline (vector search only), then hybrid, then hybrid plus reranking. The before/after table goes into the README.
- **Done for the pilot:** `eval/tests.jsonl` has 30 questions on the 10 semiconductor companies (cross-sector questions wait for the full embed). Each row also has `filters` for `search()`. Categories: `direct_fact` (12), `paraphrase` (10, worded without the keywords), `broad` (5, ticker-only filter) and `cross_company` (3, including one filing-date range). Keywords were copied from the passages that answer each question.
  - `eval/eval.py` changes two things from the course version: keywords match case-sensitively as whole words (substrings made "Arm" match "harm"), and nDCG's ideal counts every chunk in the filtered subset that contains the keyword.
  - Results (top 10; the full eval takes about 70 seconds and costs about $0.05), before and after the Phase 4 experiment:

    | Stage | MRR, `filings_v1` | MRR, `filings_v2` | nDCG@10, `filings_v2` |
    |---|---|---|---|
    | Vector search only | 0.673 | 0.866 | 0.802 |
    | BM25 only | 0.761 | 0.755 | 0.755 |
    | Hybrid (RRF) | 0.773 | 0.845 | 0.793 |
    | Hybrid + Claude rerank | 0.890 | 0.903 | 0.879 |

  - On `filings_v1`, BM25 won on exact terms and vector search on paraphrased questions. On `filings_v2`, hybrid is level with vector search (−0.02, within noise; BM25 MRR on paraphrased questions is 0.54), but its top 20 holds slightly more of the relevant chunks, which is what the reranker sees. Reranking gives the best result either way, and its two runs differ by only 0.005 MRR.
- **Experiment: done** (`eval/experiment.py`; design, pre-run review and full results in `eval/FINDINGS.md`). It ran 10 one-factor variants plus 4 combinations, for about $7.30 in total including the re-embed and checks.
  - **Winner, now in production as `filings_v2`:** heading-aligned chunks (500-token target, 800 maximum), embedded as plain text with no context header and no overlap, `text-embedding-3-large` at 1,024 dims.
  - **Removing the header or the overlap each helped clearly:** +0.15 and +0.11 MRR, with 14 and 15 of 30 questions better. But a multi-sector check (20 companies, 50 questions, `eval/FINDINGS.md` section 7) couldn't tell header-without-overlap from plain text; plain text stays as the simpler option, not a proven winner. The likely reason for the header, not proven: every query is filtered first, so the header is the same for every chunk in the filtered set. Our overlap was added only to the embedding, not the stored chunk, so it moved vectors toward the previous chunk's content. That's not the usual kind of overlap.
  - **Once the header and overlap were gone,** heading splits, chunk sizes of 500 to 1,200, the large vs small model, and 256 to 3,072 dims all made no measurable difference. 300-token chunks were worse. "No measurable difference" means within what 30 questions can detect (about 0.08 MRR); many questions sit at the 1.0 ceiling.
  - **Large was kept over small:** no measurable difference on the pilot, but ahead in every comparison on the multi-sector check (hybrid +0.057 [−0.002, +0.122]). It costs about $22 more at full scale.
  - **Gemini was skipped:** the Google key is on the free tier (1,000 embedding requests a day). `gemini-embedding-001` is deprecated anyway.
  - **Free follow-ups from saved vectors and rankings:** BM25's weight in the fusion, BM25 stemming, the company and period in the query, and more reranker candidates.

### 5. Sentiment scoring (`sentiment.py`): tone-profile scorer built and pilot-checked; baselines not yet built
- **Built (2026-10-13):** `sentiment.py` scores a (passage, topic) pair with structured output, quoting `labels/guidelines.md` word for word, with every safeguard below: evidence before each score, contrasting examples, one combined call or shuffled order or one call per dimension, rule-based cross-checks, cost cap. `sentiment.py pilot` samples about 100 passage–topic pairs from the pilot, scores them in all three modes (plus a second combined run), and reports per dimension: score spread, run-to-run agreement, combined vs separate agreement, the correlations between dimensions in each mode, and the rule-based cross-checks. With `claude-haiku-5-5` it costs about $0.50; `tests/test_sentiment.py` checks the schema, prompt and metrics offline.
- **Pilot check (2026-10-13, `claude-haiku-5-5`, 100 passage–topic pairs, 800 calls, $0.20):**
  - Consistent: 88–94% of scores identical between two runs (weighted kappa 0.89–0.97). 16% of the sampled passages were flagged off topic.
  - No halo from scoring all five in one call: mean correlation between dimensions is 0.34 combined, 0.32 shuffled and 0.34 with one call per dimension. So Phase 9 uses one combined call, 5× cheaper. Combined and separate scores agree with kappa 0.76–0.96.
  - No dimensions to merge: no pair above 0.8. The strongest are polarity–uncertainty (−0.64) and magnitude–specificity (+0.55).
  - Warning on specificity: 69% of passages get the top score, and it barely tracks the rule-based detail count (+0.08; uncertainty tracks its hedge count at +0.70). It may be too lenient. The gold set must check it, and its anchors may need sharpening.
  - Consistency isn't correctness: only the gold set shows whether the scores are right.
- **First check, the 10 practice items vs my reference key:** 84% of scores exact, 96% within one point, off-topic item caught, evidence quoted correctly; $0.005. The misses were mostly magnitude rated too high, including 2 for pure boilerplate, which is the failure the guidelines warn about. That's one rater on 10 items, so it's only a first signal; the pilot check and the gold set will measure it properly.
A single polarity score is too thin for filings: risk factors are negative by design, so polarity barely moves from year to year. The project's measure is a **tone profile** with several dimensions. The standard scorers stay as baselines for the dimensions they can measure, and as the evidence for whether the profile adds anything. Some dimensions may turn out too noisy to use; Phase 6 measures that.

**Baselines,** run on the same retrieved passages:
1. **FinBERT** (`ProsusAI/finbert`), sentence by sentence, averaged per passage. Polarity only.
2. **Loughran-McDonald dictionary:** polarity as (positive − negative words) / total words. Also the shares of its uncertainty, weak-modal, strong-modal and litigious words, as a baseline for the uncertainty dimension.
3. **LLM zero-shot polarity:** a structured output with label, score from −1 to 1, and a one-line reason. This keeps the benchmark comparable with earlier work.
4. **Optional: FinBERT-FLS** (`yiyanghkust/finbert-fls`; check it's still available), which classifies forward-looking statements as specific or non-specific. A rough baseline for realization and specificity.

**Tone profile (LLM):** one structured-output call per (passage, topic) pair. Each dimension is scored toward the topic the passage was retrieved for, not the passage as a whole, since one passage can be negative on China and positive on AI. Each point on each scale has a written definition. Small anchored scales are more reliable than a fine numeric one, which mostly adds noise.

The dimensions don't need to be statistically independent, since some correlate naturally (realized risks tend to be stated more specifically). But each must capture something the others don't. They sit at three levels, which the prompt and the analysis keep apart:

| Level | Dimension | Scale | Anchors | What it is not |
|---|---|---|---|---|
| Evaluative | Polarity toward the topic | −2 to +2 | −2: the topic clearly hurts the company; 0: neutral or mixed; +2: clearly helps | Not how big the effect is |
| Evaluative | Magnitude | 0 to 3 | 0: no effect stated; 1: minor; 2: significant for a business line or region; 3: major for the whole company | Not the direction (a large gain is high magnitude), and not whether a number is given. Words like "material" in boilerplate alone don't count |
| Epistemic | Realization | 0 to 2 | 0: a hypothetical risk; 1: under way or developing; 2: has happened, with effects reported | Not how hedged the wording is |
| Epistemic | Uncertainty | 0 to 3 | 0: the effects are stated as known; 3: the company says it can't predict or estimate the effects | Not whether the event happened: "the rules took effect in October 2022; we cannot yet estimate their impact" is realized and uncertain |
| Disclosure | Specificity | 0 to 2 | 0: boilerplate any company could write; 1: specific to this company; 2: concrete (named products, dates, amounts) | Not whether it happened: "if H20 exports are restricted, we could lose $X billion of China revenue" is hypothetical and specific |

Magnitude replaces the earlier "severity", whose top anchor ("impact quantified") overlapped with specificity. It's the dimension most likely to fail: boilerplate calls nearly everything material.

- The output also has an "off topic" flag, so a passage that retrieval got wrong doesn't count, and a one-line reason.
- Example of what polarity can't see: NVDA's China risk text goes from "export controls could restrict sales" to named chips (A100, H100, H20), dates and a $4.5 billion charge. Realization and specificity rise sharply; polarity was negative before and after.
- **Model:** from a different family than the Phase 6 labelers, picked by cost and quality on the pilot. Score a sample twice to measure how much the scores change between runs.
- **Cost caps:** a budget and a time limit per run, using the `Cap` from `eval/eval.py`, plus a 60-second timeout per call.
- **Evidence first:** for each dimension, the model quotes the phrase that justifies the score before giving it. That limits the halo effect (one overall impression leaking into every score) and makes errors easy to audit.
- **Contrasting examples in the prompt,** where the dimensions disagree: realized but uncertain, hypothetical but specific, negative but minor.
- **Pilot check before scaling (about 100 passages):** score them twice, and also once with one call per dimension and once with the dimension order shuffled. If correlations between dimensions drop when they're scored separately, the combined call is leaking, so score them separately.
- **Rule-based cross-checks:** specificity against a count of numbers, dates and names in the passage; uncertainty against the LM uncertainty word share; realization against cues such as "recorded" and "incurred".

### 6. Labels and benchmark
The benchmark needs ground truth: without labels we can only measure whether scorers agree with each other, not which one is right. LLMs do most of the labeling, and a human checks a small set.
- **Silver set (LLM-labeled, about 3,000 sentences):** sample sentences evenly across sectors, years and items. Label each with two LLMs from different model families than the Phase 5 LLM scorer. Labels: positive, negative, neutral. If the scorer and the labeler were the same model, the LLM would win the benchmark automatically.
- **Silver tone profiles (about 1,000 passage–topic pairs):** the same two labeling LLMs score the five dimensions with the Phase 5 anchors.
- **Gold polarity sentences (human-labeled, about 50):** only the sentences where the two labeling LLMs disagree, which is where a person adds most. Where they agree, their shared label stands. Together these are the polarity test set.
- **Gold tone profiles (about 40 passage–topic pairs, all five dimensions):** the 10 practice items plus about 30 drawn after the full S&P 500 embed, so they span every sector, not just the pilot's semiconductor companies. All five dimensions are needed, because the halo check compares the LLM's correlations between dimensions with the human ones, and magnitude, the dimension most at risk, has to be among them. 40 still separates a reliable dimension (kappa about 0.85) from an unreliable one (below 0.6). Priority: specificity, which the pilot check flagged as possibly too lenient.
- **How people label (`labels/HOW_TO_LABEL.md`):** in a spreadsheet, blind first, then adjudicate. Pre-filled LLM scores would pull the human labels toward the LLM's, and then the halo check would compare the LLM with itself, so the gold sheets show no LLM scores. The labeler reads each passage once and scores all five dimensions. Scoring one dimension at a time down the sheet would limit a human halo effect, but means reading every passage six times; since the pilot check found little halo in the LLM's combined calls, the time isn't worth it. Any human halo makes the halo check slightly conservative. Afterwards the labeler reviews only the items where they and the LLM disagree. Both versions are kept: the blind labels for agreement and the halo check, the adjudicated ones as the final gold set.
- **Total human time: about 1.5–2 hours** (practice 30–45 minutes, 30 more profiles about an hour, 50 sentences about 20 minutes, a 10-item re-check a week later about 15 minutes).
- **Practice first:** `labels/practice.csv` has 10 real passages chosen to separate the dimensions (realized but uncertain, hypothetical but concrete, a large positive effect, one off-topic item). Labeled blind before opening `labels/practice_key.csv`, they count as the first 10 gold items; they're flagged as hand-picked, not random. The key gives reference scores with reasons. Disagreements with it show where the rules are unclear; fix the rules before the rest of the gold set, then freeze them.
- **The gold labels test the tone scorer, not retrieval.** Retrieval is tested by the retrieval eval (Phase 4); the `off_topic` labels add a small human check of whether topic queries return on-topic passages. If the scorer does badly on gold, the fixes are in scoring, in this order: clearer guidelines and examples, then a bigger model, and only then fine-tuning an open model, which needs many hundreds of labeled examples. None of these need a re-embed.
- The labeling rules are in `labels/guidelines.md`: topics, the anchors for every dimension, what each one is not, and how to score passages that mix situations (the most advanced stage, the largest stated effect, the most concrete detail). The LLM scorer and labeler prompts quote the same file.
- Report accuracy, macro-F1 and a confusion matrix for each scorer on the gold set, and how often the human agreed with the LLM labels. Use the silver set to show results by sector and item.
- **For each tone dimension:** agreement between the two labeling LLMs and with the human (weighted kappa, since the scales are ordered), the scorer's error against gold, and how much its scores change between runs. Compare uncertainty with the LM uncertainty word share, and profile polarity with FinBERT and LM polarity. A dimension that people and LLMs can't score consistently is reported as unreliable and left out of Phases 7–9.
- **Halo check:** compare the correlation matrix between dimensions for the LLM scorer with the one for the gold labels. LLM correlations clearly above the human ones (say 0.85 vs 0.5) mean halo, not signal; fix the prompt, or score dimensions in separate calls.
- **Multitrait-multimethod matrix:** the five dimensions (traits), each measured by the LLM, by people, and by a dictionary or rule where one exists (LM uncertainty words, FinBERT polarity, the rule-based specificity count). A dimension is valid if it agrees with itself across methods more than with other dimensions within a method. Confirmatory factor analysis doesn't fit: it needs several items per dimension, and there's one.
- **Rules fixed before seeing results:** drop a dimension if weighted kappa with people is below 0.6 or run-to-run agreement is below 0.8. Merge two dimensions if they correlate above 0.8 **in the human labels**; high correlation only in the LLM's scores is halo, not a reason to merge.
- Expected finding: FinBERT rates neutral risk-factor boilerplate as negative. Check whether that's true.
- Ideally a second person labels about 20 of the same gold profiles, so we can report agreement between people (weighted Cohen's kappa). Reviewers will ask for it.

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
- **Do it for each tone dimension too,** with an expected direction per dimension. Export controls on China: realization, specificity and magnitude up, polarity down. COVID: uncertainty and realization up. SVB: magnitude and realization up for deposits and liquidity. AI boom: polarity and specificity up in MD&A. Tariffs (2025): uncertainty up first, realization later. This tests the Phase 2 hypothesis: realization and specificity should react more clearly than polarity.

### 8. Boilerplate vs new text (`diff.py`)
- For each ticker, match each year's Item 1A and Item 7 sentences against the previous year's. A sentence counts as carried over if it closely matches a sentence from last year (`difflib.SequenceMatcher` ratio above about 0.9). Everything else counts as new.
- Report how much of each filing is carried over, by sector and year.
- Score new and carried-over text separately. Check whether scorers disagree more on boilerplate, and whether tone in new text reacts more strongly to the Phase 7 events.
- A check on specificity that needs no labels: carried-over text should score lower specificity than new text. If it doesn't, the dimension isn't measuring what it claims to.
- Add a "what changed" query: summarize what was added or removed in one ticker's risk factors compared with the year before.

### 9. Topic tone over time and across sectors
- For each (topic, ticker, year): retrieve the top-k passages and score their tone profile, keeping the dimensions that Phases 6–7 found reliable. Report the dimensions separately; averaging them into one score would throw away what they add.
- Output heatmaps by ticker and year, and by sector and year, for each topic and dimension, plus line charts for 2–3 topics showing all the dimensions together.
- Test whether tone moves together more within a sector than across sectors: compare the average year-over-year tone correlation for pairs of companies in the same sector with pairs from different sectors. Treat this as a hypothesis to test, not a result to expect. Shared law firms and copied boilerplate could also make tone move together, so run the test on new text only (Phase 8) as well.

### 10. Q&A layer (`answer.py`)
- A chat interface on top of the Phase 3 retrieval, so users can ask questions about any company or sector, and follow up.
- **Follow-ups:** keep the conversation history, and rewrite each new question into a standalone search query before retrieving (for example, "what about AMD?" becomes "AMD supply chain risks FY2024").
- **Multi-part questions:** split questions such as "compare NVDA and AMD's China risk in FY2022 vs FY2024" or "how did banks talk about deposits before and after SVB?" into one sub-query per ticker or sector and year, retrieve for each, then answer from the combined passages.
- **Sentiment:** attach the tone profile to each answer, cite the passages it came from (ticker, year, item), and show how each dimension changed from the previous year.
- **Check:** add about 10 multi-part and follow-up questions to `eval/tests.jsonl`, and have an LLM judge grade the answers against reference answers for accuracy and completeness.

### 11. Publish
- **README:** the questions, the data, the retrieval eval table, the scorer benchmark, the event validation results, two charts, a Q&A demo, and limitations.
- **LinkedIn post:** the single most surprising chart, plus links to the repo, the dataset and the preprint.
- **Paper, in this order:**
  1. **Zenodo first.** Upload the gold and silver label sets, the labeling guidelines, and a preprint PDF. Connect the GitHub repo so each release is archived automatically. Everything gets a DOI and can be cited. Also put the dataset on the Hugging Face Hub with a dataset card covering source, labeling method, label counts and agreement scores.
  2. **Informal review.** Before submitting, have someone with publishing experience in finance or NLP read the paper and give feedback.
  3. **Submit.** If the review is positive, submit to the FinNLP workshop (held with ACL, EMNLP and similar conferences) or a similar venue. Optionally also post to SSRN or arXiv for visibility. Check each venue's preprint and anonymity rules before posting anywhere, so the Zenodo version doesn't break them.
- **Main claims of the paper:** the scorer benchmark on 10-K text, including whether the tone profile measures anything polarity misses (Phase 6), the event validation by dimension (Phase 7), and the boilerplate vs new text split (Phase 8). The released dataset supports all three.

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
    filings.csv              # one row per 10-K: dates, accession, section lengths, ok flag
    filings_10q.csv          # same for 10-Q MD&A
  eval/
    tests.jsonl
    eval.py
  labels/
    guidelines.md            # rules, shared by people and LLM prompts
    HOW_TO_LABEL.md          # walkthrough for human labelers
    practice.csv             # 10 practice items; practice_key.csv has reference scores
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
- **Tone dimensions may be noisy or overlap:** LLM scores on ordered scales vary between runs and models. A model scoring several dimensions in one call can make them look more alike than they are (halo). Some dimensions may also really overlap: realization, uncertainty and specificity move together in risk-factor text. Phase 6 measures reliability for each dimension, compares LLM and human correlations to separate halo from real overlap, and drops or merges dimensions by rules set in advance.
- **One person labeling the gold set introduces bias:** the guidelines are written first, the labeler re-checks 10 profiles a week later, and a second person labels about 20 if possible.
- **Cost and runtime at full scale:** check embedding and LLM costs on the pilot, then estimate the full run before starting it.

## Open items (as of 2026-10-12)

- **Status:** Phase 1 is done for 10-Ks (3,446 filings, 0.6% with a short section) and 10-Q MD&A (11,101 filings, 0.2% short), across 498 companies. Phases 2–4 are done for the pilot. The Phase 4 experiment's winner is in production as `filings_v2` (see `eval/FINDINGS.md`). Next, in order: finish the open embedding model comparison; the full 498-company embed (about $26; the big-bank sections it would have skipped are now fixed); then draw the gold sample from all sectors. The practice labeling sheet (`labels/practice.csv`) can be done any time. The full 498-company embed (about $26) can run any time.
- **The project folder was renamed** from `sec-filings-tone` to `Sec Filling Tone`. If `uv run` fails, rebuild the environment: `rm -rf .venv && uv sync`.
- **`.python-version` was deleted** in commit 949ddcb ("Pulling 10-Q data too"). uv used it to pin Python 3.13. Restore it unless the deletion was on purpose.
- **Known extraction issues**, which the Phase 4 chunking experiment should fix in `fetch.py` before the full-scale embed:
  - Deere's 10-K MD&A is missing in every year: the heading doesn't appear as text anywhere in the filing.
  - Waste Management (7 10-Qs) and GE's 2020–21 10-Qs use custom layouts and are short.
  - BNY's 10-K Item 1A runs past the end of the section. QCOM's 2019–24 10-Q MD&A (and MCD's and BEN's) include a full Risk Factors section; ingest now cuts it (39 files), but the cache and the `len_mdna` manifest column still include it.
  - Big banks' MD&As often run on into the financial statements. Ingest checks any MD&A over 400k characters, or over 3× that company's median, for a financial-statement heading at least a quarter of the way in, and cuts there. That's 72 sections: all of Fifth Third's, Capital One's and PNC's, 13 of Citi's, and JPM's 10-Ks at 94%, which keeps its real 404k-character MD&A. A section still over 500k after that, or still over 3× the median, is skipped. Only 13 sections are skipped for size (0.07%), down from 123, which had cost Phase 7's bank-failure test most of Citi, Fifth Third, Capital One, PNC and JPM. The cut applies only to long sections, because normal MD&As (Parker-Hannifin, WTW) include summary tables titled "Consolidated Statement of Income", and a cut there would delete most of the MD&A. The pilot companies are unaffected.
- **Running the fetcher:** it takes about 4.4 seconds per 10-Q, almost all of it parsing. For a full run, split the tickers across 4 parallel workers with `EDGAR_RATE_LIMIT_PER_SEC=2` each (8 requests per second in total, under the SEC's 10), then run once more without tickers to rebuild the manifest. In zsh, pass ticker lists as separate arguments; an unquoted `$VAR` is not split into words.
