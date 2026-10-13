# Phase 4: code review, data check and experiment design

Written 2026-10-12, before the paid chunking and embedding experiment. It covers what was checked,
what was wrong, what was fixed, what is still open, and how the experiment is set up so that it
runs once.

## Summary

- **Retrieval code and the vector store are correct.** Chroma holds exactly the 9,766 pilot chunks,
  with matching text and metadata, and re-embedding sampled chunks reproduces the stored vectors
  (cosine 1.0000). Chroma's filtered search matches an exact search on all 30 eval queries; the
  only difference is one float32 near-tie at rank 20.
- **The chunker had four real problems**, all fixed in `ingest.py` and covered by
  `tests/test_ingest.py`. The worst: bullet points read as headings were deleted from the chunk
  text, not only mislabeled. The fixed baseline has 8,714 chunks instead of 9,766 and 10% fewer
  tokens.
- **QCOM's 2019–24 10-Q MD&A contained its whole Risk Factors section** (72% of the text). The same
  happens for MCD and BEN in the full dataset. Ingest now cuts it.
- **The eval's keyword matching was tightened** so "15%" no longer matches "2.15%".
- **The experiment was redesigned after a dry run.** Headings, not the size setting, decide most
  chunk boundaries (median about 350 tokens at any target), so the original size variants would
  barely have changed chunk sizes. Size is now tested with heading splits turned off.
- **The experiment is ready but not run:** 11 variants, about $5.75, about 30 minutes. A smoke test
  on TXN ran every code path end to end for about $0.20.

**Important for anyone running the code now:** `data/chunks.parquet` and the Chroma collection
`filings_v1` still use the old chunking and match each other. Running `ingest` now would rewrite
`chunks.parquet` with the new chunking, and retrieval would then look up vector-search IDs that no
longer exist. Rebuild and re-embed both together, once the experiment has picked a winner.

## 1. Code review

### `retrieve.py`

| Check | Result |
|---|---|
| Metadata filter gives the same subset to both searches | Yes. One function builds both the pandas mask (BM25) and the Chroma `where` (vector). The test checks that both agree. |
| Date-range filter | It filters on `filing_date`, inclusive. Chroma can't compare strings, so the range becomes an `$in` list of the filing dates inside it, which is exact. |
| Misspelled filters | They raise an error (`year=` instead of `fy=`), so a typo can't silently return unfiltered results. |
| numpy integers as filter values | Converted, since Chroma rejects them. Phase 9 will loop over DataFrame values, where this would bite. |
| BM25 | It scores only inside the filter mask (other chunks get weight 0) and drops zero-score filler. IDF is computed over the whole corpus, not the subset, so a word doesn't count less just because the filter selected text that's full of it. |
| BM25 tokenization (limitation, not a bug) | Lowercase, English stopwords, no stemming. Tokens must be 2+ characters, so "$4.5 billion" indexes only "billion", and "restricted" doesn't match "restrictions". These can be tested later for free. |
| RRF | `sum(1 / (60 + rank))` over the vector top 20 and BM25 top 20; ties keep vector order. |
| Reranker output | The LLM's ID list is validated: unknown or repeated IDs are dropped and missing ones appended, so a sloppy reply can't lose or duplicate chunks. |
| Reranker input | Each passage gets the same context header as the embedded text (ticker, form, period, item, heading). |
| Chroma vs exact search | The top 20 matched exact cosine search on all 30 queries. The only set difference was ranks 20 and 21 of one query, whose similarities differ by 0.0001 (float32 rounding). Raising `ef_search` from 100 to 500 changed nothing. It's back at 100. |
| Vector store contents | Same 9,766 IDs as `chunks.parquet`; stored text and all 12 metadata fields identical. Five random chunks re-embedded exactly as `ingest` does gave cosine 1.0000 with the stored vectors, so batching never misaligned a vector with its chunk. |

Change: `subset()` now takes the date range inside the same filter dict that tests store. No
behavior change.

### `eval/eval.py`

- **Keyword matching** is case-sensitive and whole-word. The course version's case-insensitive
  substring match made "Arm" hit "harm" and "alarm". The match was tightened further so a keyword
  can't be part of a longer number: "15%" doesn't match "2.15%", and "$1.6" doesn't match "$1.65".
- **nDCG's ideal ranking** counts every chunk in the filtered subset that contains the keyword. The
  course version built the ideal from the retrieved list only, so finding 1 of 5 relevant chunks at
  rank 1 scored a perfect 1.0.
- **Caveats on the 30 questions:**
  - I wrote the questions and picked the keywords from the passages that answer them. Some
    `direct_fact` questions contain a keyword ("Mellanox"), which favors BM25; the `paraphrase`
    questions avoid that.
  - Thirty questions is small. One question moves a mean by about 0.03, so differences under about
    0.05 shouldn't be trusted without the bootstrap interval (see the experiment design below).
  - Reranker results vary between runs, since Haiku 5.5 rejects sampling settings such as
    `temperature`.
  - All questions are about the 10 semiconductor companies; the cross-sector questions in the plan
    wait for the full embed.

## 2. Data check (pilot: 10 companies, `chunks.parquet`)

| Check | Result |
|---|---|
| Unique `chunk_id`s, no nulls | OK: 9,766 unique, no nulls in any column. |
| `seq` contiguous within each section | OK. |
| `fy` equals the year of `report_date` | OK for every chunk. |
| `filing_date` after `report_date` | OK for every chunk. |
| One 10-K per ticker per fiscal year | OK. |
| 10-Qs per ticker per year | 3, except INTC 2019 (0: MD&A extraction failed, already in `ingest_skipped.csv`), TXN 2019 (2: Q1 failed) and 2026 (year in progress). |
| Section sizes per ticker | Consistent, except **QCOM 10-Q MD&A: median 26k tokens**, 4× its own 10-K MD&A and other companies' 10-Q MD&A. See issue 4. |
| Tiny chunks | **72 chunks under 50 tokens**, all the last chunk of a section. See issue 3. |
| Headings | **1,017 chunks** had a list item as their heading and **1,024** had a page footer or table row. See issues 1 and 2. |
| Duplicate text within a section | 2 chunks (one AMAT 10-Q paragraph that the filing itself repeats). Left as is. |
| Same text across filings | 991 chunks, as expected: 10-Qs repeat paragraphs from quarter to quarter. Phase 8 measures this on purpose. |

## 3. Problems found and fixed

All four are fixed in `ingest.py` and covered by `tests/test_ingest.py`. The fixed pilot chunks are
in `data/experiment/chunks_500.parquet`; the live `chunks.parquet` is unchanged (see the Summary).

| | Before | After |
|---|---|---|
| Chunks | 9,766 | 8,714 |
| Tokens | 3.53M | 3.19M |
| Distinct headings | 1,713 | 914 |
| Chunks under 50 tokens | 72 | 0 |
| QCOM 10-Q chunks (tokens) | 1,087 (490k) | 426 (139k) |

**1. List items treated as headings were deleted.** A paragraph under 150 characters with no
closing punctuation counted as a heading, and `chunk_section` keeps headings out of the chunk text.
Bullet items such as "•We may have to write off goodwill or other intangible assets; and" end in
"and" or "or", so they were removed from the text and stored as the next chunk's heading. That's
0.33% of the text (814 paragraphs), but they are real risk statements, and the sentiment scorers
would never have seen them. PLAN.md said only the heading field was affected; that was wrong. Fix:
a heading can't start with a bullet character or a lowercase letter, and can't end with a function
word ("and", "or", "of", "the", ...). The last two rules also catch sentence fragments split by a
page break ("in the", "...reduce existing inventories of"), which were dropped the same way.

**2. Page footers and table rows were treated as headings.** For example "Lam Research Corporation
2019 10-K 14", Intel's "Other Key Information 54", and table captions such as "Three Months Ended
March 27, ... $7,173". Each one replaced the real heading in the chunk's metadata and embedded
header, and could cut a chunk in the middle of a passage. Fix: a heading that ends in a 1–3 digit
number, contains "$", or has 3 or more numbers is skipped entirely. Real headings have at most two
numbers ("Fiscal Year 2020 Compared to Fiscal Year 2019", "7.50% Senior Notes Due 2022").

**3. A section's last few lines became their own chunk.** Examples: "None." under AMAT's "Item 1B:
Unresolved Staff Comments", or MU's "No material items." A last chunk under 50 tokens now joins the
one before it, if the result still fits the maximum size.

**4. Risk Factors inside 10-Q MD&A.** QCOM (all 17 quarters from 2019 to mid-2024), MCD (14) and
BEN (8) put a full Risk Factors section inside Item 2. edgartools returns it as part of the MD&A,
and fetch.py's fallback, which already stops at "risk factors", never runs because the edgartools
text is long enough. This affected 39 of 11,101 10-Q MD&A files. CAT, HBAN and JBHT also have a
"Risk Factors" heading there, but it's a one-line pointer to the 10-K's Item 1A, and the MD&A
continues after it; for CAT, cutting there would delete 85–91% of the MD&A. Fix: ingest cuts a 10-Q
MD&A at a standalone "Risk Factors" heading, unless the next 400 characters mention "Item 1A". That
cuts exactly the 39 files and none of the others. HBAN 2019 writes "Item 1A" with a non-breaking
space, which the pattern handles.

## 4. Known problems not fixed

- **The cut happens in ingest, not fetch.** The cached files and the manifest's `len_mdna` still
  include those risk factors. This doesn't affect any results, because ingest always cuts, but the
  Phase 1 lengths for those 39 filings overstate the MD&A.
- **Headings are found by heuristics.** A few table captions with two or fewer numbers, and the odd
  fragment ("Additionally, w"), are still headings. It's rare enough to leave.
- **AMAT's 10-K Item 1A runs into Item 1B** ("Unresolved Staff Comments: None."). It's now merged
  into the last chunk instead of being its own chunk. Harmless.
- **HBAN 2020–21 10-Qs** keep a short risk-factor block at the end of the MD&A ("Item 1A Risk
  Factors below"): about 7% of those files, outside the pilot.
- **Extraction gaps already in PLAN.md** (Deere's 10-K MD&A, WM and GE 10-Qs, BNY's Item 1A) are
  outside the pilot and unchanged. INTC's 2019 10-Q MD&A (3 quarters) and TXN's Q1 2019 are missing.
- **The current eval numbers** (vector MRR 0.673, rerank 0.890) were measured on the old chunks. The
  experiment's `base` variant measures the same setup on the fixed chunks.

## 5. Experiment design

### Variants

Each variant changes one thing. Everything not listed matches the baseline: 500-token target (800
maximum), heading splits on, the previous chunk's last sentence (up to 60 tokens) as overlap, the
full context header, and `text-embedding-3-large`.

| Variant | Change | Question it answers |
|---|---|---|
| `base` | none (fixed chunker) | Reference point |
| `flat_500` | heading splits off | Do heading boundaries help? |
| `flat_300`, `flat_800`, `flat_1200` | heading splits off, target 300 / 800 / 1,200 tokens | Which size is best? Compared with `flat_500`. |
| `overlap_none` | no overlap | Does overlap help? |
| `overlap_2` | last 2 sentences, up to 120 tokens | Does more overlap help? |
| `header_none` | no context header | Does the header help? |
| `header_no_heading` | header without the section heading | Do headings help in the header, or add noise? |
| `oai_small` | `text-embedding-3-small` | How much does the 6.5× cheaper model lose? |
| `gemini2` | `gemini-embedding-2` | Another provider |

Every embedding size is scored from the same saved vectors. Both OpenAI models and
`gemini-embedding-2` are trained so a truncated vector works as a smaller embedding. The smoke test
confirmed that a truncated 3,072-dim OpenAI vector equals asking the API for 1,024 dims (cosine
1.0000). Sizes scored: 256, 1,024 and 3,072 for the large model; 512, 1,024 and 1,536 for the small
model; 768, 1,024 and 3,072 for Gemini. Variants are compared at 1,024, the size Chroma stores.

**Why size is tested with heading splits off.** A heading starts a new chunk once the current one
has 100 tokens (`HEADING_MIN_TOKENS`). MD&A sections have a subheading every few paragraphs, so
headings set most boundaries. With heading splits on, raising the target from 300 to 1,200 only
moves the median chunk from 300 to 450 tokens, even with the heading threshold scaled to the
target. With splits off, chunks reach the target: at 500, 95% of chunks have at least 450 tokens.

| Chunking | Chunks | Median tokens | 90th percentile |
|---|---|---|---|
| target 500, heading splits on (`base`) | 8,714 | 348 | 603 |
| target 300, heading splits off | 9,304 | 337 | 424 |
| target 500, heading splits off | 5,791 | 543 | 657 |
| target 800, heading splits off | 3,821 | 844 | 964 |
| target 1,200, heading splits off | 2,681 | 1,241 | 1,359 |

### Not included, and why

- **`gemini-embedding-001`** (named in PLAN.md) is deprecated, with `gemini-embedding-2` as its
  successor, so it's a poor pick for a dataset meant to last years.
- **Voyage-4-large:** no API key.
- **A full grid** (all combinations) would be 30+ variants. One factor at a time can miss
  interactions, such as small chunks only helping without the header. If two winners need to be
  combined, the final step embeds that one combination (about $0.50).
- **Free experiments**, which can run any time from the saved vectors at no embedding cost: BM25
  stemming, a context header in the query, RRF settings, and more reranker candidates.

### Saved so nothing paid is repeated

Everything goes in `data/experiment/` (gitignored, about 1 GB for the pilot):

- **Chunk tables**, one per chunking variant.
- **Document vectors at full size**, one file per API batch, named by a hash of the batch's text. A
  crash resumes where it stopped, and changed text gets new files instead of reusing stale ones.
  The smoke test reran two variants from saved vectors in 7 seconds with no API calls.
- **Query vectors**, cached per model.
- **Rankings:** the top 50 for vector and BM25, every fused candidate, and the reranked top 20
  twice, for every question, variant and size. Any metric can be recomputed from disk with
  `--score-only`.
- **Per-question metrics** in `metrics.csv`.

### Metrics

- **MRR, nDCG@10 and keyword coverage@10**, as in the Phase 4 eval.
- **Recall@20:** how many keywords are in the top 20, the most the reranker can recover.
- **Keyword coverage within 2,000 and 5,000 tokens of ranked text.** "Top 10 chunks" favors big
  chunks: ten 1,200-token chunks hold four times the text of ten 300-token chunks. These two metrics
  read ranked chunks until the token budget runs out, which is fair across sizes.
- **Every stage** (vector, BM25, hybrid, rerank), since a variant that helps vector search might not
  help the final rerank.
- **MRR difference from the reference variant**, with a 95% paired bootstrap interval over the 30
  questions. If the interval includes 0, the difference isn't established.
- **Reranker spread:** each variant is reranked twice, and the report gives the mean per-question
  MRR difference between the two runs.
- **Breakdown by question category** (direct_fact, paraphrase, broad, cross_company).

### Cost and time caps

Every paid call is checked first against the run's budget (`--budget`, default $8 of new spend) and
time limit (`--max-minutes`, default 90). If the next call could pass the budget, or time is up,
the run stops with an error. Everything finished so far stays saved, so rerunning continues where
it stopped. Reranker spend comes from the API's token counts; embedding spend is counted from
tokens (exact for OpenAI). Each API request also has a 60-second timeout, so a stuck call fails
instead of hanging. `eval/eval.py` uses the same cap (defaults $0.50 and 15 minutes), and
`tests/test_retrieve.py` checks both limits.

### Checks done before the paid run

- Dry run: chunk tables, token counts and cost, plus a check that every test keyword appears in its
  filtered subset in every variant (none are lost at a chunk boundary).
- Smoke test on TXN, the smallest pilot company: all 11 variants embedded, ranked and scored.
- It caught one bug: the Gemini client was garbage-collected mid-request. Fixed by keeping one
  client for the run.
- `gemini-embedding-2` would silently average a plain list of strings into one embedding. The code
  sends each text as its own `Content`, and the smoke test checks that a batched vector equals a
  single-call vector (cosine 1.0000) and that two different texts get different vectors (0.70).
- `gemini-embedding-2` takes no `task_type`; queries are prefixed `task: search result | query:` and
  documents `title: none | text:`, as Google's docs specify.

### Cost and time

| Item | Cost |
|---|---|
| 9 `text-embedding-3-large` variants, 3.4M–4.1M tokens each, $0.13 per 1M | $4.26 |
| `text-embedding-3-small`, $0.02 per 1M | $0.08 |
| `gemini-embedding-2`, $0.20 per 1M | $0.75 |
| Reranking: 30 questions × 2 runs × 11 variants, Haiku 5.5 | about $0.66 |
| **Total** | **about $5.75** |

Prices were checked on 2026-10-12 on the OpenAI and Google pricing pages. Token counts are exact
for OpenAI (same tokenizer) and approximate for Gemini. Expected time is about 30 minutes, an
estimate scaled up from the TXN smoke test: about 20 minutes of embedding requests, and about 7 of
reranking at 8 calls at a time.

### How the winner is picked

1. Compare vector search at 1,024 dims (the stage the experiment changes) against the reference,
   using the bootstrap interval. A change counts only if its interval excludes 0.
2. Check that hybrid and rerank MRR don't drop. Those stages are what users get.
3. For chunk size, also weigh coverage within a token budget, and the reranker's cost (bigger
   chunks send more tokens per call).
4. If two changes win separately, embed the combination once and confirm it.
5. The winner becomes `filings_v2`: rebuild `chunks.parquet` and re-embed the pilot into Chroma.
   Then estimate the full 498-company cost from the winner's tokens per chunk.

## How to reproduce

```bash
uv run python tests/test_ingest.py                        # chunker fixes
uv run python tests/test_retrieve.py                      # filters, BM25, RRF, rerank order
uv run python eval/experiment.py --dry-run                # chunks, tokens, cost; free
uv run python eval/experiment.py --smoke                  # every variant on TXN; about $0.20
uv run python eval/experiment.py                          # the experiment; about $5.75, capped at $8 and 90 minutes
uv run python eval/experiment.py --score-only             # recompute metrics from saved rankings
```
