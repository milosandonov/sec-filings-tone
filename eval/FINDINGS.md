# Phase 4: code review, data check, experiment design and results

Sections 1–5 were written on 2026-10-12, before the paid chunking and embedding experiment: what
was checked, what was wrong, what was fixed, what is still open, and how the experiment was set up
so it would run once. Section 6, added 2026-10-13, has the results and the setup now in production.

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
- **Result (sections 6 and 7): embed the plain chunk text.** Removing the overlap, the context
  header, or both made vector search clearly better. A wider check on 20 companies (section 7)
  couldn't tell header-without-overlap apart from plain text; plain text is kept as the simpler
  option. With both removed, heading-aligned 500-token chunks couldn't be told
  apart from other sizes or flat chunking, nor `text-embedding-3-small` from `-large` (30 questions
  only detect differences of about 0.08 MRR or more; see the verification below). Vector-search
  MRR went from 0.673 to 0.866, hybrid from 0.773 to 0.845, and the final reranked result from
  0.890 to 0.903. The reranker was already making up for most of the weaker vector search.
- **In production now:** `data/chunks.parquet` is rebuilt with the fixed chunker (8,714 pilot
  chunks), and the Chroma collection `filings_v2` holds their plain-text embeddings
  (`text-embedding-3-large`, 1,024 dims). The old `filings_v1` is kept for reference.

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

## 6. Results (run 2026-10-13)

### Single changes

Vector search at 1,024 dims, 30 questions. Δ is the MRR difference from the reference, with a 95%
paired bootstrap interval over questions; **bold** means the interval excludes 0.

| Variant | Vector MRR | nDCG@10 | Cov@2k tok | ΔMRR vs reference | Rerank MRR |
|---|---|---|---|---|---|
| `base` | 0.681 | 0.670 | 0.867 | reference | 0.895 |
| `oai_small` | 0.843 | 0.794 | 0.922 | **+0.161 [+0.064, +0.263]** vs base | 0.883 |
| `header_none` | 0.830 | 0.775 | 0.894 | **+0.149 [+0.059, +0.247]** vs base | 0.893 |
| `flat_500` | 0.804 | 0.776 | 0.889 | **+0.123 [+0.032, +0.222]** vs base | 0.895 |
| `overlap_none` | 0.792 | 0.750 | 0.894 | **+0.110 [+0.029, +0.190]** vs base | 0.901 |
| `header_no_heading` | 0.718 | 0.701 | 0.900 | +0.037 [−0.019, +0.101] vs base | 0.888 |
| `overlap_2` | 0.684 | 0.669 | 0.872 | +0.003 [−0.077, +0.091] vs base | 0.898 |
| `flat_800` | 0.827 | 0.797 | 0.861 | +0.022 [−0.079, +0.124] vs flat_500 | 0.903 |
| `flat_1200` | 0.809 | 0.798 | 0.717 | +0.005 [−0.083, +0.084] vs flat_500 | 0.928 |
| `flat_300` | 0.723 | 0.704 | 0.906 | **−0.082 [−0.160, −0.010]** vs flat_500 | 0.867 |

- **The context header hurt most,** and broadly: 14 questions better without it, 2 worse. The
  likely reason, not proven: every query is filtered by company, form and period first, so inside a
  filtered set every chunk starts with the same header, which can't help tell them apart. Dropping
  only the heading part changed little, so the company, form and period part is what hurts. One
  possible mechanism, that a fixed header favors short chunks, was checked and ruled out: the top
  hits have the same length with and without the header (median 506 vs 504 tokens).
- **Overlap hurt too** (15 questions better without it, 3 worse). This overlap isn't the usual kind.
  Usually overlap is stored in the chunk, so a fact cut by a boundary appears whole in one chunk.
  Here chunks already end at paragraph breaks, and the previous sentence is added only to the
  embedding. That moves a chunk's vector toward content it doesn't contain: NVDA's chunk 0024 was
  embedded starting with a sentence about networking-product licenses that belongs to chunk 0023.
  The metric credits only a chunk's own text, so that costs rank. It was only tested with the header
  on. Two sentences were no better than one.
- **Heading splits only hurt with the header on;** see the combinations below. Why isn't
  established.
- **Size:** 300 tokens is too small; 500, 800 and 1,200 are within noise of each other. 1,200 covers
  noticeably less text within a 2,000-token budget (0.72 vs 0.89), and its reranker calls cost
  more than 2× as much (median chunk 1,241 tokens vs 543). 500–800 is the range to use.
- **Embedding size barely matters:** 256, 1,024 and 3,072 dims score within about 0.05 of each other
  for every variant.
- **Gemini wasn't tested.** The `GOOGLE_API_KEY` is on Google's free tier, which allows 1,000
  embedding requests a day for `gemini-embedding-2` and counts each text as one. It ran out after
  900 of 8,714 chunks. We decided to skip it rather than enable billing; the OpenAI small model
  already matches the large one here.
- **The reranker is stable:** the mean per-question MRR difference between its two runs was 0.005.

### Combinations of the winners

All four use no header and no overlap.

| Variant | Heading splits | Model | Vector MRR | Hybrid MRR | Rerank MRR |
|---|---|---|---|---|---|
| `combo_head` | on | large | 0.866 | 0.845 | 0.903 |
| `combo_head_small` | on | small | 0.834 | 0.804 | 0.895 |
| `combo_flat` | off | large | 0.820 | 0.840 | 0.901 |
| `combo_flat_small` | off | small | 0.852 | 0.842 | 0.902 |

Paired differences (95% intervals):

| Comparison | Vector | Hybrid | Rerank |
|---|---|---|---|
| Heading splits on vs off (large) | +0.046 [−0.017, +0.119] | +0.005 [−0.058, +0.068] | +0.002 [−0.043, +0.047] |
| Large vs small, splits on | +0.031 [−0.049, +0.104] | +0.042 [−0.023, +0.111] | +0.008 [0.000, +0.024] |
| Large vs small, splits off | −0.032 [−0.110, +0.043] | −0.002 [−0.070, +0.068] | 0.000 [−0.032, +0.033] |
| `combo_head` vs `base` | **+0.184 [+0.084, +0.292]** | +0.039 [−0.005, +0.090] | +0.008 [−0.015, +0.035] |

- With the header and overlap gone, **heading splits no longer cost anything**, so chunks can stay
  aligned with the filing's sections. That matters for Phase 5: a chunk spanning two subsections
  would get one tone profile for two topics.
- **Large vs small:** no consistent difference on these 30 questions; large is ahead with heading
  splits and behind without them. That means the test can't tell them apart, not that they're
  equal: 19 of 30 questions score the same with both, many at the 1.0 ceiling, and published
  benchmarks put the large model only a few points ahead, well below what 30 questions can detect. Large was kept because the pilot covers one sector, the full run
  will cover eleven, and the difference at full scale is about $18 (see below). Both sets of pilot
  vectors are saved, so switching the pilot costs nothing.
- **Gains don't add up.** The same seven questions (q10, q14, q16, q17, q18, q25, q28) account for
  most of the gain from removing the header, from removing the overlap and from switching to the
  small model. The changes fix the same failures, so combining them adds little. Why the baseline
  failed those seven isn't established.

### Decision and what's in production

- **Chunking:** heading splits on, 500-token target, 800 maximum (unchanged), with the fixes from
  section 3.
- **Embedded text:** the chunk's plain text, with no header and no overlap. `ingest.embed_chunks`
  now does this; `embed_text` (header and overlap) stays for the reranker's passage format and for
  the experiment.
- **Model:** `text-embedding-3-large` at 1,024 dims, in the Chroma collection `filings_v2`.
  `retrieve.py` reads it. `filings_v1` is kept, unchanged.
- **Check:** `filings_v2` was built through the normal `ingest --embed` path, and its vectors match
  the experiment's `combo_head` vectors (cosine 1.000 on a sample). `eval/eval.py` on production
  gives the same scores as the experiment:

| Stage | `filings_v1` (start of Phase 4) | `filings_v2` |
|---|---|---|
| Vector | 0.673 | 0.866 |
| BM25 | 0.761 | 0.755 |
| Hybrid | 0.773 | 0.845 |
| Hybrid + rerank | 0.890 | 0.903 |

BM25 drops slightly (0.761 → 0.755) because the fixed chunker changed the chunks.

**Hybrid is now level with vector alone:** −0.020 [−0.058, +0.006], within noise. With the old,
weaker vectors it was clearly better (+0.125 [+0.029, +0.226]); fusion adds most when the two
searches are comparably good, and BM25 is weak on paraphrased questions (MRR 0.54). Hybrid still
does its main job slightly better: its top 20, which is what the reranker sees, holds 91.3% of the
relevant chunks, against 90.0% for vector alone and 89.9% for BM25.

### Verification (2026-10-13)

The results were checked again, because several looked surprising:

- **MRR:** recomputed for 4 variants and 3 stages with code written separately from `eval.py`; the
  numbers match exactly.
- **RRF:** recomputed by hand for one question; identical to the saved hybrid ranking. A chunk at
  rank 1 in both lists scores 1/61 + 1/61 = 0.0328.
- **The RRF constant (k = 60)** is the standard value from the original RRF paper (Cormack, Clarke
  and Büttcher, 2009). Hybrid MRR for `combo_head` is 0.860, 0.850, 0.851, 0.845, 0.845 and 0.845
  for k = 1, 5, 10, 20, 60 and 100. Weighting vector search double gives 0.864. k isn't the issue.
- **Embedded text:** printed for one chunk in each variant; each matches its design.
- **Effects are broad, not driven by one or two questions:** see the counts above. Removing the
  header still gains +0.087, and removing the overlap +0.064, after dropping each one's three
  biggest gains.

### Cost

| Item | Spent |
|---|---|
| Smoke test | about $0.20 |
| Main run (10 OpenAI variants; Gemini's partial run was free) | about $5.30 |
| Four combinations | $1.35 |
| `filings_v2` (pilot re-embed) | $0.42 |
| Production eval | $0.05 |
| **Total** | **about $7.30** |

The main run includes about $0.41 from a first attempt, stopped to add the cost caps; its saved
batches were reused, not paid again.

### For the full 498-company run

- Scaling the pilot: about 434,000 chunks and 160M tokens, so about $21 with
  `text-embedding-3-large` or about $3 with `-small`. Banks' long MD&As will push this up somewhat.
- Vectors at 1,024 dims are about 1.8 GB (Chroma about 2× that). 256 dims scored the same on the
  pilot and would be about 0.45 GB; worth checking on a few non-semiconductor questions first.
- Run it with `ingest --embed --budget 30` (revised up in section 7). The budget check runs before the first request.

### Free follow-ups (from saved vectors and rankings, no new embedding)

- BM25's weight in the fusion, and stemming for BM25.
- Adding the company and period to the query instead of the document.
- More than 20 reranker candidates.

## 7. Multi-sector check (2026-10-13)

The pilot's 30 questions cover 10 semiconductor companies, and many sit at the 1.0 ceiling. Before
the full embed, the winner was rechecked against two settings on a wider set: the pilot plus one
large company from each other sector (AMZN, BAC, CAT, GOOGL, LIN, NEE, PG, PLD, UNH, XOM; BAC
because JPM's 10-K MD&A is over ingest's size cap). That's 20 companies and 23,273 chunks. 20 new
questions (`eval/tests_multi.jsonl`, since merged into `eval/tests.jsonl` as rows 30–49) were added, in three kinds:

- **Year in the question, company filter only** (8): does the header's year help find the right
  filing?
- **Unfiltered** (6): no company filter across all 20 companies, where a company name in the
  embedding might help.
- **Sector vocabulary** (6): banking, insurance, energy, utilities, real estate and materials
  terms.

The pilot companies' vectors were reused (their chunks are identical); only the 10 new companies
were embedded. Cost: $1.84.

| Variant | Vector | Hybrid | Rerank |
|---|---|---|---|
| `combo_head` (plain text, large: production) | 0.804 | 0.828 | 0.908 |
| `overlap_none` (header on, no overlap, large) | 0.772 | 0.806 | 0.921 |
| `combo_head_small` (plain text, small) | 0.762 | 0.772 | 0.907 |

Paired differences (95% intervals):

| Comparison | Questions | Vector | Hybrid | Rerank |
|---|---|---|---|---|
| Header on − plain | old 30 | −0.074 [−0.182, +0.022] | −0.012 [−0.039, +0.010] | +0.004 [−0.019, +0.025] |
| Header on − plain | new 20 | +0.033 [−0.078, +0.144] | −0.038 [−0.129, +0.046] | +0.025 [0.000, +0.075] |
| Large − small | all 50 | +0.042 [−0.035, +0.120] | +0.057 [−0.002, +0.122] | +0.001 [−0.009, +0.012] |

What changes:

- **The header finding is weaker than section 6 says.** Removing the overlap, the header, or both
  helped clearly. But once the overlap is gone, header on vs off can't be told apart. It leans
  harmful on the filtered pilot questions and helpful on the unfiltered ones (0.850 vs 0.778, 6
  questions). Plain text stays in production because it's simpler and the research pipeline always
  filters, not because it's proven better than a header without overlap.
- **Large vs small:** large is ahead in every comparison, though no single one is a clear
  difference. That supports keeping `text-embedding-3-large`.
- **Hybrid helps again** on the wider set (0.828 vs 0.804 for vector alone). Across 20 companies,
  exact terms such as "Sakhalin" or "Rivian" matter more.
- **The new questions are harder,** as intended: year-in-question MRR is 0.54–0.63, so they no
  longer sit at the ceiling. A company filter alone doesn't find the right year well, with or
  without the header. The Phase 10 query rewriter should extract the year as a filter.
- **Embedding size still doesn't matter** (256, 1,024 and 3,072 within 0.01).
- **Reranking makes every setup end up at 0.91–0.92,** so the choices above matter mostly for
  vector and hybrid search on their own.

**The full-run estimate goes up.** The 10 non-semiconductor companies produce 14,559 chunks against
8,714 for the 10 semiconductor ones (banks' MD&As especially). Scaling all 20: about 580,000
chunks and 200M tokens, about $26 with `text-embedding-3-large`, and about 2.4 GB of vectors at
1,024 dims. Run it with `ingest --embed --budget 30`.

## 8. Open embedding models (2026-10-13)

Two open models that run locally and take inputs longer than our 800-token chunks, scored on the
same 20 companies and 50 questions as section 7, with the production text (plain, heading-aligned
chunks). Embedding cost nothing; reranking cost $0.28.

| Model | Vector | Hybrid | Rerank | Full 498-company run |
|---|---|---|---|---|
| `text-embedding-3-large` (production) | 0.804 | 0.828 | 0.908 | about $26, about an hour |
| `BAAI/bge-m3` (open, 568M parameters) | 0.808 | 0.789 | 0.889 | free, about 30 hours on an Apple-silicon Mac |
| `Qwen/Qwen3-Embedding-0.6B` (open) | 0.732 | 0.765 | 0.896 | free, about 30 hours |
| `text-embedding-3-small` | 0.762 | 0.772 | 0.907 | about $3 |

Paired differences from `text-embedding-3-large` (95% intervals):

| Model | Vector | Hybrid | Rerank |
|---|---|---|---|
| bge-m3 | +0.004 [−0.053, +0.060] | −0.039 [−0.101, +0.015] | −0.019 [−0.048, +0.003] |
| Qwen3-0.6B | −0.071 [−0.146, +0.003] | **−0.063 [−0.130, −0.005]** | −0.012 [−0.037, +0.005] |

- **bge-m3 matches the OpenAI model on vector search.** It's better on sector vocabulary (0.875
  vs 0.750), unfiltered questions (0.804 vs 0.778) and year questions (0.656 vs 0.631), and worse
  on broad (0.673 vs 0.753) and paraphrased ones (0.780 vs 0.821). Hybrid and rerank lean toward
  OpenAI but within noise. Phase 9 uses hybrid search without reranking, so the hybrid lean
  matters more than the rerank one.
- **Qwen3-0.6B is worse,** mainly on paraphrased (0.617) and year (0.422) questions.
- **Speed:** about 5 chunks a second on the Mac's GPU, so about 30 hours for the full 580,000
  chunks. The batch files make that resumable.
- **Using bge-m3 in production** would also mean embedding queries locally (the 2.3 GB model loaded
  in memory), and changing `ingest.embed_chunks` and `retrieve.vector_search`.

## 9. Q&A layer (Phase 10, 2026-10-18)

The Q&A layer (`answer.py`, `app.py`) plans each turn with Claude Haiku 5.5, retrieves with the
reranked hybrid search above (or, for "which companies / sectors" questions, ranks the tone cells),
and answers from the passages with the API's citations. The eval set, metrics and pass bars were
fixed in PLAN.md (Phase 10) before the run; the reference answers have not yet been reviewed by the
owner, and no person graded the answers, so the judge's scores are not validated against a person.

**The set (`eval/qa_tests.jsonl`, 69 rows):** the 50 questions of `tests.jsonl` without their
filters, 7 two-turn conversations and 2 multi-part questions written for this eval (reference
answers and keywords from the filing text), 5 "which companies / sectors" questions with
reference answers computed from the cells, and 5 out-of-scope questions.

**Results** (`uv run python eval/qa_eval.py report`; answers at effort low, $0.270 for 69 rows,
$0.0039 a row, median 21 seconds; judge Claude Sonnet 5.5 at effort medium through the Batch API,
$0.410 for 64 answers):

| Metric | Value | Bar fixed in advance |
|---|---|---|
| Planner: tickers right (59 passage questions) | 83% | ≥ 90%: **missed** |
| Planner: fiscal years right | 54% | ≥ 90%: **missed** (see below) |
| Planner: forms right | 54% | |
| "Which" questions sent to the cells (5) | 5 of 5 (topic 5, dimension 5, unit 4) | |
| Out-of-scope questions declined (5) | 5 of 5 | ≥ 80%: met |
| In-scope questions declined | 7 of 64 | |
| Keyword MRR / coverage in the passages given to the answer model (59) | 0.703 / 81% | |
| Cited spans found word for word in their passage | 417 of 417 | |
| Judge: accuracy / completeness, mean of 1–5 (64) | 4.09 / 4.08 | |
| Judge: groundedness, mean of 1–5 | 3.89 | ≥ 4: **missed** |
| Answers with at least one claim the judge calls unsupported | 78% (50 of 64) | ≤ 5%: **missed** |

By type (judge accuracy / completeness / groundedness): conversations 5.00 / 4.86 / 4.57 (7),
multi-part 5.00 / 5.00 / 4.50 (2), single questions 3.98 / 4.00 / 3.84 (50), "which" questions
3.60 / 3.40 / 3.20 (5).

What the numbers mean:

- **The planner declines too often.** 7 of the 50 single questions were declined. Each names an
  entity outside the index or asks for one figure (Mellanox's revenue share, Intel's Altera deal and
  Mobileye IPO, Amazon's Rivian loss, NextEra's storm costs, Exxon's Pioneer closing, Prologis's
  Duke Realty price), and the planner assumed the filings wouldn't have it; they do. These 7 make
  most of the judge's lowest scores. The fix (a planner rule that such questions are searched, not
  declined) is not made: these results describe the code as it was run.
- **The planner bar is partly a test artefact.** The expected filters are the retrieval eval's,
  which the test author supplied rather than wrote into the question: s11 ("What government inquiry
  about China shipments did Applied Materials disclose?") expects FY2022, and the 8
  year-in-question rows expect no year at all, so extracting "fiscal 2019" from s31 counts as a
  miss. A fair version needs expected filters written from the question text alone.
- **The unsupported-claim rate mixes three things.** On reading the judge's list: statements about
  what the passages don't say ("the passages don't break out HBM revenue"); facts the answer model
  had but the judge didn't (period-end dates from each document's context field, the scoring model
  named in the cell note); and the 7 wrong declines. A smaller number are real errors, such as s06
  citing a span for "the AI Diffusion rule was rescinded in May 2025" that doesn't say it. How
  much of the 78% is the judge's strictness is not measured.
- **Citations are reliable and retrieval mostly finds the evidence.** Every cited span is in its
  passage (the API returns spans, so this checks the plumbing more than the model). The passages
  given to the answer model contain 81% of the keywords; the filtered retrieval eval's reranked
  top 10 has an MRR of 0.910, but its filters were supplied, and here the planner sets them.
- **Speed.** Most of the 21 seconds is Chroma's filtered vector search (4–19 seconds a query on the
  full collection, timed 2026-10-18); planning takes about 2 seconds and each rerank 1–4.

**Smoke tests (before the eval, on the analyst questions, $0.164):** 12 answers at effort low
($0.078) and at medium ($0.079) cost and took the same; low stays the default. Tuning made then, not
on the eval set: planner rules for sub-industries ("regional banks") and for change between years,
third-person answers, each dimension's scale in the cell note, and ties ranked by specificity.

## How to reproduce

```bash
uv run python eval/qa_eval.py answers --budget 0.70       # Q&A eval answers; about $0.27
uv run python eval/qa_eval.py judge --budget 1.25         # Claude Sonnet 5.5 judge, Batch API; about $0.41
uv run python eval/qa_eval.py report                      # every metric; free
uv run python tests/test_ingest.py                        # chunker fixes
uv run python tests/test_retrieve.py                      # filters, BM25, RRF, rerank order
uv run python eval/experiment.py --dry-run                # chunks, tokens, cost; free
uv run python eval/experiment.py --smoke                  # every variant on TXN; about $0.20
uv run python eval/experiment.py                          # the experiment; about $5.75, capped at $8 and 90 minutes
uv run python eval/experiment.py --score-only             # recompute metrics from saved rankings
```
