"""Settings shared across the pipeline: model ids and prices, file paths, fiscal-year bounds, API batch limits and
the retrieval constants more than one module uses. Plain constants: importing this module reads nothing."""

from pathlib import Path

# ---------- models ----------
MODEL = "claude-haiku-5-5"  # the tone scorer (sentiment, tone) and discover's cluster names; cheapest; the pilot shows whether a bigger model is needed
RERANK_MODEL = "claude-haiku-5-5"
ANSWER_MODEL = "claude-haiku-5-5"  # answer.py's planner and answers
JUDGE_MODEL = "claude-sonnet-5-5"  # eval/qa_eval.py's judge
GPT_JUDGE = "gpt-6-sol"  # eval/qa_eval.py's second judge: OpenAI's middle tier, priced like Claude Sonnet 5.5
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 1024
COLLECTION = "filings_v2"  # v1 embedded a context header and overlap; Phase 4 found both hurt (eval/FINDINGS.md)
FINBERT = "ProsusAI/finbert"

# ---------- prices ----------
# USD per 1M input and output tokens (claude.com pricing, checked 2026-10-12)
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}
GPT_PRICES = {"gpt-6-sol": (2.00, 10.00)}  # USD per 1M input and output tokens (OpenAI pricing page, checked 2026-10-18)
EMBED_PRICE_PER_M = 0.13  # USD per 1M tokens (OpenAI pricing page, checked 2026-10-12)
USD_PER_REQUEST = 0.0002  # one tone.score_cells request. Gold run: $0.016 for 40 calls at full price, halved; the smoke test measures it

# ---------- API batch limits ----------
BATCH_REQUESTS, BATCH_BYTES = 100_000, 200e6  # Message Batches (tone.score_cells): the API's limits are 100,000 requests and 256 MB
BATCH_MAX_INPUTS = 50_000  # OpenAI's cap on embedding inputs per batch
REQUEST_INPUTS = 100  # inputs per request line in an embedding batch file
QUEUE_SHARE = 0.9  # use at most this share of the account's queued-token limit (20M for ours)

# ---------- fiscal years ----------
FIRST_FY = 2019
LAST_FY = 2025  # FY2026: only companies whose fiscal year doesn't end in December had filed

# ---------- retrieval ----------
CANDIDATES = 20  # each search's top n; the fused top n goes to the reranker
RRF_K = 60
K = 3  # a tone cell's passages: hybrid search's top K in its filing

# ---------- paths ----------
ROOT = Path(__file__).parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"  # fetched section texts
COMPANIES = DATA / "companies.csv"
FILINGS, FILINGS_10Q = DATA / "filings.csv", DATA / "filings_10q.csv"  # fetch's manifests
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHUNKS = DATA / "chunks.parquet"
SUBSET = DATA / "chunks_subset.parquet"  # --tickers: never CHUNKS, which every later phase reads for all companies
SKIPPED = DATA / "ingest_skipped.csv"
VECTOR_DB = str(ROOT / "vector_db")
EMBED_BATCHES = DATA / "embed_batches"  # Batch API state, request files and downloaded results
BM25_DIR = DATA / "bm25"  # saved BM25 indexes, one folder per chunk count
RUNS = DATA / "runs.jsonl"  # one JSON line per paid run; local, git-ignored
RELEASE = DATA / "release"

OUT = DATA / "tone"  # the tone cells' passages and scores, and every analysis's results
PASSAGES = OUT / "passages.parquet"
VECTORS = OUT / "vectors.npy"  # every chunk's stored embedding, unit length, in chunks() order (2.8 GB)
VECTOR_ORDER = OUT / "vectors_order.sha256"  # hash of the chunk ids in VECTORS' row order
SCORES = OUT / "scores.parquet"  # one row per scored passage, keyed by its row number in PASSAGES (pid)
STATE = OUT / "batches.json"  # submitted batches not yet collected
BASELINES = OUT / "baselines.parquet"  # FinBERT, LM and rule-based counts per distinct chunk
# form -> passages, scores, batch state, baselines; the 10-Q cells (B5) get their own q_ files
FILES = {"10-K": (PASSAGES, SCORES, STATE, BASELINES),
         "10-Q": tuple(OUT / f"q_{f.name}" for f in (PASSAGES, SCORES, STATE, BASELINES)),
         "rerun": (*(OUT / f"rerun_{f.name}" for f in (PASSAGES, SCORES, STATE)), None),  # B6: a sample scored again
         "masked": (*(OUT / f"masked_{f.name}" for f in (PASSAGES, SCORES, STATE)), None),  # masked check: event-test passages, no company or year
         "section": tuple(OUT / f"s_{f.name}" for f in (PASSAGES, SCORES, STATE, BASELINES)),  # S_TOPICS within S_ITEM (sections)
         "new": tuple(OUT / f"n_{f.name}" for f in (PASSAGES, SCORES, STATE, BASELINES))}  # the event tests' cells on new text only (diff.py)
KEY = {"10-K": ["ticker", "fy"], "10-Q": ["ticker", "report_date"]}  # one filing: a fiscal year's 10-K, a quarter's 10-Q
QUERIES = OUT / "queries.npy"  # the query vectors passages.parquet was ranked with, in TOPICS order
EVENT_RESULTS = OUT / "events.csv"
CARRYOVER = OUT / "carryover.parquet"  # one row per 10-K chunk of a company with the year before's 10-K
N_CELLS = OUT / "n_cells.parquet"  # the cells diff.py `retrieve` searched (needed()), with or without a new chunk
COMOVE_RESULTS = OUT / "comove.csv"
DISCOVERED_TOPICS = OUT / "discovered_topics.csv"  # one row per cluster
DISCOVERED_CLUSTERS = OUT / "discovered_clusters.parquet"  # chunk_id -> cluster, per seed

SENTIMENT = DATA / "sentiment"  # the pilot's and the gold benchmark's scores
LABELS = ROOT / "labels"
GUIDELINES = LABELS / "guidelines.md"
PRACTICE = LABELS / "practice.csv"  # never sampled, so it stays a clean test
GOLD = LABELS / "gold_profiles.csv"
ADJUDICATION = LABELS / "gold_adjudication.csv"
# Loughran-McDonald Master Dictionary 1993-2025, linked from sraf.nd.edu/loughranmcdonald-master-dictionary.
# Free for academic use; kept out of git.
LM_URL = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
LM_PATH = DATA / "lm" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"

EVENTS = ROOT / "events.csv"
EVENTS_QUARTERLY = ROOT / "events_quarterly.csv"
FIGURES = ROOT / "figures"
