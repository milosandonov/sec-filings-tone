"""Phase 3: retrieve chunks for a query. A metadata filter picks the subset, vector search (Chroma)
and BM25 (bm25s) each rank it, RRF fuses the two lists, and Claude reranks the fused top 20.

Usage:
    uv run python -m sec_filings_tone.retrieve "China export restrictions" --ticker NVDA --form 10-K --fy 2023 --item 1A
    uv run python -m sec_filings_tone.retrieve H20 --ticker NVDA --no-rerank
    uv run python -m sec_filings_tone.retrieve "deposit outflows" --sector Financials --date-from 2023-03-01 --date-to 2023-12-31
"""

import argparse
import hashlib
from contextlib import nullcontext
from functools import cache

import bm25s
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from sec_filings_tone.budget import Cap, rerank_lock, rerank_usage, run_log
from sec_filings_tone.config import (BM25_DIR, CANDIDATES, CHUNKS, COLLECTION, EMBED_DIMS, EMBED_MODEL, OUT, RERANK_MODEL, RRF_K,
                                     VECTOR_DB, VECTOR_ORDER, VECTORS)
from sec_filings_tone.ingest import embed_text

load_dotenv()

RERANK_TIMEOUT = 60  # seconds per call, so a stuck request fails instead of hanging (the SDK retries twice)
FILTERS = ("ticker", "sector", "form", "item", "fy")
STAGES = ("vector", "bm25", "hybrid", "rerank")

RERANK_PROMPT = """You rerank passages from SEC filings (10-K and 10-Q) by relevance to a question.
The passages arrive roughly in order from a search; you may be able to improve on that order.
Put the passage that best answers the question first. Reply with every passage id exactly once."""


class RankOrder(BaseModel):
    order: list[int] = Field(description="Every passage id, most relevant first")


@cache
def chunks() -> pd.DataFrame:
    return pd.read_parquet(CHUNKS).set_index("chunk_id", drop=False)


@cache
def bm25_index() -> bm25s.BM25:
    """Built once and saved under BM25_DIR: building takes about a minute on the full set, loading seconds."""
    # keyed on the chunk count only; delete BM25_DIR if chunks.parquet changes but keeps its count
    path = BM25_DIR / str(len(chunks()))
    if path.exists():
        return bm25s.BM25.load(path, show_progress=False)
    index = bm25s.BM25()
    index.index(bm25s.tokenize(chunks().text.tolist(), stopwords="en", show_progress=False), show_progress=False)
    index.save(path, show_progress=False)
    return index


@cache
def collection():
    import chromadb

    col = chromadb.PersistentClient(VECTOR_DB).get_collection(COLLECTION, embedding_function=None)
    if col.count() != len(chunks()):
        print(f"warning: {COLLECTION} has {col.count()} chunks but {CHUNKS} has {len(chunks())}, "
              "so BM25 can return chunks vector search never sees")
    return col


def subset(filters: dict) -> tuple[np.ndarray, dict | None]:
    """The same filter two ways: a mask over chunks() for BM25 and a Chroma `where` for vector search.
    Keys are FILTERS (a scalar or a list each) plus date_from/date_to on filing_date, both ends inclusive."""
    filters = dict(filters)
    date_from, date_to = filters.pop("date_from", None), filters.pop("date_to", None)
    if unknown := set(filters) - set(FILTERS):
        raise ValueError(f"unknown filters {sorted(unknown)}; use {FILTERS}")
    df = chunks()
    conds = {col: [x.item() if hasattr(x, "item") else x  # numpy scalars break Chroma
                   for x in (v if isinstance(v, (list, tuple, set)) else [v])]
             for col, v in filters.items() if v is not None}
    mask = np.ones(len(df), bool)
    for col, vals in conds.items():
        mask &= df[col].isin(vals).to_numpy()
    if date_from:
        mask &= (df.filing_date >= date_from).to_numpy()
    if date_to:
        mask &= (df.filing_date <= date_to).to_numpy()
    if date_from or date_to:  # Chroma compares only numbers, so list the filing dates in range
        conds["filing_date"] = sorted(df.filing_date[mask].unique().tolist())
    clauses = [{col: {"$in": vals}} for col, vals in conds.items()]
    return mask, {"$and": clauses} if len(clauses) > 1 else clauses[0] if clauses else None


@cache
def embed_query(query: str) -> list[float]:
    """Cached, so Phase 9's ~42,000 searches over 13 topic queries embed each query once."""
    from openai import OpenAI

    return OpenAI().embeddings.create(model=EMBED_MODEL, dimensions=EMBED_DIMS, input=[query]).data[0].embedding


def vector_search(query: str, where: dict | None, n: int) -> tuple[list[str], list[float]]:
    res = collection().query(query_embeddings=[embed_query(query)], n_results=n, where=where, include=["distances"])
    return res["ids"][0], [1 - d for d in res["distances"][0]]  # cosine similarity


def order_hash(ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def matches_chroma(v: np.ndarray, ids: pd.Series, n: int = 200, seed: int = 0) -> bool:
    """Whether n random rows of v are the Chroma embeddings of the chunks in those rows of chunks()."""
    pick = np.sort(np.random.default_rng(seed).choice(len(ids), min(n, len(ids)), replace=False))
    want = ids.iloc[pick].tolist()
    got = collection().get(ids=want, include=["embeddings"])
    if sorted(got["ids"]) != sorted(want):
        return False
    e = pd.DataFrame(np.asarray(got["embeddings"], np.float32), index=got["ids"]).loc[want].to_numpy()
    return bool(np.allclose((e / np.linalg.norm(e, axis=1, keepdims=True) * v[pick]).sum(axis=1), 1, atol=1e-4))


def vectors() -> np.ndarray:
    """Read from Chroma once, then from disk. Rows follow chunks()'s order, which moves if chunks.parquet is
    rebuilt from a changed manifest, so the order is recorded (VECTOR_ORDER) and checked on every load."""
    ids = chunks().chunk_id
    if VECTORS.exists():
        v = np.load(VECTORS, mmap_mode="r")
        if not VECTOR_ORDER.exists() and len(v) == len(ids) and matches_chroma(v, ids):
            VECTOR_ORDER.write_text(order_hash(ids))  # saved before the order was recorded: checked against Chroma first
        if not VECTOR_ORDER.exists() or VECTOR_ORDER.read_text() != order_hash(ids):
            raise SystemExit(f"{VECTORS} doesn't follow the rows of {CHUNKS}; delete it and {VECTOR_ORDER.name} "
                             "to rebuild it from Chroma")
        return v
    pos = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    v = np.zeros((len(ids), EMBED_DIMS), np.float32)
    col, page = collection(), 20_000
    for off in range(0, col.count(), page):
        got = col.get(include=["embeddings"], limit=page, offset=off)
        v[pos.loc[got["ids"]].to_numpy()] = got["embeddings"]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if (norms == 0).any():
        raise SystemExit(f"{int((norms == 0).sum())} chunks have no vector in Chroma; re-embed them first")
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS, v / norms)
    VECTOR_ORDER.write_text(order_hash(ids))
    return np.load(VECTORS, mmap_mode="r")


@cache
def stored_vectors() -> np.ndarray:
    """Every chunk's embedding in chunks() order (data/tone/vectors.npy, memory-mapped; vectors() checks the order)."""
    return vectors()


def exact_vector_search(query: str, mask: np.ndarray, n: int) -> tuple[list[str], list[float]]:
    """vector_search by exact cosine over the stored vectors of the chunks in mask. Chroma's filtered
    queries take 4-19 seconds on the full set; this takes well under one. Phase 9 ranks cells the same way."""
    v, q = stored_vectors(), np.asarray(embed_query(query), np.float32)
    idx = np.arange(len(v)) if mask.all() else np.flatnonzero(mask)
    s = (v if mask.all() else v[idx]) @ (q / np.linalg.norm(q))
    top = np.argsort(-s, kind="stable")[:n] if len(s) <= 10 * n else np.argpartition(-s, n)[:n]
    top = top[np.argsort(-s[top], kind="stable")]
    return chunks().chunk_id.iloc[idx[top]].tolist(), s[top].tolist()


def bm25_search(query: str, mask: np.ndarray, n: int) -> tuple[list[str], list[float]]:
    q = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)
    docs, scores = bm25_index().retrieve(q, k=n, weight_mask=mask.astype(np.float32), show_progress=False)
    hit = scores[0] > 0  # masked-out chunks and chunks without a query term score 0
    return chunks().chunk_id.iloc[docs[0][hit]].tolist(), scores[0][hit].tolist()


def rrf(*rankings: list[str]) -> dict[str, float]:
    """Reciprocal Rank Fusion: a chunk scores sum(1 / (RRF_K + rank)) over the lists it's in, best first."""
    scores = {}
    for ids in rankings:
        for rank, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def fix_order(order: list[int], n: int) -> list[int]:
    """The LLM's 1-based ids -> 0-based positions, without unknown or repeated ids, plus any it left out."""
    pos = [i - 1 for i in dict.fromkeys(order) if 1 <= i <= n]
    return pos + [i for i in range(n) if i not in pos]


def llm_rerank(query: str, rows: pd.DataFrame) -> pd.DataFrame:
    from anthropic import Anthropic

    # same context header as the embedded chunks, so Claude knows the ticker, form, year and item
    passages = "\n\n".join(f"# PASSAGE ID {i}\n{doc}" for i, doc in enumerate(embed_text(rows, overlap_tokens=0), 1))
    resp = Anthropic(timeout=RERANK_TIMEOUT).messages.parse(
        model=RERANK_MODEL, max_tokens=8000, output_config={"effort": "low"}, system=RERANK_PROMPT,
        messages=[{"role": "user", "content": f"Question: {query}\n\n{passages}\n\n"
                                              f"Rank all {len(rows)} passage ids, most relevant first."}],
        output_format=RankOrder)
    with rerank_lock:
        rerank_usage["input_tokens"] += resp.usage.input_tokens
        rerank_usage["output_tokens"] += resp.usage.output_tokens
    if resp.parsed_output is None:
        raise RuntimeError(f"rerank returned no order (stop_reason={resp.stop_reason})")
    return rows.iloc[fix_order(resp.parsed_output.order, len(rows))].reset_index(drop=True)


def rows(ids: list[str], scores: list[float]) -> pd.DataFrame:
    return chunks().loc[ids].assign(score=scores).reset_index(drop=True)


def search(query: str, k: int = 10, rerank: bool = True, date_from: str | None = None,
           date_to: str | None = None, exact: bool = False, **filters) -> dict[str, pd.DataFrame]:
    """Top k of each stage, as chunks.parquet rows plus `score`: vector (cosine similarity), bm25,
    hybrid (RRF score) and rerank (the hybrid top CANDIDATES reordered by Claude, keeping RRF scores).
    exact: vector search by exact_vector_search instead of Chroma (the Q&A layer, for speed)."""
    mask, where = subset({**filters, "date_from": date_from, "date_to": date_to})
    n = min(CANDIDATES, int(mask.sum()))
    if n == 0:
        return dict.fromkeys(STAGES[:4 if rerank else 3], rows([], []))
    vec = exact_vector_search(query, mask, n) if exact else vector_search(query, where, n)
    bm = bm25_search(query, mask, n)
    fused = rrf(vec[0], bm[0])
    out = {"vector": rows(*vec), "bm25": rows(*bm), "hybrid": rows(list(fused), list(fused.values()))}
    if rerank:
        out["rerank"] = llm_rerank(query, out["hybrid"].head(CANDIDATES))
    return {stage: df.head(k) for stage, df in out.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    for f in FILTERS:
        p.add_argument(f"--{f}", nargs="+", type=int if f == "fy" else str)
    p.add_argument("--date-from", help="filing date, YYYY-MM-DD")
    p.add_argument("--date-to", help="filing date, YYYY-MM-DD")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--no-rerank", action="store_true")
    a = p.parse_args()
    import anthropic

    # reranking goes in the run log (a fresh Cap counts just budget.rerank_usage); without it, only the query embedding is paid
    log = nullcontext() if a.no_rerank else run_log([RERANK_MODEL, EMBED_MODEL], lambda: {"usd": {"anthropic": Cap().spent()}})
    try:
        with log:
            results = search(a.query, a.k, not a.no_rerank, a.date_from, a.date_to, **{f: getattr(a, f) for f in FILTERS})
    except anthropic.AuthenticationError:
        raise SystemExit("Anthropic rejected ANTHROPIC_API_KEY in .env (401). Keys start with sk-ant- and come from "
                         "https://console.anthropic.com/settings/keys. Use --no-rerank to skip reranking.")
    for stage, df in results.items():
        print(f"\n== {stage} ==" + ("" if len(df) else " (no chunks match the filters)"))
        for i, r in enumerate(df.itertuples(), 1):
            print(f"{i:2}. {r.score:6.3f}  {r.chunk_id}  [{r.heading[:50]}]  {r.text[:90]}")


if __name__ == "__main__":
    main()
