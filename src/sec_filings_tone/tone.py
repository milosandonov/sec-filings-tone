"""Phase 9: tone of every (company, fiscal year, topic) cell from the 10-Ks.

A cell's passages are hybrid search's top 3 in that 10-K, with no reranking (PLAN.md, Phase 9):
every topic for every company, and deposits and liquidity for Financials only.

Each passage is scored with the same call as the gold benchmark (sentiment.score: model, prompt,
schema, effort), sent through the Message Batches API at half price.

Usage:
    uv run python -m sec_filings_tone.tone retrieve    # every cell's passages; free but for 13 query embeddings
    uv run python -m sec_filings_tone.tone check       # 50 random cells against retrieve.search
    uv run python -m sec_filings_tone.tone score --tickers NVDA --budget 0.2   # smoke test on one company
    uv run python -m sec_filings_tone.tone score --budget 30                    # the rest; resumes if stopped
    uv run python -m sec_filings_tone.tone baselines   # FinBERT and LM on the same passages, local; resumes
"""

import argparse
import json
import time
from pathlib import Path

import anthropic
import bm25s
import numpy as np
import pandas as pd
from anthropic.lib._parse._transform import transform_schema  # what messages.parse sends, so both paths share one schema
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from sec_filings_tone import retrieve
from sec_filings_tone.budget import call_usd
from sec_filings_tone.ingest import CHUNKS, EMBED_DIMS
from sec_filings_tone.sentiment import DIMENSIONS, MODEL, TOPICS, finbert, lm, proxies, schema, system_prompt, user_prompt

OUT = Path("data/tone")
PASSAGES = OUT / "passages.parquet"
VECTORS = OUT / "vectors.npy"  # every chunk's stored embedding, unit length, in chunks() order (2.8 GB)
SCORES = OUT / "scores.parquet"  # one row per scored passage, keyed by its row number in PASSAGES (pid)
STATE = OUT / "batches.json"  # submitted batches not yet collected
BASELINES = OUT / "baselines.parquet"  # FinBERT, LM and rule-based counts per distinct chunk
K = 3
BATCH_REQUESTS, BATCH_BYTES = 100_000, 200e6  # the API's limits are 100,000 requests and 256 MB
USD_PER_REQUEST = 0.0002  # gold run: $0.016 for 40 calls at full price, halved; the smoke test measures it


def cells() -> pd.DataFrame:
    """One row per (10-K, topic)."""
    k = retrieve.chunks()
    k = k[k.form == "10-K"].drop_duplicates(["ticker", "fy"])[["ticker", "sector", "fy"]]
    c = k.merge(pd.DataFrame({"topic": list(TOPICS)}), how="cross")
    return c[(c.topic != "Deposits and liquidity") | (c.sector == "Financials")].reset_index(drop=True)


def vectors() -> np.ndarray:
    """Read from Chroma once, then from disk."""
    if VECTORS.exists():
        return np.load(VECTORS, mmap_mode="r")
    ids = retrieve.chunks().chunk_id
    pos = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    v = np.zeros((len(ids), EMBED_DIMS), np.float32)
    col, page = retrieve.collection(), 20_000
    for off in range(0, col.count(), page):
        got = col.get(include=["embeddings"], limit=page, offset=off)
        v[pos.loc[got["ids"]].to_numpy()] = got["embeddings"]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if (norms == 0).any():
        raise SystemExit(f"{int((norms == 0).sum())} chunks have no vector in Chroma; re-embed them first")
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS, v / norms)
    return np.load(VECTORS, mmap_mode="r")


def retrieve_cells() -> None:
    """Hybrid's top 3 for every cell, ranked as retrieve.search ranks them, but with each topic scored
    over all chunks once: search() takes about 1.5 seconds a call, about 17 hours for every cell.
    Vector scores here are exact cosine; search() asks Chroma's approximate (HNSW) index, so a few
    cells can differ. `check` measures how many."""
    if SCORES.exists() or STATE.exists():  # scores point at passage row numbers, which a new retrieval would shuffle
        raise SystemExit(f"{SCORES} or {STATE} exists; move them before retrieving again")
    start = time.time()
    df = retrieve.chunks()
    ids, vec = df.chunk_id.to_numpy(), vectors()
    groups = df.groupby(["form", "ticker", "fy"]).indices  # positions, aligned with the vectors and BM25
    c = cells()
    rows = []
    for topic, query in TOPICS.items():
        q = np.asarray(retrieve.embed_query(query), np.float32)
        vs = vec @ (q / np.linalg.norm(q))
        bs = retrieve.bm25_index().get_scores(bm25s.tokenize([query], stopwords="en", return_ids=False,
                                                             show_progress=False)[0])
        for r in c[c.topic == topic].itertuples():
            idx = groups[("10-K", r.ticker, r.fy)]
            n = min(retrieve.CANDIDATES, len(idx))
            v = idx[np.argsort(-vs[idx], kind="stable")[:n]]
            b = idx[np.argsort(-bs[idx], kind="stable")[:n]]
            fused = retrieve.rrf(list(ids[v]), list(ids[b[bs[b] > 0]]))  # BM25 drops chunks with no query word
            rows += [{"ticker": r.ticker, "sector": r.sector, "fy": r.fy, "topic": topic, "rank": i, "chunk_id": cid,
                      "rrf": s} for i, (cid, s) in enumerate(list(fused.items())[:K], 1)]
        print(f"{topic}: {(c.topic == topic).sum():,} cells, {time.time() - start:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(PASSAGES, index=False)
    print(f"{PASSAGES}: {len(out):,} passages for {out.groupby(['ticker', 'fy', 'topic']).ngroups:,} of "
          f"{len(c):,} cells")


def check(n: int, seed: int = 0) -> None:
    """n random cells' top 3 against retrieve.search's hybrid top 3."""
    p = pd.read_parquet(PASSAGES)
    same = 0
    for r in p[["ticker", "fy", "topic"]].drop_duplicates().sample(n, random_state=seed).itertuples():
        mine = p[(p.ticker == r.ticker) & (p.fy == r.fy) & (p.topic == r.topic)].sort_values("rank").chunk_id.tolist()
        theirs = retrieve.search(TOPICS[r.topic], k=K, rerank=False, ticker=r.ticker, form="10-K",
                                 fy=int(r.fy))["hybrid"].chunk_id.tolist()
        same += mine == theirs
        if mine != theirs:
            print(f"{r.ticker} {r.fy} {r.topic}: {mine} vs search() {theirs}")
    print(f"{same}/{n} cells identical to retrieve.search")


def score_cells(budget: float, tickers: list[str] | None = None) -> None:
    """Submit every passage not yet scored, in batches, then wait and collect. Failed requests aren't
    stored, so running again resubmits just those."""
    client = anthropic.Anthropic()
    state = json.loads(STATE.read_text()) if STATE.exists() else []
    if not state:
        p = pd.read_parquet(PASSAGES).reset_index(names="pid")
        if tickers:
            p = p[p.ticker.isin(tickers)]
        if SCORES.exists():
            p = p[~p.pid.isin(pd.read_parquet(SCORES, columns=["pid"]).pid)]
        est = len(p) * USD_PER_REQUEST
        print(f"{len(p):,} passages to score, about ${est:.2f}")
        if est > budget:
            raise SystemExit(f"over --budget ${budget:.2f}; nothing submitted")
        rows, system, fmt = retrieve.chunks(), system_prompt(), transform_schema(schema(list(DIMENSIONS)))
        batch, size = [], 0
        for r in [*p.itertuples(), None]:  # None flushes the last batch
            req = None if r is None else Request(custom_id=f"p{r.pid}", params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=4000, output_config={"effort": "low", "format": {"type": "json_schema", "schema": fmt}},
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                messages=[{"role": "user", "content": user_prompt(rows.loc[r.chunk_id], r.topic, list(DIMENSIONS))}]))
            n = len(json.dumps(req)) if req else 0
            if batch and (req is None or len(batch) >= BATCH_REQUESTS or size + n > BATCH_BYTES):
                while True:
                    try:
                        state.append(client.messages.batches.create(requests=batch).id)
                        break
                    except anthropic.RateLimitError:  # the queue is full of our own batches: wait for some to finish
                        print("batch queue full; retrying in 5 minutes", flush=True)
                        time.sleep(300)
                STATE.write_text(json.dumps(state))
                print(f"submitted {state[-1]}: {len(batch):,} requests, {size / 1e6:.0f} MB", flush=True)
                batch, size = [], 0
            if req:
                batch.append(req)
                size += n
    parse = schema(list(DIMENSIONS))
    while state:
        for bid in list(state):
            try:
                if client.messages.batches.retrieve(bid).processing_status != "ended":
                    continue
                results = list(client.messages.batches.results(bid))
            except anthropic.APIConnectionError as e:  # timeouts included; the batch waits on the server, try next round
                print(f"{bid}: {type(e).__name__}, retrying in a minute", flush=True)
                continue
            out, failed = [], 0
            for res in results:
                if res.result.type != "succeeded":
                    failed += 1
                    continue
                m, u = res.result.message, res.result.message.usage
                try:
                    parsed = parse.model_validate_json(next(x.text for x in m.content if x.type == "text")).model_dump()
                except (StopIteration, ValueError):  # cut off at max_tokens, or no text block
                    failed += 1
                    continue
                usd = call_usd(MODEL, u.input_tokens + (u.cache_creation_input_tokens or 0) * 2  # 1-hour cache writes cost 2x
                               + (u.cache_read_input_tokens or 0) * 0.1, u.output_tokens) / 2  # batch: half price
                out.append({"pid": int(res.custom_id[1:]), **parsed, "model": MODEL, "usd": usd, "batch": bid})
            new = pd.DataFrame(out)
            pd.concat([pd.read_parquet(SCORES), new] if SCORES.exists() else [new]).to_parquet(SCORES, index=False)
            state.remove(bid)
            STATE.write_text(json.dumps(state)) if state else STATE.unlink()
            print(f"{bid}: {len(out):,} scored, {failed} failed, ${new.usd.sum():.3f} "
                  f"(${new.usd.mean():.5f} a request)", flush=True)
        if state:
            time.sleep(60)
    s = pd.read_parquet(SCORES)
    print(f"{SCORES}: {len(s):,} passages scored, ${s.usd.sum():.2f} in all")


def baselines(part: int = 5000) -> None:
    """FinBERT, Loughran-McDonald and the rule-based counts for every passage, locally and free. They
    score the whole passage, not a topic, so each chunk is scored once however many cells it's in.
    Saved every `part` chunks, so a stopped run resumes."""
    done = pd.read_parquet(BASELINES) if BASELINES.exists() else pd.DataFrame({"chunk_id": pd.Series(dtype=str)})
    seen = set(done.chunk_id)
    todo = [c for c in pd.read_parquet(PASSAGES).chunk_id.unique() if c not in seen]
    # only these texts: all 674k chunks with FinBERT on top pushed a 16 GB machine into swap
    text = pd.read_parquet(CHUNKS, columns=["chunk_id", "text"], filters=[("chunk_id", "in", todo)]).set_index("chunk_id").text
    start = time.time()
    for k in range(0, len(todo), part):
        ids = todo[k:k + part]
        t = text.loc[ids].tolist()
        b = pd.concat([finbert(t), pd.DataFrame([lm(x) | proxies(x) for x in t])], axis=1)
        b.insert(0, "chunk_id", ids)
        done = pd.concat([done, b], ignore_index=True)
        done.to_parquet(BASELINES, index=False)
        print(f"{len(done):,} of {len(done) + len(todo) - k - len(ids):,} chunks, {time.time() - start:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["retrieve", "check", "score", "baselines"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--tickers", nargs="+")
    p.add_argument("--budget", type=float, default=0.5, help="USD; nothing is submitted if the estimate is over it")
    a = p.parse_args()
    if a.command == "retrieve":
        retrieve_cells()
    elif a.command == "check":
        check(a.n)
    elif a.command == "baselines":
        baselines()
    else:
        score_cells(a.budget, a.tickers)


if __name__ == "__main__":
    main()
