"""Phase 9: tone of every (company, fiscal year, topic) cell from the 10-Ks.

A cell's passages are hybrid search's top 3 in that 10-K, with no reranking (PLAN.md, Phase 9):
every topic for every company, and deposits and liquidity for Financials only.

With --quarterly (PLAN.md option B5), a cell is one 10-Q instead: Financials only, deposits and
liquidity and interest rates only, written to the q_ files so the annual ones are never touched.

With --section (PLAN.md Phase 7, 2026-10-18), for AI only, a cell is one 10-K's MD&A instead: its top 3
within Item 7, written to the s_ files. Passages already scored for the same cell copy their score.

Each passage is scored with the same call as the gold benchmark (sentiment.score: model, prompt,
schema, effort), sent through the Message Batches API at half price.

Usage:
    uv run python -m sec_filings_tone.tone retrieve    # every cell's passages; free but for 13 query embeddings
    uv run python -m sec_filings_tone.tone check       # 50 random cells against retrieve.search
    uv run python -m sec_filings_tone.tone score --tickers NVDA --budget 0.2   # smoke test on one company
    uv run python -m sec_filings_tone.tone score --budget 30                    # the rest; resumes if stopped
    uv run python -m sec_filings_tone.tone baselines   # FinBERT and LM on the same passages, local; resumes
    uv run python -m sec_filings_tone.tone retrieve --quarterly   # any command, on the 10-Q cells
    uv run python -m sec_filings_tone.tone rerun --budget 1.2      # B6: score 5,000 random passages again, compare
    uv run python -m sec_filings_tone.tone masked --budget 0.5     # masked check: event-test passages without company and year; --budget 0 just shows sample and cost
    uv run python -m sec_filings_tone.tone retrieve --section      # AI's top 3 within each 10-K's MD&A; then score and baselines --section
"""

import argparse
import hashlib
import json
import re
import time

import anthropic
import bm25s
import numpy as np
import pandas as pd
from anthropic import transform_schema  # what messages.parse sends, so both paths share one schema
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from sec_filings_tone import retrieve
from sec_filings_tone.budget import call_usd
from sec_filings_tone.ingest import CHUNKS, EMBED_DIMS
from sec_filings_tone.sentiment import DIMENSIONS, MODEL, TOPICS, finbert, lm, period, proxies, schema, system_prompt, user_prompt

OUT = CHUNKS.parent / "tone"
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
         "section": tuple(OUT / f"s_{f.name}" for f in (PASSAGES, SCORES, STATE, BASELINES))}  # S_TOPICS within S_ITEM (sections)
KEY = {"10-K": ["ticker", "fy"], "10-Q": ["ticker", "report_date"]}  # one filing: a fiscal year's 10-K, a quarter's 10-Q
Q_TOPICS = ["Deposits and liquidity", "Interest rates"]  # B5 scores 10-Qs for Financials on these only
S_TOPICS, S_ITEM = ["AI"], "7"  # section-level cells: AI's top 3 within each 10-K's MD&A (PLAN.md Phase 7, 2026-10-18)
QUERIES = OUT / "queries.npy"  # the query vectors passages.parquet was ranked with, in TOPICS order
K = 3
LAST_FY = 2025  # FY2026: only companies whose fiscal year doesn't end in December had filed
BATCH_REQUESTS, BATCH_BYTES = 100_000, 200e6  # the API's limits are 100,000 requests and 256 MB
USD_PER_REQUEST = 0.0002  # gold run: $0.016 for 40 calls at full price, halved; the smoke test measures it
# the masked check leaves these out: their fiscal-year labels are being fixed (2026-10-17)
MASK_SKIP = ["AVY", "CDNS", "DPZ", "ILMN", "JNJ", "LDOS", "LHX", "RVTY", "SNA", "SWK", "TDY", "TRMB", "TXT"]


def cell_grid(form: str = "10-K", k: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (filing, topic) from chunk metadata `k` (default: every chunk). 10-K: every topic,
    deposits and liquidity for Financials only. 10-Q: Financials only, Q_TOPICS only."""
    k = retrieve.chunks() if k is None else k
    k = k[(k.form == form) & ((k.sector == "Financials") if form == "10-Q" else True)]
    k = k.drop_duplicates(KEY[form])[list(dict.fromkeys(["ticker", "sector", "fy", *KEY[form]]))]
    c = k.merge(pd.DataFrame({"topic": Q_TOPICS if form == "10-Q" else list(TOPICS)}), how="cross")
    return c[(c.topic != "Deposits and liquidity") | (c.sector == "Financials")].reset_index(drop=True)


def top(c: pd.DataFrame, df: pd.DataFrame, vs: np.ndarray, bs: np.ndarray, form: str) -> list[dict]:
    """Each cell's hybrid top K within its own filing, ranked as retrieve.search ranks them: the top
    CANDIDATES by vector score `vs` and by BM25 score `bs` (both aligned with df's rows), fused with RRF."""
    ids, groups = df.chunk_id.to_numpy(), df.groupby(["form", *KEY[form]]).indices  # positions in df
    rows = []
    for r in c.to_dict("records"):
        idx = groups[(form, *[r[x] for x in KEY[form]])]
        n = min(retrieve.CANDIDATES, len(idx))
        v = idx[np.argsort(-vs[idx], kind="stable")[:n]]
        b = idx[np.argsort(-bs[idx], kind="stable")[:n]]
        fused = retrieve.rrf(list(ids[v]), list(ids[b[bs[b] > 0]]))  # BM25 drops chunks with no query word
        rows += [{**r, "rank": i, "chunk_id": cid, "rrf": s} for i, (cid, s) in enumerate(list(fused.items())[:K], 1)]
    return rows


def order_hash(ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def matches_chroma(v: np.ndarray, ids: pd.Series, n: int = 200, seed: int = 0) -> bool:
    """Whether n random rows of v are the Chroma embeddings of the chunks in those rows of chunks()."""
    pick = np.sort(np.random.default_rng(seed).choice(len(ids), min(n, len(ids)), replace=False))
    want = ids.iloc[pick].tolist()
    got = retrieve.collection().get(ids=want, include=["embeddings"])
    if sorted(got["ids"]) != sorted(want):
        return False
    e = pd.DataFrame(np.asarray(got["embeddings"], np.float32), index=got["ids"]).loc[want].to_numpy()
    return bool(np.allclose((e / np.linalg.norm(e, axis=1, keepdims=True) * v[pick]).sum(axis=1), 1, atol=1e-4))


def vectors() -> np.ndarray:
    """Read from Chroma once, then from disk. Rows follow chunks()'s order, which moves if chunks.parquet is
    rebuilt from a changed manifest, so the order is recorded (VECTOR_ORDER) and checked on every load."""
    ids = retrieve.chunks().chunk_id
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
    col, page = retrieve.collection(), 20_000
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


def retrieve_cells(form: str = "10-K") -> None:
    """Hybrid's top 3 for every cell, ranked as retrieve.search ranks them, but with each topic scored
    over all chunks once: search() takes about 1.5 seconds a call, about 17 hours for every cell.
    Vector scores here are exact cosine; search() asks Chroma's approximate (HNSW) index, so a few
    cells can differ. `check` measures how many."""
    passages, scores, state, _ = FILES[form]
    if scores.exists() or state.exists():  # scores point at passage row numbers, which a new retrieval would shuffle
        raise SystemExit(f"{scores} or {state} exists; move them before retrieving again")
    start = time.time()
    df = retrieve.chunks()
    vec, c = vectors(), cell_grid(form, df)
    rows = []
    for topic in c.topic.unique():
        q = np.asarray(retrieve.embed_query(TOPICS[topic]), np.float32)
        vs = vec @ (q / np.linalg.norm(q))
        bs = retrieve.bm25_index().get_scores(bm25s.tokenize([TOPICS[topic]], stopwords="en", return_ids=False,
                                                             show_progress=False)[0])
        rows += top(c[c.topic == topic], df, vs, bs, form)
        print(f"{topic}: {(c.topic == topic).sum():,} cells, {time.time() - start:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(passages, index=False)
    print(f"{passages}: {len(out):,} passages for {out.groupby([*KEY[form], 'topic']).ngroups:,} of "
          f"{len(c):,} cells")


def within(df: pd.DataFrame, vs: np.ndarray, bs: np.ndarray, topic: str, item: str | None) -> list[dict]:
    """Each 10-K's top K for `topic` among its chunks of `item` (None: the whole 10-K, as retrieve_cells ranks
    it). Scores `vs` and `bs` are over all of df's rows, so BM25 keeps the whole corpus's word weights."""
    m = np.ones(len(df), bool) if item is None else (df["item"] == item).to_numpy()
    c = cell_grid("10-K", df[m])  # only 10-Ks that have the item
    return top(c[c.topic == topic], df[m], vs[m], bs[m], "10-K")


def sections() -> None:
    """Section-level cells (PLAN.md Phase 7, 2026-10-18): for S_TOPICS, each 10-K's top K within S_ITEM, ranked
    as retrieve_cells ranks the whole 10-K, with the query vectors passages.parquet was ranked with (QUERIES),
    so nothing is embedded. First, the same code grouped by 10-K must reproduce passages.parquet."""
    passages, scores, state, _ = FILES["section"]
    if scores.exists() or state.exists():  # as in retrieve_cells: scores point at row numbers
        raise SystemExit(f"{scores} or {state} exists; move them before retrieving again")
    df, vec, old = retrieve.chunks(), vectors(), pd.read_parquet(PASSAGES)
    q, rows = dict(zip(TOPICS, np.load(QUERIES))), []
    for topic in S_TOPICS:
        vs = vec @ (q[topic] / np.linalg.norm(q[topic]))
        bs = retrieve.bm25_index().get_scores(bm25s.tokenize([TOPICS[topic]], stopwords="en", return_ids=False,
                                                             show_progress=False)[0])
        mine, want = pd.DataFrame(within(df, vs, bs, topic, None)), old[old.topic == topic].reset_index(drop=True)
        same = len(mine) == len(want) and (mine[want.columns].to_numpy() == want.to_numpy()).all()
        if not same:
            raise SystemExit(f"{topic}: grouped by 10-K, this code doesn't reproduce {PASSAGES}; nothing written")
        rows += within(df, vs, bs, topic, S_ITEM)
        print(f"{topic}: {len(want):,} passages of {PASSAGES} reproduced exactly; {len(rows):,} within Item {S_ITEM}")
    out = pd.DataFrame(rows)
    out.to_parquet(passages, index=False)
    print(f"{passages}: {len(out):,} passages for {out.groupby(['ticker', 'fy', 'topic']).ngroups:,} cells")


def check(n: int, form: str = "10-K", seed: int = 0) -> None:
    """n random cells' top 3 against retrieve.search's hybrid top 3 (section cells: searched within S_ITEM)."""
    p = pd.read_parquet(FILES[form][0])
    sec = "10-Q" if form == "10-Q" else "10-K"  # the SEC form; section cells are 10-K cells
    key, same = [*KEY[sec], "topic"], 0
    for _, r in p.drop_duplicates(key).sample(n, random_state=seed).iterrows():
        mine = p[(p[key] == r[key]).all(axis=1)].sort_values("rank").chunk_id.tolist()
        day = retrieve.chunks().filing_date.loc[mine[0]]  # a 10-Q is the ticker's only one filed that day
        one = {"fy": int(r.fy)} if sec == "10-K" else {"date_from": day, "date_to": day}
        if form == "section":
            one["item"] = S_ITEM
        theirs = retrieve.search(TOPICS[r.topic], k=K, rerank=False, ticker=r.ticker, form=sec,
                                 **one)["hybrid"].chunk_id.tolist()
        same += mine == theirs
        if mine != theirs:
            print(f"{' '.join(map(str, r[key]))}: {mine} vs search() {theirs}")
    print(f"{same}/{n} cells identical to retrieve.search")


def score_cells(budget: float, tickers: list[str] | None = None, form: str = "10-K", prompt=user_prompt) -> None:
    """Submit every passage not yet scored, in batches, then wait and collect. Failed requests aren't
    stored, so running again resubmits just those. `prompt` writes each user message (the masked check: masked_prompt).
    A passage whose (ticker, fy, topic, chunk_id) SCORES already has would be the same request: its score is copied."""
    passages, scores, saved, _ = FILES[form]
    client = anthropic.Anthropic()
    state = json.loads(saved.read_text()) if saved.exists() else []
    if not state:
        p = pd.read_parquet(passages).reset_index(names="pid")
        if tickers:
            p = p[p.ticker.isin(tickers)]
        if scores.exists():
            p = p[~p.pid.isin(pd.read_parquet(scores, columns=["pid"]).pid)]
        key, copied = ["ticker", "fy", "topic", "chunk_id"], pd.DataFrame()
        if prompt is user_prompt and form != "rerun" and SCORES.exists():  # B6 scores the same requests again on purpose
            done = pd.read_parquet(PASSAGES, columns=key).reset_index(names="pid").merge(pd.read_parquet(SCORES), on="pid")
            copied = p[["pid", *key]].merge(done.drop(columns="pid"), on=key, validate="1:1").drop(columns=key).assign(usd=0.0)
            p = p[~p.pid.isin(copied.pid)]
        est = len(p) * USD_PER_REQUEST
        print(f"{len(p):,} passages to score, about ${est:.2f}; {len(copied):,} scores copied from {SCORES}")
        if est > budget:
            raise SystemExit(f"over --budget ${budget:.2f}; nothing submitted or copied")
        if len(copied):  # usd 0: nothing was paid for these
            pd.concat([pd.read_parquet(scores), copied] if scores.exists() else [copied]).to_parquet(scores, index=False)
        # only these chunks' texts, not all 674k
        rows = pd.read_parquet(CHUNKS, filters=[("chunk_id", "in", p.chunk_id.unique().tolist())]).set_index("chunk_id", drop=False)
        system, fmt = system_prompt(), transform_schema(schema(list(DIMENSIONS)))
        batch, size = [], 0
        for r in [*p.itertuples(), None]:  # None flushes the last batch
            req = None if r is None else Request(custom_id=f"p{r.pid}", params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=4000, output_config={"effort": "low", "format": {"type": "json_schema", "schema": fmt}},
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                messages=[{"role": "user", "content": prompt(rows.loc[r.chunk_id], r.topic, list(DIMENSIONS))}]))
            n = len(json.dumps(req)) if req else 0
            if batch and (req is None or len(batch) >= BATCH_REQUESTS or size + n > BATCH_BYTES):
                while True:
                    try:
                        state.append(client.messages.batches.create(requests=batch).id)
                        break
                    except anthropic.RateLimitError:  # the queue is full of our own batches: wait for some to finish
                        print("batch queue full; retrying in 5 minutes", flush=True)
                        time.sleep(300)
                saved.write_text(json.dumps(state))
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
            old = pd.read_parquet(scores) if scores.exists() else None
            if len(new) and old is not None:  # collected again after a crash between saving scores and state: keep the first copy
                new = new[~new.pid.isin(old.pid)]
            if len(new):
                pd.concat([old, new] if old is not None else [new]).to_parquet(scores, index=False)
            state.remove(bid)
            saved.write_text(json.dumps(state)) if state else saved.unlink()
            usd = new.usd.sum() if len(new) else 0.0
            print(f"{bid}: {len(out):,} scored ({len(out) - len(new):,} already saved), {failed} failed, ${usd:.3f} "
                  f"(${usd / max(1, len(new)):.5f} a request)", flush=True)
        if state:
            time.sleep(60)
    s = pd.read_parquet(scores) if scores.exists() else pd.DataFrame({"usd": []})  # none saved: every request failed
    print(f"{scores}: {len(s):,} passages scored, ${s.usd.sum():.2f} in all")


def rerun(budget: float, n: int = 5000, seed: int = 0) -> None:
    """B6: score a random sample of the annual passages again with the same request, then compare the
    two runs as the pilot did. The sample is drawn once and kept, so a stopped run resumes."""
    passages, scores, _, _ = FILES["rerun"]
    if not passages.exists():
        pd.read_parquet(PASSAGES).reset_index(names="orig_pid").sample(n, random_state=seed).to_parquet(passages, index=False)
    score_cells(budget, form="rerun")
    from sec_filings_tone.sentiment import kappa_ci

    a = pd.read_parquet(passages).reset_index(names="pid").merge(pd.read_parquet(scores), on="pid")
    b = pd.read_parquet(SCORES).rename(columns={"pid": "orig_pid"})
    m = a.merge(b, on="orig_pid", suffixes=("", "_first"))
    print(f"{len(m):,} passages scored twice; off-topic flag the same for {(m.off_topic == m.off_topic_first).mean():.1%}")
    both = m[~m.off_topic & ~m.off_topic_first]
    for d, labels in DIMENSIONS.items():
        print(f"  {d:12} identical {(both[d] == both[d + '_first']).mean():.0%}, weighted kappa "
              "{:.2f} [{:.2f}, {:.2f}]".format(*kappa_ci(both[d], both[d + "_first"], labels, n_boot=500)))
    by = both.groupby("sector")
    print("identical share by sector:\n" + pd.DataFrame({d: by.apply(lambda g, d=d: (g[d] == g[d + "_first"]).mean())
                                                         for d in DIMENSIONS}).round(2).to_string())
    noise = {d: (both[d] - both[d + "_first"]).var() / 2 for d in DIMENSIONS}  # one run's scorer variance per passage
    print("scorer noise variance per passage: " + ", ".join(f"{d} {v:.3f}" for d, v in noise.items()))


def masked_prompt(row: pd.Series, topic: str, dims: list[str]) -> str:
    """sentiment.user_prompt with the company and fiscal year taken out: the Filing line keeps only the
    form and section ("Filing: a 10-K, Item 1A Risk Factors"). Every other character is the same."""
    p, hint = user_prompt(row, topic, dims), f"\nFiling: {row['ticker']} {row['form']} {period(row)}, "
    if hint not in p:  # the production prompt changed: stop rather than send the hint
        raise ValueError("sentiment.user_prompt's Filing line changed; update masked_prompt")
    return p.replace(hint, f"\nFiling: a {row['form']}, ", 1)


def masked_sample(p: pd.DataFrame, ev: pd.DataFrame, n: int = 340, seed: int = 0) -> pd.DataFrame:
    """The masked check's passages. For each event in `ev` (events.csv): each company's "before" and "after" cell on the
    event's topics, picked as events.changes picks them, with only the passages its test uses (the
    event's items), for companies with such passages in both cells and not in MASK_SKIP. Whole companies
    are drawn at random until the event has about n passages: up to two thirds from the affected group
    (all of it if that's fewer), the rest from the others, or all n from the affected group if there are
    no others. `p` is the annual passages with orig_pid, item, filing_date and report_date."""
    from sec_filings_tone.events import affected

    p, key, out = p[~p.ticker.isin(MASK_SKIP)], ["ticker", "fy", "topic"], []
    for e in ev.itertuples():
        c = p[p.topic.isin(e.topics.split(";"))]
        cell = c.drop_duplicates(key).sort_values("filing_date")
        before = cell[cell.filing_date < e.date].drop_duplicates(["ticker", "topic"], keep="last")
        after = cell[cell.report_date >= (e.after or e.date)].drop_duplicates(["ticker", "topic"], keep="first")
        x = pd.concat([c.merge(before[key]).assign(cell="before"), c.merge(after[key]).assign(cell="after")])
        x = x[x["item"].isin(e.items.split(";"))] if e.items else x
        x = x[x.groupby(["ticker", "topic"]).cell.transform("nunique") == 2]
        co = x.groupby("ticker").agg(n=("orig_pid", "size"), sector=("sector", "first"))
        co = co.assign(group=np.where(affected(e, co.sector), "affected", "others")).sample(frac=1, random_state=seed)
        a, o = co[co.group == "affected"], co[co.group == "others"]
        a = a[a.n.cumsum() <= (n * 2 // 3 if len(o) else n)]
        o = o[o.n.cumsum() <= n - a.n.sum()]
        out.append(x[x.ticker.isin([*a.index, *o.index])].assign(event=e.event, group=lambda d: d.ticker.map(co.group)))
    return pd.concat(out, ignore_index=True)


def leaks(p: pd.DataFrame) -> pd.DataFrame:
    """Per passage of `p`, whether its own text gives away what the masked header hides: its company (the
    ticker, if two letters or more, or the name in data/companies.csv without Inc., Group and the like,
    as written there or in capitals), a year from 2019 to 2026, COVID, or another word dated to the period."""
    from sec_filings_tone.events import COMPANIES

    text = pd.read_parquet(CHUNKS, columns=["chunk_id", "text"], filters=[("chunk_id", "in", p.chunk_id.unique().tolist())]
                           ).set_index("chunk_id").text.loc[p.chunk_id].reset_index(drop=True)
    name = pd.read_csv(COMPANIES).set_index("ticker").name.str.replace(
        r"(,? (Inc\.?|Corp\.|Corporation|Company|Co\.|plc|Holdings|Group|Ltd\.)| \(Class [ABC]\))+$", "", regex=True)
    words = [rf"(?<!\w)(?:{re.escape(t) + '|' if len(t) > 1 else ''}{re.escape(name[t])}|{re.escape(name[t].upper())})(?!\w)"
             for t in p.ticker]
    return pd.DataFrame({
        "company": [bool(re.search(w, s)) for w, s in zip(words, text)],
        "year": text.str.contains(r"\b20(?:19|2[0-6])\b"),
        "covid": text.str.contains(r"(?i)pandemic|covid|coronavirus|sars-cov"),
        "other dated word": text.str.contains(r"(?i)ukraine|silicon valley bank|chatgpt|generative ai|"
                                              r"inflation reduction act|chips (?:and science )?act\b")}).set_axis(p.index)


def masked(budget: float, n: int = 340, seed: int = 0) -> None:
    """Masked check: do the event results come partly from the company and fiscal year in the prompt, which the
    model can date, rather than from the text? The passages the annual event tests compare
    (masked_sample, drawn once and kept, so a stopped run resumes) are scored again with masked_prompt
    and compared with their original scores (masked_report). The sample, how often its text names the
    company or the period anyway, and the cost are printed first; with --budget 0 nothing is sent."""
    from sec_filings_tone.events import EVENTS

    passages = FILES["masked"][0]
    if not passages.exists():
        p = pd.read_parquet(PASSAGES).reset_index(names="orig_pid").merge(
            pd.read_parquet(CHUNKS, columns=["chunk_id", "item", "filing_date", "report_date"]), on="chunk_id")
        masked_sample(p, pd.read_csv(EVENTS, dtype=str, keep_default_na=False), n, seed).to_parquet(passages, index=False)
    s = pd.read_parquet(passages)
    print(s.groupby(["event", "group"], sort=False).agg(companies=("ticker", "nunique"), passages=("orig_pid", "size")).to_string())
    leak = leaks(s).assign(any=lambda d: d.any(axis=1))
    print("\nshare of passages whose text names the company or the period anyway:\n"
          + leak.groupby([s.event, s.cell], sort=False).mean().round(2).to_string()
          + f"\nall {len(s):,}: " + ", ".join(f"{k} {v:.0%}" for k, v in leak.mean().items()))
    score_cells(budget, form="masked", prompt=masked_prompt)
    masked_report()


def masked_report() -> None:
    """The masked check on the same passages, masked scores against the original ones: the off-topic flag, and per
    dimension on passages both call on topic the share identical and weighted kappa [95% CI], beside
    B6's second run against the first (the noise floor: the hint matters only if masking agrees clearly
    less). Then each event's change where it hit, and its difference from the others, with each scoring,
    on the companies whose change exists with both."""
    from sec_filings_tone.events import EVENTS, affected, changes, did
    from sec_filings_tone.sentiment import kappa_ci

    dims, first = list(DIMENSIONS), pd.read_parquet(SCORES).rename(columns={"pid": "orig_pid"})

    def paired(form):  # a sample's new scores beside its original ones (suffix _orig)
        passages, scores, _, _ = FILES[form]
        p = pd.read_parquet(passages).reset_index(names="pid").merge(pd.read_parquet(scores), on="pid")
        return p.merge(first, on="orig_pid", suffixes=("", "_orig"))

    def agree(m):
        both = m[~m.off_topic & ~m.off_topic_orig]
        row = {"passages": len(m), "off topic same": f"{(m.off_topic == m.off_topic_orig).mean():.1%}", "both on topic": len(both)}
        for d, labels in DIMENSIONS.items():
            k, lo, hi = kappa_ci(both[d], both[d + "_orig"], labels, n_boot=500)
            row[d] = f"{(both[d] == both[d + '_orig']).mean():.0%} {k:.2f} [{lo:.2f}, {hi:.2f}]"
        return row

    m, b6 = paired("masked"), paired("rerun")
    clean = ~leaks(m).any(axis=1)
    print(f"\n{len(m):,} of {len(pd.read_parquet(FILES['masked'][0])):,} passages scored masked. Per dimension: share identical, "
          "weighted kappa [95% CI], on passages both scorings call on topic")
    print(pd.DataFrame({"masked vs original": agree(m), "  text names no company or date": agree(m[clean]),
                        "B6 rerun vs original": agree(b6), "  on these topics": agree(b6[b6.topic.isin(m.topic)])}).T.to_string())

    f, rows, pooled = "{:+.2f} [{:+.2f}, {:+.2f}]".format, [], []
    for e in pd.read_csv(EVENTS, dtype=str, keep_default_na=False).itertuples():
        ch = {}
        for name, sfx in [("original", "_orig"), ("masked", "")]:  # each cell's tone as events.cells has it
            x = m[m.event == e.event]
            x = x.assign(**{c: x[c + sfx] for c in ["off_topic", *dims]})
            c = x.groupby(["ticker", "fy", "topic", "sector"], as_index=False)[["filing_date", "report_date"]].first()
            c = c.merge(x[~x.off_topic].groupby(["ticker", "fy", "topic"], as_index=False)[dims].mean(), how="left")
            ch[name] = changes(c.assign(discussed=c[dims[0]].notna().astype(float)), e.date, e.topics.split(";"), e.after,
                                 filed=True)  # the sample was drawn under the filed rule
        o, k = ch["original"], ch["masked"]
        hit = affected(e, o.sector)
        for d in [*dims, "discussed"]:
            same = o[d].notna() & k[d].notna()
            a, u, none = same & hit, same & ~hit, o[d][:0]
            if not a.any():
                continue
            row = {"event": e.event, "measure": d, "expected": getattr(e, d, "") or "·", "companies": a.sum(),
                   "change, original": f(*did(o[d][a], none)), "change, masked": f(*did(k[d][a], none)),
                   "masked - original": f(*did((k[d] - o[d])[a], none))}
            if u.any():
                row |= {"others": u.sum(), "vs others, original": f(*did(o[d][a], o[d][u])),
                        "vs others, masked": f(*did(k[d][a], k[d][u])),
                        "vs others, masked - original": f(*did((k[d] - o[d])[a], (k[d] - o[d])[u]))}
            rows.append(row)
        pooled.append((k[dims] - o[dims])[hit])
    r = pd.DataFrame(rows)
    for event, g in r.groupby("event", sort=False):
        print(f"\n{event}\n" + g.drop(columns="event").dropna(axis=1, how="all").to_string(index=False, float_format="{:.0f}".format))
    pooled = pd.concat(pooled)
    print("\nall events, affected companies, masked - original change: "
          + ", ".join(f"{d} {f(*did(pooled[d], pooled[d][:0]))}" for d in dims))


def baselines(form: str = "10-K", part: int = 5000) -> None:
    """FinBERT, Loughran-McDonald and the rule-based counts for every passage, locally and free. They
    score the whole passage, not a topic, so each chunk is scored once however many cells it's in.
    Saved every `part` chunks, so a stopped run resumes."""
    passages, _, _, out = FILES[form]
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame({"chunk_id": pd.Series(dtype=str)})
    seen = set(done.chunk_id)
    todo = [c for c in pd.read_parquet(passages).chunk_id.unique() if c not in seen]
    # only these texts: all 674k chunks with FinBERT on top pushed a 16 GB machine into swap
    text = pd.read_parquet(CHUNKS, columns=["chunk_id", "text"], filters=[("chunk_id", "in", todo)]).set_index("chunk_id").text
    start = time.time()
    for k in range(0, len(todo), part):
        ids = todo[k:k + part]
        t = text.loc[ids].tolist()
        b = pd.concat([finbert(t), pd.DataFrame([lm(x) | proxies(x) for x in t])], axis=1)
        b.insert(0, "chunk_id", ids)
        done = pd.concat([done, b], ignore_index=True)
        done.to_parquet(out, index=False)
        print(f"{len(done):,} of {len(done) + len(todo) - k - len(ids):,} chunks, {time.time() - start:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["retrieve", "check", "score", "baselines", "rerun", "masked"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--tickers", nargs="+")
    p.add_argument("--budget", type=float, default=0.5, help="USD; nothing is submitted if the estimate is over it")
    p.add_argument("--quarterly", action="store_true", help="B5: Financials' 10-Q cells, in the q_ files")
    p.add_argument("--section", action="store_true", help="AI within each 10-K's MD&A (sections), in the s_ files")
    a = p.parse_args()
    form = "10-Q" if a.quarterly else "section" if a.section else "10-K"
    if a.command == "retrieve" and a.section:
        sections()
    elif a.command == "retrieve":
        retrieve_cells(form)
    elif a.command == "check":
        check(a.n, form)
    elif a.command == "baselines":
        baselines(form)
    elif a.command == "rerun":
        rerun(a.budget)
    elif a.command == "masked":
        masked(a.budget)
    else:
        score_cells(a.budget, a.tickers, form)


if __name__ == "__main__":
    main()
