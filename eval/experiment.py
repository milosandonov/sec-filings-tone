"""Phase 4 experiment: chunking and embedding variants on the pilot, each scored with the retrieval eval.

Each variant changes one thing from the baseline (500-token chunks, the previous chunk's last sentence
as overlap, the full context header, text-embedding-3-large). Nothing paid is ever redone:
- document vectors are saved at full size, one file per API batch, named by a hash of the batch's
  texts, so a crash resumes where it stopped and smaller sizes are scored by truncating;
- every ranking (top 50 per stage) is saved, so new metrics are computed from disk;
- vector search is exact cosine over the saved vectors (Chroma matched it on all 30 queries up to a
  float32 near-tie), so differences come from the embeddings, not from the index.
BM25, RRF, the reranker and the filters are the production functions in retrieve.py.

Usage:
    uv run python eval/experiment.py --dry-run                # chunks, tokens, cost, unreachable keywords
    uv run python eval/experiment.py --smoke                  # every variant on TXN only, plus API checks
    uv run python eval/experiment.py                          # embed what's missing, rank, score
    uv run python eval/experiment.py --variants base oai_small  # some variants only
    uv run python eval/experiment.py --score-only             # metrics from saved rankings, no API calls
    uv run python eval/experiment.py --multi                  # pilot + one company per other sector, more questions
    uv run python eval/experiment.py --multi --variants oss_bge_m3 --no-rerank   # vector, BM25, hybrid only
    uv run python eval/experiment.py --budget 6 --max-minutes 60   # caps (defaults: $8 of new spend, 90 minutes)
"""

import argparse
import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from eval import K, TESTS, Cap, CapReached, contains, rerank_call_usd, score
from sec_filings_tone import retrieve
from sec_filings_tone.ingest import build_chunks, embed_text, n_tokens

PILOT = ["AMAT", "AMD", "AVGO", "INTC", "KLAC", "LRCX", "MU", "NVDA", "QCOM", "TXN"]
# --multi: one large company from each other sector (BAC for Financials: JPM's 10-K MD&A is over ingest's size cap)
EXTRA = ["AMZN", "BAC", "CAT", "GOOGL", "LIN", "NEE", "PG", "PLD", "UNH", "XOM"]
MULTI_VARIANTS = ["combo_head", "combo_head_small", "overlap_none"]  # winner, small model, header on without overlap
BASE = {"target": 500, "headings": True, "overlap_sentences": 1, "overlap_tokens": 60, "header": "full",
        "model": "text-embedding-3-large"}
# Headings set most chunk boundaries (median ~350 tokens at any target), so size is tested with
# heading splits off ("flat"), and flat_500 vs base tests the heading rule itself.
VARIANTS = {
    "base": {},
    "flat_500": {"headings": False},
    "flat_300": {"headings": False, "target": 300},
    "flat_800": {"headings": False, "target": 800},
    "flat_1200": {"headings": False, "target": 1200},
    "overlap_none": {"overlap_tokens": 0},
    "overlap_2": {"overlap_sentences": 2, "overlap_tokens": 120},
    "header_none": {"header": "none"},
    "header_no_heading": {"header": "no_heading"},
    "oai_small": {"model": "text-embedding-3-small"},
    # combinations of the single-factor winners (no header, no overlap), with heading splits on and off
    # and with both models: decides the heading question for topic-level tone, and large vs small
    "combo_flat": {"headings": False, "header": "none", "overlap_tokens": 0},
    "combo_flat_small": {"headings": False, "header": "none", "overlap_tokens": 0, "model": "text-embedding-3-small"},
    "combo_head": {"header": "none", "overlap_tokens": 0},
    "combo_head_small": {"header": "none", "overlap_tokens": 0, "model": "text-embedding-3-small"},
    # open models with the production text (plain, heading-aligned chunks)
    "oss_qwen3": {"header": "none", "overlap_tokens": 0, "model": "Qwen/Qwen3-Embedding-0.6B"},
    "oss_bge_m3": {"header": "none", "overlap_tokens": 0, "model": "BAAI/bge-m3"},
}
# full size saved, sizes scored, USD per 1M input tokens (checked 2026-10-12)
MODELS = {
    "text-embedding-3-large": {"dims": 3072, "score_dims": [256, 1024, 3072], "price": 0.13},
    "text-embedding-3-small": {"dims": 1536, "score_dims": [512, 1024, 1536], "price": 0.02},
    # open models, run locally (free apart from compute time); both take inputs well past our 800 tokens
    "Qwen/Qwen3-Embedding-0.6B": {"dims": 1024, "score_dims": [256, 1024], "price": 0.0, "local": True},
    "BAAI/bge-m3": {"dims": 1024, "score_dims": [1024], "price": 0.0, "local": True},  # not trained for truncation
}
MAIN_DIMS = 1024  # what Chroma stores; variants are compared here
DEPTH = 50  # ranks saved per stage, so recall@k is computable up to 50
RERANK_REPEATS = 2  # the reranker isn't deterministic; two runs show how much it moves
RERANK_PRICE = 0.10  # USD per 1M input tokens, claude-haiku-5-5
TOKEN_BUDGETS = (2000, 5000)  # keyword coverage within this much ranked text, fair across chunk sizes
API_TIMEOUT = 60  # seconds per request, so a stuck call fails instead of hanging
CAP = Cap()  # main() sets the run's budget and time limit; every paid call checks it first


def config(name: str) -> dict:
    c = BASE | VARIANTS[name]
    return c | {"table": (c["target"], c["headings"])}


def reference(name: str) -> str:
    """What a variant is compared with: flat sizes with flat_500, everything else with base."""
    return "flat_500" if name.startswith("flat_") and name != "flat_500" else "base"


def chunk_table(table: tuple[int, bool], out: Path, tickers: list[str]) -> pd.DataFrame:
    target, headings = table
    path = out / f"chunks_{target}{'' if headings else '_flat'}.parquet"
    if not path.exists():
        # same max/target (1.6) and heading-rule/target (0.2) ratios as the 800/500/100 baseline
        build_chunks(tickers, round(target * 1.6), target, path, round(target * 0.2) if headings else 10**9)
    # embed_text needs each section's chunks in seq order; vectors are stored in this order
    df = pd.read_parquet(path).sort_values(["ticker", "form", "item", "report_date", "seq"])
    return df.set_index("chunk_id", drop=False)


def batches(texts: list[str], model: str) -> list[list[int]]:
    """OpenAI takes up to 2,048 inputs and 300k tokens a request; local models get 100 texts at a time."""
    out, cur, tok = [], [], 0
    cap, size = (200_000, 1000) if model.startswith("text-embedding") else (10**9, 100)
    for i, t in enumerate(texts):
        tok += n_tokens(t)
        cur.append(i)
        if len(cur) == size or tok > cap:
            out.append(cur)
            cur, tok = [], 0
    return out + [cur] if cur else out


@cache
def local_model(name: str):
    from sentence_transformers import SentenceTransformer
    import torch

    m = SentenceTransformer(name, device="mps" if torch.backends.mps.is_available() else "cpu")
    m.max_seq_length = 1024  # chunks are at most 800 cl100k tokens
    return m


def embed(texts: list[str], model: str, query: bool = False) -> np.ndarray:
    """Full-size vectors, one per text, in order."""
    if model.startswith("text-embedding"):
        from openai import OpenAI

        data = OpenAI(timeout=API_TIMEOUT, max_retries=3).embeddings.create(model=model, input=texts).data
        vecs = [d.embedding for d in data]
    else:  # a local model
        m = local_model(model)
        # Qwen3 expects an instruction on queries (its "query" prompt); documents get none
        prompt = "query" if query and "query" in m.prompts else None
        vecs = m.encode(texts, batch_size=8, prompt_name=prompt, convert_to_numpy=True)
    out = np.array(vecs, dtype=np.float32)
    assert out.shape == (len(texts), MODELS[model]["dims"]), out.shape
    return out


def doc_vectors(name: str, df: pd.DataFrame, out: Path) -> np.ndarray:
    c = config(name)
    texts = embed_text(df, c["overlap_tokens"], c["overlap_sentences"], c["header"])
    folder = out / f"emb_{name}"
    folder.mkdir(exist_ok=True)
    parts = []
    for i, idx in enumerate(b := batches(texts, c["model"])):
        batch = [texts[j] for j in idx]
        f = folder / f"part_{i:04d}_{hashlib.sha1(chr(0).join(batch).encode()).hexdigest()[:10]}.npy"
        if not f.exists():
            usd = sum(map(n_tokens, batch)) / 1e6 * MODELS[c["model"]]["price"]
            CAP.check(usd)
            np.save(f, embed(batch, c["model"]))
            CAP.charge(usd)
            print(f"  {name}: batch {i + 1}/{len(b)} embedded")
        parts.append(np.load(f))
    return np.concatenate(parts)


def vectors(name: str, df: pd.DataFrame, out: Path) -> np.ndarray:
    """In --multi, the pilot companies' chunks are identical to the pilot run's, so their saved vectors
    are reused and only the other companies are embedded."""
    if out.name != "multi":
        return doc_vectors(name, df, out)
    pilot = df.ticker.isin(PILOT).to_numpy()
    ref = chunk_table(config(name)["table"], out.parent, PILOT)
    assert ref.chunk_id.tolist() == df.chunk_id[pilot].tolist() and (ref.text.to_numpy() == df.text[pilot].to_numpy()).all()
    E = np.empty((len(df), MODELS[config(name)["model"]]["dims"]), dtype=np.float32)
    E[pilot] = doc_vectors(name, ref, out.parent)
    E[~pilot] = doc_vectors(name, df[~pilot], out)
    return E


def query_vectors(model: str, questions: list[str], out: Path) -> np.ndarray:
    path = out / f"queries_{model.replace('/', '_')}.pkl"
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    if missing := [q for q in questions if q not in cache]:
        usd = sum(map(n_tokens, missing)) / 1e6 * MODELS[model]["price"]
        CAP.check(usd)
        cache |= dict(zip(missing, embed(missing, model, query=True)))
        CAP.charge(usd)
        path.write_bytes(pickle.dumps(cache))
    return np.array([cache[q] for q in questions])


def unit(m: np.ndarray, dims: int) -> np.ndarray:
    m = m[..., :dims]
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def rank(name: str, df: pd.DataFrame, E: np.ndarray, Q: np.ndarray, tests: list[dict], rerank: bool = True) -> pd.DataFrame:
    """Every stage's ranking for every test and scored size; rerank only at MAIN_DIMS."""
    retrieve.chunks = lambda: df  # filters, BM25 and rows() now use this variant's chunks
    retrieve.bm25_index.cache_clear()
    model = config(name)["model"]
    rows = []

    def add(dims, test, stage, ids, scores, repeat=0):
        rows.extend({"variant": name, "dims": dims, "test": test, "stage": stage, "repeat": repeat,
                     "rank": r, "chunk_id": c, "score": float(s)} for r, (c, s) in enumerate(zip(ids, scores), 1))

    jobs = []  # (test, repeat, question, candidates) for the reranker, run in parallel at the end
    for ti, t in enumerate(tests):
        mask = retrieve.subset(t["filters"])[0]
        n = min(DEPTH, int(mask.sum()))
        bm_ids, bm_scores = retrieve.bm25_search(t["question"], mask, n)
        for dims in MODELS[model]["score_dims"]:
            sims = unit(E, dims) @ unit(Q[ti], dims)
            sims[~mask] = -np.inf
            top = np.argsort(-sims)[:n]
            vec_ids = df.chunk_id.iloc[top].tolist()
            # production fuses each search's top CANDIDATES (retrieve.search)
            fused = retrieve.rrf(vec_ids[:retrieve.CANDIDATES], bm_ids[:retrieve.CANDIDATES])
            add(dims, ti, "vector", vec_ids, sims[top])
            add(dims, ti, "bm25", bm_ids, bm_scores)
            add(dims, ti, "hybrid", list(fused), list(fused.values()))
            if dims == MAIN_DIMS and rerank:
                cands = retrieve.rows(list(fused)[:retrieve.CANDIDATES], list(fused.values())[:retrieve.CANDIDATES])
                jobs += [(ti, rep, t["question"], cands) for rep in range(RERANK_REPEATS)]
    def rerank(job):
        CAP.check(rerank_call_usd(job[3].n_tokens.sum() + 1000))  # + prompt and headers
        return retrieve.llm_rerank(job[2], job[3])

    with ThreadPoolExecutor(8) as pool:
        for (ti, rep, _, _), rr in zip(jobs, pool.map(rerank, jobs)):
            add(MAIN_DIMS, ti, "rerank", rr.chunk_id.tolist(), rr.score.tolist(), rep)
    return pd.DataFrame(rows)


def metrics(runs: pd.DataFrame, tables: dict[int, pd.DataFrame], tests: list[dict]) -> pd.DataFrame:
    """One row per variant, size, stage, repeat and test."""
    out, relevant = [], {}
    for (name, dims, stage, rep, ti), g in runs.groupby(["variant", "dims", "stage", "repeat", "test"]):
        df, t = tables[config(name)["table"]], tests[ti]
        key = (config(name)["table"], ti)
        if key not in relevant:
            pool = df.text[retrieve_mask(df, t)]
            relevant[key] = {kw: int(pool.map(lambda x: contains(x, kw)).sum()) for kw in t["keywords"]}
        hit = df.loc[g.sort_values("rank").chunk_id]
        texts, toks = hit.text.tolist(), hit.n_tokens.to_numpy()
        m = score(texts, t["keywords"], relevant[key])
        m["recall@20"] = np.mean([any(contains(x, kw) for x in texts[:20]) for kw in t["keywords"]])
        for b in TOKEN_BUDGETS:  # read ranked chunks until the budget runs out (at least one)
            n = max(1, int((np.cumsum(toks) <= b).sum()))
            m[f"cov@{b // 1000}k_tok"] = np.mean([any(contains(x, kw) for x in texts[:n]) for kw in t["keywords"]])
        m["unreachable"] = sum(v == 0 for v in relevant[key].values())
        out.append({"variant": name, "dims": dims, "stage": stage, "repeat": rep, "test": ti,
                    "category": t["category"], **m})
    return pd.DataFrame(out)


def retrieve_mask(df: pd.DataFrame, test: dict) -> np.ndarray:
    retrieve.chunks = lambda: df
    return retrieve.subset(test["filters"])[0]


def bootstrap(diff: np.ndarray, n: int = 10_000) -> tuple[float, float]:
    """95% interval for the mean of per-test differences, resampling tests."""
    rng = np.random.default_rng(0)
    means = rng.choice(diff, (n, len(diff))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(m: pd.DataFrame, names: list[str]) -> None:
    per_test = m.groupby(["variant", "dims", "stage", "test", "category"]).mean(numeric_only=True).reset_index()
    main = per_test[per_test.dims == MAIN_DIMS]
    cols = ["mrr", "ndcg", "coverage", "recall@20", "cov@2k_tok", "cov@5k_tok"]
    print(f"\n## Vector search at {MAIN_DIMS} dims ({main.test.nunique()} tests); MRR difference vs the reference "
          "variant (flat_500 for flat sizes, base otherwise) with a 95% bootstrap interval over tests\n")
    print("| Variant | MRR | nDCG@10 | Coverage@10 | Recall@20 | Cov@2k tok | Cov@5k tok | ΔMRR vs reference |\n"
          "|---|---|---|---|---|---|---|---|")
    vec = main[main.stage == "vector"].set_index(["variant", "test"])
    for name in names:
        r = vec.loc[name][cols].mean()
        ref = reference(name) if reference(name) in names else "combo_head"
        if ref in names and ref != name:
            d = (vec.loc[name].mrr - vec.loc[ref].mrr).to_numpy()
            lo, hi = bootstrap(d)
            delta = f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] vs {ref}"
        else:
            delta = "reference"
        print(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {delta} |")

    print(f"\n## MRR by stage at {MAIN_DIMS} dims (rerank averaged over {RERANK_REPEATS} runs)\n")
    st = main.pivot_table(index="variant", columns="stage", values="mrr").reindex(names)
    st = st[[s for s in retrieve.STAGES if s in st.columns]]
    print("| Variant | " + " | ".join(st.columns) + " |\n|---|" + "---|" * len(st.columns))
    for name, r in st.iterrows():
        print(f"| {name} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    rr = m[(m.stage == "rerank")].pivot_table(index=["variant", "test"], columns="repeat", values="mrr")
    if rr.shape[1] > 1:
        print(f"\nReranker run-to-run spread: mean |MRR run 1 - run 2| per test = {(rr[0] - rr[1]).abs().mean():.3f}")

    print("\n## Vector MRR by embedding size\n")
    sizes = per_test[per_test.stage == "vector"].pivot_table(index="variant", columns="dims", values="mrr").reindex(names)
    print("| Variant | " + " | ".join(str(c) for c in sizes.columns) + " |\n|---|" + "---|" * len(sizes.columns))
    for name, r in sizes.iterrows():
        print(f"| {name} | " + " | ".join("" if np.isnan(v) else f"{v:.3f}" for v in r) + " |")

    print(f"\n## Vector MRR by question category at {MAIN_DIMS} dims\n")
    cat = main[main.stage == "vector"].pivot_table(index="variant", columns="category", values="mrr").reindex(names)
    print("| Variant | " + " | ".join(cat.columns) + " |\n|---|" + "---|" * len(cat.columns))
    for name, r in cat.iterrows():
        print(f"| {name} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    if (bad := m[m.unreachable > 0][["variant", "test"]].drop_duplicates()).size:
        print(f"\nwarning: keywords no chunk contains (split by a chunk boundary?): {bad.values.tolist()}")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--multi", action="store_true")
    p.add_argument("--no-rerank", action="store_true", help="skip the Claude reranking stage")
    p.add_argument("--budget", type=float, default=8.0, help="USD of new spend; stop before a call would pass it")
    p.add_argument("--max-minutes", type=float, default=90)
    a = p.parse_args()
    global CAP
    CAP = Cap(a.budget, a.max_minutes)

    # smoke: the smallest pilot company, so every code path runs for about $0.20
    tickers, out = (["TXN"], Path("data/experiment/smoke")) if a.smoke else (PILOT, Path("data/experiment"))
    if a.multi:
        tickers, out = PILOT + EXTRA, Path("data/experiment/multi")
        if a.variants == list(VARIANTS):
            a.variants = MULTI_VARIANTS
    out.mkdir(parents=True, exist_ok=True)
    tests = [json.loads(line) for line in TESTS.read_text().splitlines() if line.strip()]
    if not a.multi:  # the first 30 are the pilot's questions; the other 20 need --multi's companies
        tests = tests[:30]
    if a.smoke:  # only questions answerable from TXN alone
        tests = [t for t in tests if t["filters"].get("ticker") == "TXN"]
    tables = {t: chunk_table(t, out, tickers) for t in sorted({config(n)["table"] for n in a.variants})}

    if a.dry_run:
        total = 0.0
        print("| Variant | Chunks | Tokens to embed | Cost | Unreachable keywords |\n|---|---|---|---|---|")
        for name in a.variants:
            c, df = config(name), tables[config(name)["table"]]
            tok = sum(map(n_tokens, embed_text(df, c["overlap_tokens"], c["overlap_sentences"], c["header"])))
            cost = tok / 1e6 * MODELS[c["model"]]["price"]
            total += cost
            miss = [(i, kw) for i, t in enumerate(tests) for kw in t["keywords"]
                    if not (retrieve_mask(df, t) & df.text.map(lambda x: contains(x, kw)).to_numpy()).any()]
            print(f"| {name} | {len(df):,} | {tok:,} | ${cost:.2f} | {miss or 'none'} |")
        rr = sum(len(tests) * RERANK_REPEATS * 20 * df.n_tokens.mean() / 1e6 * RERANK_PRICE
                 for df in (tables[config(n)["table"]] for n in a.variants))
        print(f"\nembedding total ${total:.2f}; reranking about ${rr:.2f} "
              f"({len(tests)} questions x {RERANK_REPEATS} runs x {len(a.variants)} variants)")
        return

    try:
        if a.smoke:
            check_apis({config(n)["model"] for n in a.variants})
        for name in a.variants:
            path = out / f"runs_{name}.parquet"
            if path.exists() or a.score_only:
                continue
            df = tables[config(name)["table"]]
            E = vectors(name, df, out)
            Q = query_vectors(config(name)["model"], [t["question"] for t in tests], out)
            rank(name, df, E, Q, tests, not a.no_rerank).to_parquet(path, index=False)
            print(f"{name}: ranked, saved {path} (spent so far ${CAP.spent():.2f})", flush=True)
    except CapReached as e:
        raise SystemExit(f"stopped: {e}. Finished variants and embedded batches are saved; rerun to continue.")
    print(f"spent this run: ${CAP.spent():.2f}")

    if not (have := [n for n in a.variants if (out / f"runs_{n}.parquet").exists()]):
        raise SystemExit(f"no saved rankings in {out}")
    runs = pd.concat([pd.read_parquet(out / f"runs_{n}.parquet") for n in have])
    m = metrics(runs, tables, tests)
    m.to_csv(out / "metrics.csv", index=False)
    report(m, have)


def check_apis(models: set[str]) -> None:
    """Cheap check before the paid run: truncating a 3072 OpenAI vector equals asking for 1024."""
    from openai import OpenAI

    texts = ["Export controls restrict sales of our A100 and H100 chips to China.",
             "Demand for data center products grew strongly in fiscal 2024."]
    if "text-embedding-3-large" not in models:
        return
    full = embed(texts, "text-embedding-3-large")
    small = np.array([d.embedding for d in OpenAI().embeddings.create(
        model="text-embedding-3-large", dimensions=1024, input=texts).data])
    cos = (unit(full, 1024) * unit(small, 1024)).sum(axis=1)
    print(f"check: OpenAI 3072 truncated to 1024 vs dimensions=1024, cosine {cos.round(4).tolist()}")
    assert (cos > 0.999).all()


if __name__ == "__main__":
    main()
