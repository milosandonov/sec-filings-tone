"""Phase 4: score each retrieval stage (vector, bm25, hybrid, rerank) on eval/tests.jsonl.

Usage:
    uv run python eval/eval.py              # every test, every stage; prints the README table
    uv run python eval/eval.py --no-rerank  # skip the Claude calls
    uv run python eval/eval.py 3            # one test, with each keyword's first-hit rank per stage
    uv run python eval/eval.py --budget 0.25 --max-minutes 10   # caps (defaults: $0.50, 15 minutes)

Metrics follow week5/evaluation/eval.py in github.com/ed-donner/llm_engineering (MIT): MRR, nDCG@k and keyword coverage per keyword,
averaged per test, then over tests. Two changes:
- A chunk contains a keyword only as a case-sensitive whole word; substring matching makes "Arm" hit
  "harm" and "CAC" hit inside other words.
- nDCG's ideal ranking counts every chunk in the filtered subset that has the keyword, not just the
  retrieved ones, so finding 1 of 5 relevant chunks at rank 1 doesn't score a perfect 1.0.
"""

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from sec_filings_tone.budget import Cap, CapReached, rerank_call_usd, run_log
from sec_filings_tone.config import CANDIDATES, EMBED_MODEL, RERANK_MODEL
from sec_filings_tone.retrieve import STAGES, chunks, search, subset

TESTS = Path(__file__).parent / "tests.jsonl"
K = 10
def contains(text: str, keyword: str) -> bool:
    # whole word, and not part of a longer number: "15%" doesn't match "2.15%", "$1.6" not "$1.65"
    return re.search(rf"(?<![\w.,]){re.escape(keyword)}(?!\w|[.,]\d)", text) is not None


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def score(texts: list[str], keywords: list[str], n_relevant: dict[str, int]) -> dict[str, float]:
    """MRR, nDCG@K and coverage for one ranked list, averaged over the keywords."""
    mrr, ndcg = [], []
    for kw in keywords:
        rels = [int(contains(t, kw)) for t in texts[:K]]
        mrr.append(next((1 / r for r, hit in enumerate(rels, 1) if hit), 0.0))
        ideal = dcg([1] * min(n_relevant[kw], K))
        ndcg.append(dcg(rels) / ideal if ideal else 0.0)
    return {"mrr": sum(mrr) / len(mrr), "ndcg": sum(ndcg) / len(ndcg),
            "coverage": sum(m > 0 for m in mrr) / len(mrr)}


def run(test: dict, rerank: bool) -> tuple[dict, dict]:
    """Each stage's metrics for one test, plus the ranked chunks per stage."""
    pool = chunks().text[subset(test["filters"])[0]]
    n_relevant = {kw: int(pool.map(lambda t: contains(t, kw)).sum()) for kw in test["keywords"]}
    if missing := [kw for kw, n in n_relevant.items() if n == 0]:
        print(f"  warning: no chunk in the filtered subset contains {missing}; fix the test")
    results = search(test["question"], K, rerank, **test["filters"])
    return {stage: score(df.text.tolist(), test["keywords"], n_relevant) for stage, df in results.items()}, results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("test", nargs="?", type=int, help="row number in tests.jsonl (0-based)")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--budget", type=float, default=0.50, help="USD; stop before a call would pass it")
    p.add_argument("--max-minutes", type=float, default=15)
    a = p.parse_args()
    cap = Cap(a.budget, a.max_minutes)
    if a.no_rerank:  # free but for the query embeddings
        return evaluate(a, cap)
    with run_log([RERANK_MODEL, EMBED_MODEL], lambda: {"usd": {"anthropic": cap.spent()}}):
        evaluate(a, cap)


def evaluate(a: argparse.Namespace, cap: Cap) -> None:
    tests = [json.loads(line) for line in TESTS.read_text().splitlines() if line.strip()]
    if a.test is not None:
        t = tests[a.test]
        print(f"Q: {t['question']}\nfilters: {t['filters']}\nkeywords: {t['keywords']}\n")
        metrics, results = run(t, not a.no_rerank)
        for stage, df in results.items():
            ranks = {kw: next((i for i, x in enumerate(df.text, 1) if contains(x, kw)), None) for kw in t["keywords"]}
            m = metrics[stage]
            print(f"{stage:7} MRR {m['mrr']:.3f}  nDCG {m['ndcg']:.3f}  coverage {m['coverage']:.0%}  first hit: {ranks}")
            print("        " + " ".join(df.chunk_id.head(5)))
        return

    rows = []
    for i, t in enumerate(tests):
        try:
            cap.check(rerank_call_usd(CANDIDATES * 800) if not a.no_rerank else 0)
        except CapReached as e:
            raise SystemExit(f"stopped before test {i}: {e}")
        metrics, _ = run(t, not a.no_rerank)
        rows += [{"test": i, "category": t["category"], "stage": s, **m} for s, m in metrics.items()]
        print(f"{i:2} {t['category']:13} " + "  ".join(f"{s} {m['mrr']:.2f}" for s, m in metrics.items()))
    df = pd.DataFrame(rows)
    stages = [s for s in STAGES if s in set(df.stage)]
    overall = df.groupby("stage")[["mrr", "ndcg", "coverage"]].mean().loc[stages]
    print(f"\nspent on reranking: ${cap.spent():.3f}")
    print(f"\n{len(tests)} tests, top {K} per stage\n\n| Stage | MRR | nDCG | Keyword coverage |\n|---|---|---|---|")
    for stage, r in overall.iterrows():
        print(f"| {stage} | {r.mrr:.3f} | {r.ndcg:.3f} | {r.coverage:.1%} |")
    by_cat = df.pivot_table(index="category", columns="stage", values="mrr")[stages]
    print("\nMRR by category\n\n| Category | " + " | ".join(stages) + " |\n|---|" + "---|" * len(stages))
    for cat, r in by_cat.iterrows():
        print(f"| {cat} ({(df[df.stage == stages[0]].category == cat).sum()}) | " + " | ".join(f"{v:.3f}" for v in r) + " |")


if __name__ == "__main__":
    main()
