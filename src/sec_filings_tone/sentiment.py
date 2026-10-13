"""Phase 5: score a passage's tone profile toward a topic with an LLM.

The rules are labels/guidelines.md, quoted into the prompt word for word, so the model and human
labelers work from the same text. Safeguards against one overall impression leaking into every
score (the halo effect):
- the model quotes the evidence for each dimension before giving its score;
- the prompt has contrasting examples, where the dimensions disagree;
- dimensions can be scored in one call, in shuffled order, or one call each, and the pilot check
  compares the three;
- rule-based measures cross-check specificity, uncertainty and realization.

Usage:
    uv run python -m sec_filings_tone.sentiment pilot                     # reliability check, 100 passages
    uv run python -m sec_filings_tone.sentiment pilot --n 10 --budget 0.2 # a small first run
    uv run python -m sec_filings_tone.sentiment report                    # rescore the saved pilot, no API calls
"""

import argparse
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, create_model

from sec_filings_tone import retrieve
from sec_filings_tone.budget import Cap, CapReached, call_usd
from sec_filings_tone.ingest import ITEM_NAMES

GUIDELINES = Path(__file__).parents[2] / "labels" / "guidelines.md"
OUT = Path("data/sentiment")
MODEL = "claude-haiku-5-5"  # cheapest; the pilot shows whether a bigger model is needed
TIMEOUT = 60  # seconds per call

# dimension -> allowed scores, in the guidelines' order (evaluative, epistemic, disclosure)
DIMENSIONS = {
    "polarity": (-2, -1, 0, 1, 2),
    "magnitude": (0, 1, 2, 3),
    "realization": (0, 1, 2),
    "uncertainty": (0, 1, 2, 3),
    "specificity": (0, 1, 2),
}
TOPICS = {  # topic -> search query used to find passages about it
    "China": "China export controls, Chinese customers, suppliers and regulators, US-China trade tensions",
    "AI": "artificial intelligence demand, AI products, AI data centers",
    "Supply chain": "supply chain, suppliers, shortages, manufacturing capacity, inventory and purchase commitments",
    "Demand": "customer demand for our products and services",
    "Interest rates": "interest rates and their effect on borrowing, investment income and customers",
    "Tariffs": "tariffs and trade investigations",
    "Government policy": "government regulation, subsidies, tax credits, CHIPS Act, antitrust",
    "Deposits and liquidity": "bank deposits, deposit flows and costs, liquidity and funding",
}
PRACTICE = Path(__file__).parents[2] / "labels" / "practice.csv"  # never sampled, so it stays a clean test

EXAMPLES = """\
Short examples where the dimensions disagree (scores in the order polarity, magnitude, realization,
uncertainty, specificity):

1. Topic China. "On March 31 the regulator opened a review of our products, and on May 21 it barred
critical infrastructure operators from buying them. The impact on our business remains uncertain."
-> -2, 2, 2, 3, 2. It happened (realization 2) and the effects are unknown (uncertainty 3).

2. Topic China. "If export rules were extended to our H20 products, we could lose up to $5 billion of
annual China revenue." -> -1, 2, 0, 2, 2. Hypothetical (realization 0) but concrete (specificity 2).

3. Topic Supply chain. "The outbreak has disrupted some supplier shipments, but we do not expect a
material impact on our results for the year." -> -1, 1, 2, 1, 1. Negative but minor.

4. Topic AI. "Demand for our accelerators from AI data centers drove a 70% increase in revenue."
-> +2, 3, 2, 0, 2. A large effect in the positive direction is still magnitude 3.

5. Topic Tariffs. "Changes in trade policy, including tariffs, could adversely affect our business."
-> -1, 0, 0, 2, 0. Boilerplate: no described effect, so magnitude 0."""

INSTRUCTIONS = """\
For each dimension asked for, first quote the words from the passage that justify the score: the
shortest span that does, or "" if nothing in the passage bears on it. Then give the score. Score
each dimension on its own question, and check its "Not" rule before deciding. If the passage says
nothing about the topic, set off_topic to true; the scores are then ignored."""


def system_prompt() -> str:
    return (f"You score the tone of a passage from an SEC filing toward one topic, following these labeling "
            f"guidelines exactly.\n\n<guidelines>\n{GUIDELINES.read_text()}\n</guidelines>\n\n"
            f"<examples>\n{EXAMPLES}\n</examples>\n\n{INSTRUCTIONS}")


def schema(dims: list[str], with_off_topic: bool = True):
    """Output model with each dimension's evidence before its score, in the given order. The model
    writes fields in schema order, so the evidence comes first."""
    fields = {"off_topic": (bool, ...)} if with_off_topic else {}
    for d in dims:
        fields[f"{d}_evidence"] = (str, Field(description=f"words from the passage that justify the {d} score"))
        fields[d] = (Literal[DIMENSIONS[d]], ...)
    return create_model("Profile_" + "_".join(dims), **fields)


def user_prompt(row: pd.Series, topic: str, dims: list[str]) -> str:
    # row["item"], not row.item: on a pandas row, .item is a method
    when = f"FY{row['fy']}" if row["form"] == "10-K" else f"quarter ending {row['report_date']}"
    return (f"Topic: {topic}\nFiling: {row['ticker']} {row['form']} {when}, {ITEM_NAMES[row['item']]}\n\n"
            f"Passage:\n{row['text']}\n\nScore: {', '.join(dims)}.")


def score(row, topic: str, dims: list[str], cap: Cap, model: str = MODEL, with_off_topic: bool = True) -> dict:
    """One call: {dimension: score, dimension_evidence: text, off_topic: bool}, or {"error": ...}."""
    from anthropic import Anthropic

    prompt = user_prompt(row, topic, dims)
    cap.check(call_usd(model, 4000 + len(prompt) / 3, 4000))  # upper bound: prompt + generous output
    resp = Anthropic(timeout=TIMEOUT, max_retries=3).messages.parse(
        model=model, max_tokens=4000, output_config={"effort": "low"},
        system=[{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}], output_format=schema(dims, with_off_topic))
    u = resp.usage
    cap.charge(call_usd(model, u.input_tokens + (u.cache_creation_input_tokens or 0) * 1.25
                        + (u.cache_read_input_tokens or 0) * 0.1, u.output_tokens))
    if resp.parsed_output is None:
        return {"error": f"no parsed output (stop_reason={resp.stop_reason})"}
    return resp.parsed_output.model_dump()


# ---------- rule-based cross-checks (no model) ----------

HEDGES = r"\b(may|might|could|would|possibly|potential(ly)?|uncertain(ty|ties)?|cannot (predict|estimate)|unable to (predict|estimate)|unpredictable|risk)\b"
# ponytail: a short hedge list; swap for the Loughran-McDonald uncertainty list once the LM baseline is added
REALIZED = r"\b(recorded|incurred|recognized|resulted in|took effect|was (imposed|enacted|announced|completed)|we (lost|received|paid))\b"


def proxies(text: str) -> dict:
    """Per 100 words: concrete details (amounts, percentages, years, product codes), hedge words, realized-event cues."""
    words = max(1, len(text.split()))
    details = len(re.findall(r"\$[\d.,]+|\b\d[\d,.]*%|\b(?:19|20)\d\d\b|\b[A-Z]{1,4}\d{2,4}[A-Z]?\b", text))
    return {"details_per_100w": 100 * details / words,
            "hedges_per_100w": 100 * len(re.findall(HEDGES, text, re.I)) / words,
            "realized_per_100w": 100 * len(re.findall(REALIZED, text, re.I)) / words}


# ---------- pilot reliability check ----------

def sample(n: int, seed: int = 0) -> pd.DataFrame:
    """n (passage, topic) pairs: for each topic in turn, a random company, form and year, and the top
    hybrid hit there. Banking topics are skipped when no bank is embedded."""
    rng = random.Random(seed)
    df = retrieve.chunks()
    skip = set(pd.read_csv(PRACTICE).chunk_id) if PRACTICE.exists() else set()
    topics = [t for t in TOPICS if t != "Deposits and liquidity" or (df.sector == "Financials").any()]
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 20:
        tries += 1
        topic = topics[len(out) % len(topics)]
        ticker = rng.choice(sorted(df.ticker.unique()))
        form = rng.choice(["10-K", "10-Q"])
        fy = rng.choice(sorted(df[(df.ticker == ticker) & (df.form == form)].fy.unique()))
        hits = retrieve.search(TOPICS[topic], k=1, rerank=False, ticker=ticker, form=form, fy=int(fy))["hybrid"]
        if len(hits) and (cid := hits.chunk_id.iloc[0]) not in seen | skip:
            seen.add(cid)
            out.append({"pid": len(out), "topic": topic, **hits.iloc[0].drop("score").to_dict()})
    return pd.DataFrame(out)


def pilot(n: int, cap: Cap, model: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pilot_sample_{n}.parquet"
    items = pd.read_parquet(path) if path.exists() else sample(n)
    items.to_parquet(path, index=False)
    dims = list(DIMENSIONS)
    rng = random.Random(1)
    # (pilot item, mode, run, dims, with_off_topic): combined twice, shuffled once, each dimension alone
    jobs = [(i, "combined", run, dims, True) for i in range(len(items)) for run in (1, 2)]
    jobs += [(i, "shuffled", 1, rng.sample(dims, len(dims)), True) for i in range(len(items))]
    jobs += [(i, "separate", 1, [d], False) for i in range(len(items)) for d in dims]
    print(f"{len(items)} items, {len(jobs)} calls with {model}, budget ${cap.usd:.2f}")

    def run(job):
        i, mode, r, ds, ot = job
        try:
            res = score(items.iloc[i], items.topic.iloc[i], ds, cap, model, ot)
        except CapReached:
            raise
        except Exception as e:  # one failed call shouldn't lose the rest of the run
            res = {"error": f"{type(e).__name__}: {e}"}
        return {"pid": i, "mode": mode, "run": r, "order": ",".join(ds), "model": model, **res}

    rows = []
    try:
        with ThreadPoolExecutor(8) as pool:
            for k, row in enumerate(pool.map(run, jobs), 1):
                rows.append(row)
                if k % 100 == 0:
                    print(f"  {k}/{len(jobs)} calls, ${cap.spent():.2f}", flush=True)
    except CapReached as e:
        print(f"stopped: {e}")
    pd.DataFrame(rows).to_parquet(OUT / f"pilot_scores_{n}_{model}.parquet", index=False)
    print(f"spent ${cap.spent():.2f}; saved {len(rows)} results")
    report(n, model)


def kappa(a, b, labels) -> float:
    """Cohen's kappa with quadratic weights, for ordered scales."""
    idx = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[idx[x], idx[y]] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return 1 - (w * o).sum() / (w * e).sum() if (w * e).sum() else float("nan")


def report(n: int, model: str) -> None:
    items = pd.read_parquet(OUT / f"pilot_sample_{n}.parquet")
    s = pd.read_parquet(OUT / f"pilot_scores_{n}_{model}.parquet")
    dims = list(DIMENSIONS)
    errors = s["error"].notna().sum() if "error" in s else 0
    comb = s[(s["mode"] == "combined") & s.get("error", pd.Series(index=s.index)).isna()]
    r1, r2 = (comb[comb.run == r].set_index("pid") for r in (1, 2))
    on = r1.index[~r1.off_topic.astype(bool)]
    print(f"\n{model}: {len(items)} items, {len(s)} calls, {errors} failed; off topic in run 1: {r1.off_topic.mean():.0%}")

    print("\n| Dimension | Most common score (share) | Run 1 vs run 2: same score | weighted kappa | Combined vs separate kappa |")
    print("|---|---|---|---|---|")
    sep = s[s["mode"] == "separate"].dropna(subset=dims, how="all")
    for d in dims:
        both = on.intersection(r2.index)
        a, b = r1.loc[both, d].astype(int), r2.loc[both, d].astype(int)
        top = r1.loc[on, d].value_counts(normalize=True)
        sd = sep.dropna(subset=[d]).set_index("pid")[d].astype(int)
        common = on.intersection(sd.index)
        print(f"| {d} | {top.index[0]} ({top.iloc[0]:.0%}) | {(a == b).mean():.0%} | {kappa(a, b, DIMENSIONS[d]):.2f} | "
              f"{kappa(r1.loc[common, d].astype(int), sd.loc[common], DIMENSIONS[d]):.2f} |")

    def corr(frame):
        return frame[dims].astype(float).corr(method="spearman")

    shuf = s[(s["mode"] == "shuffled")].dropna(subset=dims).set_index("pid")
    wide_sep = sep.pivot_table(index="pid", values=dims, aggfunc="first")
    print("\nCorrelation between dimensions (Spearman), on-topic items. If 'separate' is clearly lower than "
          "'combined', scoring together leaks between dimensions.")
    for name, frame in [("combined", r1.loc[on]), ("shuffled", shuf.loc[shuf.index.intersection(on)]),
                        ("separate", wide_sep.loc[wide_sep.index.intersection(on)])]:
        c = corr(frame)
        off = c.where(~np.eye(len(dims), dtype=bool)).abs().stack()
        print(f"\n{name}: mean |r| between dimensions {off.mean():.2f}\n{c.round(2).to_string()}")

    px = pd.DataFrame([proxies(t) for t in items.text], index=items.index).loc[on]
    pairs = [("specificity", "details_per_100w"), ("uncertainty", "hedges_per_100w"), ("realization", "realized_per_100w")]
    print("\nRule-based cross-checks (Spearman with the run 1 score): " +
          ", ".join(f"{d} vs {p} {r1.loc[on, d].astype(float).corr(px[p], method='spearman'):+.2f}" for d, p in pairs))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["pilot", "report"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--budget", type=float, default=2.0, help="USD; stop before a call would pass it")
    p.add_argument("--max-minutes", type=float, default=60)
    a = p.parse_args()
    if a.command == "pilot":
        pilot(a.n, Cap(a.budget, a.max_minutes), a.model)
    else:
        report(a.n, a.model)


if __name__ == "__main__":
    main()
