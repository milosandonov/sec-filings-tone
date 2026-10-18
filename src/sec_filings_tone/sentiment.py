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
    uv run python -m sec_filings_tone.sentiment baselines                 # FinBERT and LM on the pilot, local
    uv run python -m sec_filings_tone.sentiment gold                      # draw labels/gold_profiles.csv, 30 items
    uv run python -m sec_filings_tone.sentiment gold-report --budget 0.1  # score the labeled items, compare, adjudication sheet
    uv run python -m sec_filings_tone.sentiment gold-baselines            # FinBERT, LM and the counts vs the labels, local
"""

import argparse
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, create_model

from sec_filings_tone import retrieve
from sec_filings_tone.budget import Cap, CapReached, call_usd, rerank_call_usd
from sec_filings_tone.ingest import CHUNKS, ITEM_NAMES

GUIDELINES = Path(__file__).parents[2] / "labels" / "guidelines.md"
OUT = CHUNKS.parent / "sentiment"
MODEL = "claude-haiku-5-5"  # cheapest; the pilot shows whether a bigger model is needed
TIMEOUT = 60  # seconds per call
SELECTION = "reranked"  # how pilot passages are picked; in the file names, so results from different pickers don't mix

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
    "Inflation and costs": "inflation, rising costs of materials, labor, energy and freight, and price increases",
    "Geopolitics and war": "war, sanctions and geopolitical conflict, such as Russia and Ukraine or the Middle East",
    "Cybersecurity": "cybersecurity incidents, cyberattacks, data breaches and protecting information systems",
    "Labor and workforce": "labor shortages, wages, hiring, layoffs, strikes and unions",
    "Energy transition and climate": "climate change, emissions regulation, renewable energy and clean energy incentives",
}
# Phase 9 scores every topic for every company, except deposits and liquidity, for Financials only
PRACTICE = Path(__file__).parents[2] / "labels" / "practice.csv"  # never sampled, so it stays a clean test
GOLD = Path(__file__).parents[2] / "labels" / "gold_profiles.csv"
ADJUDICATION = Path(__file__).parents[2] / "labels" / "gold_adjudication.csv"

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


def period(row) -> str:
    return f"FY{row['fy']}" if row["form"] == "10-K" else f"quarter ending {row['report_date']}"


def user_prompt(row: pd.Series, topic: str, dims: list[str]) -> str:
    # row["item"], not row.item: on a pandas row, .item is a method
    return (f"Topic: {topic}\nFiling: {row['ticker']} {row['form']} {period(row)}, {ITEM_NAMES[row['item']]}\n\n"
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
# a short hedge list; on the pilot it tracks the LLM's uncertainty at least as well as LM's lists (+0.79 vs +0.69/+0.71)
REALIZED = r"\b(recorded|incurred|recognized|resulted in|took effect|was (imposed|enacted|announced|completed)|we (lost|received|paid))\b"


def proxies(text: str) -> dict:
    """Per 100 words: concrete details (amounts, percentages, years, product codes), hedge words, realized-event cues."""
    words = max(1, len(text.split()))
    details = len(re.findall(r"\$[\d.,]+|\b\d[\d,.]*%|\b(?:19|20)\d\d\b|\b[A-Z]{1,4}\d{2,4}[A-Z]?\b", text))
    return {"details_per_100w": 100 * details / words,
            "hedges_per_100w": 100 * len(re.findall(HEDGES, text, re.I)) / words,
            "realized_per_100w": 100 * len(re.findall(REALIZED, text, re.I)) / words}


# ---------- baselines (local, free) ----------

# Loughran-McDonald Master Dictionary 1993-2025, linked from sraf.nd.edu/loughranmcdonald-master-dictionary.
# Free for academic use; kept out of git.
LM_URL = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
LM_PATH = CHUNKS.parent / "lm" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
LM_CATEGORIES = ["negative", "positive", "uncertainty", "litigious", "strong_modal", "weak_modal", "constraining"]
FINBERT = "ProsusAI/finbert"


@cache
def lm_lists() -> tuple[frozenset, dict[str, frozenset]]:
    """The LM master word list, and each category's words. In the file, a category column holds the year
    a word was added, or minus the year it was removed."""
    if not LM_PATH.exists():
        LM_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LM_URL, LM_PATH)
        if not LM_PATH.read_text()[:13] == "Word,Seq_num,":  # Drive returned a page, not the file
            LM_PATH.unlink()
            raise SystemExit(f"LM dictionary download failed; save the CSV from sraf.nd.edu to {LM_PATH}")
    d = pd.read_csv(LM_PATH, keep_default_na=False)  # else the words NA and NULL become NaN
    return frozenset(d.Word), {c: frozenset(d.Word[d[c.title()] > 0]) for c in LM_CATEGORIES}


def lm(text: str) -> dict:
    """Loughran-McDonald: each category's share of the words, and polarity = (positive - negative) / words.
    Words are runs of letters found in the master dictionary, as LM count them."""
    words, cats = lm_lists()
    toks = [t for t in re.findall(r"[A-Z]+", text.upper()) if t in words]
    n = max(1, len(toks))
    out = {f"lm_{c}": sum(t in cats[c] for t in toks) / n for c in LM_CATEGORIES}
    return {"lm_polarity": out["lm_positive"] - out["lm_negative"], **out}


def sentences(text: str) -> list[str]:
    # a regex split; "U.S. Government" splits in two. Fine for an average over a passage.
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"“])|\n+", text) if s.strip()] or [text]


def finbert(texts: list[str], batch_size: int = 64) -> pd.DataFrame:
    """FinBERT per sentence, averaged per passage: polarity = P(positive) - P(negative), and the share of
    sentences whose top label is negative or positive."""
    import torch
    from transformers import pipeline

    clf = pipeline("text-classification", model=FINBERT, top_k=None, truncation=True,
                   device="mps" if torch.backends.mps.is_available() else "cpu")
    sents = [(i, s) for i, t in enumerate(texts) for s in sentences(t)]
    rows = []
    for (i, _), res in zip(sents, clf([s for _, s in sents], batch_size=batch_size)):
        p = {r["label"]: r["score"] for r in res}
        top = max(p, key=p.get)
        rows.append({"i": i, "finbert_polarity": p["positive"] - p["negative"],
                     "finbert_negative": top == "negative", "finbert_positive": top == "positive"})
    return pd.DataFrame(rows).groupby("i").mean().reindex(range(len(texts))).reset_index(drop=True)


def spearman_ci(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Spearman correlation with a 95% bootstrap interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    r = pd.Series(a).corr(pd.Series(b), method="spearman")
    boot = [pd.Series(a[k]).corr(pd.Series(b[k]), method="spearman")
            for k in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return r, lo, hi


def spearman_gap_ci(h: pd.Series, a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Spearman(h, a) minus Spearman(h, b) on the same items, with a 95% bootstrap interval that resamples
    the items once for both, so the two correlations share each resample."""
    h, a, b = (np.asarray(v, float) for v in (h, a, b))
    rng = np.random.default_rng(seed)

    def gap(k):
        hk = pd.Series(h[k])
        return hk.corr(pd.Series(a[k]), method="spearman") - hk.corr(pd.Series(b[k]), method="spearman")

    boot = [gap(rng.integers(0, len(h), len(h))) for _ in range(n_boot)]
    return gap(np.arange(len(h))), *np.nanpercentile(boot, [2.5, 97.5])


def baselines(n: int, model: str) -> None:
    """Score the pilot passages with FinBERT and LM, and compare with the LLM's first combined run."""
    items = pd.read_parquet(OUT / f"pilot_sample_{n}_{SELECTION}.parquet")
    path = OUT / f"pilot_baselines_{n}_{SELECTION}.parquet"
    if not path.exists():
        b = pd.concat([finbert(items.text.tolist()), pd.DataFrame([lm(t) for t in items.text])], axis=1)
        b.insert(0, "pid", items.pid.values)
        b.to_parquet(path, index=False)
    b = pd.read_parquet(path).set_index("pid")
    s = pd.read_parquet(OUT / f"pilot_scores_{n}_{SELECTION}_{model}.parquet")
    r1 = s[(s["mode"] == "combined") & (s.run == 1)].set_index("pid")
    r1 = r1[r1.off_topic.eq(False)]  # errors have no off_topic and drop out too
    x = b.loc[r1.index].join(r1[list(DIMENSIONS)].astype(float))
    x = x.join(pd.DataFrame([proxies(t) for t in items.set_index("pid").loc[r1.index].text], index=r1.index))
    print(f"\n{len(x)} on-topic pilot passages. LLM scores are toward the topic; the baselines score the whole passage.")
    print("\n| LLM dimension | Baseline | Spearman [95% CI] |\n|---|---|---|")
    for d, c in [("polarity", "finbert_polarity"), ("polarity", "lm_polarity"), ("uncertainty", "lm_uncertainty"),
                 ("uncertainty", "lm_weak_modal"), ("uncertainty", "hedges_per_100w"), ("magnitude", "finbert_polarity"),
                 ("specificity", "details_per_100w")]:
        r, lo, hi = spearman_ci(x[d], x[c])
        print(f"| {d} | {c} | {r:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    print(f"\nFinBERT vs LM polarity: {spearman_ci(x.finbert_polarity, x.lm_polarity)[0]:+.2f}")
    print("\nBy LLM polarity (does FinBERT call neutral text negative?):")
    print(x.groupby("polarity").agg(n=("lm_polarity", "size"), finbert_polarity=("finbert_polarity", "mean"),
                                    finbert_negative_share=("finbert_negative", "mean"),
                                    lm_polarity_x100=("lm_polarity", lambda v: 100 * v.mean())).round(2).to_string())


# ---------- pilot reliability check ----------

def sample(n: int, cap: Cap, seed: int = 0) -> pd.DataFrame:
    """n (passage, topic) pairs: for each topic in turn, a random company, form and year, and the top
    reranked hit there, as Phase 9 will pick passages. (An earlier pilot used hybrid's top hit;
    reranking found an on-topic passage for half of the 16% it left off topic.) Banking topics are
    skipped when no bank is embedded."""
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
        cap.check(rerank_call_usd(retrieve.CANDIDATES * 800))
        hits = retrieve.search(TOPICS[topic], k=1, rerank=True, ticker=ticker, form=form, fy=int(fy))["rerank"]
        if len(hits) and (cid := hits.chunk_id.iloc[0]) not in seen | skip:
            seen.add(cid)
            out.append({"pid": len(out), "topic": topic, **hits.iloc[0].drop("score").to_dict()})
    return pd.DataFrame(out)


def pilot(n: int, cap: Cap, model: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pilot_sample_{n}_{SELECTION}.parquet"
    items = pd.read_parquet(path) if path.exists() else sample(n, cap)
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
    pd.DataFrame(rows).to_parquet(OUT / f"pilot_scores_{n}_{SELECTION}_{model}.parquet", index=False)
    print(f"spent ${cap.spent():.2f}; saved {len(rows)} results")
    report(n, model)


# ---------- gold labeling sheet ----------

def sheet(items: pd.DataFrame) -> pd.DataFrame:
    """Labeling-sheet rows in practice.csv's columns, label columns empty. `items` are chunks.parquet
    rows plus `topic`."""
    return pd.DataFrame({
        "topic": items.topic.values, "chunk_id": items.chunk_id.values, "company": items.ticker.values,
        "form": items.form.values, "period": [period(r) for _, r in items.iterrows()],
        "section": items["item"].map(ITEM_NAMES).values, "passage": items.text.values,
        **dict.fromkeys(["off_topic", *DIMENSIONS, "notes"], "")})


def gold(n: int, seed: int = 0) -> None:
    """Write labels/gold_profiles.csv: n passage-topic pairs picked the way Phase 9 picks passages, so
    the gold set tests the scorer on what it will score. Each item is a random filing from the next
    sector in turn (about 1 in 6 a 10-Q, as Phase 9 scores 10-Qs in event windows) with the next topic
    in turn, and one of hybrid search's top 3 there, at random. Off-topic hits are kept: Phase 9
    scores them too. No LLM scores go in the sheet, so labeling stays blind. Costs only n query
    embeddings."""
    if GOLD.exists():
        raise SystemExit(f"{GOLD} exists and may hold labels; move it before drawing again")
    rng = random.Random(seed)
    df = retrieve.chunks()
    filings = df.drop_duplicates(["ticker", "form", "filing_date"])
    skip = set(pd.read_csv(PRACTICE).chunk_id)
    topics, sectors = list(TOPICS), sorted(df.sector.unique())
    rng.shuffle(topics)
    rng.shuffle(sectors)
    out = []
    while len(out) < n:
        i = len(out)
        topic = topics[i % len(topics)]
        sector = "Financials" if topic == "Deposits and liquidity" else sectors[i % len(sectors)]
        pool = filings[(filings.sector == sector) & (filings.form == ("10-Q" if rng.random() < 1 / 6 else "10-K"))]
        f = pool.iloc[rng.randrange(len(pool))]
        hits = retrieve.search(TOPICS[topic], k=3, rerank=False, ticker=f.ticker, form=f.form,
                               date_from=f.filing_date, date_to=f.filing_date)["hybrid"]
        hit = hits.iloc[rng.randrange(len(hits))]
        if hit.chunk_id not in skip | {o["chunk_id"] for o in out}:
            out.append({"topic": topic, **hit.drop("score").to_dict()})
            print(f"{len(out):2} {sector:22} {topic:30} {hit.chunk_id}")
    items = pd.DataFrame(out)
    s = sheet(items)
    s.insert(0, "item_id", [f"G{i:02}" for i in range(1, n + 1)])
    s.to_csv(GOLD, index=False)
    print(f"\nwrote {GOLD}: {n} items, {items.sector.nunique()} sectors, {items.topic.nunique()} topics, "
          f"{(items.form == '10-Q').sum()} from 10-Qs")


# ---------- gold benchmark ----------

def labels(path: Path, final: pd.DataFrame | None = None) -> pd.DataFrame:
    """A labeled sheet, checked: off_topic yes or no (a spreadsheet's TRUE/FALSE is accepted) and every
    on-topic item scored on each dimension's scale. Stops with the cells to fix. With `final` (the
    adjudication sheet), each of its rows replaces that item's label first."""
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    for r in (final.itertuples() if final is not None else ()):
        d.loc[d.item_id == r.item_id, r.field] = r.final
    ot = d.off_topic.str.strip().str.lower().map({"yes": True, "true": True, "no": False, "false": False})
    bad = [f"{i} off_topic" for i in d.item_id[ot.isna()]]
    on = ot.eq(False)
    for c, scale in DIMENSIONS.items():
        d[c] = pd.to_numeric(d[c], errors="coerce")
        bad += [f"{i} {c}" for i, x in zip(d.item_id[on], d[c][on]) if x not in scale]
    if bad:
        raise SystemExit(f"{path.name}: fix {', '.join(bad)}")
    return d.assign(off_topic=ot.astype(bool))


def gold_scores(cap: Cap, model: str) -> pd.DataFrame:
    """The scorer's profile of every labeled item (practice and gold sheets): one combined call each,
    as Phase 9 scores. Saved, so the report reruns for free; delete the file to rescore."""
    path = OUT / f"gold_scores_{model}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    items = pd.concat([pd.read_csv(p, usecols=["item_id", "topic", "chunk_id"]) for p in (PRACTICE, GOLD)])
    rows = retrieve.chunks().loc[items.chunk_id]

    def run(k):
        try:
            res = score(rows.iloc[k], items.topic.iloc[k], list(DIMENSIONS), cap, model)
        except CapReached:
            raise
        except Exception as e:  # report every failure at once instead of stopping at the first
            res = {"error": f"{type(e).__name__}: {e}"}
        return {"item_id": items.item_id.iloc[k], "model": model, **res}

    with ThreadPoolExecutor(8) as pool:
        out = pd.DataFrame(pool.map(run, range(len(items))))
    print(f"scored {len(out)} items with {model}: ${cap.spent():.3f}")
    if "error" in out and out.error.notna().any():
        raise SystemExit(f"{out.error.notna().sum()} calls failed, nothing saved:\n{out.error.dropna().to_string()}")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def kappa_ci(a, b, labels, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Weighted kappa with a 95% bootstrap interval."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    rng = np.random.default_rng(seed)
    boot = [kappa(a[k], b[k], labels) for k in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return kappa(a, b, labels), *np.nanpercentile(boot, [2.5, 97.5])


def gold_report(cap: Cap, model: str, adjudicated: bool = False) -> None:
    """Phase 6 on the human labels: off-topic agreement, agreement per dimension, how the scores
    spread, the halo and merge checks, and labels/gold_adjudication.csv with one row per
    disagreement for the labeler to review. The gold sheet (random, labeled blind under the frozen
    rules) is the main set; the practice items are hand-picked and were labeled before the rules
    changed, so they're shown added in, not alone. `adjudicated` applies the labeler's decisions;
    disagreements resolved toward the model raise agreement by construction, so the report counts them."""
    final = pd.read_csv(ADJUDICATION, dtype=str, keep_default_na=False) if adjudicated else None
    human = pd.concat([labels(PRACTICE, final), labels(GOLD, final)]).set_index("item_id")
    llm = gold_scores(cap, model).set_index("item_id").loc[human.index]
    llm["off_topic"] = llm.off_topic.astype(bool)
    dims = list(DIMENSIONS)
    print(f"\n{model} vs the {'adjudicated' if adjudicated else 'blind'} human labels: {len(human)} items")
    if adjudicated:
        changed = final.final.str.strip() != final.your_label.str.strip()
        print(f"adjudication: {changed.sum()} of {len(final)} disagreements changed, "
              f"{(changed & (final.final.str.strip() == final.model_label.str.strip())).sum()} to the model's label")
    print("\nOff topic (rows: human, columns: model)")
    print(pd.crosstab(human.off_topic, llm.off_topic).to_string())
    on = human.index[~human.off_topic & ~llm.off_topic]
    gold_on = on[on.str.startswith("G")]

    def spread(v, scale):
        return " ".join(f"{s}:{(v == s).sum()}" for s in scale)

    print("\n| Dimension | Exact, gold | Within 1, gold | Weighted kappa, gold [95% CI] | Kappa, gold + practice | "
          "Human scores, gold | Model scores, gold |\n|---|---|---|---|---|---|---|")
    for d, scale in DIMENSIONS.items():
        h, m = human.loc[gold_on, d].astype(int), llm.loc[gold_on, d].astype(int)
        k, lo, hi = kappa_ci(h, m, scale)
        print(f"| {d} | {(h == m).mean():.0%} | {((h - m).abs() <= 1).mean():.0%} | {k:.2f} [{lo:.2f}, {hi:.2f}] | "
              f"{kappa(human.loc[on, d].astype(int), llm.loc[on, d].astype(int), scale):.2f} | "
              f"{spread(h, scale)} | {spread(m, scale)} |")
    print(f"\n{len(gold_on)} gold and {len(on) - len(gold_on)} practice items on topic for both. "
          "Rule fixed in advance: drop a dimension with kappa below 0.6.")

    print("\nCorrelation between dimensions (Spearman), gold + practice on topic. Halo: the model's clearly above "
          "the human's. Merge: a human pair above 0.8. A dimension with one score throughout shows NaN.")
    for name, frame in [("human", human.loc[on]), (model, llm.loc[on])]:
        c = frame[dims].astype(float).corr(method="spearman")
        pairs = c.where(np.triu(np.ones(c.shape, bool), 1)).stack()
        print(f"\n{name}: mean |r| between dimensions {pairs.abs().mean():.2f}\n{c.round(2).to_string()}")
        if name == "human" and (big := pairs[pairs.abs() > 0.8]).size:
            print(f"merge candidates: {big.round(2).to_dict()}")

    if ADJUDICATION.exists():
        print(f"\n{ADJUDICATION.name} exists and may hold your decisions; not overwritten")
        return
    text = pd.concat([pd.read_csv(p) for p in (PRACTICE, GOLD)]).set_index("item_id")
    out = []
    for i in human.index:
        h, m = human.loc[i], llm.loc[i]
        fields = ["off_topic"] if h.off_topic != m.off_topic else [] if h.off_topic else [d for d in dims if h[d] != m[d]]
        for f in fields:
            yours, theirs = (("yes" if h[f] else "no"), ("yes" if m[f] else "no")) if f == "off_topic" else (int(h[f]), int(m[f]))
            out.append({"item_id": i, **text.loc[i, ["topic", "company", "period", "section"]].to_dict(), "field": f,
                        "your_label": yours, "model_label": theirs, "model_evidence": m.get(f"{f}_evidence", ""),
                        "final": yours, "note": "", "passage": text.loc[i, "passage"]})
    pd.DataFrame(out).to_csv(ADJUDICATION, index=False)
    print(f"\nwrote {ADJUDICATION.name}: {len(out)} disagreements on {len({o['item_id'] for o in out})} items")


# the person's dimension -> scorer column; llm_ columns are the LLM's saved scores on the same items
GOLD_PAIRS = [("polarity", "finbert_polarity"), ("polarity", "lm_polarity"), ("polarity", "llm_polarity"),
              ("uncertainty", "lm_uncertainty"), ("uncertainty", "hedges_per_100w"), ("uncertainty", "llm_uncertainty"),
              ("specificity", "details_per_100w"), ("specificity", "llm_specificity"),
              ("realization", "realized_per_100w"), ("realization", "llm_realization")]
# dimension -> the free baseline the LLM's gap is measured against (RESULTS.md section 1, "Paired gaps")
GAP_PAIRS = [("realization", "realized_per_100w"), ("uncertainty", "hedges_per_100w"),
             ("specificity", "details_per_100w"), ("polarity", "finbert_polarity")]


def label_spearman(human: pd.DataFrame, x: pd.DataFrame, ids) -> list[tuple]:
    """(dimension, scorer column, n, Spearman, low, high) for each of GOLD_PAIRS, on the items in `ids`
    the scorer has a score for (the LLM has none where it calls an item off topic)."""
    out = []
    for d, c in GOLD_PAIRS:
        k = [i for i in ids if pd.notna(x.at[i, c])]
        out.append((d, c, len(k), *spearman_ci(human.loc[k, d], x.loc[k, c])))
    return out


def gold_baselines(model: str) -> None:
    """Research question 1 on the labeled items: FinBERT, LM and the rule-based counts against the
    person's labels, and the LLM's saved scores on the same items with the same statistic (Spearman).
    Local and free: it stops if the LLM's scores aren't saved, never rescoring. Item sets: (a) the gold
    items both the person and the LLM call on topic, as behind gold-report's kappas; (b) every item the
    person calls on topic. Blind labels first; the adjudicated ones lean toward the LLM by construction."""
    scores = OUT / f"gold_scores_{model}.parquet"
    if not scores.exists():
        raise SystemExit(f"{scores} not found; gold-report makes it (paid)")
    llm = pd.read_parquet(scores).set_index("item_id")
    llm["off_topic"] = llm.off_topic.astype(bool)
    path = OUT / "gold_baselines.parquet"
    if not path.exists():
        items = pd.concat([pd.read_csv(p, usecols=["item_id", "chunk_id"]) for p in (PRACTICE, GOLD)])
        text = pd.read_parquet(CHUNKS, columns=["chunk_id", "text"], filters=[("chunk_id", "in", items.chunk_id.tolist())]
                               ).set_index("chunk_id").text.loc[items.chunk_id]
        b = pd.concat([finbert(text.tolist()), pd.DataFrame([lm(t) | proxies(t) for t in text])], axis=1)
        b.insert(0, "item_id", items.item_id.values)
        b.to_parquet(path, index=False)
    x = pd.read_parquet(path).set_index("item_id").join(
        llm.loc[~llm.off_topic, list(DIMENSIONS)].astype(float).add_prefix("llm_"))
    print(f"\n{len(x)} labeled items. The person and {model} (llm_) score toward the topic; the baselines score "
          f"the whole passage. llm_ counts only items {model} also calls on topic.")
    for adjudicated in (False, True):
        final = pd.read_csv(ADJUDICATION, dtype=str, keep_default_na=False) if adjudicated else None
        human = pd.concat([labels(PRACTICE, final), labels(GOLD, final)]).set_index("item_id")
        on = human.index[~human.off_topic]
        both = on[on.str.startswith("G") & ~llm.loc[on].off_topic.values]
        print(f"\n{'Adjudicated (leans toward the LLM by construction)' if adjudicated else 'Blind'} labels. "
              "(a): gold items both call on topic. (b): every item the person calls on topic.")
        print("\n| Person's dimension | Scorer | n (a) | Spearman (a) [95% CI] | n (b) | Spearman (b) [95% CI] |"
              "\n|---|---|---|---|---|---|")
        for (d, c, na, r, lo, hi), (*_, nb, rb, lob, hib) in zip(label_spearman(human, x, both), label_spearman(human, x, on)):
            print(f"| {d} | {c} | {na} | {r:+.2f} [{lo:+.2f}, {hi:+.2f}] | {nb} | {rb:+.2f} [{lob:+.2f}, {hib:+.2f}] |")
        print("\nIs the LLM closer to the person than the free baseline? Spearman of each with the person, LLM minus "
              "baseline, on items (a), resampling the items for both:")
        for d, c in GAP_PAIRS:
            k = [i for i in both if pd.notna(x.at[i, f"llm_{d}"]) and pd.notna(x.at[i, c])]
            est, lo, hi = spearman_gap_ci(human.loc[k, d], x.loc[k, f"llm_{d}"], x.loc[k, c])
            print(f"  {d:12} llm_{d} - {c}: {est:+.2f} [{lo:+.2f}, {hi:+.2f}]  n={len(k)}")


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
    items = pd.read_parquet(OUT / f"pilot_sample_{n}_{SELECTION}.parquet")
    s = pd.read_parquet(OUT / f"pilot_scores_{n}_{SELECTION}_{model}.parquet")
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
    p.add_argument("command", choices=["pilot", "report", "baselines", "gold", "gold-report", "gold-baselines"])
    p.add_argument("--n", type=int, help="items; default 100 (pilot), 30 (gold)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--budget", type=float, default=2.0, help="USD; stop before a call would pass it")
    p.add_argument("--max-minutes", type=float, default=60)
    p.add_argument("--adjudicated", action="store_true", help="gold-report: apply labels/gold_adjudication.csv")
    a = p.parse_args()
    n = a.n or (30 if a.command == "gold" else 100)
    if a.command == "pilot":
        pilot(n, Cap(a.budget, a.max_minutes), a.model)
    elif a.command == "baselines":
        baselines(n, a.model)
    elif a.command == "gold":
        gold(n)
    elif a.command == "gold-report":
        gold_report(Cap(a.budget, a.max_minutes), a.model, a.adjudicated)
    elif a.command == "gold-baselines":
        gold_baselines(a.model)
    else:
        report(n, a.model)


if __name__ == "__main__":
    main()
