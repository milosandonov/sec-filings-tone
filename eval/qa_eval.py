"""Phase 10: evaluate the Q&A layer (answer.py) on eval/qa_tests.jsonl. Design and pass bars: PLAN.md Phase 10.

Usage:
    uv run python eval/qa_eval.py answers --budget 0.70   # paid: every row through answer.run, resumable
    uv run python eval/qa_eval.py judge --budget 1.00     # paid: Claude Sonnet 5.5 grades them, Batch API
    uv run python eval/qa_eval.py report                  # free: every metric, against the pass bars
    uv run python eval/qa_eval.py answers --name _v2      # a new run, in qa_answers_v2.jsonl (judge, report the same)
    uv run python eval/qa_eval.py judge --name _v2 --judge-name _v2draft --draft  # drafts saved before a (dropped) check
    uv run python eval/qa_eval.py judge --judge-name _rejudge --sample 15  # regrade 15 published answers: judge noise
    uv run python eval/qa_eval.py diff --name _v2         # free: paired differences from the published run
    uv run python eval/qa_eval.py answers --tests eval/qa_dev.jsonl --name _dev  # the dev questions

Free metrics: planner accuracy (the tickers, fiscal years and forms over a plan's sub-queries against the
expected ones; the plan kind), keywords in the passages given to the answer model, citation validity
(every cited span is in its passage), declines. Paid: the judge's accuracy, completeness and groundedness.
"""

import argparse
import json
import random
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anthropic
import numpy as np
import pandas as pd
from anthropic import transform_schema
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv
from pydantic import BaseModel

from sec_filings_tone import answer, retrieve
from sec_filings_tone.budget import Cap, call_usd, run_log
from sec_filings_tone.ingest import EMBED_MODEL

from eval import contains

load_dotenv()
HERE = Path(__file__).parent
TESTS, ANSWERS, JUDGED, BATCH, RUNS = (HERE / f for f in ("qa_tests.jsonl", "qa_answers.jsonl", "qa_judge.jsonl",
                                                          "qa_batch.json", "qa_runs.jsonl"))
HAND = HERE / "qa_hand_grades.csv"
JUDGE_MODEL, JUDGE_EFFORT = "claude-sonnet-5-5", "medium"
WORKERS = 4

JUDGE_SYSTEM = """You grade answers from a question-answering system over SEC filings (10-K and 10-Q text). You get the question (with any earlier turns of the conversation), a reference answer written from the filing text, the system's answer, the passages it cites (its [n] markers number the cited spans) and, for questions about many companies, the table of model-scored tone cells it was given.

Score each from 1 (poor) to 5 (excellent):
- accuracy: the answer's facts agree with the reference and the passages; wrong amounts, dates, companies or years count heavily.
- completeness: it covers the reference's key points. Extra correct detail is fine.
- groundedness: every claim about what a filing says is supported by the cited passages or the cell table. Background the passages don't support lowers it even if it is true.
List each unsupported claim in unsupported_claims (empty if none). Grade the content, not the style or length."""


class Grade(BaseModel):
    accuracy: Literal[1, 2, 3, 4, 5]
    completeness: Literal[1, 2, 3, 4, 5]
    groundedness: Literal[1, 2, 3, 4, 5]
    unsupported_claims: list[str]
    notes: str


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def run_row(row: dict, cap: Cap, model: str, effort: str) -> dict:
    """The row's turns asked in order as one conversation; the last turn's plan, passages, citations and answer."""
    history, t = [], time.time()
    for q in row["turns"]:
        r = answer.run(q, history, cap, model=model, effort=effort)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r["reply"]}]
    docs = r["docs"]
    return {"id": row["id"], "plan": r["plan"].model_dump() if r["plan"] else None, "text": r["text"],
            "citations": r["citations"], "cells": r["cells"], "secs": round(time.time() - t, 1),
            "docs": [{"chunk_id": d.chunk_id, "title": answer.title(d)} for d in docs.itertuples()] if len(docs) else [],
            "stopped": r["text"].startswith(("Stopped:", "The API call failed")), "model": model, "effort": effort}


def answers(cap: Cap, model: str, effort: str, rec: dict | None = None) -> None:
    """Every row not yet answered; rec (the run log's) gets how many stopped, and status "stopped" if any did."""
    done = {a["id"] for a in load(ANSWERS) if not a["stopped"]}
    todo = [r for r in load(TESTS) if r["id"] not in done]
    print(f"{len(todo)} rows to answer ({len(done)} already saved); about ${0.007 * sum(len(r['turns']) for r in todo):.2f}")
    answer.warm()
    kept = [a for a in load(ANSWERS) if not a["stopped"]]
    with ThreadPoolExecutor(WORKERS) as ex:
        for out in ex.map(lambda r: run_row(r, cap, model, effort), todo):
            kept.append(out)
            ANSWERS.write_text("".join(json.dumps(a) + "\n" for a in kept))  # saved as it goes, so a rerun resumes
            print(f"{out['id']}: {out['plan']['kind'] if out['plan'] else '-'}, {len(out['docs'])} passages, "
                  f"{len(out['citations'])} citations, {out['secs']}s{' STOPPED' if out['stopped'] else ''}", flush=True)
    if rec is not None:
        rec["stopped"] = sum(a["stopped"] for a in kept)  # kept holds this run's stopped rows only
        if rec["stopped"]:
            rec["status"] = "stopped"
    # rows run in parallel, so only the run's total is a true cost (a row's share of the session total isn't)
    with RUNS.open("a") as f:
        f.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "rows": len(todo), "usd": cap.spent(), "model": model,
                            "effort": effort}) + "\n")
    print(f"spent ${cap.spent():.3f}")


def judge_prompt(row: dict, a: dict, draft: bool = False) -> str:
    """draft: grade the saved draft instead (qa_answers_v2.jsonl: the answer before a sentence check, since
    removed, dropped sentences; same citations and passages)."""
    texts = retrieve.chunks().text
    turns = "\n".join(f"Turn {i}: {q}" for i, q in enumerate(row["turns"], 1))
    cited = "\n".join(f"[{n}] {a['docs'][d]['title']}: “{span}”" for n, (d, span) in enumerate(a["citations"], 1))
    docs = sorted({d for d, _ in a["citations"]})
    passages = "\n\n".join(f"<passage title=\"{a['docs'][d]['title']}\">\n{texts[a['docs'][d]['chunk_id']]}\n</passage>" for d in docs)
    cells = f"\n\n<cell_table>\n{a['cells']}\n</cell_table>" if a["cells"] else ""
    return (f"<question>\n{turns}\n</question>\n\n<reference_answer>\n{row['reference_answer']}\n</reference_answer>\n\n"
            f"<answer>\n{a['draft'] if draft else a['text']}\n</answer>\n\n<cited_spans>\n{cited or '(none)'}\n</cited_spans>\n\n"
            f"<cited_passages>\n{passages or '(none)'}\n</cited_passages>{cells}\n\nGrade the answer to the last turn.")


def judge(budget: float, sample: int = 0, draft: bool = False, rec: dict | None = None) -> None:
    """sample: grade only a random `sample` of the answers (seed 0), e.g. to regrade them and measure the judge's
    own noise. draft: grade the saved drafts of the answers the check changed (qa_answers_v2.jsonl only).
    rec (the run log's) gets the batch's cost as printed, ungraded results included."""
    client = anthropic.Anthropic()
    rows = {r["id"]: r for r in load(TESTS)}
    graded = {j["id"] for j in load(JUDGED)}
    todo = [a for a in load(ANSWERS) if not a["stopped"] and rows[a["id"]]["type"] != "out_of_scope" and a["id"] not in graded
            and (not draft or a.get("dropped"))]
    if sample:
        todo = random.Random(0).sample(todo, min(sample, len(todo)))
    if not BATCH.exists():
        prompts = {a["id"]: judge_prompt(rows[a["id"]], a, draft) for a in todo}
        # upper bound: ~3.5 characters a token, 3,000 output tokens each, at the batch's half price
        est = sum(call_usd(JUDGE_MODEL, (len(JUDGE_SYSTEM) + len(p)) / 3.5, 3000) for p in prompts.values()) / 2
        print(f"{len(prompts)} answers to judge, at most about ${est:.2f}")
        if est > budget:
            raise SystemExit(f"over --budget ${budget:.2f}; nothing submitted")
        fmt = transform_schema(Grade)
        b = client.messages.batches.create(requests=[Request(custom_id=i, params=MessageCreateParamsNonStreaming(
            model=JUDGE_MODEL, max_tokens=8000, system=JUDGE_SYSTEM, messages=[{"role": "user", "content": p}],
            output_config={"effort": JUDGE_EFFORT, "format": {"type": "json_schema", "schema": fmt}})) for i, p in prompts.items()])
        BATCH.write_text(json.dumps({"id": b.id}))
        print(f"submitted {b.id}", flush=True)
    bid = json.loads(BATCH.read_text())["id"]
    while (b := client.messages.batches.retrieve(bid)).processing_status != "ended":
        print(f"{bid}: {b.request_counts.processing} processing", flush=True)
        time.sleep(60)
    out, usd = load(JUDGED), 0.0
    for res in client.messages.batches.results(bid):
        if res.result.type != "succeeded":
            print(f"{res.custom_id}: {res.result.type}")
            continue
        m, u = res.result.message, res.result.message.usage
        cost = call_usd(JUDGE_MODEL, u.input_tokens, u.output_tokens) / 2
        usd += cost
        try:
            g = Grade.model_validate_json(next(x.text for x in m.content if x.type == "text")).model_dump()
        except (StopIteration, ValueError):
            print(f"{res.custom_id}: no grade (stop_reason {m.stop_reason})")
            continue
        out.append({"id": res.custom_id, **g, "usd": cost, "batch": bid})
    if rec is not None:
        rec["usd"] = {"anthropic": usd}
    JUDGED.write_text("".join(json.dumps(j) + "\n" for j in out))
    BATCH.unlink()
    print(f"{len(out)} graded, ${usd:.3f}")


def plan_sets(p: dict | None) -> dict[str, set]:
    qs = p["queries"] if p else []
    return {"ticker": {t for q in qs for t in q["tickers"]}, "fy": {y for q in qs for y in q["fy"]},
            "form": {q["form"] for q in qs if q["form"]}}


def records(answers_path: Path, judged_path: Path) -> pd.DataFrame:
    """One row per answered question: planner, retrieval and citation checks, and the judge's scores."""
    rows = {r["id"]: r for r in load(TESTS)}
    ans = {a["id"]: a for a in load(answers_path) if not a["stopped"]}
    texts = retrieve.chunks().text
    recs = []
    for i, a in ans.items():
        r, p = rows[i], a["plan"]
        e, got = r["expected"], plan_sets(p)
        rec = {"id": i, "type": r["type"], "kind": p["kind"] if p else None, "kind_ok": bool(p) and p["kind"] == e["kind"],
               "secs": a["secs"]}
        if e["kind"] == "passages":
            rec |= {f"{k}_ok": got[k] == set(e.get(k, [])) for k in ("ticker", "fy", "form")}
        if e["kind"] == "aggregate" and p and p["aggregate"]:
            rec |= {f"agg_{k}_ok": p["aggregate"][k] == e[k] for k in ("topic", "dimension", "unit")}
        if r["keywords"]:
            ctx = [texts[d["chunk_id"]] for d in a["docs"]]
            first = [next((n for n, t in enumerate(ctx, 1) if contains(t, kw)), None) for kw in r["keywords"]]
            rec |= {"mrr": sum(1 / f for f in first if f) / len(first), "coverage": sum(f is not None for f in first) / len(first)}
        valid = [d < len(a["docs"]) and " ".join(s.split()) in " ".join(texts[a["docs"][d]["chunk_id"]].split())
                 for d, s in a["citations"]]
        rec |= {"citations": len(valid), "citations_valid": sum(valid)}
        recs.append(rec)
    df = pd.DataFrame(recs)
    for j in load(judged_path):
        for k in ("accuracy", "completeness", "groundedness"):
            df.loc[df.id == j["id"], k] = j[k]
        df.loc[df.id == j["id"], "unsupported"] = bool(j["unsupported_claims"])
        df.loc[df.id == j["id"], "claims"] = len(j["unsupported_claims"])
    return df


def report() -> None:
    rows, ans = {r["id"]: r for r in load(TESTS)}, {a["id"]: a for a in load(ANSWERS) if not a["stopped"]}
    df = records(ANSWERS, JUDGED)
    p = df[df.type.isin(["single", "conversation", "multi_part"])]
    oos = df[df.type == "out_of_scope"]
    usd, judge_usd = sum(x["usd"] for x in load(RUNS)), sum(j["usd"] for j in load(JUDGED))
    lines = [f"{len(df)} of {len(rows)} rows answered for ${usd:.3f} (${usd / len(df):.4f} a row; median {df.secs.median():.0f} s); "
             f"judged: {df.accuracy.notna().sum() if 'accuracy' in df else 0}, ${judge_usd:.3f}", "",
             "| Metric | Value | Bar |", "|---|---|---|"]
    def row(name, v, bar=""):
        lines.append(f"| {name} | {v} | {bar} |")
    for k in ("ticker", "fy", "form"):
        row(f"Planner {k} right ({len(p)} passage questions)", f"{p[f'{k}_ok'].mean():.0%}", "≥ 90%" if k != "form" else "")
    row("Plan kind right, passage questions", f"{p.kind_ok.mean():.0%}")
    agg = df[df.type == "aggregate"]
    row(f"Aggregate questions routed to the cells ({len(agg)})", f"{agg.kind_ok.mean():.0%}")
    if "agg_topic_ok" in agg:
        row("… with the expected topic / dimension / unit", " / ".join(f"{agg[f'agg_{k}_ok'].mean():.0%}" for k in ("topic", "dimension", "unit")))
    row(f"Out-of-scope declined ({len(oos)})", f"{oos.kind_ok.mean():.0%}", "≥ 80%")
    row("In-scope questions declined", f"{((df.type != 'out_of_scope') & (df.kind == 'decline')).sum()} of {len(df) - len(oos)}")
    kw = df[df.mrr.notna()] if "mrr" in df else df[:0]
    row(f"Keyword MRR / coverage in the answer's passages ({len(kw)})", f"{kw.mrr.mean():.3f} / {kw.coverage.mean():.0%}")
    row("Cited spans found in their passage", f"{df.citations_valid.sum()} of {df.citations.sum()}")
    if "accuracy" in df and df.accuracy.notna().any():
        g = df[df.accuracy.notna()]
        for k in ("accuracy", "completeness", "groundedness"):
            row(f"Judge {k}, mean of 1–5 ({len(g)})", f"{g[k].mean():.2f}", "≥ 4" if k == "groundedness" else "")
        row("Answers with an unsupported claim", f"{g.unsupported.mean():.0%} ({int(g.unsupported.sum())})", "≤ 5%")
        lines += ["", "By type (judge means): " + "; ".join(
            f"{t} {x.accuracy.mean():.2f} / {x.completeness.mean():.2f} / {x.groundedness.mean():.2f} ({len(x)})" for t, x in g.groupby("type"))]
    print("\n".join(lines))
    misses = df[(df.get("ticker_ok") == False) | (df.get("fy_ok") == False)] if "ticker_ok" in df else df[:0]  # noqa: E712
    for i in misses.id:
        print(f"  planner miss {i}: expected {rows[i]['expected']}, got {plan_sets(ans[i]['plan'])}")
    if ANSWERS == HERE / "qa_answers.jsonl" and not HAND.exists() and "accuracy" in df and df.accuracy.notna().any():
        pick = random.Random(0).sample(sorted(df[df.accuracy.notna()].id), 10)
        pd.DataFrame({"id": pick, "question": [rows[i]["turns"][-1] for i in pick], "accuracy": "", "completeness": "",
                      "groundedness": "", "unsupported": ""}).to_csv(HAND, index=False)
        print(f"\nwrote {HAND}: 10 answers to grade by hand, to check the judge")


def diff(base: str, base_judge: str | None = None, n_boot: int = 10_000) -> None:
    """Each metric for the run named `base` (its answers, and judge file base_judge or its own) and the current
    one, on the questions both have, with a 95% bootstrap interval (resampling questions) for the paired difference."""
    bj = base if base_judge is None else base_judge
    a, b = records(HERE / f"qa_answers{base}.jsonl", HERE / f"qa_judge{bj}.jsonl"), records(ANSWERS, JUDGED)
    a, b = a.set_index("id"), b.set_index("id")
    passage = b.type.isin(["single", "conversation", "multi_part"])
    metrics = {"ticker_ok": passage, "fy_ok": passage, "form_ok": passage, "kind_ok": None,
               "mrr": passage, "coverage": passage, "accuracy": None, "completeness": None, "groundedness": None,
               "unsupported": None, "claims": None, "secs": None}
    rng = np.random.default_rng(0)
    print(f"| Metric | n | {bj or 'published'} | {JUDGED.stem.removeprefix('qa_judge') or 'published'} | new − old [95% CI] |"
          "\n|---|---|---|---|---|")
    for m, keep in metrics.items():
        if m not in a or m not in b:
            continue
        ids = b.index[keep if keep is not None else slice(None)]
        x = pd.DataFrame({"old": a[m].reindex(ids), "new": b[m].reindex(ids)}).dropna().astype(float)
        if x.empty:
            continue
        d = (x.new - x.old).to_numpy()
        boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"| {m} | {len(x)} | {x.old.mean():.3f} | {x.new.mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")


GPT_JUDGE = "gpt-6-sol"  # OpenAI's middle tier, priced like Claude Sonnet 5.5
GPT_PRICES = {"gpt-6-sol": (2.00, 10.00)}  # USD per 1M input and output tokens (OpenAI pricing page, checked 2026-10-18)
GPT_BATCH = HERE / "qa_batch_gpt.json"
GPT_MAX_OUTPUT = 1500  # reasoning included; bounds the cost a batch can reach, since a running batch can't be capped
SCORES = ("accuracy", "completeness", "groundedness")


def gpt_judged(model: str) -> Path:
    return HERE / f"qa_judge_{model}.jsonl"


def gpt_body(model: str, prompt: str) -> dict:
    from openai.lib._pydantic import to_strict_json_schema

    return {"model": model, "instructions": JUDGE_SYSTEM, "input": prompt, "reasoning": {"effort": "medium"},
            "max_output_tokens": GPT_MAX_OUTPUT,
            "text": {"format": {"type": "json_schema", "name": "grade", "schema": to_strict_json_schema(Grade), "strict": True}}}


def judge_gpt(model: str, budget: float, test: bool, rec: dict | None = None) -> None:
    """A second judge from another model family, on the answers the Claude judge graded: the same system
    prompt, user prompt and output schema, through OpenAI's Batch API (half price). --test sends one
    request directly first, to check the model takes the request; its cost goes in `rec` (the run log's)."""
    from openai import OpenAI

    client, (pin, pout) = OpenAI(), GPT_PRICES[model]
    rows, ans = {r["id"]: r for r in load(TESTS)}, {a["id"]: a for a in load(ANSWERS)}
    ids = [j["id"] for j in load(JUDGED) if j["id"] not in {g["id"] for g in load(gpt_judged(model))}]
    prompts = {i: judge_prompt(rows[i], ans[i]) for i in ids}
    if test:
        r = client.responses.create(**gpt_body(model, prompts[ids[0]]))
        usd = (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
        if rec is not None:
            rec["usd"] = {"openai": usd}
        print(ids[0], Grade.model_validate_json(r.output_text), f"${usd:.4f}")
        return
    if not GPT_BATCH.exists():
        est = sum(((len(JUDGE_SYSTEM) + len(p)) / 3.5 * pin + GPT_MAX_OUTPUT * pout) / 1e6 for p in prompts.values()) / 2
        print(f"{len(prompts)} answers to judge with {model}, at most about ${est:.2f}")
        if est > budget:
            raise SystemExit(f"over --budget ${budget:.2f}; nothing submitted")
        lines = [json.dumps({"custom_id": i, "method": "POST", "url": "/v1/responses", "body": gpt_body(model, p)})
                 for i, p in prompts.items()]
        f = client.files.create(file=("qa_judge.jsonl", "\n".join(lines).encode()), purpose="batch")
        b = client.batches.create(input_file_id=f.id, endpoint="/v1/responses", completion_window="24h")
        GPT_BATCH.write_text(json.dumps({"id": b.id}))
        print(f"submitted {b.id}", flush=True)
    bid = json.loads(GPT_BATCH.read_text())["id"]
    while (b := client.batches.retrieve(bid)).status not in ("completed", "failed", "expired", "cancelled"):
        print(f"{bid}: {b.status}, {b.request_counts.completed} of {b.request_counts.total} done", flush=True)
        time.sleep(60)
    out, usd = load(gpt_judged(model)), 0.0
    for line in client.files.content(b.output_file_id).text.splitlines() if b.output_file_id else []:
        res = json.loads(line)
        body = (res.get("response") or {}).get("body") or {}
        u = body.get("usage") or {}
        cost = (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout) / 1e6 / 2
        usd += cost
        text = next((c["text"] for o in body.get("output", []) if o.get("type") == "message"
                     for c in o.get("content", []) if c.get("type") == "output_text"), None)
        try:
            out.append({"id": res["custom_id"], **Grade.model_validate_json(text).model_dump(), "usd": cost, "batch": bid})
        except (TypeError, ValueError):
            print(f"{res['custom_id']}: no grade ({res.get('error') or body.get('status')})")
    if rec is not None:
        rec["usd"] = {"openai": usd}
    gpt_judged(model).write_text("".join(json.dumps(j) + "\n" for j in out))
    GPT_BATCH.unlink()
    print(f"{b.status}: {len(out)} graded, ${usd:.3f}")


def compare(model: str) -> None:
    """How the two judges agree, and the answers where they differ most (eval/qa_judge_divergences.md)."""
    from sec_filings_tone.sentiment import kappa_ci

    a, b = {j["id"]: j for j in load(JUDGED)}, {j["id"]: j for j in load(gpt_judged(model))}
    ids = sorted(a.keys() & b.keys())
    print(f"{len(ids)} answers graded by both {JUDGE_MODEL} and {model}\n\n| Score | {JUDGE_MODEL} | {model} | "
          "identical | within 1 | weighted kappa [95%] |\n|---|---|---|---|---|---|")
    for k in SCORES:
        x, y = [a[i][k] for i in ids], [b[i][k] for i in ids]
        kap, lo, hi = kappa_ci(x, y, [1, 2, 3, 4, 5])
        print(f"| {k} | {sum(x) / len(x):.2f} | {sum(y) / len(y):.2f} | {sum(p == q for p, q in zip(x, y)) / len(x):.0%} | "
              f"{sum(abs(p - q) <= 1 for p, q in zip(x, y)) / len(x):.0%} | {kap:.2f} [{lo:.2f}, {hi:.2f}] |")
    fa, fb = [bool(a[i]["unsupported_claims"]) for i in ids], [bool(b[i]["unsupported_claims"]) for i in ids]
    print(f"| an unsupported claim | {sum(fa) / len(ids):.0%} | {sum(fb) / len(ids):.0%} | "
          f"{sum(p == q for p, q in zip(fa, fb)) / len(ids):.0%} agree | | |")
    gap = {i: sum(abs(a[i][k] - b[i][k]) for k in SCORES) for i in ids}
    big = [i for i in sorted(ids, key=lambda i: -gap[i]) if gap[i] >= 3 or max(abs(a[i][k] - b[i][k]) for k in SCORES) >= 2]
    rows, ans = {r["id"]: r for r in load(TESTS)}, {x["id"]: x for x in load(ANSWERS)}
    doc = [f"# Where the two judges differ most ({len(big)} of {len(ids)} answers)\n",
           "Scores are accuracy / completeness / groundedness, 1–5. Listed when the summed difference is at least 3 "
           "or one score differs by 2 or more. Written by `eval/qa_eval.py compare`.\n"]
    for i in big:
        g = lambda j: f"{j['accuracy']} / {j['completeness']} / {j['groundedness']}"
        claims = lambda j: "".join(f"\n  - {c}" for c in j["unsupported_claims"]) or " none"
        doc.append(f"## {i} (difference {gap[i]})\n\n**Question:** {' → '.join(rows[i]['turns'])}\n\n"
                   f"**Reference:** {rows[i]['reference_answer']}\n\n**Answer:**\n\n{ans[i]['text']}\n\n"
                   f"**{JUDGE_MODEL}:** {g(a[i])}. {a[i]['notes']}\n- unsupported:{claims(a[i])}\n\n"
                   f"**{model}:** {g(b[i])}. {b[i]['notes']}\n- unsupported:{claims(b[i])}\n")
    (HERE / "qa_judge_divergences.md").write_text("\n".join(doc))
    print(f"\n{len(big)} answers with a large difference: eval/qa_judge_divergences.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["answers", "judge", "report", "diff", "judge-gpt", "compare"])
    ap.add_argument("--budget", type=float, default=0.70)
    ap.add_argument("--model", default=answer.MODEL, help="answers: the answer model; judge-gpt and compare: the OpenAI judge")
    ap.add_argument("--effort", default=answer.EFFORT)
    ap.add_argument("--test", action="store_true", help="judge-gpt: one direct request, no batch")
    ap.add_argument("--tests", type=Path, default=TESTS, help="the questions (e.g. eval/qa_dev.jsonl)")
    ap.add_argument("--name", default="", help="run name: writes qa_answers{name}.jsonl, qa_judge{name}.jsonl, "
                    "qa_runs{name}.jsonl instead of the published files (e.g. _v2)")
    ap.add_argument("--judge-name", help="judge, report, diff: the judge file's name, if not --name's")
    ap.add_argument("--sample", type=int, default=0, help="judge: only a random N of the answers (seed 0)")
    ap.add_argument("--draft", action="store_true", help="judge: grade the saved drafts the check changed (_v2 only)")
    ap.add_argument("--base", default="", help="diff: the run to compare with (default the published one)")
    ap.add_argument("--base-judge", help="diff: the base run's judge file name, if not --base's")
    a = ap.parse_args()
    jn = a.name if a.judge_name is None else a.judge_name
    TESTS, ANSWERS, RUNS = a.tests, HERE / f"qa_answers{a.name}.jsonl", HERE / f"qa_runs{a.name}.jsonl"
    JUDGED, BATCH = HERE / f"qa_judge{jn}.jsonl", HERE / f"qa_batch{jn}.json"
    gpt = a.model if a.model in GPT_PRICES else GPT_JUDGE
    cap = Cap(a.budget)

    def collected(path: Path, pending: Path, provider: str):
        """usage() for a batch judge: the grades saved during the run, their batch, and a batch still waiting."""
        before = {j["id"] for j in load(path)}

        def usage():
            new = [j for j in load(path) if j["id"] not in before]
            return {"usd": {provider: sum((j["usd"] for j in new), 0.0)}, "rows": len(new),
                    "batches_collected": sorted({j["batch"] for j in new}),
                    "batch_pending": json.loads(pending.read_text())["id"] if pending.exists() else None}
        return usage

    # the paid steps write one line each to the run log
    log = {"answers": lambda: run_log([a.model, retrieve.RERANK_MODEL, EMBED_MODEL],  # query embeddings: not costed
                                      lambda: {"usd": {"anthropic": cap.spent()}}, [ANSWERS, RUNS], effort=a.effort),
           "judge": lambda: run_log([JUDGE_MODEL], collected(JUDGED, BATCH, "anthropic"), [JUDGED]),
           "judge-gpt": lambda: run_log([gpt], collected(gpt_judged(gpt), GPT_BATCH, "openai"), [gpt_judged(gpt)])}
    with log.get(a.step, nullcontext)() as rec:
        {"answers": lambda: answers(cap, a.model, a.effort, rec), "judge": lambda: judge(a.budget, a.sample, a.draft, rec),
         "report": report, "diff": lambda: diff(a.base, a.base_judge),
         "judge-gpt": lambda: judge_gpt(gpt, a.budget, a.test, rec), "compare": lambda: compare(gpt)}[a.step]()
