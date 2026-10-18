"""Phase 10: answer questions about the filings, with cited passages and the tone profile.

Each turn is a fixed workflow. A planner call (structured output) turns the question and the
conversation so far into one of three plans: sub-queries with filters, a lookup in the tone cells
for "which companies / which sectors" questions (a search returns only a few passages, so it can't
list every company in a sector), or a decline. retrieve.search (hybrid + rerank) finds passages for
each sub-query; an aggregate plan ranks events.cells() and quotes the top cells' own passages. Then
Claude answers from the passages, given as documents with citations on, streamed. Under the answer:
the cited spans, the tone profile of the cells it draws on with the change from the year before,
and the cost.

Usage:
    uv run python -m sec_filings_tone.answer "What charge did NVIDIA record for H20 exports to China?"
    uv run python -m sec_filings_tone.answer "Apple's supply-chain risks in FY2023?" "How did that change in FY2025?"
    uv run python -m sec_filings_tone.answer "Which sectors treated tariffs as realized in FY2025?" --plan-only
    uv run python -m sec_filings_tone.answer "..." --model claude-sonnet-5-5 --effort medium --budget 0.10
"""

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import cache
from typing import Literal

import anthropic
import openai
import pandas as pd
from pydantic import BaseModel, Field

from sec_filings_tone import events, retrieve
from sec_filings_tone.budget import PRICES, Cap, CapReached, call_usd
from sec_filings_tone.ingest import CHUNKS, ITEM_NAMES
from sec_filings_tone.sentiment import DIMENSIONS, TOPICS
from sec_filings_tone.tone import FILES

MODEL = "claude-haiku-5-5"
EFFORT = "low"
TIMEOUT = 120  # seconds per call
PLAN_TOKENS, ANSWER_TOKENS = 8000, 8000  # thinking counts toward both
MAX_QUERIES = 6
K = 6  # passages kept per sub-query
MAX_PASSAGES = 24
MAX_ROWS = 30  # rows of a cell table
EXAMPLES = 6  # passages quoted for an aggregate answer
FYS = range(2019, 2027)
DIMS = list(DIMENSIONS)
FOOTER = "\n\n---\n"  # starts the sources, tables and cost under an answer; cut off before it goes back to the model
SECTORS = tuple(sorted(pd.read_csv(events.COMPANIES).sector.unique()))

Sector = Literal[SECTORS]
Topic = Literal[tuple(TOPICS)]


class SubQuery(BaseModel):
    query: str = Field(description="Standalone search text in the filings' own words")
    tickers: list[str] = Field(description="Tickers from the company list; empty for any company")
    sector: Sector | None
    form: Literal["10-K", "10-Q"] | None
    item: Literal["1A", "7", "mdna"] | None = Field(description="1A: 10-K risk factors, 7: 10-K MD&A, mdna: 10-Q MD&A")
    fy: list[int] = Field(description="Fiscal years; empty for any")
    date_from: str | None = Field(description="Earliest filing date, YYYY-MM-DD")
    date_to: str | None = Field(description="Latest filing date, YYYY-MM-DD")
    topic: Topic | None = Field(description="The topic from the list this sub-query is about, if any")


class Aggregate(BaseModel):
    topic: Topic
    sectors: list[Sector] = Field(description="Empty for every sector")
    fy: list[int] = Field(description="Fiscal years to compare, in order")
    item: Literal["1A", "7"] | None = Field(description="1A: risk factors, 7: MD&A, empty: both")
    dimension: Literal[(*DIMS, "discussed")]
    lowest_first: bool
    by_change: bool = Field(description="Rank on the change from the first to the last fiscal year")
    unit: Literal["companies", "sectors"]


class Plan(BaseModel):
    kind: Literal["passages", "aggregate", "decline"]
    message: str = Field(description="For decline: the reply to the user. Otherwise empty")
    queries: list[SubQuery]
    aggregate: Aggregate | None


PLANNER_RULES = """You plan how to answer a question about S&P 500 companies' SEC filings. You don't answer it; you decide what to look up.

The data: the 10-K risk factors (item "1A") and MD&A (item "7") of the current S&P 500 companies for fiscal years 2019 to 2025 (2026 only for companies whose fiscal year doesn't end in December), and their 10-Q MD&A (item "mdna"), quarters from 2019 to mid-2026. Nothing else: no stock prices, earnings calls, news or 8-Ks.

A fiscal year (fy) is the year a company's fiscal period ends. NVIDIA's ends in late January, so its FY2024 is mostly calendar 2023; a 52/53-week year ending January 1-7 counts as the year before. "Fiscal 2023", "FY2023" and "the 2023 10-K" all mean fy 2023. For a calendar date window ("the first 10-Qs after March 2023"), use 10-Qs with a filing-date range.

Choose kind:
- "passages": the answer is in the text of particular filings. Write one sub-query per company, fiscal year and aspect: comparing 2 companies in 2 years is 4 sub-queries. At most 6. Each query is standalone search text in the filings' own words; rewrite follow-ups from the conversation ("what about AMD?" after a question on NVIDIA's FY2024 supply chain becomes "supply chain risks" with ticker AMD and fy 2024). Put every company and year the question names into the filters: tickers from the company list, fy, form, item, sector, or a filing-date range. Leave a filter empty when the question doesn't limit it. Set topic when the sub-query is about one of the topics below.
- "aggregate": the question asks which or how many companies or sectors did something, across a sector or the whole index ("which Energy companies ...", "which sectors ..."). A search returns only a few passages, so these are answered from a table of tone scores: one row per company, 10-K and topic. Fill aggregate: topic (one of the topics below; if none fits, use "passages" with a sector filter instead), sectors (empty for all), fy (each year compared; for "after 2022", 2022 and the years after it), item ("1A" for risk factors, "7" for MD&A, empty for both), the dimension asked about, lowest_first (true for "most negative", "least ..."), by_change (true when the question asks what changed or was added between the first and the last year) and unit.
- "decline": the filings can't answer it: predictions, stock prices or returns, investment advice, companies outside the list, years outside the data, or anything else not in these filings. Write message: one or two sentences saying what you can't do and what the filings do cover. Also decline when the company or year is ambiguous, and ask in message which one the user means.

Tone dimensions, scored by Claude Haiku 5.5 for each company, 10-K and topic:
- polarity: does the topic help or hurt the company (-2 to +2)
- magnitude: how big is the described effect (0 to 3)
- realization: has it happened (0 hypothetical, 1 under way, 2 happened); use it for "actual", "already happened", "reported a hit", "has experienced"
- uncertainty: how unpredictable are the effects (0 to 3)
- specificity: how concrete (0 boilerplate, 1 company-specific, 2 named products, dates or amounts)
- discussed: whether the 10-K has a passage on the topic at all; use it for "mention", "discuss", "added"
"""

ANSWER_SYSTEM = """You answer an analyst's questions about S&P 500 companies' SEC filings (10-K risk factors and MD&A, 10-Q MD&A), using the passages you are given.
- Use only the given passages and, when given, the table of tone cells. Don't add outside knowledge of the companies or events.
- Cite the passages for every claim about what a filing says.
- Say plainly what the passages don't cover. If they don't answer the question, say so instead of guessing.
- Don't compute numbers that aren't in the text (sums, growth rates); quote amounts as written.
- No investment advice, price targets or predictions.
- Name the company, form and fiscal year when you report what a filing says. Mention a company's fiscal-year convention when it matters: NVIDIA's fiscal year ends in late January, so its FY2024 is mostly calendar 2023.
- Be concise: the short answer first, then the details, in a few short paragraphs or bullets."""

CELLS_NOTE = """The table below comes from model-scored cells: for each company's 10-K and topic, Claude Haiku 5.5 scored the 3 passages a search ranked highest on the topic, and a cell's value is the mean over the passages it judged on topic ({version}). Answer the "which" part from this table, and say that the list comes from model-scored cells and which version. Use the passages for examples and details. Don't say that companies outside the table did or didn't do something."""

PROFILE_NOTE = ("**Tone profile** of the cells this answer draws on: the mean over each 10-K's on-topic passages, "
                "scored by Claude Haiku 5.5 (polarity −2..+2, magnitude 0–3, realization 0–2, uncertainty 0–3, "
                "specificity 0–2), with the change from the year before in brackets. * word-checked: AI, China and "
                "tariff passages count only if they name the topic. Magnitude is the least stable dimension run to "
                "run, and 86% of polarity scores are −1.")


@cache
def client():
    return anthropic.Anthropic(timeout=TIMEOUT)


@cache
def companies() -> pd.DataFrame:
    """companies.csv rows for the tickers that have filings in the data."""
    have = set(pd.read_parquet(CHUNKS, columns=["ticker"]).ticker)
    c = pd.read_csv(events.COMPANIES)
    return c[c.ticker.isin(have)].sort_values("ticker").reset_index(drop=True)


@cache
def planner_system() -> str:
    """The same bytes every call, so it's read from the prompt cache after the first."""
    topics = "\n".join(f"- {t}: {q}" + (" (Financials only)" if t == "Deposits and liquidity" else "")
                       for t, q in TOPICS.items())
    rows = "\n".join(f"{r.ticker} | {r.name} | {r.sector} | {r.sub_industry}" for r in companies().itertuples())
    return f"{PLANNER_RULES}\n<topics>\n{topics}\n</topics>\n\n<companies>\nticker | name | sector | sub-industry\n{rows}\n</companies>"


def usd(model: str, u) -> float:
    """One call's cost from its usage: cache writes at 1.25x the input price, reads at 0.1x."""
    pin, pout = PRICES[model]
    tokens_in = u.input_tokens + 1.25 * (u.cache_creation_input_tokens or 0) + 0.1 * (u.cache_read_input_tokens or 0)
    return (tokens_in * pin + u.output_tokens * pout) / 1e6


def decline(message: str) -> Plan:
    return Plan(kind="decline", message=message, queries=[], aggregate=None)


def check(p: Plan, tickers: set[str]) -> Plan:
    """The plan with its filters checked against the data. An unknown ticker or fiscal year or a bad date
    becomes a question back to the user instead of a guess; sub-queries past MAX_QUERIES are dropped;
    an item that doesn't exist in the form (10-Qs have only "mdna") is mapped to the one that does."""
    if p.kind == "passages" and not p.queries or p.kind == "aggregate" and p.aggregate is None:
        return decline("I couldn't work out what to look up. Could you rephrase the question?")
    qs = [q.model_copy(update={"tickers": [t.upper() for t in q.tickers]}) for q in p.queries[:MAX_QUERIES]] \
        if p.kind == "passages" else []
    if bad := sorted({t for q in qs for t in q.tickers} - tickers):
        return decline(f"I have no filings for {', '.join(bad)}. The data covers the 10-Ks and 10-Qs of the "
                       "current S&P 500 companies. Which company did you mean?")
    years = {y for q in qs for y in q.fy} | set(p.aggregate.fy if p.kind == "aggregate" else [])
    if bad := sorted(years - set(FYS)):
        return decline(f"The filings cover fiscal years 2019 to 2025 (2026 for some companies), not "
                       f"{', '.join(map(str, bad))}. Which year did you mean?")
    for q in qs:
        for d in filter(None, (q.date_from, q.date_to)):
            try:
                date.fromisoformat(d)
            except ValueError:
                return decline(f"I couldn't read the date {d!r}. Which period did you mean?")
        if q.item and q.form:
            q.item = "mdna" if q.form == "10-Q" else "7" if q.item == "mdna" else q.item
    return p.model_copy(update={"queries": qs})


def plan(question: str, history: list[dict], cap: Cap, model: str = MODEL, effort: str = EFFORT) -> tuple[Plan, object]:
    """The checked plan and the call's usage."""
    cap.check(call_usd(model, 20_000, PLAN_TOKENS))  # upper bound: system prompt and history, full output
    convo = "\n\n".join(f"{m['role'].title()}: {m['content'][:1500]}" for m in history)
    text = (f"<conversation>\n{convo}\n</conversation>\n\n" if convo else "") + f"Question: {question}"
    resp = client().messages.parse(
        model=model, max_tokens=PLAN_TOKENS, output_config={"effort": effort},
        system=[{"type": "text", "text": planner_system(), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": text}], output_format=Plan)
    cap.charge(usd(model, resp.usage))
    if resp.parsed_output is None:
        return decline(f"I couldn't plan an answer (stop reason: {resp.stop_reason}). Try rephrasing the question."), resp.usage
    return check(resp.parsed_output, set(companies().ticker)), resp.usage


def filters(q: SubQuery) -> dict:
    f = {"ticker": q.tickers, "sector": q.sector, "form": q.form, "item": q.item, "fy": q.fy,
         "date_from": q.date_from, "date_to": q.date_to}
    return {k: v for k, v in f.items() if v}


def passages(qs: list[SubQuery], cap: Cap) -> list[pd.DataFrame]:
    """Each sub-query's reranked top K, tagged with its topic. Searches run in parallel."""
    cap.check(len(qs) * call_usd(retrieve.RERANK_MODEL, retrieve.CANDIDATES * 800, 8000))  # upper bound per rerank
    with ThreadPoolExecutor(len(qs)) as ex:
        return list(ex.map(lambda q: retrieve.search(q.query, K, True, **filters(q))["rerank"].assign(topic=q.topic), qs))


def merge(lists: list[pd.DataFrame], n: int = MAX_PASSAGES) -> pd.DataFrame:
    """Round robin over the ranked lists (every list's first, then every second, ...), so each sub-query
    keeps its best passages under the cap; a chunk found twice keeps its first place."""
    rows = [df.iloc[[i]] for i in range(max(map(len, lists), default=0)) for df in lists if i < len(df)]
    return pd.concat(rows).drop_duplicates("chunk_id").head(n).reset_index(drop=True) if rows else pd.DataFrame()


named = cache(events.named)


@cache
def all_cells(item: str | None, word_checked: bool) -> pd.DataFrame:
    return events.cells([item] if item else None, keep=named() if word_checked else None)


@cache
def profile_cells() -> pd.DataFrame:
    """Every 10-K cell, word-checked for the topics that have a vocabulary (events.TOPIC_WORDS)."""
    a, w = all_cells(None, False), all_cells(None, True)
    return pd.concat([a[~a.topic.isin(events.TOPIC_WORDS)], w[w.topic.isin(events.TOPIC_WORDS)]], ignore_index=True)


@cache
def scored(mdna_ai: bool) -> pd.DataFrame:
    """The annual passages with their scores and item, numbered (pid) as events.named numbers them. With
    mdna_ai, AI's passages are its top 3 within MD&A (events.section), as events.cells uses for MD&A."""
    p = pd.read_parquet(FILES["10-K"][0]).reset_index(names="pid").merge(pd.read_parquet(FILES["10-K"][1]), on="pid")
    if mdna_ai:
        s = events.section()
        p = pd.concat([p[~p.topic.isin(s.topic)], s[p.columns]], ignore_index=True)
    return p.assign(item=retrieve.chunks().item.loc[p.chunk_id].to_numpy())


def fmt(v, dim: str) -> str:
    if pd.isna(v):
        return "–"
    if dim == "discussed":
        return "yes" if v else "no"
    return f"{v:.0%}" if dim == "share" else f"{v:+.2f}" if dim == "polarity" else f"{v:.2f}"


def md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def rank(c: pd.DataFrame, a: Aggregate, names: pd.Series) -> tuple[pd.DataFrame, str]:
    """The cell table for an aggregate question, ranked, and a line of counts. Rows are companies or
    sectors, columns fiscal years. A company's value is its cell's; a sector's is the mean over its
    companies discussing the topic (for `discussed`: the share discussing it). Ranked on the last year,
    or with by_change on the last year minus the first; companies with no value are left out."""
    c = c[(c.topic == a.topic) & (c.sector.isin(a.sectors) if a.sectors else True)
          & (c.fy.isin(a.fy) if a.fy else c.fy.le(2025))]
    if c.empty:
        return pd.DataFrame(), "No cells match."
    t = c.pivot_table(index="sector" if a.unit == "sectors" else "ticker", columns="fy", values=a.dimension, aggfunc="mean")
    years = sorted(t.columns)
    if len(years) > 1:
        t["change"] = t[years[-1]] - t[years[0]]
    key = t["change"] if a.by_change and len(years) > 1 else t[years[-1]]
    t = t.loc[key.sort_values(ascending=a.lowest_first, na_position="last").index].dropna(how="all")
    n = c.groupby("fy").agg(companies=("ticker", "nunique"), discussing=("discussed", "sum"))
    counts = "; ".join(f"FY{y}: {int(r.discussing)} of {r.companies} companies' 10-Ks discuss {a.topic}" for y, r in n.iterrows())
    shown = t.head(MAX_ROWS)
    dim = a.dimension if a.dimension != "discussed" else "discussed" if a.unit == "companies" else "share"
    out = pd.DataFrame({"Sector" if a.unit == "sectors" else "Company":
                        shown.index if a.unit == "sectors" else [f"{x} ({names.get(x, '')})" for x in shown.index]})
    for y in years:
        out[f"{a.dimension} FY{y}"] = [fmt(v, dim) for v in shown[y]]
    if "change" in shown:
        out["change"] = [fmt(v, "polarity") for v in shown["change"]]
    if len(t) > MAX_ROWS:
        counts += f". Showing the top {MAX_ROWS} of {len(t)}."
    return out.assign(_id=shown.index.to_numpy()), counts


def top_cells(c: pd.DataFrame, ids: list[str], a: Aggregate, n: int = 3) -> list[tuple[str, int]]:
    """(ticker, fy) of the cells to quote: the top n companies in the last fiscal year asked, or for
    sectors, the company with the highest (lowest_first: lowest) value in each of the top n sectors."""
    fy = max(a.fy) if a.fy else 2025
    c = c[(c.topic == a.topic) & (c.fy == fy) & c[a.dimension].notna()]
    if a.unit == "companies":
        return [(t, fy) for t in ids if t in set(c.ticker)][:n]
    c = c.sort_values([a.dimension, "ticker"], ascending=[a.lowest_first, True])
    return [(c[c.sector == s].ticker.iloc[0], fy) for s in ids if s in set(c.sector)][:n]


def cell_passages(p: pd.DataFrame, cells_: list[tuple[str, int]], a: Aggregate, keep: set | None, per: int = 2) -> list[str]:
    """chunk_ids of each cell's on-topic passages (word-checked where keep is given), the ones scoring
    highest on the dimension first (lowest with lowest_first), then by search rank."""
    p = p[(p.topic == a.topic) & ~p.off_topic & (p["item"] == a.item if a.item else True)
          & (p.pid.isin(keep) if keep is not None else True)]
    by = ([a.dimension] if a.dimension in DIMS else []) + ["rank"]
    out = []
    for t, fy in cells_:
        x = p[(p.ticker == t) & (p.fy == fy)].sort_values(by, ascending=[a.lowest_first] * (len(by) - 1) + [True])
        out += x.chunk_id.head(per).tolist()
    return list(dict.fromkeys(out))[:EXAMPLES]


def aggregate(a: Aggregate) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """The cell table (markdown) for the answer model, the counts line, and the passages to quote."""
    checked = a.topic in events.TOPIC_WORDS
    c = all_cells(a.item, checked)
    names = companies().set_index("ticker").name
    table, counts = rank(c, a, names)
    if table.empty:
        return table, counts, pd.DataFrame()
    ids = cell_passages(scored(a.item == "7"), top_cells(c, table._id.tolist(), a), a, named() if checked else None)
    return table.drop(columns="_id"), counts, retrieve.chunks().loc[ids].reset_index(drop=True).assign(topic=a.topic)


def profile(c: pd.DataFrame, keys: list[tuple[str, int, str]]) -> pd.DataFrame:
    """Each (ticker, fy, topic) cell's dimensions and `discussed`, with each dimension's change from the
    same company and topic a fiscal year earlier (NaN where either isn't discussed). Keys without a
    cell are skipped."""
    c = c.set_index(["ticker", "fy", "topic"])
    rows = []
    for t, fy, topic in keys:
        if (t, fy, topic) in c.index:
            now = c.loc[(t, fy, topic)]
            prev = c.loc[(t, fy - 1, topic)] if (t, fy - 1, topic) in c.index else None
            rows.append({"ticker": t, "fy": fy, "topic": topic, "discussed": now.discussed,
                         **{d: now[d] for d in DIMS},
                         **{f"{d}_change": now[d] - prev[d] if prev is not None else math.nan for d in DIMS}})
    return pd.DataFrame(rows)


def profile_md(pr: pd.DataFrame) -> str:
    def cell(r, d):
        v, ch = r[d], r[f"{d}_change"]
        return fmt(v, d) + ("" if pd.isna(ch) else f" ({ch:+.2f})")
    out = pd.DataFrame({"Company": pr.ticker, "FY": pr.fy,
                        "Topic": [t + ("*" if t in events.TOPIC_WORDS else "") for t in pr.topic]})
    for d in DIMS:
        out[d.title()] = [cell(r, d) if r.discussed else "not discussed" for _, r in pr.iterrows()]
    return f"{PROFILE_NOTE}\n\n{md(out)}"


def title(r) -> str:
    when = f"FY{r.fy}" if r.form == "10-K" else f"quarter ending {r.report_date}"
    return f"{r.ticker} {r.form} {when}, {ITEM_NAMES[r.item]}"


def document(r, names: pd.Series) -> dict:
    return {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": r.text},
            "title": title(r), "citations": {"enabled": True},
            "context": f"{names.get(r.ticker, r.ticker)}, {r.sector}. Period ended {r.report_date}, filed "
                       f"{r.filing_date}. Section heading: {r.heading}"}


def stream(blocks: list, history: list[dict], cap: Cap, model: str, effort: str):
    """Yields (text so far, citations) as the answer streams; each cited block gets its citation
    numbers [n] at its end. Citations are unique (document, span) pairs, numbered from 1."""
    cap.check(call_usd(model, 40_000, ANSWER_TOKENS))  # upper bound: 24 passages and history, full output
    text, cites, block = "", [], []
    with client().messages.stream(model=model, max_tokens=ANSWER_TOKENS, output_config={"effort": effort},
                                  system=ANSWER_SYSTEM, messages=[*history, {"role": "user", "content": blocks}]) as s:
        for ev in s:
            if ev.type == "text":
                text += ev.text
            elif ev.type == "citation":
                block.append(ev.citation)
                continue
            elif ev.type == "content_block_stop" and block:
                keys = [(x.document_index, x.cited_text) for x in block]
                cites += [k for k in dict.fromkeys(keys) if k not in cites]
                text += "".join(f"[{cites.index(k) + 1}]" for k in dict.fromkeys(keys))
                block = []
            else:
                continue
            yield text, cites
        msg = s.get_final_message()
    cap.charge(usd(model, msg.usage))
    if msg.stop_reason == "refusal":
        yield ("The model declined to answer this (a safety filter"
               + (f", category {msg.stop_details.category}" if msg.stop_details else "") + "). Try rephrasing the question."), []
    elif msg.stop_reason == "max_tokens":
        yield text + "\n\n*(The answer was cut off at the length limit.)*", cites


def footer(out: dict, cap: Cap, model: str, effort: str) -> str:
    parts = []
    if out["citations"]:
        docs = out["docs"]
        parts.append("**Sources**\n\n" + "\n".join(
            f"{i}. {title(docs.iloc[d])}: “{' '.join(span.split())[:300]}”" for i, (d, span) in enumerate(out["citations"], 1)))
    if out["cells"]:
        parts.append(out["cells"])
    if out["profile"] is not None and len(out["profile"]):
        parts.append(profile_md(out["profile"]))
    of = f" of ${cap.usd:.2f}" if cap.usd < math.inf else ""
    parts.append(f"<sub>Cost of this answer ${out['usd']:.4f} · session ${cap.spent():.3f}{of} · {model}, effort {effort}</sub>")
    return FOOTER + "\n\n".join(parts)


def answer(question: str, history: list[dict] = (), cap: Cap | None = None, model: str = MODEL, effort: str = EFFORT):
    """Yields the reply as markdown, longer each time: the answer as it streams, then with the footer.
    Returns a dict with the plan, the passages given to the model, the citations, the text and the cost.
    history: earlier {"role", "content"} turns; anything after FOOTER in them is dropped."""
    cap = cap if cap is not None else Cap()
    history = [{"role": m["role"], "content": str(m["content"]).split(FOOTER)[0]} for m in history]
    start = cap.spent()
    out = {"question": question, "plan": None, "plan_usage": None, "docs": pd.DataFrame(), "citations": [],
           "cells": "", "profile": None, "text": ""}
    try:
        p, out["plan_usage"] = plan(question, history, cap, model, effort)
        out["plan"] = p
        notes, docs = [], pd.DataFrame()
        if p.kind == "passages":
            lists = passages(p.queries, cap)
            docs = merge(lists)
            notes = [f"No passage matched the search {q.query!r} with filters {filters(q)}."
                     for q, df in zip(p.queries, lists) if df.empty]
        elif p.kind == "aggregate":
            table, counts, docs = aggregate(p.aggregate)
            version = ("word-checked: passages count only if they name the topic" if p.aggregate.topic in events.TOPIC_WORDS
                       else "the model's own on-topic judgment, not word-checked")
            if len(table):
                out["cells"] = (f"**Tone cells**, {p.aggregate.topic}, {p.aggregate.dimension} "
                                f"({version}; {counts})\n\n{md(table)}")
                notes = [CELLS_NOTE.format(version=version) + f"\n{counts}\n\n{md(table)}"]
        out["docs"] = docs
        if p.kind == "decline":
            out["text"] = p.message
        elif out["docs"].empty and not out["cells"]:
            out["text"] = "I found no passages in the filings for that. " + " ".join(notes)
        else:
            names = companies().set_index("ticker").name
            blocks = [document(r, names) for r in out["docs"].itertuples()]
            blocks.append({"type": "text", "text": "\n\n".join([*notes, f"Question: {question}"])})
            for text, cites in stream(blocks, history, cap, model, effort):
                out["text"], out["citations"] = text, cites
                yield text
            keys = [(r.ticker, r.fy, r.topic) for r in out["docs"].itertuples() if r.form == "10-K" and r.topic]
            out["profile"] = profile(profile_cells(), list(dict.fromkeys(keys))[:8])
    except CapReached as e:
        out["text"] = f"Stopped: {e}. Start a new session or raise the budget."
    except (anthropic.APIError, openai.APIError) as e:  # openai: the query embeddings
        out["text"] = f"The API call failed: {e}"
    out["usd"] = cap.spent() - start
    out["reply"] = out["text"] + footer(out, cap, model, effort)
    yield out["reply"]
    return out


def run(question: str, history: list[dict] = (), cap: Cap | None = None, **kw) -> dict:
    """answer() without streaming: its result dict."""
    gen = answer(question, history, cap, **kw)
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


def warm() -> None:
    """Load what every question needs (chunks, the BM25 index, Chroma, the cells), so the first answer
    doesn't wait for it."""
    retrieve.chunks(), retrieve.bm25_index(), retrieve.collection(), profile_cells(), scored(False), companies()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="+", help="one question, or several asked in turn as one conversation")
    ap.add_argument("--model", default=MODEL, choices=list(PRICES))
    ap.add_argument("--effort", default=EFFORT, choices=["low", "medium", "high"])
    ap.add_argument("--budget", type=float, default=0.10, help="USD for the whole run")
    ap.add_argument("--plan-only", action="store_true", help="print each plan; no search or answer")
    a = ap.parse_args()
    cap, history = Cap(a.budget), []
    for q in a.questions:
        print(f"\n> {q}\n", flush=True)
        if a.plan_only:
            p, u = plan(q, history, cap, a.model, a.effort)
            print(p.model_dump_json(indent=1, exclude_defaults=True))
            print(f"tokens: {u.input_tokens} in, {u.cache_read_input_tokens} cache read, "
                  f"{u.cache_creation_input_tokens} cache write, {u.output_tokens} out; session ${cap.spent():.4f}")
            reply = "(plan only, no answer)"
        else:
            gen, reply = answer(q, history, cap, a.model, a.effort), ""
            while True:
                try:
                    new = next(gen)
                except StopIteration as e:
                    out = e.value
                    break
                print(new[len(reply):] if new.startswith(reply) else "\n" + new, end="", flush=True)
                reply = new
            if out["plan"]:
                print("\n\nplan: " + out["plan"].model_dump_json(exclude_defaults=True))
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
    print(f"\nspent ${cap.spent():.4f}")


if __name__ == "__main__":
    main()
