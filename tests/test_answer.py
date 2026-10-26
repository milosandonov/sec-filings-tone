"""Run: uv run pytest tests/test_answer.py  (offline: fixed plans and synthetic cells, no API calls)"""

import math

import pandas as pd

from sec_filings_tone import answer as A

TICKERS = {"NVDA", "AMD", "JPM"}


def sub(**kw):
    q = dict(query="export controls", tickers=[], sector=None, form=None, item=None, fy=[], date_from=None,
             date_to=None, topic=None)
    return A.SubQuery(**(q | kw))


def plan(*qs, **kw):
    return A.Plan(**({"kind": "passages", "message": "", "queries": list(qs), "aggregate": None} | kw))


def agg(**kw):
    a = dict(topic="Tariffs", sectors=[], fy=[2025], item=None, dimension="realization", lowest_first=False,
             by_change=False, unit="companies")
    return A.Aggregate(**(a | kw))



def test_check():
    # the planner's filters, checked against the data
    p = A.check(plan(sub(tickers=["nvda"], fy=[2022]), sub(tickers=["AMD"], fy=[2024])), TICKERS)
    assert p.kind == "passages" and [q.tickers for q in p.queries] == [["NVDA"], ["AMD"]]
    assert A.filters(p.queries[0]) == {"ticker": ["NVDA"], "fy": [2022]}
    p = A.check(plan(sub(tickers=["RIVN"])), TICKERS)  # not in the data: ask, don't guess
    assert p.kind == "decline" and "RIVN" in p.message and not p.queries
    p = A.check(plan(sub(fy=[2031])), TICKERS)
    assert p.kind == "decline" and "2031" in p.message
    assert A.check(plan(sub(date_from="March 2023")), TICKERS).kind == "decline"
    assert A.check(plan(sub(date_from="2023-03-10", date_to="2023-09-30")), TICKERS).kind == "passages"
    assert len(A.check(plan(*[sub(query=str(i)) for i in range(9)]), TICKERS).queries) == A.MAX_QUERIES
    assert A.check(plan(), TICKERS).kind == "decline"  # passages with nothing to search
    assert A.check(plan(kind="aggregate"), TICKERS).kind == "decline"
    assert A.check(plan(kind="aggregate", aggregate=agg(fy=[2018])), TICKERS).kind == "decline"
    # an item the form doesn't have: 10-Qs have only MD&A ("mdna"), 10-Ks call theirs "7"
    assert A.check(plan(sub(form="10-Q", item="7")), TICKERS).queries[0].item == "mdna"
    assert A.check(plan(sub(form="10-K", item="mdna")), TICKERS).queries[0].item == "7"
    assert A.check(plan(sub(form="10-K", item="1A")), TICKERS).queries[0].item == "1A"
    d = A.check(plan(kind="decline", message="I can't predict stock prices."), TICKERS)
    assert d.kind == "decline" and d.message.startswith("I can't")


def test_merge():
    # merging sub-queries: round robin, first place kept, capped
    a = pd.DataFrame({"chunk_id": ["a1", "a2", "x"], "fy": [2022] * 3})
    b = pd.DataFrame({"chunk_id": ["x", "b2"], "fy": [2024] * 2})
    assert A.merge([a, b]).chunk_id.tolist() == ["a1", "x", "a2", "b2"]
    assert A.merge([a, b], n=2).chunk_id.tolist() == ["a1", "x"]
    assert A.merge([a, b]).fy.dtype == a.fy.dtype  # rows keep their dtypes
    assert A.merge([a.head(0), b.head(0)]).empty and A.merge([]).empty


def test_rank_and_top_cells():
    # the cell table: synthetic cells, three companies in two sectors
    c = pd.DataFrame({
        "ticker": ["A", "A", "B", "B", "C", "C", "A"], "fy": [2023, 2025, 2023, 2025, 2023, 2025, 2025],
        "topic": ["Tariffs"] * 6 + ["China"], "sector": ["Energy", "Energy", "Energy", "Energy", "Utilities", "Utilities", "Energy"],
        "polarity": [-1.0] * 7, "magnitude": [1.0] * 7, "uncertainty": [2.0] * 7, "specificity": [1.0] * 7,
        "realization": [0.0, 2.0, 1.0, 1.5, math.nan, 0.5, 2.0], "discussed": [1, 1, 1, 1, 0, 1, 1.0]})
    names = pd.Series({"A": "Alpha", "B": "Beta", "C": "Gamma"})
    t, counts = A.rank(c, agg(), names)
    assert t._id.tolist() == ["A", "B", "C"] and t["realization FY2025"].tolist() == ["2.00", "1.50", "0.50"]
    assert t.Company[0] == "A (Alpha)" and "FY2025: 3 of 3 companies' 10-Ks discuss Tariffs" in counts
    t, _ = A.rank(c, agg(lowest_first=True), names)
    assert t._id.tolist() == ["C", "B", "A"]
    t, _ = A.rank(c, agg(fy=[2023, 2025], by_change=True), names)  # A +2, B +0.5, C has no FY2023 value
    assert t._id.tolist() == ["A", "B", "C"] and t.change.tolist() == ["+2.00", "+0.50", "–"]
    t, _ = A.rank(c, agg(sectors=["Utilities"]), names)
    assert t._id.tolist() == ["C"]
    t, _ = A.rank(c, agg(unit="sectors", fy=[2023, 2025]), names)  # Energy: (0+1)/2 -> (2+1.5)/2; Utilities: only C, not discussed in 2023
    assert t._id.tolist() == ["Energy", "Utilities"] and t["realization FY2023"].tolist() == ["0.50", "–"]
    t, _ = A.rank(c, agg(unit="sectors", fy=[2023], dimension="discussed"), names)  # share discussing
    assert t["discussed FY2023"].tolist() == ["100%", "0%"]
    t, _ = A.rank(c, agg(fy=[2023], dimension="discussed"), names)
    assert t["discussed FY2023"].tolist() == ["yes", "yes", "no"]
    assert A.rank(c, agg(topic="AI"), names)[0].empty
    # ties: the more specific cell first, then alphabetical; the count of tied rows is reported
    tie = c.assign(realization=[0, 2, 0, 2, 0, 2, 0], specificity=[1, 1, 1, 2, 1, 1, 1], discussed=1.0)
    t, counts = A.rank(tie, agg(), names)
    assert t._id.tolist() == ["B", "A", "C"] and "3 companies share the highest value, 2.00" in counts
    # which cells to quote: top companies, or the best company in each top sector
    assert A.top_cells(c, ["A", "B", "C"], agg()) == [("A", 2025), ("B", 2025), ("C", 2025)]
    assert A.top_cells(c, ["A", "B"], agg(), n=1) == [("A", 2025)]
    assert A.top_cells(c, ["Utilities", "Energy"], agg(unit="sectors")) == [("C", 2025), ("A", 2025)]


def test_cell_passages():
    # a cell's own on-topic passages, highest on the dimension first, word-checked when keep is given
    p = pd.DataFrame({"pid": [1, 2, 3, 4, 5], "ticker": ["A"] * 4 + ["B"], "fy": [2025] * 5, "topic": ["Tariffs"] * 5,
                      "rank": [1, 2, 3, 1, 1], "chunk_id": ["a1", "a2", "a3", "a4", "b1"],
                      "off_topic": [False, False, True, False, False], "item": ["1A", "7", "1A", "1A", "1A"],
                      "realization": [1, 2, 2, 0, 2]})
    p.loc[3, "fy"] = 2024
    assert A.cell_passages(p, [("A", 2025)], agg()) == ["a2", "a1"]  # a3 is off topic, a4 another year
    assert A.cell_passages(p, [("A", 2025)], agg(item="1A")) == ["a1"]
    assert A.cell_passages(p, [("A", 2025), ("B", 2025)], agg(), keep={1, 5}) == ["a1", "b1"]
    assert A.cell_passages(p, [("A", 2025)], agg(lowest_first=True), per=1) == ["a1"]
    assert A.cell_passages(p, [("A", 2025)], agg(dimension="discussed")) == ["a1", "a2"]  # by search rank


def test_profile():
    # the tone profile: each cell with its change from the year before
    y = pd.DataFrame({"ticker": ["A", "A", "C", "C"], "fy": [2024, 2025, 2024, 2025], "topic": ["Tariffs"] * 4,
                      "polarity": [-1.0] * 4, "magnitude": [1.0] * 4, "uncertainty": [2.0] * 4, "specificity": [1.0] * 4,
                      "realization": [0.5, 2.0, math.nan, 0.5], "discussed": [1, 1, 0, 1.0]})
    pr = A.profile(y, [("A", 2025, "Tariffs"), ("C", 2025, "Tariffs"), ("A", 2024, "Tariffs"), ("Z", 2025, "Tariffs")])
    assert pr.ticker.tolist() == ["A", "C", "A"]  # Z has no cell
    assert pr.realization_change[0] == 1.5 and pr.polarity_change[0] == 0.0
    assert math.isnan(pr.realization_change[1])  # C wasn't discussed in FY2024
    assert math.isnan(pr.realization_change[2])  # no FY2023 cell: only the year just before counts
    table = A.profile_md(pr)
    assert "| A | 2025 | Tariffs* | -1.00 (+0.00) |" in table and "2.00 (+1.50)" in table


def test_reply_and_footer():
    # replies sent back as history lose everything from FOOTER on
    reply = "NVIDIA recorded a charge.[1]" + A.FOOTER + "**Sources**\n\n1. NVDA ..."
    assert reply.split(A.FOOTER)[0] == "NVIDIA recorded a charge.[1]"
    docs = pd.DataFrame({"ticker": ["NVDA", "NVDA"], "form": ["10-Q", "10-K"], "fy": [2026, 2025],
                         "report_date": ["2025-04-27", "2025-01-26"], "item": ["mdna", "1A"]})
    assert A.title(next(docs.itertuples())) == "NVDA 10-Q quarter ending 2025-04-27, MD&A"
    assert A.title(list(docs.itertuples())[1]) == "NVDA 10-K FY2025 (year ended 2025-01-26), Item 1A Risk Factors"
    out = {"docs": docs, "citations": [(1, "Export  controls\nmay hurt."), (0, "H20")], "cells": "", "profile": None, "usd": 0.002}
    f = A.footer(out, A.Cap(0.5), "claude-haiku-5-5", "low")
    assert "1. NVDA 10-K FY2025 (year ended 2025-01-26), Item 1A Risk Factors: “Export controls may hurt.”" in f
    assert "2. NVDA 10-Q" in f and "removed" not in f
    assert "2 sentences removed" in A.footer(out | {"dropped": ["a", "b"]}, A.Cap(0.5), "claude-haiku-5-5", "low")
    assert f.startswith(A.FOOTER) and "$0.0020" in f and "of $0.50" in f
    assert A.md(pd.DataFrame({"a": [1], "b": ["x"]})) == "| a | b |\n|---|---|\n| 1 | x |"


def test_drop():
    # the check's unsupported sentences come out of the answer; ones not found word for word are ignored
    text = ("**Short answer:** NVIDIA took a charge.[1] It was large.\n\n"
            "- **Costs:** The filing doesn't say more.\n- Sales fell.[2] Mainly in China.\n\nNot in the passages found: Q3.")
    new, gone = A.drop(text, ["It was large.", "The filing doesn't say more.", "Mainly in China.", "Not in the answer."])
    assert gone == ["It was large.", "The filing doesn't say more.", "Mainly in China."]
    assert new == "**Short answer:** NVIDIA took a charge.[1]\n\n- Sales fell.[2]\n\nNot in the passages found: Q3."
    assert A.drop(text, []) == (text, [])
