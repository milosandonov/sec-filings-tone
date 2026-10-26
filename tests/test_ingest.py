"""Run: uv run pytest tests/test_ingest.py"""

import pandas as pd
import pytest

from sec_filings_tone.ingest import (build_chunks, chunk_section, drop_financial_statements, drop_risk_factors,
                                     embed_text, n_tokens, queue_limit, take)

PARA = "Demand for our products may decline if customers delay purchases. " * 8  # ~100 tokens
LONG = "Export rules from the U.S. Government limit sales to China. " * 120  # ~1,400 tokens

DOC = f"""\
Table of Contents

Risks Related to Demand

{PARA}

Cash paid during the year 883 1,627 2,076 - 4,586 7,001 9,999
| % of net revenue | 14.2 | % | 27.2 | % |
| --- | --- | --- |

{PARA}

Risks Related to Export Controls

{LONG}

**Regulation\\.** ![chart](https://www.sec.gov/x.jpg) {PARA}
"""

CHUNKS = chunk_section(DOC, max_tokens=300, target_tokens=200)


def test_chunk_section():
    texts = " ".join(c["text"] for c in CHUNKS)
    assert all(c["n_tokens"] <= 300 and n_tokens(c["text"]) <= 300 for c in CHUNKS), [c["n_tokens"] for c in CHUNKS]
    assert "Table of Contents" not in texts and "14.2" not in texts and "---" not in texts  # furniture and figures dropped
    assert "Risks Related" not in texts  # headings go to metadata, not text
    assert CHUNKS[0]["heading"] == "Risks Related to Demand"
    assert CHUNKS[-1]["heading"] == "Risks Related to Export Controls"
    assert "Regulation." in texts and "\\" not in texts and "sec.gov" not in texts  # markup cleaned
    assert "U.S. Government" in texts  # sentence split doesn't break abbreviations
    # short paragraphs are packed whole, not split
    assert sum(PARA.strip() in c["text"] for c in CHUNKS) >= 2
    assert chunk_section(DOC, 300, 200) == CHUNKS  # deterministic


def test_embed_text():
    # embedded text: header + overlap from the previous chunk; the stored text stays clean
    rows = pd.DataFrame([{"ticker": "NVDA", "sector": "IT", "form": "10-K", "item": "1A", "fy": 2024,
                          "report_date": "2024-01-28", "seq": i, **c} for i, c in enumerate(CHUNKS)])
    emb = embed_text(rows, overlap_tokens=20)
    assert emb[0].startswith("NVDA (IT) 10-K FY2024, Item 1A Risk Factors — Risks Related to Demand\n\n")
    assert emb[1].endswith(CHUNKS[1]["text"]) and len(emb[1]) > len(CHUNKS[1]["text"]) + 60  # has overlap
    assert embed_text(rows, overlap_tokens=0)[1].split("\n\n", 1)[1] == CHUNKS[1]["text"]
    assert embed_text(rows, header="no_heading")[0].startswith("NVDA (IT) 10-K FY2024, Item 1A Risk Factors\n\n")
    assert embed_text(rows, overlap_tokens=0, header="none")[1] == CHUNKS[1]["text"]
    two = embed_text(rows, overlap_tokens=200, overlap_sentences=2)[1]
    assert len(two) > len(emb[1]) and two.endswith(CHUNKS[1]["text"])  # two sentences of overlap


def test_lists_footers_and_tails():
    # list items aren't headings, so their text stays; page footers are dropped, not used as headings
    doc = f"""\
Risks Related to Demand

{PARA}

•our IP rights may be weak in some countries; or

•we may have to write off goodwill; and

Lam Research Corporation 2019 10-K 14

Three Months Ended March 27, September 26, Change Cash and cash equivalents $7,173 $7,116

{PARA}

Fiscal Year 2020 Compared to Fiscal Year 2019

{PARA}
"""
    c = chunk_section(doc, 300, 200)
    assert "our IP rights may be weak" in c[0]["text"] and "write off goodwill" in c[0]["text"]
    assert c[0]["heading"] == "Risks Related to Demand" and c[-1]["heading"] == "Fiscal Year 2020 Compared to Fiscal Year 2019"
    assert not any("10-K 14" in x["heading"] + x["text"] or "$7,173" in x["heading"] + x["text"] for x in c)
    # a tiny last chunk joins the one before it
    tail = chunk_section(f"{PARA}\n\n{PARA}\n\nItem 1B: Unresolved Staff Comments\n\nNone.\n", 300, 150)
    assert tail[-1]["text"].endswith("None.") and all(x["n_tokens"] >= 50 for x in tail)


def test_drop_risk_factors():
    # a Risk Factors section inside a 10-Q MD&A is cut; a stub pointing to the 10-K's Item 1A is not
    mdna = "Results of Operations\n\nRevenue grew.\n\n"
    assert drop_risk_factors(mdna + "Risk Factors\n\nYou should consider each of the following factors.\n") == mdna
    stub = mdna + "Risk Factors \n\nRisk factors are disclosed within Item 1A. Risk Factors of the 2022 Form 10-K.\n\nLiquidity\n"
    assert drop_risk_factors(stub) == stub


def test_drop_financial_statements():
    # an MD&A that runs on into the financial statements is cut there; a table of contents near the top isn't a cut point
    toc = "Consolidated Statements of Income 45\n" + "Overview of the year.\n\n" * 120
    body = "Results of Operations\n\nNet interest income rose.\n\n" * 50
    fin = "CONSOLIDATED STATEMENTS OF INCOME\n\n| Interest income | 100 |\n"
    assert drop_financial_statements(toc + body + fin) == toc + body
    assert drop_financial_statements(toc + body) == toc + body
    q = "Notes to Condensed Consolidated Financial Statements (unaudited) |\n"
    assert drop_financial_statements(body + q + "Note 1.") == body
    cite = "Notes to Consolidated Financial Statements are referenced throughout this report for more detail.\n"
    assert drop_financial_statements(body + cite + body) == body + cite + body  # a sentence, not a heading


def test_batch_sizing():
    # Batch API sizing: batches stay under the queued-token limit, which is read from OpenAI's error message
    assert take([400] * 10, max_inputs=100, max_tokens=1500) == 3  # 3 x 400 fits under 1,500, a 4th doesn't
    assert take([400] * 10, max_inputs=2, max_tokens=1e9) == 2
    assert take([400] * 10, max_inputs=100, max_tokens=300) == 0  # queue full: wait
    assert queue_limit(["Enqueued token limit reached for text-embedding-3-large in organization org-x. "
                        "Limit: 20,000,000 enqueued tokens. Please try again"]) == 20_000_000
    assert queue_limit(["some other error"]) is None


def test_subset_never_replaces_chunks():
    # a ticker subset never replaces chunks.parquet, which every later phase inner-joins for all companies
    with pytest.raises(SystemExit, match="would replace"):
        build_chunks(["NVDA"])
