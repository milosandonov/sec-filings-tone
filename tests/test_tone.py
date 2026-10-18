"""Run: uv run python tests/test_tone.py  (offline: synthetic chunks)"""

import numpy as np
import pandas as pd

from sec_filings_tone.sentiment import DIMENSIONS, TOPICS, user_prompt
from sec_filings_tone.tone import MASK_SKIP, Q_TOPICS, cell_grid, masked_prompt, masked_sample, top, within

# a bank with two 10-Qs in the same fiscal year and a 10-K, and a non-financial company's 10-Q
k = pd.DataFrame({
    "chunk_id": [f"q1-{i}" for i in range(4)] + [f"q2-{i}" for i in range(4)] + ["k-0", "t-0"],
    "ticker": ["B"] * 9 + ["T"], "sector": ["Financials"] * 9 + ["Information Technology"],
    "form": ["10-Q"] * 8 + ["10-K", "10-Q"], "fy": [2023] * 10,
    "report_date": ["2023-03-31"] * 4 + ["2023-06-30"] * 4 + ["2023-12-31", "2023-03-31"]})

c = cell_grid("10-Q", k)
# one cell per (Financials 10-Q, topic): the IT company's 10-Q and the 10-K are left out
assert len(c) == 4 and set(c.ticker) == {"B"} and set(c.topic) == set(Q_TOPICS), c
assert sorted(c.report_date.unique()) == ["2023-03-31", "2023-06-30"], c
# the annual cells are unchanged: one per 10-K and topic, keyed by fiscal year
a = cell_grid("10-K", k)
assert list(a.columns) == ["ticker", "sector", "fy", "topic"] and len(a) == len(TOPICS), a

# the June 10-Q's chunks and the 10-K score highest, but each cell ranks only its own filing's chunks
vs = np.array([.1, .4, .3, .2, .9, .9, .9, .9, 1., 1.])
bs = np.array([0., 2., 1., 3., 5., 5., 5., 5., 9., 9.])
p = pd.DataFrame(top(c[c.topic == Q_TOPICS[0]], k, vs, bs, "10-Q"))
march = p[p.report_date == "2023-03-31"].sort_values("rank")
# RRF of the vector order (q1-1, q1-2, q1-3, q1-0) and the BM25 order (q1-3, q1-1, q1-2; q1-0 has no query word)
assert march.chunk_id.tolist() == ["q1-1", "q1-3", "q1-2"], march
assert set(p[p.report_date == "2023-06-30"].chunk_id) <= {f"q2-{i}" for i in range(4)}, p

# section-level cells: A's risk factors outrank its MD&A, and B's 10-K has no MD&A
k = pd.DataFrame({"chunk_id": ["a1-0", "a1-1", "a1-2", "a7-0", "a7-1", "a7-2", "b1-0"], "ticker": ["A"] * 6 + ["B"],
                  "sector": ["Energy"] * 7, "form": ["10-K"] * 7, "fy": [2023] * 7, "item": ["1A"] * 3 + ["7"] * 3 + ["1A"]})
vs, bs = np.array([.9, .8, .7, .2, .3, .1, .5]), np.array([9., 8., 7., 3., 0., 2., 5.])
whole = pd.DataFrame(within(k, vs, bs, "AI", None))
assert whole[whole.ticker == "A"].chunk_id.tolist() == ["a1-0", "a1-1", "a1-2"], whole  # no MD&A chunk makes A's top 3
s = pd.DataFrame(within(k, vs, bs, "AI", "7"))
# one cell, A's, ranked within its MD&A: RRF of the vector order (a7-1, a7-0, a7-2) and BM25's (a7-0, a7-2; a7-1 has no query word)
assert s.chunk_id.tolist() == ["a7-0", "a7-2", "a7-1"] and set(s.ticker) == {"A"} and s["rank"].tolist() == [1, 2, 3], s

# masked check: the masked prompt has neither the company nor the fiscal year, and is otherwise the production prompt
row = pd.Series({"ticker": "NVDA", "form": "10-K", "fy": 2023, "item": "1A", "text": "Export rules may change."})
prod, mask = (f(row, "China", list(DIMENSIONS)).splitlines() for f in (user_prompt, masked_prompt))
assert prod[1] == "Filing: NVDA 10-K FY2023, Item 1A Risk Factors" and mask[1] == "Filing: a 10-K, Item 1A Risk Factors", mask
assert "NVDA" not in "\n".join(mask) and "2023" not in "\n".join(mask), mask
assert mask[:1] + mask[2:] == prod[:1] + prod[2:], mask

# the masked check's sample: an export-control-style event dated 2022-10-07; a company in MASK_SKIP is left out, and each
# company's before cell is its FY2021 10-K (last filed before the date) and its after cell FY2022
f = pd.DataFrame({"fy": [2021, 2022, 2023], "report_date": ["2021-12-31", "2022-12-31", "2023-12-31"],
                  "filing_date": ["2022-02-20", "2023-02-20", "2024-02-20"]})
co = pd.DataFrame({"ticker": ["NVDA", "JNJ", "XOM"], "sector": ["Information Technology", "Health Care", "Energy"]})
p = co.merge(f, how="cross").merge(pd.DataFrame({"rank": [1, 2, 3]}), how="cross").assign(topic="China", item="1A")
ev = pd.DataFrame([{"event": "e", "date": "2022-10-07", "after": "", "sectors": "Information Technology",
                    "sub_industries": "", "topics": "China", "items": ""}])
s = masked_sample(p.assign(orig_pid=range(len(p))), ev)
assert "JNJ" in MASK_SKIP and set(s.ticker) == {"NVDA", "XOM"} and len(s) == 12, s
assert s.groupby("cell").fy.unique().map(list).to_dict() == {"after": [2022], "before": [2021]}, s
assert s.groupby("ticker").group.first().to_dict() == {"NVDA": "affected", "XOM": "others"}, s
print("ok")

# vectors.npy is used only while its rows follow chunks()'s order: recorded when it's built, and checked
# against Chroma first for a file saved before the order was recorded
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import sec_filings_tone.tone as tone  # noqa: E402
from sec_filings_tone import retrieve  # noqa: E402

ids = pd.Series([f"c{i}" for i in range(300)])
emb = dict(zip(ids, np.random.default_rng(1).normal(size=(300, 8)).astype(np.float32)))


class FakeChroma:
    def count(self):
        return len(emb)

    def get(self, ids=None, include=(), limit=None, offset=0):
        got = ids if ids is not None else list(emb)[offset:offset + limit]
        return {"ids": got, "embeddings": np.array([emb[i] for i in got])}


retrieve.chunks, retrieve.collection = (lambda: pd.DataFrame({"chunk_id": ids})), FakeChroma
tone.EMBED_DIMS = 8
with tempfile.TemporaryDirectory() as tmp:
    tone.VECTORS, tone.VECTOR_ORDER = Path(tmp, "vectors.npy"), Path(tmp, "vectors_order.sha256")
    v = tone.vectors()  # built from Chroma: unit rows in chunks() order, order recorded
    assert np.allclose(v[5], emb["c5"] / np.linalg.norm(emb["c5"])) and tone.VECTOR_ORDER.exists()
    tone.VECTOR_ORDER.unlink()  # a file saved before the order was recorded
    assert tone.vectors().shape == (300, 8) and tone.VECTOR_ORDER.exists()  # matches Chroma: order recorded
    ids = ids[::-1].reset_index(drop=True)  # chunks.parquet rebuilt in another order
    for unlink in (False, True):  # with the recorded order, and for an older file without it
        if unlink:
            tone.VECTOR_ORDER.unlink()
        try:
            tone.vectors()
            raise AssertionError("vectors in another order than chunks() must stop the run")
        except SystemExit as e:
            assert "doesn't follow" in str(e), e
print("ok")
