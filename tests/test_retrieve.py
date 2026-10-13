"""Run: uv run python tests/test_retrieve.py  (offline: no API calls, no data files)"""

import numpy as np
import pandas as pd

from sec_filings_tone import retrieve as r

toy = pd.DataFrame({
    "chunk_id": ["a", "b", "c", "d"],
    "ticker": ["NVDA", "NVDA", "AMD", "JPM"],
    "sector": ["IT", "IT", "IT", "Financials"],
    "form": ["10-K", "10-K", "10-Q", "10-Q"],
    "item": ["1A", "1A", "mdna", "mdna"],
    "fy": [2023, 2024, 2023, 2023],
    "filing_date": ["2023-02-24", "2024-02-21", "2023-05-03", "2023-05-02"],
    "text": ["The H20 needs an export license for China.", "Demand for data center products grew.",
             "Export controls on China may hurt sales.", "Deposit outflows rose after SVB failed."],
}).set_index("chunk_id", drop=False)
r.chunks = lambda: toy

# the same filter as a BM25 mask and a Chroma where
mask, where = r.subset({"ticker": "NVDA"})
assert mask.tolist() == [True, True, False, False] and where == {"ticker": {"$in": ["NVDA"]}}
mask, where = r.subset({"ticker": ["NVDA", "AMD"], "fy": np.int64(2023), "item": None})
assert mask.tolist() == [True, False, True, False]
assert where == {"$and": [{"ticker": {"$in": ["NVDA", "AMD"]}}, {"fy": {"$in": [2023]}}]}
assert type(where["$and"][1]["fy"]["$in"][0]) is int  # Chroma rejects numpy ints
mask, where = r.subset({"date_from": "2023-05-01", "date_to": "2023-12-31"})
assert mask.tolist() == [False, False, True, True]
assert where == {"filing_date": {"$in": ["2023-05-02", "2023-05-03"]}}
mask, where = r.subset({})
assert mask.all() and where is None
try:
    r.subset({"year": 2023})
    raise AssertionError("a typo'd filter must not be ignored")
except ValueError:
    pass

# BM25: exact terms, scored only inside the mask, no zero-score filler
assert r.bm25_search("H20", np.ones(4, bool), 4)[0] == ["a"]
assert r.bm25_search("China", np.array([False, True, True, True]), 4)[0] == ["c"]
assert r.bm25_search("the of", np.ones(4, bool), 4)[0] == []  # stopwords only

# RRF: in both lists beats first in one; ranks only, not scores
assert list(r.rrf(["a", "b"], ["b", "c"])) == ["b", "a", "c"]
assert r.rrf(["a"], ["a"])["a"] == 2 / (r.RRF_K + 1)

# the LLM's order: 1-based, deduped, out-of-range dropped, missing ids appended
assert r.fix_order([3, 3, 9, 1], 4) == [2, 0, 1, 3]
assert r.fix_order([], 2) == [0, 1]

# nothing matches the filters -> empty stages, no API calls
res = r.search("anything", ticker="TSLA")
assert list(res) == list(r.STAGES) and all(df.empty for df in res.values())
assert list(r.search("anything", rerank=False, ticker="TSLA")) == list(r.STAGES[:3])
print("ok")

# cost and time caps (eval/eval.py): stop before a call would pass the budget, or after the deadline
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from eval import Cap, CapReached  # noqa: E402

c = Cap(usd=0.01)
c.check(0.005)
c.charge(0.008)
for cap, nxt in [(c, 0.005), (Cap(minutes=0), 0)]:
    try:
        cap.check(nxt)
        raise AssertionError("cap should have stopped the run")
    except CapReached:
        pass
print("caps ok")
