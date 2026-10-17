"""Run: uv run python tests/test_tone.py  (offline: synthetic chunks)"""

import numpy as np
import pandas as pd

from sec_filings_tone.sentiment import TOPICS
from sec_filings_tone.tone import Q_TOPICS, cells, top

# a bank with two 10-Qs in the same fiscal year and a 10-K, and a non-financial company's 10-Q
k = pd.DataFrame({
    "chunk_id": [f"q1-{i}" for i in range(4)] + [f"q2-{i}" for i in range(4)] + ["k-0", "t-0"],
    "ticker": ["B"] * 9 + ["T"], "sector": ["Financials"] * 9 + ["Information Technology"],
    "form": ["10-Q"] * 8 + ["10-K", "10-Q"], "fy": [2023] * 10,
    "report_date": ["2023-03-31"] * 4 + ["2023-06-30"] * 4 + ["2023-12-31", "2023-03-31"]})

c = cells("10-Q", k)
# one cell per (Financials 10-Q, topic): the IT company's 10-Q and the 10-K are left out
assert len(c) == 4 and set(c.ticker) == {"B"} and set(c.topic) == set(Q_TOPICS), c
assert sorted(c.report_date.unique()) == ["2023-03-31", "2023-06-30"], c
# the annual cells are unchanged: one per 10-K and topic, keyed by fiscal year
a = cells("10-K", k)
assert list(a.columns) == ["ticker", "sector", "fy", "topic"] and len(a) == len(TOPICS), a

# the June 10-Q's chunks and the 10-K score highest, but each cell ranks only its own filing's chunks
vs = np.array([.1, .4, .3, .2, .9, .9, .9, .9, 1., 1.])
bs = np.array([0., 2., 1., 3., 5., 5., 5., 5., 9., 9.])
p = pd.DataFrame(top(c[c.topic == Q_TOPICS[0]], k, vs, bs, "10-Q"))
march = p[p.report_date == "2023-03-31"].sort_values("rank")
# RRF of the vector order (q1-1, q1-2, q1-3, q1-0) and the BM25 order (q1-3, q1-1, q1-2; q1-0 has no query word)
assert march.chunk_id.tolist() == ["q1-1", "q1-3", "q1-2"], march
assert set(p[p.report_date == "2023-06-30"].chunk_id) <= {f"q2-{i}" for i in range(4)}, p
print("ok")
