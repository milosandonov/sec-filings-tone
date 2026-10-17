"""Run: uv run python tests/test_events.py  (offline: synthetic cells)"""

import pandas as pd

import re

from sec_filings_tone.events import TOPIC_WORDS, changes, did

c = pd.DataFrame({
    "ticker": ["A", "A", "A", "B", "B"], "fy": [2021, 2022, 2023, 2021, 2022], "topic": ["T"] * 5,
    "sector": ["X", "X", "X", "Y", "Y"],
    "filing_date": ["2022-02-01", "2023-02-01", "2024-02-01", "2022-02-01", "2022-11-01"],
    "report_date": ["2021-12-31", "2022-12-31", "2023-12-31", "2021-12-31", "2022-09-30"],
    "realization": [0, 2, 1, 1, 2], "discussed": [1.0] * 5})
ch = changes(c, "2022-10-07", ["T"])
# A: the last 10-K filed before the date (FY2021) against the first whose year ends after it (FY2022)
assert ch.loc["A", "realization"] == 2, ch
# B's FY2022 10-K was filed after the date but covers a year that ended before it: not an 'after' filing
assert "B" not in ch.index, ch

est, lo, hi = did(pd.Series([1.0, 1.0, 1.0]), pd.Series([0.0, 0.0]))
assert est == 1 and lo == hi == 1
est, *_ = did(pd.Series([2.0, 4.0]), pd.Series([], dtype=float))  # no comparison group: the plain mean
assert est == 3
# the word check: names the topic, not look-alikes
for topic, yes, no in [("AI", ["AI-driven demand", "artificial intelligence", "A.I. tools", "machine learning"], ["AIG", "fiduciary", "maintain"]),
                       ("China", ["China", "Chinese customers", "the PRC", "Hong Kong"], ["Indochina", "chinaware"]),
                       ("Tariffs", ["tariffs", "Section 301", "anti-dumping duties"], ["fiduciary duties", "trade secrets"])]:
    assert all(re.search(TOPIC_WORDS[topic], x) for x in yes), topic
    assert not any(re.search(TOPIC_WORDS[topic], x) for x in no), topic
print("ok")
