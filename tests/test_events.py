"""Run: uv run python tests/test_events.py  (offline: synthetic cells)"""

import pandas as pd

import re

from types import SimpleNamespace

from sec_filings_tone.events import MEASURES, TOPIC_WORDS, breaks, changes, did, study

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
# a date between A's FY2022 year end and its filing (the 2025 tariffs' case): the filed rule skips FY2022
# (FY2021 -> FY2023: 1 - 0); fy_before takes it as "before" (FY2022 -> FY2023: 1 - 2)
assert changes(c, "2023-01-15", ["T"]).loc["A", "realization"] == 1
assert changes(c, "2023-01-15", ["T"], fy_before=True).loc["A", "realization"] == -1

est, lo, hi = did(pd.Series([1.0, 1.0, 1.0]), pd.Series([0.0, 0.0]))
assert est == 1 and lo == hi == 1
est, *_ = did(pd.Series([2.0, 4.0]), pd.Series([], dtype=float))  # no comparison group: the plain mean
assert est == 3
# A4: event study and break check. A: 0 -> 1 -> 1.5 (FY2020-22, event between the FY2021 and FY2022
# 10-Ks): the change before -> after (+0.5) minus the change the year before (+1) is -0.5. B has no FY2020.
y = pd.DataFrame({"ticker": ["A"] * 3 + ["B"] * 2, "fy": [2020, 2021, 2022, 2021, 2022], "topic": ["T"] * 5,
                  "sector": ["X"] * 5, "filing_date": ["2021-02-01", "2022-02-01", "2023-02-01"] * 1 + ["2022-02-01", "2023-02-01"],
                  "report_date": ["2020-12-31", "2021-12-31", "2022-12-31", "2021-12-31", "2022-12-31"],
                  **{m: [0.0, 1.0, 1.5, 0.0, 2.0] if m == "realization" else [0.0] * 5 for m in MEASURES}})
e = SimpleNamespace(topics="T", date="2022-03-16", after="")
b = breaks(y, e)
assert b.ticker.tolist() == ["A"] and b.realization.iloc[0] == -0.5, b
st = study(y, e).set_index(["ticker", "k"]).realization
assert st[("A", -1)] == -1 and st[("A", 0)] == 0 and st[("A", 1)] == 0.5 and st[("B", 1)] == 2, st

# the word check: names the topic, not look-alikes
for topic, yes, no in [("AI", ["AI-driven demand", "artificial intelligence", "A.I. tools", "machine learning"], ["AIG", "fiduciary", "maintain"]),
                       ("China", ["China", "Chinese customers", "the PRC", "Hong Kong"], ["Indochina", "chinaware"]),
                       ("Tariffs", ["tariffs", "Section 301", "anti-dumping duties"], ["fiduciary duties", "trade secrets"])]:
    assert all(re.search(TOPIC_WORDS[topic], x) for x in yes), topic
    assert not any(re.search(TOPIC_WORDS[topic], x) for x in no), topic
print("ok")
