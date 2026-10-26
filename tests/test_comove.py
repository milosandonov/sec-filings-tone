"""Run: uv run pytest tests/test_comove.py  (offline: synthetic cells)"""

import numpy as np
import pandas as pd

from sec_filings_tone.comove import changes, test as sector_test


def test_changes():
    # two companies, one topic, three years; B isn't discussed in 2020, so its 2020 and 2021 changes are missing
    c = pd.DataFrame({"ticker": ["A"] * 3 + ["B"] * 3, "topic": ["T"] * 6, "fy": [2019, 2020, 2021] * 2,
                      "realization": [0.0, 1.0, 1.5, 1.0, np.nan, 2.0]})
    d = changes(c, "realization", demean=False)
    assert d.loc[("T", 2020), "A"] == 1.0 and d.loc[("T", 2021), "A"] == 0.5, d
    assert d["B"].isna().all(), d
    dm = changes(c, "realization")  # minus each topic-year's mean change: A is the only one, so 0
    assert (dm["A"] == 0).all(), dm


def test_sector_effect():
    # 12 companies in two sectors; same-sector pairs correlate 0.8, the rest 0: a planted sector effect
    s = pd.Series(["X"] * 6 + ["Y"] * 6, index=[f"c{i}" for i in range(12)])
    r = pd.DataFrame(np.where(np.equal.outer(s.to_numpy(), s.to_numpy()), 0.8, 0.0), index=s.index, columns=s.index)
    t = sector_test(r, s)
    assert abs(t["difference"] - 0.8) < 1e-9 and t["p"] < 0.02, t
    assert t["pairs"] == 66 and t["same_pairs"] == 30, t
