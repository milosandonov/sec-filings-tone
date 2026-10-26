"""Run: uv run pytest tests/test_diff.py  (offline: synthetic filings)"""

import pandas as pd

from sec_filings_tone.diff import matcher, new_minus_carried, shares

OLD = ["Changes in trade policy, including tariffs, could adversely affect our business and results.",
       "We sell our products in over 100 countries around the world."]


def test_matcher():
    match = matcher(OLD)
    assert match(OLD[0])  # an exact copy
    assert match("Changes in trade policy, including tariffs, could adversely affect our business and our results.")  # small edit
    assert not match("In 2025 new tariffs on imports from Canada and Mexico raised our costs by $40 million.")
    assert not match("We sell")  # too short to share a 4-word run, and not an exact copy


def test_shares():
    # one company, two years: the FY2024 chunk repeats one sentence and adds one; FY2023 has no year before
    g = pd.DataFrame({
        "chunk_id": ["a23", "a24"], "fy": [2023, 2024], "item": ["1A", "1A"], "seq": [0, 0],
        "text": [OLD[0], OLD[0] + " In 2025 new tariffs on imports raised our costs by $40 million."]})
    r = shares(g)
    assert [x["chunk_id"] for x in r] == ["a24"], r
    assert r[0]["carried"] == len(OLD[0]) and r[0]["chars"] > r[0]["carried"], r


def test_new_minus_carried():
    # pooled over a company's topics, a topic with more new text looks like a gap; within topic and year there
    # is none. A: T1 scores 2 and is mostly new, T2 scores 0 and is mostly copied
    x = pd.DataFrame({"ticker": "A", "fy": 2024, "topic": ["T1"] * 4 + ["T2"] * 4 + ["T3"],
                      "kind": ["new"] * 3 + ["carried over"] + ["new"] + ["carried over"] * 3 + ["mixed"],
                      "realization": [2.0] * 4 + [0.0] * 4 + [9.0]})
    assert new_minus_carried(x, "realization", []).loc["A"] == 1.0  # (2+2+2+0)/4 - (2+0+0+0)/4; mixed left out
    assert new_minus_carried(x, "realization", ["topic", "fy"]).loc["A"] == 0.0
    # a topic-year with only one kind doesn't count; a company with no topic-year having both drops out
    y = pd.concat([x, pd.DataFrame({"ticker": "B", "fy": 2024, "topic": "T1", "kind": ["new"], "realization": [1.0]})])
    assert list(new_minus_carried(y, "realization", ["topic", "fy"]).index) == ["A"]
