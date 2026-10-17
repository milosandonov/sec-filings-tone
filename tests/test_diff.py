"""Run: uv run python tests/test_diff.py  (offline: synthetic filings)"""

import pandas as pd

from sec_filings_tone.diff import matcher, shares

old = ["Changes in trade policy, including tariffs, could adversely affect our business and results.",
       "We sell our products in over 100 countries around the world."]
match = matcher(old)
assert match(old[0])  # an exact copy
assert match("Changes in trade policy, including tariffs, could adversely affect our business and our results.")  # small edit
assert not match("In 2025 new tariffs on imports from Canada and Mexico raised our costs by $40 million.")
assert not match("We sell")  # too short to share a 4-word run, and not an exact copy

# one company, two years: the FY2024 chunk repeats one sentence and adds one; FY2023 has no year before
g = pd.DataFrame({
    "chunk_id": ["a23", "a24"], "fy": [2023, 2024], "item": ["1A", "1A"], "seq": [0, 0],
    "text": [old[0], old[0] + " In 2025 new tariffs on imports raised our costs by $40 million."]})
r = shares(g)
assert [x["chunk_id"] for x in r] == ["a24"], r
assert r[0]["carried"] == len(old[0]) and r[0]["chars"] > r[0]["carried"], r
print("ok")
