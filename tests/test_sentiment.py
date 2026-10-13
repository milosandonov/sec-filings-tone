"""Run: uv run python tests/test_sentiment.py  (offline: no API calls)"""

import json

from sec_filings_tone.sentiment import DIMENSIONS, kappa, proxies, schema, system_prompt

# evidence comes before each score, in the order asked; scores are limited to the scale
m = schema(["realization", "polarity"])
assert list(m.model_fields) == ["off_topic", "realization_evidence", "realization", "polarity_evidence", "polarity"]
props = m.model_json_schema()["properties"]
assert props["polarity"]["enum"] == [-2, -1, 0, 1, 2] and props["realization"]["enum"] == [0, 1, 2]
assert list(schema(["magnitude"], with_off_topic=False).model_fields) == ["magnitude_evidence", "magnitude"]
try:
    m.model_validate_json(json.dumps({"off_topic": False, "realization_evidence": "", "realization": 3,
                                      "polarity_evidence": "", "polarity": 0}))
    raise AssertionError("a score off the scale must be rejected")
except ValueError:
    pass

# the prompt quotes the guidelines file word for word
from sec_filings_tone.sentiment import GUIDELINES  # noqa: E402

assert GUIDELINES.read_text() in system_prompt()
assert set(DIMENSIONS) == {"polarity", "magnitude", "realization", "uncertainty", "specificity"}

# rule-based cross-checks move the right way
concrete = proxies("We recorded a $4.5 billion charge in fiscal 2026 for H20 inventory, 12% of revenue.")
generic = proxies("Changes in trade policy could adversely affect our business and may harm results.")
assert concrete["details_per_100w"] > generic["details_per_100w"] and concrete["realized_per_100w"] > 0
assert generic["hedges_per_100w"] > concrete["hedges_per_100w"]

# weighted kappa: 1 for identical, below 0 for reversed, 1-step misses cost less than 2-step
assert abs(kappa([0, 1, 2, 2], [0, 1, 2, 2], (0, 1, 2)) - 1) < 1e-9
assert kappa([0, 1, 2], [2, 1, 0], (0, 1, 2)) < 0
assert kappa([0, 1, 2, 0], [1, 1, 2, 0], (0, 1, 2)) > kappa([0, 1, 2, 0], [2, 1, 2, 0], (0, 1, 2))
print("ok")

# the prompt reads the section from the "item" column, not pandas' .item method
import pandas as pd  # noqa: E402

from sec_filings_tone.sentiment import user_prompt  # noqa: E402

row = pd.Series({"ticker": "NVDA", "form": "10-K", "fy": 2023, "report_date": "2023-01-29", "item": "1A", "text": "Export rules."})
assert "NVDA 10-K FY2023, Item 1A Risk Factors" in user_prompt(row, "China", ["polarity"])
print("ok")

# baselines: LM counts dictionary words only, by category; the sentence splitter keeps bullets apart
from sec_filings_tone.sentiment import LM_PATH, lm, sentences  # noqa: E402

assert sentences("Sales fell. Costs rose.\n• supply shortages\n• tariffs") == ["Sales fell.", "Costs rose.", "• supply shortages", "• tariffs"]
if LM_PATH.exists():  # downloaded on first use
    s = lm("We may lose customers, and litigation could adversely affect profitability. H20 X")
    assert s["lm_negative"] > 0 and s["lm_uncertainty"] > 0 and s["lm_litigious"] > 0 and s["lm_polarity"] < 0
    from sec_filings_tone.sentiment import lm_lists
    assert "NULL" in lm_lists()[0]  # read as a word, not as a missing value
    print("ok")
