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

# gold sheets come out in the practice sheet's columns and formats, so one loader reads both
from sec_filings_tone.ingest import CHUNKS  # noqa: E402
from sec_filings_tone.sentiment import PRACTICE, sheet  # noqa: E402

if CHUNKS.exists():  # local build, gitignored
    key = pd.read_csv(PRACTICE.with_name("practice_key.csv"))
    items = pd.read_parquet(CHUNKS).set_index("chunk_id", drop=False).loc[key.chunk_id].assign(topic=key.topic.values)
    s = sheet(items)
    meta = ["topic", "chunk_id", "company", "form", "period", "section"]
    assert s[meta].equals(key[meta]), s[meta].compare(key[meta])
    assert ["item_id", *s.columns] == list(pd.read_csv(PRACTICE, nrows=0).columns)
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

# labeled sheets: yes/no or TRUE/FALSE read the same; a score off its scale or missing on an on-topic item stops
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from sec_filings_tone.sentiment import kappa_ci, labels  # noqa: E402

head = "item_id,off_topic,polarity,magnitude,realization,uncertainty,specificity\n"
with tempfile.TemporaryDirectory() as tmp:
    ok, bad = Path(tmp, "ok.csv"), Path(tmp, "bad.csv")
    ok.write_text(head + "G01,no,-2,3,2,0,1\nG02,TRUE,,,,,\nG03,FALSE,0,0,0,0,0\n")
    d = labels(ok)
    assert d.off_topic.tolist() == [False, True, False] and d.polarity.tolist()[0] == -2
    final = pd.DataFrame({"item_id": ["G01", "G03"], "field": ["polarity", "off_topic"], "final": ["-1", "yes"]})
    d = labels(ok, final)  # adjudicated: each final replaces that label
    assert d.polarity.tolist()[0] == -1 and d.off_topic.tolist() == [False, True, True]
    bad.write_text(head + "G01,no,-2,3,3,0,1\nG02,no,0,0,0,,0\nG03,maybe,0,0,0,0,0\n")
    try:
        labels(bad)
        raise AssertionError("bad cells must stop the run")
    except SystemExit as e:
        assert all(c in str(e) for c in ("G01 realization", "G02 uncertainty", "G03 off_topic")), e

k, lo, hi = kappa_ci([0, 1, 2, 2, 1, 0], [0, 1, 2, 2, 1, 0], (0, 1, 2))
assert abs(k - 1) < 1e-9 and lo <= k <= hi
print("ok")

# gold baselines: each scorer is compared only on the items it has a score for (the LLM has none off topic)
from sec_filings_tone.sentiment import GOLD_PAIRS, label_spearman  # noqa: E402

ids = ["G01", "G02", "G03", "G04", "G05"]
human = pd.DataFrame({d: [0, 1, 2, 3, 4] for d in DIMENSIONS}, index=ids)
x = pd.DataFrame({c: [0.0, 1, 2, 3, 4] for _, c in GOLD_PAIRS}, index=ids)
x["lm_polarity"] = [4.0, 3, 2, 1, 0]
x.loc["G05", "llm_polarity"] = float("nan")
res = {c: (n, r) for _, c, n, r, _, _ in label_spearman(human, x, ids)}
assert res["finbert_polarity"][0] == 5 and abs(res["finbert_polarity"][1] - 1) < 1e-9
assert abs(res["lm_polarity"][1] + 1) < 1e-9
assert res["llm_polarity"][0] == 4 and abs(res["llm_polarity"][1] - 1) < 1e-9
assert all(n == 3 for _, _, n, *_ in label_spearman(human, x, ids[:3]))
print("ok")
