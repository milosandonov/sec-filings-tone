"""Run: uv run python tests/test_ingest.py"""

from sec_filings_tone.ingest import chunk_section, embed_text, n_tokens

PARA = "Demand for our products may decline if customers delay purchases. " * 8  # ~100 tokens
LONG = "Export rules from the U.S. Government limit sales to China. " * 120  # ~1,400 tokens

DOC = f"""\
Table of Contents

Risks Related to Demand

{PARA}

Cash paid during the year 883 1,627 2,076 - 4,586 7,001 9,999
| % of net revenue | 14.2 | % | 27.2 | % |
| --- | --- | --- |

{PARA}

Risks Related to Export Controls

{LONG}

**Regulation\\.** ![chart](https://www.sec.gov/x.jpg) {PARA}
"""

chunks = chunk_section(DOC, max_tokens=300, target_tokens=200)
texts = " ".join(c["text"] for c in chunks)

assert all(c["n_tokens"] <= 300 and n_tokens(c["text"]) <= 300 for c in chunks), [c["n_tokens"] for c in chunks]
assert "Table of Contents" not in texts and "14.2" not in texts and "---" not in texts  # furniture and figures dropped
assert "Risks Related" not in texts  # headings go to metadata, not text
assert chunks[0]["heading"] == "Risks Related to Demand"
assert chunks[-1]["heading"] == "Risks Related to Export Controls"
assert "Regulation." in texts and "\\" not in texts and "sec.gov" not in texts  # markup cleaned
assert "U.S. Government" in texts  # sentence split doesn't break abbreviations
# short paragraphs are packed whole, not split
assert sum(PARA.strip() in c["text"] for c in chunks) >= 2
assert chunk_section(DOC, 300, 200) == chunks  # deterministic

# embedded text: header + overlap from the previous chunk; the stored text stays clean
import pandas as pd

rows = pd.DataFrame([{"ticker": "NVDA", "sector": "IT", "form": "10-K", "item": "1A", "fy": 2024,
                      "report_date": "2024-01-28", "seq": i, **c} for i, c in enumerate(chunks)])
emb = embed_text(rows, overlap_tokens=20)
assert emb[0].startswith("NVDA (IT) 10-K FY2024, Item 1A Risk Factors — Risks Related to Demand\n\n")
assert emb[1].endswith(chunks[1]["text"]) and len(emb[1]) > len(chunks[1]["text"]) + 60  # has overlap
assert embed_text(rows, overlap_tokens=0)[1].split("\n\n", 1)[1] == chunks[1]["text"]
print("ok")
