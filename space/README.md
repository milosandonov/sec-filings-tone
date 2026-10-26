---
title: SEC Filings Tone Explorer
sdk: gradio
sdk_version: 6.30.0
python_version: "3.13"
app_file: app.py
license: mit
short_description: Tone of S&P 500 10-Ks by topic, sector and year
datasets:
- milosand1/sec-filings-tone
---

# SEC Filings Tone: explorer

How S&P 500 companies write about 13 topics (AI, China, tariffs, interest rates, ...) in their 10-K risk factors
and MD&A, FY2019–FY2025: a sector × fiscal-year heatmap of five tone dimensions scored by Claude Haiku 5.5.
Click a sector for its companies, and a company's year for the passages behind that cell, with each score and
the words the model quoted as evidence.

- Code, method and results: [github.com/milosandonov/sec-filings-tone](https://github.com/milosandonov/sec-filings-tone)
- Data (read at startup, data version 0.4.0): [milosand1/sec-filings-tone](https://huggingface.co/datasets/milosand1/sec-filings-tone)

No API calls: the Space shows the released scores only. `explore.py` is the project's
`src/sec_filings_tone/explore.py`; the numbers are the same as the project's local explorer and charts.
