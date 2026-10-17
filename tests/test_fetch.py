"""Run: uv run python tests/test_fetch.py"""

from sec_filings_tone.fetch import find_section, fiscal_year

DOC = """\
| Item 7. | Management's Discussion and Analysis | 29 |
| Item 8. | Financial Statements and Supplementary Data | 50 |
# Item 7. Management's Discussion and Analysis
See "Management's Discussion and Analysis" on page 50.
# Item 7A. Quantitative and Qualitative Disclosures About Market Risk
### MANAGEMENT’S DISCUSSION AND ANALYSIS
Revenue grew.
### MANAGEMENT’S DISCUSSION AND ANALYSIS
Margins fell.
**Report of Independent Registered Public Accounting Firm**
We audited."""

mdna = find_section(DOC, "7")
assert "Revenue grew." in mdna and "Margins fell." in mdna, mdna  # page headers don't split the span
assert "See " not in mdna and "| 29 |" not in mdna, mdna  # stub and TOC row lose to the real section
assert "We audited." not in mdna, mdna  # stops at the next section
assert find_section(DOC, "1A") == ""

# a 52/53-week year ending in the first week of January is the year before (JNJ, DPZ, LHX)
assert fiscal_year("2021-01-03") == 2020 and fiscal_year("2023-01-01") == 2022 and fiscal_year("2020-01-07") == 2019
assert fiscal_year("2020-01-08") == 2020  # later in January: that year (no S&P 500 10-K ends Jan 5-24)
assert fiscal_year("2022-12-31") == 2022 and fiscal_year("2023-02-03") == 2023 and fiscal_year("2019-06-28") == 2019
print("ok")
