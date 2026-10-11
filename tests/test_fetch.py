"""Run: uv run python tests/test_fetch.py"""

from sec_filings_tone.fetch import find_section

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
print("ok")
