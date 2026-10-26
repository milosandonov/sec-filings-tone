"""Run: uv run pytest tests/test_explore.py  (offline; the data check needs the local data/tone/ files)"""

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("gradio")  # the app extra: uv sync --extra app
from sec_filings_tone import explore as E  # noqa: E402
from sec_filings_tone.tone import FILES  # noqa: E402


def test_click_drills_down():
    # a sector's row opens its companies, whichever column was clicked
    assert E.click("AI", "Both sections", None, SimpleNamespace(row_value=["Energy", 0.1], index=[3, 2])) == ("Energy", "")
    # in a sector, a click on the company column shows no passages
    assert E.click("AI", "Both sections", "Energy", SimpleNamespace(row_value=["XOM"], index=[1, 0])) == ("Energy", "")


@pytest.mark.skipif(not FILES["10-K"][1].exists(), reason="needs the local scores (data/tone/, gitignored)")
def test_matches_published_chart():
    # the explorer's cells are the charts' cells: AI in risk factors, word-checked (figures/lines_ai_named.csv)
    c = E.cells("AI", "1A")
    pub = pd.read_csv("figures/lines_ai_named.csv").query("group == 'Risk factors (Item 1A)' and measure == 'polarity'")
    for r in pub.itertuples():
        x = c[c.fy == r.fy].polarity
        assert x.count() == r.n and abs(x.mean() - r.mean) < 1e-9, r
