"""Run: uv run pytest tests/test_explore.py  (offline; the data checks need the local data/tone/ or data/release/ files)"""

from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("gradio")  # the app extra: uv sync --extra app
from sec_filings_tone import explore as E  # noqa: E402
from sec_filings_tone.config import FILES  # noqa: E402
from sec_filings_tone.sentiment import DIMENSIONS  # noqa: E402

RELEASE = Path("data/release")  # the files uploaded to the dataset repo, as the Space reads them
SOURCES = {"local": (FILES["10-K"][1], E.local), "release": (RELEASE / "passages.parquet", cache(lambda: E.release(str(RELEASE))))}


def test_click_drills_down():
    # a sector's row opens its companies, whichever column was clicked
    assert E.click("AI", "Both sections", None, SimpleNamespace(row_value=["Energy", 0.1], index=[3, 2])) == ("Energy", "")
    # in a sector, a click on the company column shows no passages
    assert E.click("AI", "Both sections", "Energy", SimpleNamespace(row_value=["XOM"], index=[1, 0])) == ("Energy", "")


def test_dimensions():
    assert E.DIMENSIONS == list(DIMENSIONS)  # explore.py keeps its own copy, so the Space needn't import sentiment


def test_release_word_check():
    # the passage view's "doesn't name the topic" uses these pids; section passages are numbered apart (-1 - row)
    if not all(path.exists() for path, _ in SOURCES.values()):
        pytest.skip("needs data/tone/ and data/release/ (gitignored)")
    assert SOURCES["release"][1]().named() == E.local().named()


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("item, group", [("1A", "Risk factors (Item 1A)"), ("7", "MD&A (Item 7)")])
def test_matches_published_chart(source, item, group, monkeypatch):
    # the explorer's cells are the charts' cells: AI by section, word-checked (figures/lines_ai_named.csv), from the
    # local scores or rebuilt from the released passages (MD&A: AI's top 3 within MD&A, s_passages)
    path, load = SOURCES[source]
    if not path.exists():
        pytest.skip(f"needs {path} (gitignored)")
    monkeypatch.setattr(E, "D", load())
    c = E.cells("AI", item)
    pub = pd.read_csv("figures/lines_ai_named.csv").query("group == @group and measure != 'discussed'")
    for r in pub.itertuples():
        x = c[c.fy == r.fy][r.measure]
        assert x.count() == r.n and abs(x.mean() - r.mean) < 1e-9, r


def test_missing_cells_not_black(monkeypatch):
    # pandas' background_gradient paints NaN #000000; empty cells ("–") must stay transparent
    path, load = SOURCES["local"]
    if not path.exists():
        pytest.skip(f"needs {path} (gitignored)")
    monkeypatch.setattr(E, "D", load())
    s = E.heatmap("AI", "realization", "1A", "Information Technology")
    ctx = s._compute().ctx
    empty = [(r, c) for r in range(len(s.data)) for c in range(1, s.data.shape[1]) if pd.isna(s.data.iat[r, c])]
    assert empty, "the IT company view has empty cells to check"
    for rc in empty:
        bg = [v for k, v in ctx[rc] if k == "background-color"]
        assert bg[-1] == "transparent", (rc, ctx[rc])
