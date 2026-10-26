"""Run: uv run pytest tests/test_discover.py  (offline: synthetic vectors and chunks)"""

import numpy as np
import pandas as pd

from sec_filings_tone.discover import FIRST_FY, LAST_FY, growth, kmeans, top_words


def test_kmeans_recovers_planted_clusters():
    # three tight groups around orthogonal directions: each group must land in one cluster of its own
    rng = np.random.default_rng(0)
    x = np.repeat(np.eye(3, 16), 50, axis=0) + rng.normal(0, 0.05, (150, 16))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    labels, cent = kmeans(x, 3, seed=0)
    groups = labels.reshape(3, 50)
    assert all(len(set(g)) == 1 for g in groups) and len({g[0] for g in groups}) == 3, labels
    assert np.allclose(np.linalg.norm(cent, axis=1), 1, atol=1e-5)


def test_growth_counts_companies_not_chunks():
    # A: cluster 1 in both years (no change), five chunks in cluster 0 in the last year only;
    # B: cluster 1 only in the last year; C files only the first year, so it's out of the shares and the change
    rows = [("A", FIRST_FY, 1), *[("A", LAST_FY, 0)] * 5, ("A", LAST_FY, 1), ("B", FIRST_FY, 0), ("B", LAST_FY, 1),
            ("C", FIRST_FY, 1)]
    meta = pd.DataFrame(rows, columns=["ticker", "fy", "cluster"])
    share, ch = growth(meta, meta.cluster.to_numpy(), 3)
    assert share.loc[FIRST_FY].tolist() == [0.5, 0.5, 0] and share.loc[LAST_FY].tolist() == [0.5, 1, 0]
    assert ch.change.tolist() == (share.loc[LAST_FY] - share.loc[FIRST_FY]).tolist() == [0, 0.5, 0], ch
    assert (ch.companies == 2).all(), ch
    assert (ch.lo <= ch.change).all() and (ch.change <= ch.hi).all()


def test_top_words():
    texts = ["stablecoin custody wallet", "stablecoin wallet token", "tariffs duties imports", "tariffs imports 2025"]
    words = top_words(texts, np.array([0, 0, 1, 1]), 2, n=2)
    assert set(words[0]) == {"stablecoin", "wallet"} and set(words[1]) == {"tariffs", "imports"}, words
