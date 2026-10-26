"""Phase 9, research question 4 (PLAN.md option A2): do companies' year-over-year tone changes move
together more within a sector than across sectors?

For each dimension: each company's change in each topic's cell from one fiscal year to the next
(both years discussed), minus that topic-year's mean change over all companies, so the shift every
company shared doesn't count. Pairs of companies are correlated over the topic-years they share
(at least MIN_SHARED). The statistic is the mean correlation of same-sector pairs minus that of
different-sector pairs; its one-sided p-value comes from shuffling the sector labels across companies.

Usage:
    uv run python -m sec_filings_tone.comove
"""

import numpy as np
import pandas as pd

from sec_filings_tone.config import COMOVE_RESULTS, LAST_FY
from sec_filings_tone.events import DIMS, cells, named

MIN_SHARED = 10
PERMUTATIONS = 2000


def changes(c: pd.DataFrame, m: str, demean: bool = True) -> pd.DataFrame:
    """Rows (topic, fy), columns tickers: the change in dimension m from fy - 1, NaN unless both years
    are discussed; minus each row's mean over companies if demean."""
    c = c[(c.topic != "Deposits and liquidity") & (c.fy <= LAST_FY)]  # deposits: Financials only
    full = pd.MultiIndex.from_product([sorted(c.topic.unique()), range(c.fy.min(), LAST_FY + 1)], names=["topic", "fy"])
    d = c.pivot(index=["topic", "fy"], columns="ticker", values=m).reindex(full).groupby(level="topic").diff()
    d = d.dropna(how="all")
    return d.sub(d.mean(axis=1), axis=0) if demean else d


def test(r: pd.DataFrame, sector: pd.Series, seed: int = 0) -> dict:
    """Same-sector minus different-sector mean pair correlation, with a one-sided permutation p-value."""
    s = sector.reindex(r.index).to_numpy()
    i, j = np.triu_indices(len(s), 1)
    v = r.to_numpy()[i, j]
    ok = ~np.isnan(v)
    v, i, j = v[ok], i[ok], j[ok]
    same = s[i] == s[j]
    obs = v[same].mean() - v[~same].mean()
    rng = np.random.default_rng(seed)
    null = np.empty(PERMUTATIONS)
    for k in range(PERMUTATIONS):
        p = rng.permutation(s)
        sm = p[i] == p[j]
        null[k] = v[sm].mean() - v[~sm].mean()
    return {"same": v[same].mean(), "different": v[~same].mean(), "difference": obs,
            "p": (np.sum(null >= obs) + 1) / (PERMUTATIONS + 1), "pairs": len(v), "same_pairs": int(same.sum())}


def report() -> None:
    from sec_filings_tone.diff import passages  # Phase 8's carryover classes

    p = passages()
    versions = {"primary": (None, True), "raw changes": (None, False), "word-checked": (named(), True),
                "not mostly copied": (set(p.pid[p.share < 0.8]), True), "new text only": (set(p.pid[p.kind == "new"]), True)}
    rows = []
    for name, (keep, demean) in versions.items():
        c = cells(keep=keep)
        sector = c.drop_duplicates("ticker").set_index("ticker").sector
        for m in DIMS:
            r = changes(c, m, demean).corr(min_periods=MIN_SHARED)
            rows.append({"version": name, "dimension": m, **test(r, sector)})
            if name == "primary":
                s = sector.reindex(r.index)
                same = r.where(np.equal.outer(s.to_numpy(), s.to_numpy()) & ~np.eye(len(s), dtype=bool))
                by = {g: np.nanmean(same.loc[s == g, s == g].to_numpy()) for g in sorted(s.unique())}
                rows[-1] |= {f"sector:{g}": x for g, x in by.items()}
        print(f"{name}: done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(COMOVE_RESULTS, index=False)
    pd.set_option("display.width", 200)
    print(out[["version", "dimension", "same", "different", "difference", "p", "pairs", "same_pairs"]].round(4).to_string(index=False))
    sec = out[out.version == "primary"].set_index("dimension").filter(like="sector:").T
    print("\nmean same-sector pair correlation, primary version:\n" + sec.rename(index=lambda x: x[7:]).round(3).to_string())
    print(f"\nwrote {COMOVE_RESULTS}")


if __name__ == "__main__":
    report()
