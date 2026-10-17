"""Phase 7: does tone move where a known event hit (events.csv), and stay flat where it didn't?

For each event, each company's last 10-K filed before the event date is compared with its first
10-K whose fiscal year ends on or after it (so it covers the event and was written after it; the
first version took the first 10-K filed after the date, which for the 2025 tariffs meant 10-Ks filed
weeks after the first orders), on the event's topics. A cell's tone is the mean of each dimension over its
on-topic passages; `discussed` is whether it has any. The change is averaged over the event's
topics, then difference-in-differences: the mean change in the affected sectors minus the mean
change elsewhere, with a 95% bootstrap interval over companies. Where there's no comparison group
(an event that hit every sector, or a topic scored only for the affected sector), the plain change
in the affected sectors is reported instead.

With --quarterly, the same tests on the 10-Q cells (PLAN.md option B5) and events_quarterly.csv.

Usage:
    uv run python -m sec_filings_tone.events
    uv run python -m sec_filings_tone.events --quarterly
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sec_filings_tone.ingest import CHUNKS
from sec_filings_tone.sentiment import DIMENSIONS
from sec_filings_tone.tone import FILES, KEY, OUT

EVENTS = Path(__file__).parents[2] / "events.csv"
RESULTS = OUT / "events.csv"
QUARTERLY = (EVENTS.with_name("events_quarterly.csv"), "10-Q", OUT / "q_events.csv")  # report()'s arguments for B5
DIMS = list(DIMENSIONS)
# baseline -> the dimension whose expected direction it's tested against
BASELINE_FOR = {"finbert_polarity": "polarity", "lm_polarity": "polarity", "lm_uncertainty": "uncertainty",
                "details_per_100w": "specificity"}


def cells(items: list[str] | None = None, form: str = "10-K") -> pd.DataFrame:
    """One row per cell (a 10-K: ticker, fy, topic; a 10-Q: ticker, report_date, topic): sector, dates,
    `discussed`, and the mean of each dimension (and each baseline, once computed for every chunk)
    over the on-topic passages, using only passages from `items` if given."""
    passages, scores, _, baselines = FILES[form]
    p = pd.read_parquet(passages).reset_index(names="pid").merge(pd.read_parquet(scores), on="pid")
    p = p.merge(pd.read_parquet(CHUNKS, columns=["chunk_id", *[x for x in ("filing_date", "report_date", "item")
                                                               if x not in p]]), on="chunk_id")
    cols = DIMS.copy()
    if baselines.exists():
        b = pd.read_parquet(baselines, columns=["chunk_id", *BASELINE_FOR])
        if set(p.chunk_id) <= set(b.chunk_id):  # partial baselines would compare different companies
            p, cols = p.merge(b, on="chunk_id"), cols + list(BASELINE_FOR)
    key = [*KEY[form], "topic"]
    grid = p.groupby([*key, "sector"], as_index=False)[[x for x in ("fy", "filing_date", "report_date")
                                                        if x not in key]].first()
    on = p[~p.off_topic & (p["item"].isin(items) if items else True)]
    grid = grid.merge(on.groupby(key)[cols].mean().reset_index(), on=key, how="left")
    return grid.assign(discussed=grid[DIMS[0]].notna().astype(float))


def changes(c: pd.DataFrame, date: str, topics: list[str], after_date: str = "") -> pd.DataFrame:
    """Per company: the last cell filed before `date` subtracted from the first whose fiscal year ends
    on or after `after_date` (default `date`), per topic, then averaged over the topics. A dimension
    is missing where either cell isn't discussed."""
    c = c[c.topic.isin(topics)].sort_values("filing_date")
    before = c[c.filing_date < date].drop_duplicates(["ticker", "topic"], keep="last").set_index(["ticker", "topic"])
    after = c[c.report_date >= (after_date or date)].drop_duplicates(["ticker", "topic"], keep="first").set_index(["ticker", "topic"])
    both = before.index.intersection(after.index)
    m = [x for x in c.columns if x not in ("ticker", "fy", "topic", "sector", "filing_date", "report_date")]
    d = (after.loc[both, m] - before.loc[both, m]).groupby(level="ticker").mean()
    return d.join(c.drop_duplicates("ticker").set_index("ticker").sector)


def did(a: pd.Series, u: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a minus mean of u (or a's mean when u is empty), with a 95% bootstrap interval."""
    a, u = a.dropna().to_numpy(), u.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    est = a.mean() - (u.mean() if len(u) else 0)
    boot = [rng.choice(a, len(a)).mean() - (rng.choice(u, len(u)).mean() if len(u) else 0) for _ in range(n_boot)]
    return est, *np.percentile(boot, [2.5, 97.5])


def report(events: Path = EVENTS, form: str = "10-K", results: Path = RESULTS) -> None:
    ev = pd.read_csv(events, dtype=str, keep_default_na=False)
    sub = pd.read_csv(Path(__file__).parents[2] / "data" / "companies.csv").set_index("ticker").sub_industry
    rows = []
    for e in ev.itertuples():
        items = e.items.split(";") if e.items else None
        ch = changes(cells(items, form), e.date, e.topics.split(";"), e.after)
        hit = ch.sector.isin(e.sectors.split(";")) if e.sectors != "all" else pd.Series(True, ch.index)
        if e.sub_industries:
            hit &= sub.reindex(ch.index).fillna("").str.contains("|".join(e.sub_industries.split(";")))
        for m in [c for c in ch.columns if c != "sector"]:
            a, u = ch.loc[hit, m], ch.loc[~hit, m]
            want = getattr(e, BASELINE_FOR.get(m, m), "") if m != "discussed" else ""
            row = {"event": e.event, "measure": m, "expected": want, "n_affected": a.notna().sum(),
                   "n_others": u.notna().sum()}
            # the change where the event hit, the change elsewhere, and the difference (the planned test)
            for name, (x, y) in {"affected": (a, a[:0]), "others": (u, u[:0]), "did": (a, u)}.items():
                if x.notna().any():
                    est, lo, hi = did(x, y)
                    row |= {name: est, f"{name}_lo": lo, f"{name}_hi": hi,
                            f"{name}_sign": "+" if lo > 0 else "-" if hi < 0 else "0"}
            rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(results, index=False)

    def fmt(x, k):
        return f"{x[k]:+.2f} [{x[k + '_lo']:+.2f}, {x[k + '_hi']:+.2f}]" if pd.notna(x.get(k)) else "-"

    for event, g in r.groupby("event", sort=False):
        print(f"\n{event} (companies: {g.n_affected.max()} affected, {g.n_others.max()} others)")
        print(f"  {'measure':18} {'expected':8} {'affected: after - before':26} {'others':22} difference")
        for _, x in g.iterrows():
            mark = lambda k: "" if not x.expected or pd.isna(x.get(k)) else " ok" if x[f"{k}_sign"] == x.expected else " no"
            print(f"  {x.measure:18} {x.expected or '·':8} {fmt(x, 'affected') + mark('affected'):26} "
                  f"{fmt(x, 'others'):22} {fmt(x, 'did') if x.n_others else '-'}{mark('did') if x.n_others else ''}")
    print(f"\nwrote {results}. 'ok' = the 95% interval is on the expected side of zero.")


if __name__ == "__main__":
    report(*QUARTERLY) if "--quarterly" in sys.argv else report()
