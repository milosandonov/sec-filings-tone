"""Phase 7: does tone move where a known event hit (events.csv), and stay flat where it didn't?

For each event, each company's last 10-K whose fiscal year ended before the event date is compared
with its first 10-K whose fiscal year ends on or after it, on the event's topics: one fiscal year
apart for nearly every company. (The first version took the first 10-K filed after the date as
"after", which for the 2025 tariffs meant 10-Ks filed weeks after the first orders; the second took
the last 10-K filed before the date as "before", which for the tariffs skipped December companies'
FY2024 10-Ks and compared FY2023 with FY2025. That rule is kept as --filed; the fiscal-year rule is
primary since 2026-10-17.) A cell's tone is the mean of each dimension over its
on-topic passages; `discussed` is whether it has any. The change is averaged over the event's
topics, then difference-in-differences: the mean change in the affected sectors minus the mean
change elsewhere, with a 95% bootstrap interval over companies. Where there's no comparison group
(an event that hit every sector, or a topic scored only for the affected sector), the plain change
in the affected sectors is reported instead.

With --quarterly, the same tests on the 10-Q cells (PLAN.md option B5) and events_quarterly.csv, with
the filed rule its definition fixed in advance.
With --named, the word-checked version: a passage of a TOPIC_WORDS topic counts only if it names the
topic, since the model calls many passages on topic that don't (RESULTS.md, limitations).

With --robust (PLAN.md option A4): an event study (every fiscal year around each event), a check that
the change is a break rather than a trend already under way, and every test on risk factors and MD&A
separately; each also word-checked.

With --filed, every test (model only and word-checked) with "before" = the last 10-K filed before
the date, in events_filed*.csv. The primary rule's "before" 10-K can be filed after the event and so
already react to it, which biases the change toward zero; the filed rule mixes in a second year.

Usage:
    uv run python -m sec_filings_tone.events
    uv run python -m sec_filings_tone.events --quarterly
    uv run python -m sec_filings_tone.events --named
    uv run python -m sec_filings_tone.events --robust
    uv run python -m sec_filings_tone.events --filed
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sec_filings_tone.config import CHUNKS, COMPANIES, EVENT_RESULTS, EVENTS, EVENTS_QUARTERLY, FILES, KEY, LAST_FY, OUT
from sec_filings_tone.sentiment import DIMENSIONS

QUARTERLY = (EVENTS_QUARTERLY, "10-Q", OUT / "q_events.csv")  # report()'s arguments for B5
# topics with a clear vocabulary; added 2026-10-17 after the model was found calling passages on topic that name none of it
TOPIC_WORDS = {
    "AI": r"(?i:artificial intelligence|machine learning|deep learning|neural network|large language model|genai)|\bA\.?I\.?\b",
    "China": r"(?i)\bchina\b|chinese|\bPRC\b|hong kong|beijing|shanghai|huawei",
    "Tariffs": r"(?i)tariff|section 301|section 232|anti-dumping|countervailing dut|import dut|customs dut|trade war",
}
DIMS = list(DIMENSIONS)
MEASURES = [*DIMS, "discussed"]
# baseline -> the dimension whose expected direction it's tested against
BASELINE_FOR = {"finbert_polarity": "polarity", "lm_polarity": "polarity", "lm_uncertainty": "uncertainty",
                "details_per_100w": "specificity"}


def section() -> pd.DataFrame:
    """The section-level passages (tone.sections: AI's top 3 within each 10-K's MD&A) with their scores, pid
    -1 - row number, so cells() and named() can pool them with the 10-K passages without a clash."""
    s = pd.read_parquet(FILES["section"][0]).reset_index(names="pid").merge(pd.read_parquet(FILES["section"][1]), on="pid")
    return s.assign(pid=-1 - s.pid)


def named(form: str = "10-K") -> set[int]:
    """Passages (pid: row number in the passages file; the 10-K's section passages too, as section() numbers
    them) that pass the word check: all of a topic without a vocabulary, and those of a TOPIC_WORDS topic whose text names it."""
    p = pd.read_parquet(FILES[form][0]).reset_index(names="pid")
    if form == "10-K" and FILES["section"][1].exists():
        p = pd.concat([p, section()[p.columns]], ignore_index=True)
    q = p[p.topic.isin(TOPIC_WORDS)]
    text = pd.read_parquet(CHUNKS, columns=["chunk_id", "text"],
                           filters=[("chunk_id", "in", q.chunk_id.unique().tolist())]).set_index("chunk_id").text
    ok = [bool(re.search(TOPIC_WORDS[t], text[c])) for c, t in zip(q.chunk_id, q.topic)]
    return set(p.pid[~p.topic.isin(TOPIC_WORDS)]) | set(q.pid[ok])


def cells(items: list[str] | None = None, form: str = "10-K", keep: set | None = None) -> pd.DataFrame:
    """One row per cell (a 10-K: ticker, fy, topic; a 10-Q: ticker, report_date, topic): sector, dates,
    `discussed`, and the mean of each dimension (and each baseline, once computed for every chunk)
    over the on-topic passages, using only passages from `items`, and only passages (pid) in `keep`, if given.
    A 10-K analysis of MD&A alone takes AI's cells from section(): its top 3 searched within MD&A (PLAN.md
    Phase 7, 2026-10-18), not the MD&A part of the whole 10-K's top 3.
    form "new" (diff.py, 2026-10-26): the tone of each cell's top 3 among new chunks, on the 10-K cells' grid, so
    "before" and "after" are the same 10-Ks as in the 10-K tests and a cell with no new on-topic passage isn't discussed;
    so a cell never retrieved would count as not discussed too: diff.new_events checks every cell it needs was retrieved."""
    passages, scores, _, baselines = FILES[form]
    p = pd.read_parquet(passages).reset_index(names="pid").merge(pd.read_parquet(scores), on="pid")
    bases = [baselines]
    if form == "10-K" and items and set(items) <= {"7", "mdna"} and FILES["section"][1].exists():
        s = section()
        p, bases = pd.concat([p[~p.topic.isin(s.topic)], s], ignore_index=True), [baselines, FILES["section"][3]]
    p = p.merge(pd.read_parquet(CHUNKS, columns=["chunk_id", *[x for x in ("filing_date", "report_date", "item")
                                                               if x not in p]]), on="chunk_id")
    cols = DIMS.copy()
    if all(f.exists() for f in bases):
        b = pd.concat([pd.read_parquet(f, columns=["chunk_id", *BASELINE_FOR]) for f in bases]).drop_duplicates("chunk_id")
        if set(p.chunk_id) <= set(b.chunk_id):  # partial baselines would compare different companies
            p, cols = p.merge(b, on="chunk_id"), cols + list(BASELINE_FOR)
    key = [*KEY["10-Q" if form == "10-Q" else "10-K"], "topic"]
    grid = p.groupby([*key, "sector"], as_index=False)[[x for x in ("fy", "filing_date", "report_date")
                                                        if x not in key]].first()
    if form == "new":
        grid = cells(items)[grid.columns]
    on = p[~p.off_topic & (p["item"].isin(items) if items else True) & (p.pid.isin(keep) if keep is not None else True)]
    grid = grid.merge(on.groupby(key)[cols].mean().reset_index(), on=key, how="left")
    return grid.assign(discussed=grid[DIMS[0]].notna().astype(float))


def before_cells(c: pd.DataFrame, date: str, filed: bool = False) -> pd.DataFrame:
    """Each company and topic's "before" cell: its last 10-K whose fiscal year ended before `date`, or
    with `filed` (the rule until 2026-10-17), its last 10-K filed before `date`."""
    c = c[(c.filing_date if filed else c.report_date) < date]
    return c.sort_values("filing_date").drop_duplicates(["ticker", "topic"], keep="last")


def changes(c: pd.DataFrame, date: str, topics: list[str], after_date: str = "", filed: bool = False) -> pd.DataFrame:
    """Per company: the "before" cell (before_cells) subtracted from the first whose fiscal year ends
    on or after `after_date` (default `date`), per topic, then averaged over the topics. A dimension
    is missing where either cell isn't discussed."""
    c = c[c.topic.isin(topics)].sort_values("filing_date")
    before = before_cells(c, date, filed).set_index(["ticker", "topic"])
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


def affected(e, sector: pd.Series) -> pd.Series:
    """Whether each company (`sector` indexed by ticker) is in the event's affected group."""
    hit = sector.isin(e.sectors.split(";")) if e.sectors != "all" else pd.Series(True, sector.index)
    if e.sub_industries:
        sub = pd.read_csv(COMPANIES).set_index("ticker").sub_industry
        hit &= sub.reindex(sector.index).fillna("").str.contains("|".join(e.sub_industries.split(";")))
    return hit


def report(events: Path | pd.DataFrame = EVENTS, form: str = "10-K", results: Path = EVENT_RESULTS,
           keep: set | None = None, filed: bool = False) -> None:
    ev = events if isinstance(events, pd.DataFrame) else pd.read_csv(events, dtype=str, keep_default_na=False)
    rows = []
    for e in ev.itertuples():
        items = e.items.split(";") if e.items else None
        ch = changes(cells(items, form, keep), e.date, e.topics.split(";"), e.after, filed)
        hit = affected(e, ch.sector)
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


def study(c: pd.DataFrame, e) -> pd.DataFrame:
    """Per company and event year k (fiscal year minus that of its "before" 10-K, before_cells): each
    measure's change from that 10-K, averaged over the event's topics. FY2019-FY2025."""
    c = c[c.topic.isin(e.topics.split(";")) & (c.fy <= LAST_FY)]
    before = before_cells(c, e.date)
    x = c.merge(before[["ticker", "topic", "fy", *MEASURES]], on=["ticker", "topic"], suffixes=("", "_b"))
    x = x.assign(k=x.fy - x.fy_b, **{m: x[m] - x[f"{m}_b"] for m in MEASURES})
    return x.groupby(["ticker", "sector", "k"], as_index=False)[MEASURES].mean()


def breaks(c: pd.DataFrame, e) -> pd.DataFrame:
    """Per company: the change before -> after minus the change in the year before the "before" 10-K,
    averaged over the event's topics; only where "after" is the next fiscal year."""
    key = ["ticker", "topic"]
    c = c[c.topic.isin(e.topics.split(";"))].sort_values("filing_date")
    b = before_cells(c, e.date)
    a = c[c.report_date >= (e.after or e.date)].drop_duplicates(key, keep="first")
    x = b.merge(a, on=key, suffixes=("_b", "_a"))
    x = x[x.fy_a == x.fy_b + 1]
    prev = c.assign(fy_b=c.fy + 1)[[*key, "fy_b", *MEASURES]].rename(columns={m: f"{m}_p" for m in MEASURES})
    x = x.merge(prev, on=[*key, "fy_b"])
    out = pd.DataFrame({m: (x[f"{m}_a"] - x[f"{m}_b"]) - (x[f"{m}_b"] - x[f"{m}_p"]) for m in MEASURES})
    return out.assign(ticker=x.ticker, sector=x.sector_b).groupby(["ticker", "sector"], as_index=False)[MEASURES].mean()


def break_rows(c: pd.DataFrame, e) -> list[dict]:
    """The break check for event e on cells c: each dimension with an expected direction, on the affected companies."""
    b = breaks(c, e)
    b = b[b.ticker.map(affected(e, c.drop_duplicates("ticker").set_index("ticker").sector))]
    rows = []
    for m in [d for d in DIMS if getattr(e, d)]:
        if not e.after and (v := b[m].dropna()).size:
            est, lo, hi = did(v, v[:0])
            sign = "+" if lo > 0 else "-" if hi < 0 else "0"
            rows.append({"event": e.event, "measure": m, "expected": getattr(e, m), "break": est, "lo": lo,
                         "hi": hi, "n": v.size, "ok": sign == getattr(e, m)})
    return rows


def robust() -> None:
    """A4: the event study, the break check, and the tests on each 10-K section alone; model only and
    word-checked."""
    ev = pd.read_csv(EVENTS, dtype=str, keep_default_na=False)
    for suffix, keep in [("", None), ("_named", named())]:
        rows, brk = [], []
        for e in ev.itertuples():
            c = cells(e.items.split(";") if e.items else None, keep=keep)
            hit = affected(e, c.drop_duplicates("ticker").set_index("ticker").sector)
            s = study(c, e)
            for (k, g), y in s.assign(affected=s.ticker.map(hit)).groupby(["k", "affected"]):
                for m in MEASURES:
                    if (v := y[m].dropna()).size:
                        rows.append({"event": e.event, "affected": g, "k": k, "measure": m, "n": v.size,
                                     **dict(zip(["mean", "lo", "hi"], did(v, v[:0])))})
            brk += break_rows(c, e)
        pd.DataFrame(rows).to_csv(OUT / f"event_study{suffix}.csv", index=False)
        brk = pd.DataFrame(brk)
        brk.to_csv(OUT / f"event_breaks{suffix}.csv", index=False)
        print(f"\nbreak check{suffix or ' (model only)'}:\n" + brk.round(3).to_string(index=False))
        for item in ["1A", "7"]:
            report(ev.assign(items=item), results=OUT / f"events_{item}{suffix}.csv", keep=keep)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 7 event tests; with no option, the primary tests (events.csv)")
    run = p.add_mutually_exclusive_group()  # one version per run; an unknown option is an error, not the default run
    run.add_argument("--quarterly", action="store_true", help="B5: the 10-Q cells and events_quarterly.csv")
    run.add_argument("--named", action="store_true", help="word-checked: TOPIC_WORDS topics count only passages naming them")
    run.add_argument("--robust", action="store_true", help="A4: event studies, break check, each 10-K section alone")
    run.add_argument("--filed", action="store_true", help='robustness: "before" is the last 10-K filed before the date')
    a = p.parse_args()
    if a.quarterly:  # events_quarterly.csv fixed "before" as the last 10-Q filed before the date
        report(*QUARTERLY, filed=True)
    elif a.named:
        report(results=OUT / "events_named.csv", keep=named())
    elif a.robust:
        robust()
    elif a.filed:
        for suffix, keep in [("", None), ("_named", named())]:
            report(results=OUT / f"events_filed{suffix}.csv", keep=keep, filed=True)
    else:
        report()


if __name__ == "__main__":
    main()
