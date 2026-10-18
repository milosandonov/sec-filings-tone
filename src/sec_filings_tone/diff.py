"""Phase 8 (PLAN.md option A3): how much of each 10-K is carried over from the year before, and how the
saved tone scores differ between new and carried-over text. Free: nothing is scored again.

A sentence of a 10-K's Item 1A or Item 7 is carried over if its difflib ratio with some sentence of
the same item in the company's 10-K for the year before is at least RATIO. A passage's carryover
share is the share of its characters in carried-over sentences; a filing's, over all its chunks.

Usage:
    uv run python -m sec_filings_tone.diff carryover   # every 10-K chunk's carryover share, minutes
    uv run python -m sec_filings_tone.diff report      # carryover by sector and year, and the tests
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher

import pandas as pd

from sec_filings_tone.events import DIMS, EVENTS, did, named, report as event_report, section
from sec_filings_tone.ingest import CHUNKS
from sec_filings_tone.sentiment import sentences, spearman_ci
from sec_filings_tone.tone import BASELINES, FILES, OUT, PASSAGES, SCORES

CARRYOVER = OUT / "carryover.parquet"  # one row per 10-K chunk of a company with the year before's 10-K
RATIO = 0.9
NEW, CARRIED = 0.2, 0.8  # a passage is new at or below NEW, carried over at or above CARRIED
ITEMS = {"1A": "Risk factors", "7": "MD&A"}


def matcher(prev: list[str]):
    """Whether a sentence closely matches (difflib ratio >= RATIO) one of `prev`. Exact copies are
    looked up; otherwise only the 5 sentences sharing the most 4-word runs with it are compared."""
    # shortcut: top 5 by shared 4-word runs; a sentence at ratio >= 0.9 shares most of its runs, so it ranks first
    exact, runs = set(prev), defaultdict(set)
    for i, s in enumerate(prev):
        w = s.split()
        for j in range(len(w) - 3):
            runs[tuple(w[j:j + 4])].add(i)

    def match(s: str) -> bool:
        if s in exact:
            return True
        w = s.split()
        hits = Counter(i for j in range(len(w) - 3) for i in runs.get(tuple(w[j:j + 4]), ()))
        sm = SequenceMatcher(None, autojunk=False)
        sm.set_seq2(s)  # difflib caches the second sequence
        for i, _ in hits.most_common(5):
            sm.set_seq1(prev[i])
            if sm.real_quick_ratio() >= RATIO and sm.quick_ratio() >= RATIO and sm.ratio() >= RATIO:
                return True
        return False

    return match


def shares(g: pd.DataFrame) -> list[dict]:
    """One company's 10-K chunks -> characters and carried-over characters per chunk, for each fiscal
    year whose year before is there too."""
    out = []
    for _, h in g.groupby("item"):
        years = {fy: [(cid, [" ".join(x.split()) for x in sentences(t)]) for cid, t in zip(y.chunk_id, y.text)]
                 for fy, y in h.sort_values("seq").groupby("fy")}
        for fy, chunks in years.items():
            if fy - 1 not in years:
                continue
            match = matcher([s for _, ss in years[fy - 1] for s in ss])
            for cid, ss in chunks:
                out.append({"chunk_id": cid, "chars": sum(map(len, ss)), "carried": sum(len(s) for s in ss if match(s))})
    return out


def carryover() -> None:
    k = pd.read_parquet(CHUNKS, columns=["chunk_id", "ticker", "sector", "fy", "item", "seq", "text"],
                        filters=[("form", "==", "10-K")])
    with ProcessPoolExecutor(6) as ex:
        rows = [r for part in ex.map(shares, [g for _, g in k.groupby("ticker")], chunksize=4) for r in part]
    out = pd.DataFrame(rows).merge(k.drop(columns=["text", "seq"]), on="chunk_id")
    out["share"] = out.carried / out.chars
    out.to_parquet(CARRYOVER, index=False)
    print(f"{CARRYOVER}: {len(out):,} chunks of {out.groupby(['ticker', 'fy']).ngroups:,} 10-Ks")


def filings() -> pd.DataFrame:
    """Carryover share per 10-K item, FY2020-FY2025 (FY2026: only companies whose fiscal year doesn't end in December had filed)."""
    c = pd.read_parquet(CARRYOVER).query("fy <= 2025")
    f = c.groupby(["ticker", "sector", "fy", "item"], as_index=False)[["chars", "carried"]].sum()
    return f.assign(share=f.carried / f.chars)


def passages() -> pd.DataFrame:
    """Scored, on-topic annual passages with their chunk's carryover share and class."""
    p = pd.read_parquet(PASSAGES).reset_index(names="pid").merge(pd.read_parquet(SCORES), on="pid")
    p = p[~p.off_topic].merge(pd.read_parquet(CARRYOVER, columns=["chunk_id", "item", "share"]), on="chunk_id")
    p = p.merge(pd.read_parquet(BASELINES, columns=["chunk_id", "finbert_polarity", "lm_polarity"]), on="chunk_id")
    p["kind"] = pd.cut(p.share, [-0.01, NEW, CARRIED - 1e-9, 1.0], labels=["new", "mixed", "carried over"])
    return p


def new_minus_carried(x: pd.DataFrame, m: str, within: list[str]) -> pd.Series:
    """Per company: dimension m's mean on new passages minus its mean on carried-over ones, compared within
    each group of `within` and then averaged over the company's groups; only groups with both kinds count.
    With topic and fiscal year as `within`, a topic or year that simply has more new text (supply chain in
    FY2020, tariffs in FY2025) doesn't show up as a difference between new and carried-over text."""
    g = x[x.kind != "mixed"].groupby(["ticker", *within, "kind"], observed=True)[m].mean().unstack("kind")
    g = g.reindex(columns=["new", "carried over"])
    return (g["new"] - g["carried over"]).dropna().groupby(level="ticker").mean()


def fmt(e: float, lo: float, hi: float) -> str:
    return f"{e:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def report() -> None:
    f = filings()
    print("Share of each 10-K item carried over from the year before (mean over companies):")
    t = f.groupby(["item", "fy"]).share.mean().unstack("fy")
    print(t.rename(index=ITEMS).map("{:.0%}".format).to_string())
    for item, name in ITEMS.items():
        print(f"\n{name}, by sector:")
        print(f[f["item"] == item].groupby(["sector", "fy"]).share.mean().unstack("fy").map("{:.0%}".format).to_string())

    p, keep = passages(), named()
    for label, q in [("model only", p), ("word-checked", p[p.pid.isin(keep)])]:
        print(f"\n=== {label}: on-topic passages by class: {q.kind.value_counts().to_dict()}")
        for item, name in ITEMS.items():
            x = q[q["item"] == item]
            for how, within in [("within company, pooling its topics and years", []),
                                ("within company x topic x fiscal year, then per company", ["topic", "fy"])]:
                print(f"{name}: new minus carried over, {how} (expected for specificity: > 0)")
                for m in DIMS:
                    d = new_minus_carried(x, m, within)
                    print(f"  {m:12} {fmt(*did(d, d[:0]))}  companies={len(d)}")
            r = spearman_ci(x.share, x.specificity, n_boot=500)
            print(f"  Spearman(carryover share, specificity) {fmt(*r)}  passages={len(x)}")
        print("Agreement of the LLM's polarity with the baselines (Spearman; intervals resample passages):")
        for b in ["finbert_polarity", "lm_polarity"]:
            for kind in ["new", "carried over"]:
                x = q[q.kind == kind]
                print(f"  {b:16} {kind:12} {fmt(*spearman_ci(x.polarity, x[b], n_boot=500))}  passages={len(x)}")

    new = set(p.pid[p.kind == "new"])
    if FILES["section"][1].exists():  # the MD&A tests' AI passages (events.cells), as events.section numbers them
        s = section().merge(pd.read_parquet(CARRYOVER, columns=["chunk_id", "share"]), on="chunk_id")
        new |= set(s.pid[s.share <= NEW])
    for label, k, out in [("model only", new, OUT / "events_new.csv"), ("word-checked", new & keep, OUT / "events_new_named.csv")]:
        print(f"\n=== event tests on new passages only ({label})")
        event_report(EVENTS, "10-K", out, keep=k)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 8: carryover from the year before, and new vs carried-over text")
    p.add_argument("command", choices=["carryover", "report"])
    {"carryover": carryover, "report": report}[p.parse_args().command]()


if __name__ == "__main__":
    main()
