"""Phase 9 charts (PLAN.md option A1): tone toward each topic in the S&P 500's 10-Ks, by fiscal year.

- Line charts for AI, tariffs and interest rates: per fiscal year, the mean over companies of each
  cell's tone (the mean over its on-topic passages), risk factors and MD&A separately, with 95%
  bootstrap intervals over companies, and the share of companies with an on-topic passage. The
  topic's events.csv dates are marked.
- Heatmaps, one per topic: each dimension's mean per sector and fiscal year over the companies
  discussing the topic (whole 10-K), on one color scale per dimension for every topic.
- Word-checked versions (`_named`) for AI, China and tariffs: their passages count only if they name the
  topic (events.TOPIC_WORDS), on the same color scales.
- Phase 8: how much of each 10-K section is carried over from the year before, by sector and year.
- A4 event studies: each event's change from every company's last 10-K before it, by year around it.
Each PNG in figures/ has its numbers in a CSV beside it.

Usage:
    uv run python -m sec_filings_tone.charts
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from sec_filings_tone.diff import CARRYOVER, ITEMS, filings  # noqa: E402
from sec_filings_tone.events import DIMS, EVENTS, TOPIC_WORDS, cells, did, named  # noqa: E402
from sec_filings_tone.sentiment import DIMENSIONS  # noqa: E402
from sec_filings_tone.tone import FILES, LAST_FY, OUT, S_TOPICS  # noqa: E402

FIGURES = Path(__file__).parents[2] / "figures"
LINES = ["AI", "Tariffs", "Interest rates"]
SECTIONS = {"Risk factors (Item 1A)": ["1A"], "MD&A (Item 7)": ["7"]}
MIN_N = 5  # heatmap cells with fewer companies discussing the topic are hatched instead
MEASURES = [*DIMS, "discussed"]
# dataviz reference palette: categorical slots 1-2 (validated), the blue sequential ramp, the blue-red diverging pair
SERIES = ["#2a78d6", "#eb6834"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#e34948", "#f0efec", "#2a78d6"]  # hurts, neutral, helps
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NOTE = ("S&P 500 10-Ks. A company's cell is the mean over its on-topic passages (hybrid search's top 3 in that 10-K, "
        "scored by Claude Haiku 5.5). A fiscal year is labelled by the calendar year it ends in; most end in December, "
        "and the 9% of 10-Ks whose year ends January to May mostly cover the year before. FY2026 is left out: only "
        "companies whose fiscal year doesn't end in December had filed.")
CHECKED = " Word-checked: a passage counts only if it names the topic."
WITHIN = " MD&A line: each 10-K's top 3 searched within its MD&A, not the MD&A part of the whole 10-K's top 3."  # S_TOPICS (PLAN.md Phase 7, 2026-10-18)


def title(m: str) -> str:
    scale = DIMENSIONS.get(m)
    return f"{m.capitalize()} ({scale[0]} to {scale[-1]})" if scale else "Share of companies with an on-topic passage"


def header(fig, title_text: str, sub: str, note: str) -> None:
    fig.suptitle(title_text, x=0.012, y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.012, 0.952, sub, color=MUTED, fontsize=9.5, va="top")
    fig.text(0.012, 0.012, note, color=MUTED, fontsize=8, wrap=True, va="bottom")


def table(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per group, fiscal year and measure: the mean over companies, its 95% bootstrap interval, n."""
    rows = []
    for g, x in groups.items():
        for fy, y in x.groupby("fy"):
            for m in MEASURES:
                v = y[m].dropna()
                est, lo, hi = did(v, v[:0])
                rows.append({"group": g, "fy": fy, "measure": m, "mean": est, "lo": lo, "hi": hi, "n": len(v)})
    return pd.DataFrame(rows)


def panels(t: pd.DataFrame, x: str, marks: dict[str, float], title_text: str, sub: str, note: str, path: Path,
           change: bool = False) -> None:
    """Six line panels (the five dimensions and `discussed`), one line per `group` over `x`, with
    vertical marks; `change` when the values are changes rather than levels."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, facecolor=SURFACE)
    groups = list(dict.fromkeys(t.group))
    for ax, m in zip(axes.flat, MEASURES):
        ax.set_facecolor(SURFACE)
        for color, g in zip(SERIES, groups):
            s = t[(t.group == g) & (t.measure == m)].sort_values(x)
            ax.fill_between(s[x], s.lo, s.hi, color=color, alpha=0.12, lw=0)
            ax.plot(s[x], s["mean"], color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=g)
        for name, at in marks.items():
            ax.axvline(at, color=AXIS, lw=1, zorder=0)
            ax.text(at, 1.0, f" {name} ", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=7.5, va="bottom")
        if change:
            ax.axhline(0, color=AXIS, lw=1, zorder=0)
        ax.set_title(("Change in: " if change else "") + title(m), loc="left", color=INK, fontsize=10, pad=16)
        if m == "discussed":
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            if not change:
                ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0, pad=4)
    header(fig, title_text, sub, note)
    if len(groups) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.005, 0.93),
                   ncol=len(groups), frameon=False, labelcolor=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9 if len(groups) > 1 else 0.93), h_pad=2.5)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    t.to_csv(path.with_suffix(".csv"), index=False)
    print(f"wrote {path} and its .csv")


def lines(t: pd.DataFrame, topic: str, path: Path, checked: bool = False) -> None:
    ev = pd.read_csv(EVENTS, dtype=str, keep_default_na=False)
    # FY t is plotted where it ends, at x = t
    marks = {e.event: (d := pd.Timestamp(e.date)).year - 1 + d.dayofyear / 365.25
             for e in ev.itertuples() if topic in e.topics.split(";")}
    n = t[(t.measure == "discussed") & (t.group == t.group.iloc[0])].n
    panels(t, "fy", marks, f"{topic}: tone in S&P 500 10-Ks, risk factors and MD&A, by fiscal year" + (" (word-checked)" if checked else ""),
           f"{n.min()}–{n.max()} companies a year; dimension means cover the companies with an on-topic passage "
           "from that section", NOTE + " Bands: 95% bootstrap intervals over companies." + (CHECKED if checked else "")
           # events.cells switches the MD&A line to the section-level cells only once they're scored
           + (WITHIN if topic in S_TOPICS and FILES["section"][1].exists() else ""), path)


def event_studies() -> None:
    """A4: each event's change from every company's last 10-K before it, by fiscal year around it."""
    ev = pd.read_csv(EVENTS, dtype=str, keep_default_na=False).set_index("event")
    for suffix in ["", "_named"]:
        st = pd.read_csv(OUT / f"event_study{suffix}.csv")
        for event, t in st.groupby("event", sort=False):
            e = ev.loc[event]
            if suffix and not set(e.topics.split(";")) & set(TOPIC_WORDS):
                continue  # the word check changes nothing for this event
            hit = e.sectors.replace(";", ", ") if e.sectors != "all" else "All companies"
            hit += f" ({e.sub_industries.replace(';', ', ')})" if e.sub_industries else ""
            t = t[t.n >= MIN_N].sort_values("affected", ascending=False, kind="stable")  # affected first: slot 1
            t = t.assign(group=t.affected.map({True: hit, False: "Other companies"}))
            slug = "_".join("".join(ch if ch.isalnum() else " " for ch in event.lower()).split())
            panels(t, "k", {"event": 0.5}, f"{event}: change from each company's last 10-K before {e.date}" + (" (word-checked)" if suffix else ""),
                   f"Topics: {e.topics.replace(';', ', ')}. Event year 0 is that 10-K, 1 the next fiscal year; means over "
                   f"companies discussing the topic in both years (at least {MIN_N}).",
                   NOTE + " Bands: 95% bootstrap intervals over companies." + (CHECKED if suffix else ""),
                   FIGURES / f"event_study_{slug}{suffix}.png", change=True)


def heat(fig, ax, v: pd.DataFrame, colors: list[str], lo: float, hi: float, name: str, pct: bool = False) -> None:
    """One sector x fiscal year panel; missing cells (NaN) are hatched."""
    im = ax.imshow(v.to_numpy(float), cmap=LinearSegmentedColormap.from_list(name, colors), vmin=lo, vmax=hi, aspect="auto")
    for i, j in zip(*np.where(v.isna().to_numpy())):
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////", edgecolor=AXIS, lw=0))
    ax.set_xticks(range(v.shape[1]), v.columns)
    ax.set_yticks(range(v.shape[0]), v.index)
    ax.set_xticks(np.arange(-.5, v.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, v.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)  # the 2px surface gap between cells
    ax.tick_params(which="both", length=0, colors=MUTED)
    ax.spines[:].set_visible(False)
    ax.set_title(name, loc="left", color=INK, fontsize=10, pad=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    if pct:
        cb.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def heatmaps(c: pd.DataFrame, lims: pd.DataFrame | None = None, checked: bool = False) -> pd.DataFrame:
    c = c[c.topic != "Deposits and liquidity"]  # scored for Financials only: a single row
    g = c.groupby(["topic", "sector", "fy"])
    long = pd.concat({m: pd.DataFrame({"mean": g[m].mean(), "n": g[m].count()}) for m in MEASURES},
                     names=["measure"]).reset_index()
    if lims is None:
        lims = long[long.n >= MIN_N].groupby("measure")["mean"].agg(["min", "max"])
    for topic, x in long.groupby("topic"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharey=True, facecolor=SURFACE)
        for ax, m in zip(axes.flat, MEASURES):
            y = x[x.measure == m]
            v = y.pivot(index="sector", columns="fy", values="mean").where(
                y.pivot(index="sector", columns="fy", values="n") >= MIN_N)
            lo, hi = (0, 1) if m == "discussed" else lims.loc[m]
            if m == "polarity":  # symmetric around neutral
                lo, hi = -max(-lo, hi), max(-lo, hi)
            heat(fig, ax, v, DIVERGING if m == "polarity" else BLUES, lo, hi, title(m), pct=m == "discussed")
        header(fig, f"{topic}: tone by sector and fiscal year, S&P 500 10-Ks" + (" (word-checked)" if checked else ""),
               f"Mean over the companies discussing the topic (risk factors and MD&A together); hatched: fewer than "
               f"{MIN_N} companies discuss it. Each dimension has the same color scale for every topic.",
               NOTE + (CHECKED if checked else ""))
        fig.tight_layout(rect=(0, 0.05, 1, 0.93), h_pad=2)
        path = FIGURES / f"heatmap_{topic.lower().replace(' ', '_')}{'_named' if checked else ''}.png"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        x.to_csv(path.with_suffix(".csv"), index=False)
        print(f"wrote {path} and its .csv")
    return lims


def carryover() -> None:
    """Phase 8: the share of each 10-K item carried over from the year before, by sector and year."""
    f = filings()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True, facecolor=SURFACE)
    for ax, (item, name) in zip(axes, ITEMS.items()):
        v = f[f["item"] == item].groupby(["sector", "fy"]).share.mean().unstack("fy")
        heat(fig, ax, v, BLUES, 0, 1, f"{name} (Item {item})", pct=True)
    header(fig, "How much of each 10-K is carried over from the year before",
           "Mean over companies of the share of each section's text in sentences that closely match (difflib ratio "
           ">= 0.9) a sentence of the same section a year earlier", NOTE.split(". A company's")[0] + ". FY2019 has no "
           "year before in the data; FY2026 is left out (only part of the year had filed).")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    path = FIGURES / "carryover.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    f.groupby(["item", "sector", "fy"]).share.mean().reset_index().to_csv(path.with_suffix(".csv"), index=False)
    print(f"wrote {path} and its .csv")


def main() -> None:
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})
    FIGURES.mkdir(exist_ok=True)
    sec = {g: cells(items).query("fy <= @LAST_FY") for g, items in SECTIONS.items()}
    for topic in LINES:
        lines(table({g: x[x.topic == topic] for g, x in sec.items()}), topic,
              FIGURES / f"lines_{topic.lower().replace(' ', '_')}.png")
    lims = heatmaps(cells().query("fy <= @LAST_FY"))
    keep = named()
    sec = {g: cells(items, keep=keep).query("fy <= @LAST_FY") for g, items in SECTIONS.items()}
    for topic in [t for t in LINES if t in TOPIC_WORDS]:
        lines(table({g: x[x.topic == topic] for g, x in sec.items()}), topic,
              FIGURES / f"lines_{topic.lower().replace(' ', '_')}_named.png", checked=True)
    heatmaps(cells(keep=keep).query("fy <= @LAST_FY and topic in @TOPIC_WORDS"), lims, checked=True)
    if CARRYOVER.exists():
        carryover()
    if (OUT / "event_study.csv").exists():
        event_studies()


if __name__ == "__main__":
    main()
