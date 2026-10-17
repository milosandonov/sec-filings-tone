"""B5 charts: tone toward a topic in Financials' 10-Q MD&A, per calendar quarter.

Per quarter: the mean over companies of each cell's tone (the mean over its on-topic passages), with
a 95% bootstrap interval over companies, and the share of companies whose cell has any on-topic
passage. Each PNG in figures/ has its numbers in a CSV beside it.

Usage:
    uv run python -m sec_filings_tone.charts
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from sec_filings_tone.events import DIMS, cells, did  # noqa: E402
from sec_filings_tone.sentiment import DIMENSIONS  # noqa: E402
from sec_filings_tone.tone import Q_TOPICS  # noqa: E402

FIGURES = Path(__file__).parents[2] / "figures"
COMPANIES = Path(__file__).parents[2] / "data" / "companies.csv"
BANKS = ["Diversified Banks", "Regional Banks"]
EVENTS = {"First rate hike": "2022-03-16", "SVB closed": "2023-03-10"}
SERIES = ["#2a78d6", "#eb6834"]  # dataviz palette slots 1 and 2, validated on the light surface
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NOTE = ("10-Q MD&A only. No Q4: the Q4 report is the 10-K, which isn't comparable because it includes risk factors. "
        "A company's cell is the mean over its on-topic passages (hybrid search's top 3 in that 10-Q, scored by Claude "
        "Haiku 5.5); dimension means cover the companies discussing the topic. Bands: 95% bootstrap intervals over "
        "companies. Five companies with non-calendar fiscal years (BEN, FDS, JKHY, RJF, V) are placed by the calendar "
        "quarter their period ends in; their 10-Qs ending in calendar Q4 are left out.")


def table(c: pd.DataFrame, groups: dict[str, pd.Series]) -> pd.DataFrame:
    """Per group, calendar quarter and measure: the mean over companies, its 95% bootstrap interval, n."""
    rows = []
    for g, mask in groups.items():
        for q, x in c[mask].groupby("quarter"):
            for m in [*DIMS, "discussed"]:
                v = x[m].dropna()
                est, lo, hi = did(v, v[:0])
                rows.append({"group": g, "quarter": q, "measure": m, "mean": est, "lo": lo, "hi": hi, "n": len(v)})
    return pd.DataFrame(rows)


def plot(t: pd.DataFrame, title: str, path: Path) -> None:
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, facecolor=SURFACE)
    groups = list(dict.fromkeys(t.group))
    for ax, m in zip(axes.flat, [*DIMS, "discussed"]):
        ax.set_facecolor(SURFACE)
        for color, g in zip(SERIES, groups):
            s = t[(t.group == g) & (t.measure == m)].sort_values("quarter")
            x = s.quarter.dt.to_timestamp(how="end")
            ax.fill_between(x, s.lo, s.hi, color=color, alpha=0.12, lw=0)
            ax.plot(x, s["mean"], color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=g)
        for name, d in EVENTS.items():
            ax.axvline(pd.Timestamp(d), color=AXIS, lw=1, zorder=0)
            ax.text(pd.Timestamp(d), 1.0, f" {name} ", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=7.5,
                    ha="right" if name == "First rate hike" else "left", va="bottom")
        scale = DIMENSIONS.get(m)
        ax.set_title(f"{m.capitalize()} ({scale[0]} to {scale[-1]})" if scale else "Share of companies discussing it",
                     loc="left", color=INK, fontsize=10, pad=16)
        if m == "discussed":
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_ylim(0, 1.05)  # a full axis: near 100% throughout, and one company is about 1.5 points
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0, pad=4)
    n = t[t.measure == "discussed"].groupby("group").n
    sub = "; ".join(f"{g}: {n.min()[g]}–{n.max()[g]} companies a quarter" for g in groups)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.012, 0.952, sub, color=MUTED, fontsize=9.5, va="top")
    if len(groups) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.99, 0.958),
                   ncol=len(groups), frameon=False, labelcolor=INK, fontsize=9.5)
    fig.text(0.012, 0.012, NOTE, color=MUTED, fontsize=8, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93), h_pad=2.5)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    t.to_csv(path.with_suffix(".csv"), index=False)
    print(f"wrote {path} and its .csv")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    c = cells(form="10-Q")
    c["quarter"] = pd.to_datetime(c.report_date).dt.to_period("Q")
    c = c[c.quarter.dt.quarter != 4]  # only non-calendar fiscal years have a 10-Q ending in calendar Q4
    bank = c.ticker.map(pd.read_csv(COMPANIES).set_index("ticker").sub_industry).isin(BANKS)
    for topic in Q_TOPICS:
        x = c[c.topic == topic]
        slug = topic.lower().replace(" ", "_")
        plot(table(x, {"All Financials": bank[x.index] | True}), f"{topic}: tone in Financials' 10-Q MD&A, by quarter",
             FIGURES / f"q_{slug}.png")
    x = c[c.topic == "Deposits and liquidity"]
    nb = x[bank[x.index]].ticker.nunique()
    plot(table(x, {f"Banks ({nb})": bank[x.index], f"Other Financials ({x.ticker.nunique() - nb})": ~bank[x.index]}),
         "Deposits and liquidity: banks against other Financials, 10-Q MD&A, by quarter",
         FIGURES / "q_deposits_banks_vs_others.png")


if __name__ == "__main__":
    main()
