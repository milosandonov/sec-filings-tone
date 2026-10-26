"""Browse the tone cells: a sector x fiscal-year heatmap for one topic and dimension. Click a sector for its
companies, and a company's year for the passages behind that cell, with their scores and the model's evidence.
Free: no API calls; reads data/tone/ and chunks.parquet, the same cells as the charts and the Q&A layer.

Usage:
    uv run python -m sec_filings_tone.explore           # http://127.0.0.1:7861 (needs --extra app)
"""

import argparse
from functools import cache

import gradio as gr
import pandas as pd

from sec_filings_tone import answer, events, retrieve
from sec_filings_tone.sentiment import DIMENSIONS, TOPICS

FYS = list(range(2019, 2026))  # FY2026 has only the companies whose fiscal year doesn't end in December
DIMS = [*DIMENSIONS, "discussed"]
RANGES = {"polarity": (-2, 2), "magnitude": (0, 3), "realization": (0, 2), "uncertainty": (0, 3), "specificity": (0, 2),
          "discussed": (0, 1)}
ITEMS = {"Both sections": None, "Risk factors (Item 1A)": "1A", "MD&A (Item 7)": "7"}
named = cache(events.named)

HEADER = """Tone profiles of S&P 500 10-Ks, FY2019–FY2025: each cell is the mean over the on-topic passages among
the 3 a search ranked highest for that company, 10-K and topic, scored by Claude Haiku 5.5. AI, China and Tariffs
count only passages that name the topic (the word check). **Discussed** is the share of companies with an
on-topic passage. Click a sector to see its companies, then a company's year to see the passages and the evidence
behind each score. Only realization is validated against a person's labels; see RESULTS.md."""


def cells(topic: str, item: str | None) -> pd.DataFrame:
    c = answer.all_cells(item, topic in events.TOPIC_WORDS)
    return c[(c.topic == topic) & c.fy.isin(FYS)]


def heatmap(topic: str, dim: str, item: str | None, sector: str | None):
    """Sectors x fiscal years, or one sector's companies x fiscal years: the mean of `dim` over discussed cells."""
    c = cells(topic, item)
    if sector:
        c = c[c.sector == sector]
    by = "ticker" if sector else "sector"
    t = c.pivot_table(index=by, columns="fy", values=dim, aggfunc="mean", dropna=False).reindex(columns=FYS)
    t = t.set_axis([f"FY{y}" for y in FYS], axis=1).reset_index()
    years, (lo, hi) = t.columns[1:], RANGES[dim]
    fmt = "{:.0%}" if dim == "discussed" else "{:+.2f}" if dim == "polarity" else "{:.2f}"
    return (t.style.background_gradient(cmap="RdBu" if dim == "polarity" else "Blues", vmin=lo, vmax=hi, subset=years)
            .format(fmt, subset=years, na_rep="–"))


def passages(topic: str, item: str | None, ticker: str, fy: int) -> str:
    """The cell's passages, best search rank first: whether each counts toward the cell, its scores and evidence."""
    p = answer.scored(item == "7")
    p = p[(p.ticker == ticker) & (p.fy == fy) & (p.topic == topic)].sort_values("rank")
    if p.empty:
        return f"No passages for {ticker} FY{fy}, {topic}."
    keep, text = named() if topic in events.TOPIC_WORDS else None, retrieve.chunks().text
    out = [f"### {ticker} FY{fy} · {topic}"]
    for r in p.itertuples():
        why = ("off topic (model's flag)" if r.off_topic else f"Item {r.item}, outside the section shown" if item and r.item != item
               else "doesn't name the topic (word check)" if keep is not None and r.pid not in keep else "")
        out.append(f"\n**Passage {r.rank}** · Item {r.item} · " + (f"not counted: {why}" if why else "counted"))
        if not r.off_topic:
            out.append("\n| Dimension | Score | Evidence |\n|---|---|---|")
            out += [f"| {d} | {getattr(r, d):{'+' if d == 'polarity' else ''}.0f} | {str(getattr(r, d + '_evidence')).replace('|', '/')} |" for d in DIMENSIONS]
        out.append("\n> " + text[r.chunk_id].replace("\n", " "))
    return "\n".join(out)


def show(topic: str, dim: str, item: str, sector: str | None):
    label = f"**{sector}:** click a company's year for its passages." if sector else "**All sectors.** Click a sector."
    return heatmap(topic, dim, ITEMS[item], sector), label, gr.update(visible=bool(sector))


def click(topic: str, item: str, sector: str | None, evt: gr.SelectData):
    """A sector's row drills down to its companies; a company's year shows that cell's passages."""
    row, col = evt.row_value[0], evt.index[1]
    if not sector:
        return row, ""
    return sector, passages(topic, ITEMS[item], row, FYS[col - 1]) if col > 0 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    print("loading the cells and passages ...", flush=True)
    answer.all_cells(None, False), answer.scored(False), retrieve.chunks(), named()

    with gr.Blocks(title="SEC Filings Tone: explorer") as demo:
        gr.Markdown("# SEC Filings Tone: explorer\n" + HEADER)
        sector = gr.State(None)
        with gr.Row():
            topic = gr.Dropdown(list(TOPICS), value="AI", label="Topic")
            dim = gr.Dropdown(DIMS, value="discussed", label="Dimension")
            item = gr.Radio(list(ITEMS), value="Both sections", label="Section")
        where = gr.Markdown("**All sectors.** Click a sector.")
        back = gr.Button("Back to all sectors", visible=False)
        table = gr.Dataframe(interactive=False)
        detail = gr.Markdown()

        inputs = [topic, dim, item, sector]
        for ctl in (topic, dim, item):
            ctl.change(show, inputs, [table, where, back])
        table.select(click, [topic, item, sector], [sector, detail]).then(show, inputs, [table, where, back])
        back.click(lambda: (None, ""), None, [sector, detail]).then(show, inputs, [table, where, back])
        demo.load(show, inputs, [table, where, back])
    demo.launch(server_name="127.0.0.1", server_port=a.port, share=False)


if __name__ == "__main__":
    main()
