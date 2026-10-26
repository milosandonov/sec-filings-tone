"""Browse the tone cells: a sector x fiscal-year heatmap for one topic and dimension. Click a sector for its
companies, and a company's year for the passages behind that cell, with their scores and the model's evidence.
Free: no API calls; reads data/tone/ and chunks.parquet, the same cells as the charts and the Q&A layer. The
Hugging Face Space (space/app.py) runs this file on the released data instead (release()), with only gradio,
pandas, pyarrow and matplotlib installed.

Usage:
    uv run python -m sec_filings_tone.explore           # http://127.0.0.1:7861 (needs --extra app)
"""

import argparse
import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd

FYS = list(range(2019, 2026))  # FY2026 has only the companies whose fiscal year doesn't end in December
RANGES = {"polarity": (-2, 2), "magnitude": (0, 3), "realization": (0, 2), "uncertainty": (0, 3), "specificity": (0, 2),
          "discussed": (0, 1)}
DIMS = list(RANGES)
DIMENSIONS = DIMS[:-1]  # sentiment.DIMENSIONS, which the Space can't import (tests/test_explore.py checks)
ITEMS = {"Both sections": None, "Risk factors (Item 1A)": "1A", "MD&A (Item 7)": "7"}
REPO, REVISION = "milosand1/sec-filings-tone", "10b5bf8aeedbe6768a0a872f9c2b592e9d91fe7d"  # the dataset at data version 0.4.0

HEADER = """Tone profiles of S&P 500 10-Ks, FY2019–FY2025: each cell is the mean over the on-topic passages among
the 3 a search ranked highest for that company, 10-K and topic, scored by Claude Haiku 5.5. AI, China and Tariffs
count only passages that name the topic (the word check). **Discussed** is the share of companies with an
on-topic passage. Click a sector to see its companies, then a company's year to see the passages and the evidence
behind each score. Only realization is validated against a person's labels; see
[RESULTS.md](https://github.com/milosandonov/sec-filings-tone/blob/main/RESULTS.md)."""


def local() -> SimpleNamespace:
    """The data from the local pipeline (data/tone/, chunks.parquet); imported here as it needs the full install."""
    from sec_filings_tone import answer, events, retrieve
    from sec_filings_tone.sentiment import TOPICS
    return SimpleNamespace(topics=list(TOPICS), worded=set(events.TOPIC_WORDS), cells=answer.all_cells,
                           scored=answer.scored, named=answer.named, text=lambda: retrieve.chunks().text)


def release(folder: str | None = None) -> SimpleNamespace:
    """The same data from the released files: those in `folder`, or the dataset on Hugging Face at REVISION.
    Reads only the columns shown here and the text of the scored passages."""
    if folder:
        get = lambda f: f"{folder}/{f}"  # noqa: E731
    else:
        from huggingface_hub import hf_hub_download
        get = lambda f: hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION)  # noqa: E731
    prov = json.loads(Path(get("provenance.json")).read_text())
    cols = ["pid", "ticker", "fy", "topic", "sector", "rank", "chunk_id", "item", "off_topic", "names_topic",
            *DIMENSIONS, *(d + "_evidence" for d in DIMENSIONS)]
    p = pd.read_parquet(get("passages.parquet"), columns=cols)
    s = pd.read_parquet(get("s_passages.parquet"), columns=cols).assign(pid=lambda s: -1 - s.pid)  # as events.section numbers them
    both = pd.concat([p, s], ignore_index=True)
    keep = set(both.pid[both.names_topic])  # events.named(): the passages that pass the word check
    # ponytail: the file is one row group, decompressed whole here (3.7 GB peak); pq.ParquetFile.iter_batches
    # cuts that to 1 GB but triples the load time, worth it only if the Space runs short of its 16 GB
    text = pd.read_parquet(get("chunks.parquet"), columns=["chunk_id", "text"],
                           filters=[("chunk_id", "in", both.chunk_id.unique().tolist())]).set_index("chunk_id").text

    @cache
    def scored(mdna_ai: bool) -> pd.DataFrame:
        """answer.scored: with mdna_ai, AI's passages are its top 3 within MD&A."""
        return pd.concat([p[~p.topic.isin(s.topic)], s], ignore_index=True) if mdna_ai else p

    @cache
    def cells(item: str | None, word_checked: bool) -> pd.DataFrame:
        """answer.all_cells (events.cells): each cell's mean over its on-topic passages from `item` (and naming the
        topic, if word_checked), discussed if it has any. cells.parquet has only the whole-10-K cells."""
        q, key = scored(item == "7"), ["ticker", "fy", "topic", "sector"]
        on = q[~q.off_topic & (q["item"] == item if item else True) & (q.names_topic if word_checked else True)]
        c = q[key].drop_duplicates().merge(on.groupby(key)[DIMENSIONS].mean().reset_index(), on=key, how="left")
        return c.assign(discussed=c[DIMENSIONS[0]].notna().astype(float))

    return SimpleNamespace(topics=list(prov["retrieval"]["topic_queries"]), worded=set(prov["word_check"]),
                           cells=cells, scored=scored, named=lambda: keep, text=lambda: text)


D = None  # the data: local() unless set to release() first


def data() -> SimpleNamespace:
    global D
    D = D or local()
    return D


def cells(topic: str, item: str | None) -> pd.DataFrame:
    c = data().cells(item, topic in data().worded)
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
    p = data().scored(item == "7")
    p = p[(p.ticker == ticker) & (p.fy == fy) & (p.topic == topic)].sort_values("rank")
    if p.empty:
        return f"No passages for {ticker} FY{fy}, {topic}."
    keep, text = data().named() if topic in data().worded else None, data().text()
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


def build() -> gr.Blocks:
    print("loading the cells and passages ...", flush=True)
    d = data()
    d.cells(None, False), d.scored(False), d.text(), d.named()

    with gr.Blocks(title="SEC Filings Tone: explorer") as demo:
        gr.Markdown("# SEC Filings Tone: explorer\n" + HEADER)
        sector = gr.State(None)
        with gr.Row():
            topic = gr.Dropdown(d.topics, value="AI", label="Topic")
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
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    build().launch(server_name="127.0.0.1", server_port=a.port, share=False)


if __name__ == "__main__":
    main()
