"""Exploratory topic discovery (2026-10-26): risk themes in 10-K risk factors that the 13 fixed topics
miss, and which of them grew most from FY2019 to FY2025. Found by looking, not a pre-registered test;
the design below was written down and committed before any clustering was run.

- Corpus: every 10-K Item 1A chunk of FY2019-FY2025 (92,120 chunks, 466-496 companies a year; FY2026 is
  partial), with its stored embedding (tone.vectors, unit length). Nothing is embedded.
- Clusters: spherical k-means (cosine) with k = 80, k-means++ start, seed 0, until no chunk moves (at
  most 100 rounds). k = 80 gives ~1,150 chunks a cluster on average: fine enough to split the 13 topics'
  ground into narrower themes, coarse enough that most clusters reach many companies each year.
  Stability: the same run with seed 1; each seed-0 cluster is matched to the seed-1 cluster whose
  centroid is closest (cosine), and the top growing clusters' matches and growth are reported.
- Labels, free: each cluster's 12 most distinctive words (bm25s tokens, English stop words and pure
  numbers removed), by the log-odds z-score of the cluster's word counts against the rest of the corpus
  with an informative Dirichlet prior (Monroe et al. 2008; prior = corpus frequencies, 1,000 pseudo-tokens);
  plus the 3 chunks nearest the centroid (from 3 different companies: changed 2026-10-26, see below).
- Covered: a cluster's best cosine between its centroid and the 13 topic query vectors (tone.QUERIES).
  Threshold, from the project's own retrieval: for each topic, the cosine between its query and the
  centroid of the Item 1A passages it retrieved and the scorer kept as on topic (FY2019-FY2025); the
  threshold is the lowest of the 13. A cluster below it matches every topic less well than the loosest
  topic matches its own scored passages: "uncovered". The min errs toward calling clusters covered.
  Check that the threshold isn't doing all the work: the share of each cluster's chunks that the
  13-topic retrieval picked (passages.parquet, any topic), against the corpus-wide share.
- Growth: per cluster and fiscal year, the share of companies with at least one chunk in it (companies,
  not chunks, so long filings don't dominate). FY2019 -> FY2025 change over the companies with Item 1A
  chunks in both years, 95% bootstrap interval over those companies (2,000 resamples, seed 0). Filings
  got longer, so most shares rise; the median change over all clusters is reported as that drift.
  Uncovered clusters are ranked by change.
- Optional names: Claude Haiku names the top 15 uncovered clusters by growth from their top words and
  3 central chunks, capped at $0.10 (Anthropic). Names are labels only; nothing is computed from them.

Added 2026-10-26 after seeing the first results: the threshold leaves only 13 clusters uncovered, mostly
governance boilerplate, while clusters plainly outside the 13 topics (litigation, patents, acquisitions)
clear it by a hair and the topic retrieval seldom picks their chunks. So a second flag is kept beside the
first: `rarely_retrieved`, fewer than half as many of the cluster's chunks retrieved as in the corpus
overall (the cut at half is a post-hoc choice). Names and the chart cover the clusters either flag
marks; the pre-registered `covered` column is unchanged. Also changed then: the example chunks come from 3
different companies, since one cluster's 3 nearest were one company's risk factor in 3 years, and Haiku
named the cluster after it.

Usage:
    uv run python -m sec_filings_tone.discover                    # clusters, tables, chart; free, a few minutes
    uv run python -m sec_filings_tone.discover --name --budget 0.1   # also Haiku names for the top 15
"""

import argparse
from textwrap import shorten

import bm25s
import numpy as np
import pandas as pd
from scipy import sparse

from sec_filings_tone import retrieve
from sec_filings_tone.budget import Cap, call_usd
from sec_filings_tone.sentiment import MODEL, TOPICS
from sec_filings_tone.tone import FILES, LAST_FY, OUT, QUERIES

FIRST_FY = 2019
K, SEEDS = 80, (0, 1)
N_WORDS, N_EXAMPLES, N_NAMED = 12, 3, 15
PRIOR = 1000  # pseudo-tokens in the log-odds prior
N_BOOT = 2000
TABLE = OUT / "discovered_topics.csv"  # one row per cluster
CLUSTERS = OUT / "discovered_clusters.parquet"  # chunk_id -> cluster, per seed

NAME_PROMPT = """These passages come from the risk factors (Item 1A) of S&P 500 10-K filings and were grouped
together by a clustering algorithm. Below are the group's most distinctive words and the passages nearest its
center. Name the risk theme they share in 2 to 6 words. If they share only generic boilerplate or no clear
theme, answer "mixed: " and the closest description. Reply with the name only."""


def corpus() -> tuple[pd.DataFrame, np.ndarray]:
    """10-K Item 1A chunks, FY2019-LAST_FY, and their vectors; only the selected rows are read from the memmap."""
    c = retrieve.chunks()
    mask = ((c.form == "10-K") & (c["item"] == "1A") & c.fy.between(FIRST_FY, LAST_FY)).to_numpy()
    idx = np.flatnonzero(mask)
    return c.iloc[idx][["chunk_id", "ticker", "sector", "fy", "text"]].reset_index(drop=True), \
        np.asarray(retrieve.stored_vectors()[idx], np.float32)


def kmeans(x: np.ndarray, k: int, seed: int, rounds: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means on unit vectors: k-means++ start, then each chunk to the centroid with the highest
    cosine and each centroid to its chunks' normalized mean, until no chunk moves. Returns labels, centroids."""
    rng = np.random.default_rng(seed)
    pick = [int(rng.integers(len(x)))]
    d = 1 - x @ x[pick[0]]  # 1 - cos = half the squared distance between unit vectors: k-means++'s weight
    for _ in range(k - 1):
        p = d.clip(0).astype(np.float64)
        pick.append(int(rng.choice(len(x), p=p / p.sum())))
        d = np.minimum(d, 1 - x @ x[pick[-1]])
    cent, labels = x[pick].copy(), np.full(len(x), -1)
    for _ in range(rounds):
        new = np.argmax(x @ cent.T, axis=1)
        if (new == labels).all():
            break
        labels = new
        sums = sparse.csr_matrix((np.ones(len(x)), (labels, np.arange(len(x)))), shape=(k, len(x))) @ x
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        cent = np.where(norms > 0, sums / np.where(norms > 0, norms, 1), cent).astype(np.float32)  # empty: keep
    return labels, cent


def top_words(texts: list[str], labels: np.ndarray, k: int, n: int = N_WORDS) -> list[list[str]]:
    """Each cluster's n most distinctive words: log-odds z-score against the rest, informative Dirichlet prior."""
    tok = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    vocab = np.array(sorted(tok.vocab, key=tok.vocab.get))
    ids = np.concatenate([np.asarray(t, np.int64) for t in tok.ids])
    cl = np.repeat(labels, [len(t) for t in tok.ids])
    counts = np.bincount(cl * len(vocab) + ids, minlength=k * len(vocab)).reshape(k, len(vocab)).astype(float)
    total = counts.sum(0)
    alpha = PRIOR * total / total.sum()
    rest = total - counts
    ni, nr = counts.sum(1, keepdims=True), rest.sum(1, keepdims=True)
    delta = np.log((counts + alpha) / (ni + PRIOR - counts - alpha)) - np.log((rest + alpha) / (nr + PRIOR - rest - alpha))
    z = delta / np.sqrt(1 / (counts + alpha) + 1 / (rest + alpha))
    z[:, np.char.isdigit(vocab.astype(str))] = -np.inf
    return [vocab[np.argsort(-row)[:n]].tolist() for row in z]


def threshold(meta: pd.DataFrame, x: np.ndarray, q: np.ndarray) -> tuple[float, pd.Series, set]:
    """Per topic, the cosine between its query and the centroid of its scored on-topic Item 1A passages in
    the corpus; the threshold is the lowest. Also returns the chunk ids retrieved for any topic."""
    passages, scores, _, _ = FILES["10-K"]
    p = pd.read_parquet(passages).join(pd.read_parquet(scores, columns=["pid", "off_topic"]).set_index("pid"))
    p = p[p.off_topic == False]  # noqa: E712 (missing scores drop out too)
    pos = pd.Series(np.arange(len(meta)), index=meta.chunk_id)
    ref = {}
    for i, topic in enumerate(TOPICS):
        rows = pos.reindex(p.chunk_id[p.topic == topic]).dropna().astype(int).to_numpy()
        m = x[np.unique(rows)].mean(0)
        ref[topic] = float(q[i] @ (m / np.linalg.norm(m)))
    ref = pd.Series(ref)
    return ref.min(), ref, set(p.chunk_id)


def growth(meta: pd.DataFrame, labels: np.ndarray, k: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per cluster: the share of companies filing that year with a chunk in it (by fiscal year), and the
    FY2019 -> LAST_FY change over the companies in both years with a 95% bootstrap interval over companies."""
    has = pd.crosstab([meta.ticker, meta.fy], labels).reindex(columns=range(k), fill_value=0) > 0
    share = has.groupby(level="fy").mean()
    a, b = has.xs(FIRST_FY, level="fy"), has.xs(LAST_FY, level="fy")
    both = a.index.intersection(b.index)
    d = (b.loc[both].to_numpy(float) - a.loc[both].to_numpy(float))
    boot = np.stack([d[i].mean(0) for i in np.random.default_rng(seed).integers(0, len(d), (N_BOOT, len(d)))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    ch = pd.DataFrame({f"share_{FIRST_FY}": a.loc[both].mean().to_numpy(), f"share_{LAST_FY}": b.loc[both].mean().to_numpy(),
                       "change": d.mean(0), "lo": lo, "hi": hi, "companies": len(both)})
    return share, ch


def name_clusters(t: pd.DataFrame, meta: pd.DataFrame, cap: Cap) -> dict[int, str]:
    """Haiku's 2-6 word name for each row of t (top words and example chunks), within the cap."""
    from anthropic import Anthropic

    client, names = Anthropic(timeout=60), {}
    text = meta.set_index("chunk_id").text
    for r in t.itertuples():
        ex = "\n\n".join(f"# PASSAGE {i}\n{text[cid][:1500]}" for i, cid in enumerate(r.examples.split(";"), 1))
        msg = f"Distinctive words: {r.words}\n\n{ex}"
        cap.check(call_usd(MODEL, len(msg) / 3 + 200, 1024))  # ~3 characters a token, generously
        resp = client.messages.create(model=MODEL, max_tokens=1024, output_config={"effort": "low"},
                                      system=NAME_PROMPT, messages=[{"role": "user", "content": msg}])
        cap.charge(call_usd(MODEL, resp.usage.input_tokens, resp.usage.output_tokens))
        text_out = next((b.text.strip() for b in resp.content if b.type == "text"), "")
        names[r.cluster] = text_out.splitlines()[-1].strip() if text_out else ""  # once it explained first: keep the name
        print(f"  {r.cluster:2}: {names[r.cluster]}  (${cap.spent():.4f} so far)")
    return names


def chart(t: pd.DataFrame, drift: float, path) -> None:
    """Dot-and-interval chart of the FY2019 -> LAST_FY change for the top uncovered clusters."""
    import matplotlib.pyplot as plt

    from sec_filings_tone.charts import AXIS, GRID, INK, MUTED, SERIES, SURFACE, header

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(t) + 2.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for color, (lab, g) in zip(SERIES, [("Below the topic-match threshold (pre-registered)", t[~t.covered]),
                                        ("Above it, but rarely retrieved by the 13 topics (post hoc)", t[t.covered])]):
        ax.hlines(g.index, g.lo, g.hi, color=color, lw=2, alpha=0.35)
        ax.plot(g.change, g.index, "o", color=color, ms=8, mec=SURFACE, mew=1.5, label=lab)
    y = np.arange(len(t))
    ax.axvline(0, color=AXIS, lw=1, zorder=0)
    ax.axvline(drift, color=MUTED, lw=1, ls="--", zorder=0)
    ax.text(drift, len(t) - 0.4, " median cluster", color=MUTED, fontsize=8, va="bottom")
    ax.set_yticks(y, [f"{shorten(lab, 60, placeholder=" …")}  ({a:.0%} → {b:.0%})" for lab, a, b in
                      zip(t.label, t[f"share_{FIRST_FY}"], t[f"share_{LAST_FY}"])], color=INK)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:+.0f} pts"))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, pad=6)
    header(fig, "Risk themes the 13 topics miss, fastest growing first",
           f"Change from FY{FIRST_FY} to FY{LAST_FY} in the share of companies with a risk-factor passage in each cluster; "
           "in brackets the share in each year",
           f"S&P 500 10-K Item 1A chunks, FY{FIRST_FY}-FY{LAST_FY}, k-means on their embeddings (k = {K}). Threshold: a "
           "cluster's centroid matches no topic query as well as the loosest topic matches its own scored passages. Rarely "
           "retrieved: under half the corpus-wide share of its chunks among the 13 topics' passages. Exploratory: found by "
           "looking, not a pre-registered test. Bars: 95% bootstrap intervals over companies. Dashed: the median change over "
           "all clusters (filings got longer, so most shares rose). Labels: Claude Haiku names.")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=2,
               frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", action="store_true", help=f"name the top {N_NAMED} uncovered clusters with Claude Haiku")
    p.add_argument("--budget", type=float, default=0.10, help="USD cap for --name (Anthropic)")
    a = p.parse_args()
    from sec_filings_tone.charts import FIGURES

    meta, x = corpus()
    print(f"{len(meta):,} Item 1A chunks, FY{FIRST_FY}-FY{LAST_FY}, {meta.ticker.nunique()} companies")
    runs = {s: kmeans(x, K, s) for s in SEEDS}
    labels, cent = runs[SEEDS[0]]
    pd.DataFrame({"chunk_id": meta.chunk_id, **{f"cluster_seed{s}": r[0] for s, r in runs.items()}}).to_parquet(CLUSTERS)

    q = np.load(QUERIES).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    thr, ref, retrieved = threshold(meta, x, q)
    print(f"threshold {thr:.3f} (lowest of the topics' own-passage cosines):\n{ref.round(3).to_string()}")
    sims = cent @ q.T
    share, t = growth(meta, labels, K)
    # each cluster's member chunks nearest its centroid, one per company (2026-10-26, after one cluster's three
    # were the same company's risk factor in three years)
    order = np.argsort(-np.einsum("ij,ij->i", x, cent[labels]))
    near = (meta[["ticker", "chunk_id"]].iloc[order].assign(cluster=labels[order]).drop_duplicates(["cluster", "ticker"])
            .groupby("cluster").head(N_EXAMPLES).groupby("cluster").chunk_id.agg(";".join))
    in_ret = meta.chunk_id.isin(retrieved).to_numpy()
    t = t.assign(cluster=range(K), chunks=np.bincount(labels, minlength=K),
                 best_topic=np.array(list(TOPICS))[sims.argmax(1)], best_cos=sims.max(1), covered=sims.max(1) >= thr,
                 retrieved=[in_ret[labels == c].mean() for c in range(K)],
                 words=[" ".join(w) for w in top_words(meta.text.tolist(), labels, K)],
                 examples=near.reindex(range(K)).to_numpy())

    t["rarely_retrieved"] = t.retrieved < in_ret.mean() / 2  # post hoc, 2026-10-26 (module docstring)
    t["flagged"] = ~t.covered | t.rarely_retrieved

    # stability: the seed-1 cluster closest to each seed-0 cluster, its growth and its rank among seed 1's flagged
    l1, c1 = runs[SEEDS[1]]
    _, t1 = growth(meta, l1, K)
    m = cent @ c1.T
    t1["flagged"] = ((c1 @ q.T).max(1) < thr) | (np.array([in_ret[l1 == c].mean() for c in range(K)]) < in_ret.mean() / 2)
    rank1 = t1[t1.flagged].change.rank(ascending=False, method="first")
    t = t.assign(match_cos=m.max(1), match_change=t1.change.to_numpy()[m.argmax(1)],
                 match_flagged=t1.flagged.to_numpy()[m.argmax(1)], match_rank=rank1.reindex(m.argmax(1)).to_numpy())

    t["name"] = ""
    t = t.sort_values(["flagged", "change"], ascending=[False, False])
    top = t[t.flagged].head(N_NAMED)
    if a.name:
        cap = Cap(a.budget)
        names = name_clusters(top, meta, cap)
        t["name"] = t.cluster.map(names).fillna("")
        print(f"naming spent ${cap.spent():.4f} (Anthropic)")
    t["label"] = np.where(t.name != "", t.name, t.words.str.split().str[:4].str.join(" "))
    t.to_csv(TABLE, index=False)
    print(f"wrote {TABLE} and {CLUSTERS}")

    drift = t.change.median()
    print(f"\n{(~t.covered).sum()} of {K} clusters below the threshold, {t.flagged.sum()} flagged by either rule; median "
          f"change over all clusters {drift:+.3f}; corpus-wide retrieved share {in_ret.mean():.3f}")
    cols = ["cluster", "chunks", "best_topic", "best_cos", "covered", "retrieved", f"share_{FIRST_FY}", f"share_{LAST_FY}",
            "change", "lo", "hi", "match_cos", "match_rank", "label"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 60, "display.float_format", "{:.3f}".format):
        print(t[t.flagged][cols].to_string(index=False))

    top = t[t.flagged].head(N_NAMED)
    path = FIGURES / "discovered_topics.png"
    chart(top, drift, path)
    long = share[top.cluster].rename_axis(columns="cluster").stack().rename("share").reset_index()
    long.merge(top[["cluster", "label", "change", "lo", "hi"]], on="cluster").to_csv(path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
