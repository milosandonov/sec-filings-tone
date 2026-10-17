"""Phase 11: write the data release to data/release/: the derived tables every published result is
computed from, so events.py, diff.py report, comove.py and charts.py can be rerun without the SEC
fetch, the embeddings or any API call. Free and local. Rerun it after any change to the data, with
a new --version; files the previous export wrote are removed first, README.md (the card) is kept.

What goes in (data/release/README.md describes each file): every scored passage keyed by (ticker,
fy or report_date, topic, rank, chunk_id), the baselines, the cells, the carryover shares, the
passage text, the label sheets and guidelines, the event definitions, companies.csv (sectors and
sub-industries, which the export-control test needs) and the result CSVs; plus
provenance.json (model, prompt and schema hashes, row counts) and SHA256SUMS. Left out: the
embeddings (vectors.npy, 2.8 GB), backups, logs, batch state and the Loughran-McDonald dictionary
itself (only the per-passage shares derived from it are in). --no-text also leaves out the passage
text and the model's quoted evidence; the label sheets still quote their 40 passages, as the repo does.

Usage:
    uv run python -m sec_filings_tone.export --version 0.1.0
    uv run python -m sec_filings_tone.export --version 0.1.0 --no-text --out <folder>
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from anthropic.lib._parse._transform import transform_schema  # what tone.py sends, so the hash is of that

from sec_filings_tone import events
from sec_filings_tone.diff import CARRYOVER
from sec_filings_tone.ingest import CHUNKS, EMBED_DIMS, EMBED_MODEL
from sec_filings_tone.retrieve import CANDIDATES, RRF_K
from sec_filings_tone.sentiment import (DIMENSIONS, FINBERT, LM_PATH, MODEL, OUT as SENTIMENT, TOPICS, schema,
                                        system_prompt, user_prompt)
from sec_filings_tone.tone import FILES, K, KEY, OUT

RELEASE = Path("data/release")
ROOT = Path(__file__).parents[2]
META = ["item", "filing_date", "report_date", "accession"]  # chunk metadata the event tests need, kept without the text


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def scored(form: str, chunks: pd.DataFrame) -> pd.DataFrame:
    """One row per passage with its scores and chunk metadata, key columns first. pid is the passage's
    row number in data/tone's passages file, which the scores there are keyed by."""
    passages, scores = FILES[form][:2]
    p = pd.read_parquet(passages).reset_index(names="pid")
    p = p.merge(pd.read_parquet(scores).drop(columns="batch"), on="pid", how="left", validate="1:1")
    if n := p.off_topic.isna().sum():  # e.g. passages retrieved again after a fix and not yet scored
        raise SystemExit(f"{passages}: {n} passages have no score; score them before releasing")
    p = p.merge(chunks[["chunk_id", *[c for c in META if c not in p]]], on="chunk_id", how="left", validate="m:1")
    key = [*KEY.get(form, KEY["10-K"]), "topic", "rank", "chunk_id"]
    if p.duplicated(key).any() or p.item.isna().any():
        raise SystemExit(f"{passages}: duplicate keys or chunks missing from {CHUNKS}")
    return p[[*key, *[c for c in p if c not in key]]]


def export(version: str, text: bool = True, out: Path = RELEASE) -> None:
    ids = pd.concat([pd.read_parquet(FILES[f][0], columns=["chunk_id"]) for f in ("10-K", "10-Q")]).chunk_id
    chunks = pd.read_parquet(CHUNKS, filters=[("chunk_id", "in", ids.unique().tolist())])
    keep = events.named()
    k = scored("10-K", chunks)
    k["names_topic"] = k.pid.isin(keep)  # the word check (events.TOPIC_WORDS); True for topics without a word list
    rerun = scored("rerun", chunks).drop(columns="pid")  # B6's second run; orig_pid points into passages
    tables = {
        "passages": k,
        "q_passages": scored("10-Q", chunks),
        "rerun": rerun,
        "cells": pd.concat([events.cells().assign(version="model"), events.cells(keep=keep).assign(version="word-checked")]),
        "q_cells": events.cells(form="10-Q"),
        "baselines": pd.concat([pd.read_parquet(FILES[f][3]) for f in ("10-K", "10-Q")], ignore_index=True),
        "carryover": pd.read_parquet(CARRYOVER),
        "gold_scores": pd.read_parquet(SENTIMENT / f"gold_scores_{MODEL}.parquet"),
        "chunks": chunks,
    }
    if not text:
        del tables["chunks"]
        tables = {n: t.drop(columns=t.filter(like="_evidence").columns) for n, t in tables.items()}
    copies = ({f"labels/{f.name}": f for f in sorted((ROOT / "labels").glob("*")) if f.suffix in (".csv", ".md")}
              | {f.name: f for f in (events.EVENTS, events.QUARTERLY[0], ROOT / "data" / "companies.csv")}  # definitions; sub-industries
              | {f"results/{f.name}": f for f in sorted(OUT.glob("*.csv"))})  # what events, diff and comove wrote

    system, fmt = system_prompt(), json.dumps(transform_schema(schema(list(DIMENSIONS))))
    template = user_prompt({"ticker": "{ticker}", "form": "10-K", "fy": "{fy}", "item": "1A", "text": "{text}"},
                           "{topic}", list(DIMENSIONS))
    hashes = {"system_prompt_sha256": sha(system.encode()), "output_schema_sha256": sha(fmt.encode()),
              "user_prompt_template_sha256": sha(template.encode())}
    git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True)
    prov = {
        "data_version": version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git.stdout.strip() or None,  # "-dirty": uncommitted changes in the tree at export time
        "text_included": text,
        "scorer": {"model": MODEL, **hashes,
                   # one short key for this prompt, schema and template; stamp it on future score files
                   "prompt_key": sha("".join(hashes.values()).encode())[:12],
                   "hash_of": {"system_prompt": "sentiment.system_prompt()",
                               "output_schema": "json.dumps(transform_schema(sentiment.schema(list(DIMENSIONS))))",
                               "user_prompt_template": "sentiment.user_prompt with {ticker} 10-K FY{fy} Item 1A, {topic}, {text}"},
                   "request": "one call per passage, max_tokens 4000, effort low, Message Batches API (tone.score_cells)"},
        "retrieval": {"top_k": K, "candidates": CANDIDATES, "rrf_k": RRF_K, "topic_queries": TOPICS},
        "embedding": {"model": EMBED_MODEL, "dims": EMBED_DIMS},
        "baselines": {"finbert": FINBERT, "loughran_mcdonald": f"{LM_PATH.name} (not included)"},
        "word_check": events.TOPIC_WORDS,
        "coverage": {"companies": int(k.ticker.nunique()), "10-Ks": len(k.drop_duplicates(["ticker", "fy"])),
                     "fiscal_years": [int(k.fy.min()), int(k.fy.max())],
                     "cells": int((tables["cells"].version == "model").sum()),
                     "10-Qs": len(tables["q_passages"].drop_duplicates(["ticker", "report_date"]))},
        "rows": {f"{n}.parquet": len(t) for n, t in tables.items()},
    }

    out.mkdir(parents=True, exist_ok=True)
    if (old := out / "SHA256SUMS").exists():  # remove what the last export wrote, so files dropped since don't linger
        for line in old.read_text().splitlines():
            (out / line.split("  ", 1)[1]).unlink(missing_ok=True)
    for n, t in tables.items():
        t.to_parquet(out / f"{n}.parquet", index=False, compression="zstd")
    for dest, src in copies.items():
        (out / dest).parent.mkdir(exist_ok=True)
        shutil.copyfile(src, out / dest)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    written = [f"{n}.parquet" for n in tables] + list(copies) + ["provenance.json"]
    old.write_text("".join(f"{sha((out / f).read_bytes())}  {f}\n" for f in written))  # check: sha256sum -c SHA256SUMS

    for f in written:
        print(f"{(out / f).stat().st_size / 1e6:9.2f} MB  {f}")
    print(f"{sum((out / f).stat().st_size for f in written) / 1e6:9.2f} MB  in all, {len(written)} files, in {out}")
    card = out / "README.md"
    if not card.exists() or f'data_version: "{version}"' not in card.read_text():
        print(f'{card} does not say data_version: "{version}"; update the card')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True, help="data version, e.g. 0.1.0; bump it whenever the data changes")
    p.add_argument("--no-text", action="store_true", help="leave out the passage text and the model's quoted evidence")
    p.add_argument("--out", type=Path, default=RELEASE)
    a = p.parse_args()
    export(a.version, not a.no_text, a.out)


if __name__ == "__main__":
    main()
