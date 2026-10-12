"""Phase 2: clean and chunk cached sections into data/chunks.parquet, then embed into Chroma.

Usage:
    uv run python -m sec_filings_tone.ingest --tickers NVDA AMD          # chunk only (local, free)
    uv run python -m sec_filings_tone.ingest --tickers NVDA AMD --embed  # chunk, then embed (costs money)
    uv run python -m sec_filings_tone.ingest                             # chunk everything
"""

import argparse
import os
import re

import pandas as pd
import tiktoken
from dotenv import load_dotenv

from sec_filings_tone.fetch import CACHE, DATA, FORMS, MIN_CHARS

CHUNKS = DATA / "chunks.parquet"
SKIPPED = DATA / "ingest_skipped.csv"
VECTOR_DB = "vector_db"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 1024
EMBED_PRICE_PER_M = 0.13  # USD per 1M tokens, from memory: check before the full run
MAX_SECTION_CHARS = 400_000  # longer means fetch.py ran past the section (Citi 10-Q at 1.28M)
MAX_MEDIAN_RATIO = 3
HEADING_MIN_TOKENS = 100  # a heading starts a new chunk only once the current one is this big
ITEM_NAMES = {"1A": "Item 1A Risk Factors", "7": "Item 7 MD&A", "mdna": "MD&A"}

ENC = tiktoken.get_encoding("cl100k_base")
# Lowercase/digit before the period keeps "U.S. Government" together
SENTENCE_END = re.compile(r"(?<=[a-z0-9%)”\"][.!?])\s+(?=[A-Z“\"])")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def clean_line(line: str) -> str:
    """Strip the Markdown that fetch.py's fallback leaves in (Intel, bank exhibits)."""
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)  # images
    line = re.sub(r"\\(.)", r"\1", line)  # \. \- \(
    line = re.sub(r"</?u>|\*\*|\*|^\s*#+\s*", " ", line)
    return re.sub(r"\s+", " ", line.replace("|", " ")).strip()


def is_prose(line: str) -> bool:
    """False for table rows of figures and page furniture: fewer than half the tokens
    are words (2+ letters), e.g. "% of net revenue 14.2 % 27.2 %" or "| MD&A | 20 |"."""
    tokens = line.split()
    words = sum(len(re.findall(r"[A-Za-z]", t)) >= 2 for t in tokens)
    return words * 2 >= len(tokens) and line.lower() != "table of contents"


def paragraphs(text: str) -> list[str]:
    paras, cur = [], []
    for raw in text.splitlines() + [""]:
        line = clean_line(raw)
        if not raw.strip():
            if cur:
                paras.append(" ".join(cur))
            cur = []
        elif line and is_prose(line):
            cur.append(line)
    return paras


def is_heading(para: str) -> bool:
    return len(para) < 150 and not re.search(r"[.!?:;,]$", para)


def split_long(para: str, max_tokens: int) -> list[str]:
    """Sentences for a paragraph over max_tokens; a sentence still over it is cut by tokens."""
    pieces = []
    for sent in SENTENCE_END.split(para):
        toks = ENC.encode(sent)
        pieces += [ENC.decode(toks[i:i + max_tokens]) for i in range(0, len(toks), max_tokens)]
    return pieces


def chunk_section(text: str, max_tokens: int = 800, target_tokens: int = 500) -> list[dict]:
    """Section text -> [{"heading", "text", "n_tokens"}]. Packs paragraphs until a chunk
    reaches target_tokens, never past max_tokens; headings go in metadata, not text."""
    chunks, cur, cur_tokens, heading, cur_heading = [], [], 0, "", ""

    def flush():
        nonlocal cur, cur_tokens
        if cur:
            chunks.append({"heading": cur_heading, "text": " ".join(cur), "n_tokens": cur_tokens})
        cur, cur_tokens = [], 0

    for para in paragraphs(text):
        if is_heading(para):
            if cur_tokens >= HEADING_MIN_TOKENS:
                flush()
            heading = para
            if not cur:
                cur_heading = heading
            continue
        pieces = split_long(para, max_tokens) if n_tokens(para) > max_tokens else [para]
        for piece in pieces:
            if cur and n_tokens(" ".join(cur + [piece])) > max_tokens:
                flush()
            if not cur:
                cur_heading = heading
            cur.append(piece)
            cur_tokens = n_tokens(" ".join(cur))
            if cur_tokens >= target_tokens:
                flush()
    flush()
    return chunks


def sections(tickers: list[str] | None) -> tuple[list[dict], list[dict]]:
    """Every cached section to chunk, plus the ones skipped and why."""
    keep, skipped = [], []
    for form, (manifest, items) in FORMS.items():
        df = pd.read_csv(manifest).dropna(subset=["report_date"])
        if tickers:
            df = df[df.ticker.isin(tickers)]
        for item in items:
            col = f"len_{item}"
            median = df.groupby("ticker")[col].transform("median")
            for row, med in zip(df.itertuples(), median):
                fy = int(row.fy)
                key = fy if form == "10-K" else row.report_date
                sec = {"ticker": row.ticker, "sector": row.sector, "form": form, "item": item, "fy": fy,
                       "report_date": row.report_date, "filing_date": row.filing_date,
                       "accession": row.accession, "key": key, "chars": int(getattr(row, col))}
                reason = ("short" if sec["chars"] < MIN_CHARS
                          else f"over {MAX_SECTION_CHARS:,} chars" if sec["chars"] > MAX_SECTION_CHARS
                          else f"over {MAX_MEDIAN_RATIO}x ticker median" if sec["chars"] > MAX_MEDIAN_RATIO * med
                          else None)
                (skipped if reason else keep).append({**sec, "reason": reason} if reason else sec)
    return keep, skipped


def build_chunks(tickers: list[str] | None = None) -> pd.DataFrame:
    keep, skipped = sections(tickers)
    rows = []
    for sec in keep:
        text = (CACHE / f"{sec['ticker']}_{sec['key']}_{sec['item']}.txt").read_text()
        for seq, c in enumerate(chunk_section(text)):
            rows.append({"chunk_id": f"{sec['ticker']}_{sec['form']}_{sec['key']}_{sec['item']}_{seq:04d}",
                         **{k: sec[k] for k in ("ticker", "sector", "form", "item", "fy",
                                                 "report_date", "filing_date", "accession")},
                         # calendar quarter of the period end; fiscal labels never matter
                         "period_quarter": (int(sec["report_date"][5:7]) - 1) // 3 + 1,
                         "seq": seq, **c})
    chunks = pd.DataFrame(rows)
    chunks.to_parquet(CHUNKS, index=False)
    pd.DataFrame(skipped).to_csv(SKIPPED, index=False)

    raw = sum(s["chars"] for s in keep)
    kept = chunks.text.str.len().sum()
    print(f"{len(keep)} sections -> {len(chunks)} chunks in {CHUNKS}; {len(skipped)} sections skipped ({SKIPPED})")
    print(f"tokens per chunk: {chunks.n_tokens.describe(percentiles=[.05, .5, .95]).round().to_dict()}")
    print(f"chunks per section: median {chunks.groupby(['ticker', 'form', 'item', 'report_date']).size().median():.0f}")
    print(f"text kept after cleaning: {kept / raw:.0%} of raw characters (rest is tables, markup, headings)")
    if skipped:
        print(pd.DataFrame(skipped).groupby("reason").size().to_string())
    return chunks


def embed_text(chunks: pd.DataFrame, overlap_tokens: int = 60) -> list[str]:
    """What gets embedded: context header + last sentence of the previous chunk (capped) + text.
    The stored text stays clean, so scorers never see the header or count overlap twice."""
    out, prev = [], {}
    for r in chunks.itertuples():
        when = f"FY{r.fy}" if r.form == "10-K" else f"quarter ending {r.report_date}"
        header = f"{r.ticker} ({r.sector}) {r.form} {when}, {ITEM_NAMES[r.item]}" + (f" — {r.heading}" if r.heading else "")
        section = (r.ticker, r.form, r.item, r.report_date)
        overlap = ""
        if overlap_tokens and section in prev and prev[section][0] == r.seq - 1:
            last = SENTENCE_END.split(prev[section][1])[-1]
            overlap = ENC.decode(ENC.encode(last)[-overlap_tokens:]) + " "
        prev[section] = (r.seq, r.text)
        out.append(f"{header}\n\n{overlap}{r.text}")
    return out


def embed_chunks(chunks: pd.DataFrame, collection: str = "filings_v1", overlap_tokens: int = 60) -> None:
    """Embed into Chroma. Chunk ids already in the collection are skipped, so it can be stopped and rerun."""
    import chromadb
    from openai import OpenAI

    chunks = chunks.sort_values(["ticker", "form", "item", "report_date", "seq"])
    docs = embed_text(chunks, overlap_tokens)
    col = chromadb.PersistentClient(VECTOR_DB).get_or_create_collection(
        collection, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)

    done = set()
    ids = chunks.chunk_id.tolist()
    for i in range(0, len(ids), 5000):
        done |= set(col.get(ids=ids[i:i + 5000], include=[])["ids"])
    todo = [i for i, cid in enumerate(ids) if cid not in done]
    tokens = [n_tokens(docs[i]) for i in todo]
    print(f"{len(todo)} of {len(ids)} chunks to embed, {sum(tokens):,} tokens, "
          f"about ${sum(tokens) / 1e6 * EMBED_PRICE_PER_M:.2f} at ${EMBED_PRICE_PER_M}/M (check current price)")

    client = OpenAI()
    meta_cols = ["ticker", "sector", "form", "item", "fy", "report_date", "filing_date",
                 "accession", "period_quarter", "heading", "seq", "n_tokens"]
    batch, batch_tokens = [], 0
    for j, (i, t) in enumerate(zip(todo, tokens)):
        batch.append(i)
        batch_tokens += t
        # the API caps a request at 2,048 inputs and 300k tokens
        if len(batch) == 1000 or batch_tokens > 200_000 or j == len(todo) - 1:
            resp = client.embeddings.create(model=EMBED_MODEL, dimensions=EMBED_DIMS, input=[docs[k] for k in batch])
            rows = chunks.iloc[batch]
            col.add(ids=rows.chunk_id.tolist(), embeddings=[d.embedding for d in resp.data],
                    documents=rows.text.tolist(),
                    metadatas=[{k: (v.item() if hasattr(v, "item") else v) for k, v in m.items()}
                               for m in rows[meta_cols].to_dict("records")])
            print(f"  embedded {j + 1}/{len(todo)}")
            batch, batch_tokens = [], 0
    print(f"collection {collection}: {col.count()} chunks")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--embed", action="store_true")
    args = p.parse_args()
    chunks = build_chunks(args.tickers)
    if args.embed:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (add it to .env)")
        embed_chunks(chunks)


if __name__ == "__main__":
    main()
