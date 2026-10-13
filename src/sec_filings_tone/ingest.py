"""Phase 2: clean and chunk cached sections into data/chunks.parquet, then embed into Chroma.

Usage:
    uv run python -m sec_filings_tone.ingest --tickers NVDA AMD          # chunk only (local, free)
    uv run python -m sec_filings_tone.ingest --tickers NVDA AMD --embed  # chunk, then embed (costs money)
    uv run python -m sec_filings_tone.ingest --embed --budget 30         # everything; stops first if over $30
    uv run python -m sec_filings_tone.ingest                             # chunk everything
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import tiktoken
from dotenv import load_dotenv

from sec_filings_tone.fetch import CACHE, DATA, FORMS, MIN_CHARS

CHUNKS = DATA / "chunks.parquet"
SKIPPED = DATA / "ingest_skipped.csv"
VECTOR_DB = "vector_db"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 1024
EMBED_PRICE_PER_M = 0.13  # USD per 1M tokens (OpenAI pricing page, checked 2026-10-12)
COLLECTION = "filings_v2"  # v1 embedded a context header and overlap; Phase 4 found both hurt (eval/FINDINGS.md)
CUT_STATEMENTS_OVER = 400_000  # an MD&A longer than this is checked for run-on financial statements (Citi 10-Q at 1.28M)
MAX_SECTION_CHARS = 500_000  # still longer after that cut means extraction went wrong; JPM's real 10-K MD&A is about 404k
MAX_MEDIAN_RATIO = 3
HEADING_MIN_TOKENS = 100  # a heading starts a new chunk only once the current one is this big
MIN_TAIL_TOKENS = 50  # a smaller last chunk of a section joins the one before it
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
    """Short and unpunctuated, but not a list item ("•our IP rights may be weak; or") or a sentence
    fragment split by a page break ("in the", "reduce existing inventories of")."""
    return (len(para) < 150 and not re.search(r"[.!?:;,]$", para)
            and not re.match(r"[•◦▪■●+\-–—]|[a-z]", para)
            and not re.search(r"\b(and|or|of|for|the|in|to|by|a|an|with|on|our|at|from|that|as)$", para))


def is_furniture(heading: str) -> bool:
    """A "heading" ending in a page number is a running header or footer ("Lam Research Corporation
    2019 10-K 14", "Other Key Information 54"); one with a $ or 3+ numbers is a table caption or row
    ("Three Months Ended March 27, ... $7,173"). Real headings have at most two ("Fiscal 2020 Compared
    to Fiscal 2019")."""
    return (re.search(r"\s\d{1,3}$", heading) is not None or "$" in heading
            or len(re.findall(r"\d[\d,.]*", heading)) >= 3)


FIN_STATEMENTS = re.compile(
    r"^[ \t#*|]*((condensed )?consolidated (statements?|balance sheets?) of (income|operations|financial condition|comprehensive income)"
    r"|report of independent registered public accounting firm|notes to (the )?(condensed )?consolidated financial statements)\b"
    r"[^\n]*$", re.I | re.M)


def drop_financial_statements(text: str) -> str:
    """Big banks' MD&A (C, COF, PNC, USB, FITB, JPM) often runs on into the financial statements. Cut at
    the first statement heading at least a quarter of the way in. Only for sections over the size
    limits: normal MD&As (PH, WTW) include summary tables titled "Consolidated Statement of Income",
    and a cut there would delete most of the MD&A. Only heading-like lines count: short and not
    ending in a period, so a sentence citing the notes doesn't."""
    for m in FIN_STATEMENTS.finditer(text, len(text) // 4):
        line = m.group(0).strip(" \t|*#")
        if len(line) < 120 and not line.endswith("."):
            return text[:m.start()]
    return text


def section_text(sec: dict) -> str:
    """A cached section with the parts that aren't MD&A or risk factors cut off."""
    text = (CACHE / f"{sec['ticker']}_{sec['key']}_{sec['item']}.txt").read_text()
    if sec["item"] == "mdna":
        text = drop_risk_factors(text)
    if sec.get("cut_statements"):
        text = drop_financial_statements(text)
    return text


def drop_risk_factors(text: str) -> str:
    """QCOM (2019-24), MCD and BEN put a full Risk Factors section inside 10-Q Item 2, so the MD&A
    ends at its heading. CAT, HBAN and JBHT have a stub there pointing to the 10-K's Item 1A,
    and the MD&A goes on after it."""
    m = re.search(r"^[ \t]*risk factors[ \t]*$", text, re.M | re.I)
    if m and not re.search(r"item\s+1a", text[m.end():m.end() + 400], re.I):
        return text[:m.start()]
    return text


def split_long(para: str, max_tokens: int) -> list[str]:
    """Sentences for a paragraph over max_tokens; a sentence still over it is cut by tokens."""
    pieces = []
    for sent in SENTENCE_END.split(para):
        toks = ENC.encode(sent)
        pieces += [ENC.decode(toks[i:i + max_tokens]) for i in range(0, len(toks), max_tokens)]
    return pieces


def chunk_section(text: str, max_tokens: int = 800, target_tokens: int = 500,
                  heading_min_tokens: int = HEADING_MIN_TOKENS) -> list[dict]:
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
            if is_furniture(para):
                continue
            if cur_tokens >= heading_min_tokens:
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
    # a section's last few lines ("Item 1B: Unresolved Staff Comments. None.") join the chunk before
    if len(chunks) > 1 and chunks[-1]["n_tokens"] < MIN_TAIL_TOKENS:
        merged = chunks[-2]["text"] + " " + chunks[-1]["text"]
        if n_tokens(merged) <= max_tokens:
            chunks.pop()
            chunks[-1] |= {"text": merged, "n_tokens": n_tokens(merged)}
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
                def why(chars):
                    return ("short" if chars < MIN_CHARS
                            else f"over {MAX_SECTION_CHARS:,} chars" if chars > MAX_SECTION_CHARS
                            else f"over {MAX_MEDIAN_RATIO}x ticker median" if chars > MAX_MEDIAN_RATIO * med
                            else None)
                if item in ("7", "mdna") and (sec["chars"] > CUT_STATEMENTS_OVER or sec["chars"] > MAX_MEDIAN_RATIO * med):
                    # too long may only mean it ran on into the financial statements: judge the cut text
                    sec["cut_statements"] = True
                    sec["chars"] = len(section_text(sec))
                reason = why(sec["chars"])
                (skipped if reason else keep).append({**sec, "reason": reason} if reason else sec)
    return keep, skipped


def build_chunks(tickers: list[str] | None = None, max_tokens: int = 800, target_tokens: int = 500,
                 out: Path = CHUNKS, heading_min_tokens: int = HEADING_MIN_TOKENS) -> pd.DataFrame:
    keep, skipped = sections(tickers)
    rows = []
    for sec in keep:
        text = section_text(sec)
        for seq, c in enumerate(chunk_section(text, max_tokens, target_tokens, heading_min_tokens)):
            rows.append({"chunk_id": f"{sec['ticker']}_{sec['form']}_{sec['key']}_{sec['item']}_{seq:04d}",
                         **{k: sec[k] for k in ("ticker", "sector", "form", "item", "fy",
                                                 "report_date", "filing_date", "accession")},
                         # calendar quarter of the period end; fiscal labels never matter
                         "period_quarter": (int(sec["report_date"][5:7]) - 1) // 3 + 1,
                         "seq": seq, **c})
    chunks = pd.DataFrame(rows)
    chunks.to_parquet(out, index=False)
    skipped_path = SKIPPED if out == CHUNKS else out.with_suffix(".skipped.csv")
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)

    raw = sum(s["chars"] for s in keep)
    kept = chunks.text.str.len().sum()
    print(f"{len(keep)} sections -> {len(chunks)} chunks in {out}; {len(skipped)} sections skipped ({skipped_path})")
    print(f"tokens per chunk: {chunks.n_tokens.describe(percentiles=[.05, .5, .95]).round().to_dict()}")
    print(f"chunks per section: median {chunks.groupby(['ticker', 'form', 'item', 'report_date']).size().median():.0f}")
    print(f"text kept after cleaning: {kept / raw:.0%} of raw characters (rest is tables, markup, headings, "
          "risk factors inside 10-Q MD&A)")
    if skipped:
        print(pd.DataFrame(skipped).groupby("reason").size().to_string())
    return chunks


def embed_text(chunks: pd.DataFrame, overlap_tokens: int = 60, overlap_sentences: int = 1,
               header: str = "full") -> list[str]:
    """Context header + the previous chunk's last sentence(s), capped at overlap_tokens + text.
    header is "full", "no_heading" (company, form, period, item) or "none". The reranker reads
    passages in this format; filings_v1 embedded it, but Phase 4 found the header and overlap
    hurt vector search, so embed_chunks now embeds the plain text."""
    out, prev = [], {}
    for r in chunks.itertuples():
        when = f"FY{r.fy}" if r.form == "10-K" else f"quarter ending {r.report_date}"
        head = f"{r.ticker} ({r.sector}) {r.form} {when}, {ITEM_NAMES[r.item]}"
        if header == "full" and r.heading:
            head += f" — {r.heading}"
        section = (r.ticker, r.form, r.item, r.report_date)
        overlap = ""
        if overlap_tokens and section in prev and prev[section][0] == r.seq - 1:
            last = " ".join(SENTENCE_END.split(prev[section][1])[-overlap_sentences:])
            overlap = ENC.decode(ENC.encode(last)[-overlap_tokens:]) + " "
        prev[section] = (r.seq, r.text)
        out.append(f"{overlap}{r.text}" if header == "none" else f"{head}\n\n{overlap}{r.text}")
    return out


def embed_chunks(chunks: pd.DataFrame, collection: str = COLLECTION, budget: float = 5.0) -> None:
    """Embed each chunk's plain text into Chroma. Chunk ids already in the collection are skipped,
    so it can be stopped and rerun. Stops before any request if the estimate is over budget (USD)."""
    import chromadb
    from openai import OpenAI

    chunks = chunks.sort_values(["ticker", "form", "item", "report_date", "seq"])
    docs = chunks.text.tolist()
    col = chromadb.PersistentClient(VECTOR_DB).get_or_create_collection(
        collection, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)

    done = set()
    ids = chunks.chunk_id.tolist()
    for i in range(0, len(ids), 5000):
        done |= set(col.get(ids=ids[i:i + 5000], include=[])["ids"])
    todo = [i for i, cid in enumerate(ids) if cid not in done]
    tokens = [n_tokens(docs[i]) for i in todo]
    cost = sum(tokens) / 1e6 * EMBED_PRICE_PER_M
    print(f"{len(todo)} of {len(ids)} chunks to embed, {sum(tokens):,} tokens, about ${cost:.2f} at ${EMBED_PRICE_PER_M}/M")
    if cost > budget:
        raise SystemExit(f"estimated ${cost:.2f} is over the ${budget:.2f} budget; rerun with --budget {cost * 1.1:.0f}")

    client = OpenAI(timeout=60, max_retries=3)  # a stuck request fails instead of hanging
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
    p.add_argument("--budget", type=float, default=5.0, help="USD; embedding stops before starting if over it")
    args = p.parse_args()
    chunks = build_chunks(args.tickers)
    if args.embed:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (add it to .env)")
        embed_chunks(chunks, budget=args.budget)


if __name__ == "__main__":
    main()
