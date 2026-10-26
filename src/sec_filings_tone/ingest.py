"""Phase 2: clean and chunk cached sections into data/chunks.parquet, then embed into Chroma.

Usage:
    uv run python -m sec_filings_tone.ingest --tickers NVDA AMD          # chunk only, into data/chunks_subset.parquet (local, free)
    uv run python -m sec_filings_tone.ingest --tickers NVDA AMD --embed  # chunk those, then embed them (costs money)
    uv run python -m sec_filings_tone.ingest --embed --budget 30         # everything; stops first if over $30
    uv run python -m sec_filings_tone.ingest --embed --batch --wait --budget 17 --no-rebuild  # Batch API: half price
    uv run python -m sec_filings_tone.ingest                             # chunk everything
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import tiktoken
from dotenv import load_dotenv

from sec_filings_tone.config import (BATCH_MAX_INPUTS, CACHE, CHUNKS, COLLECTION, EMBED_BATCHES, EMBED_DIMS, EMBED_MODEL,
                                     EMBED_PRICE_PER_M, QUEUE_SHARE, REQUEST_INPUTS, SKIPPED, SUBSET, VECTOR_DB)
from sec_filings_tone.fetch import FORMS, MIN_CHARS

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
    if tickers and out == CHUNKS:  # the analyses inner-join CHUNKS, so a subset there would silently drop every other company
        raise SystemExit(f"{len(tickers)} tickers would replace {CHUNKS}, which holds every company; write them to another file")
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


META_COLS = ["ticker", "sector", "form", "item", "fy", "report_date", "filing_date",
             "accession", "period_quarter", "heading", "seq", "n_tokens"]


def take(tokens: list[int], max_inputs: int, max_tokens: float) -> int:
    """How many of the next chunks fit in one batch: at most max_inputs, with at most max_tokens in total."""
    n = total = 0
    for t in tokens[:max_inputs]:
        if total + t > max_tokens:
            break
        total += t
        n += 1
    return n


def queue_limit(messages: list[str]) -> int | None:
    """The account's queued-token limit from a token_limit_exceeded error ("Limit: 20,000,000 enqueued tokens")."""
    m = re.search(r"Limit: ([\d,]+) enqueued tokens", " ".join(messages))
    return int(m.group(1).replace(",", "")) if m else None


def open_collection(name: str = COLLECTION):
    import chromadb

    return chromadb.PersistentClient(VECTOR_DB).get_or_create_collection(
        name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)


def add_to_chroma(col, rows: pd.DataFrame, vectors) -> None:
    for i in range(0, len(rows), 5000):  # Chroma caps how many rows one add takes
        part = rows.iloc[i:i + 5000]
        col.add(ids=part.chunk_id.tolist(), embeddings=list(vectors[i:i + 5000]), documents=part.text.tolist(),
                metadatas=[{k: (v.item() if hasattr(v, "item") else v) for k, v in m.items()}
                           for m in part[META_COLS].to_dict("records")])


def not_embedded(col, ids: list[str]) -> list[str]:
    done = set()
    for i in range(0, len(ids), 5000):
        done |= set(col.get(ids=ids[i:i + 5000], include=[])["ids"])
    return [c for c in ids if c not in done]


def embed_chunks(chunks: pd.DataFrame, collection: str = COLLECTION, budget: float = 5.0, rec: dict | None = None) -> None:
    """Embed each chunk's plain text into Chroma. Chunk ids already in the collection are skipped,
    so it can be stopped and rerun. Stops before any request if the estimate is over budget (USD).
    rec (the run's budget.run_log record) gets the USD billed and the chunks embedded as they go."""
    from openai import OpenAI

    rec = {} if rec is None else rec
    rec |= {"usd": {"openai": 0.0}, "rows": 0}

    chunks = chunks.sort_values(["ticker", "form", "item", "report_date", "seq"]).set_index("chunk_id", drop=False)
    col = open_collection(collection)
    todo = chunks.loc[not_embedded(col, chunks.chunk_id.tolist())]
    cost = todo.n_tokens.sum() / 1e6 * EMBED_PRICE_PER_M
    print(f"{len(todo)} of {len(chunks)} chunks to embed, {todo.n_tokens.sum():,} tokens, about ${cost:.2f} at ${EMBED_PRICE_PER_M}/M")
    if cost > budget:
        raise SystemExit(f"estimated ${cost:.2f} is over the ${budget:.2f} budget; rerun with --budget {cost * 1.1:.0f}")

    client = OpenAI(timeout=60, max_retries=3)  # a stuck request fails instead of hanging
    batch, batch_tokens = [], 0
    for j, (cid, t) in enumerate(zip(todo.chunk_id, todo.n_tokens)):
        batch.append(cid)
        batch_tokens += t
        # the API caps a request at 2,048 inputs and 300k tokens
        if len(batch) == 1000 or batch_tokens > 200_000 or j == len(todo) - 1:
            rows = todo.loc[batch]
            resp = client.embeddings.create(model=EMBED_MODEL, dimensions=EMBED_DIMS, input=rows.text.tolist())
            rec["usd"]["openai"] += resp.usage.total_tokens / 1e6 * EMBED_PRICE_PER_M
            add_to_chroma(col, rows, [d.embedding for d in resp.data])
            rec["rows"] += len(rows)
            print(f"  embedded {j + 1}/{len(todo)}")
            batch, batch_tokens = [], 0
    print(f"collection {collection}: {col.count()} chunks")


def embed_batches(chunks: pd.DataFrame, collection: str = COLLECTION, budget: float = 15.0,
                  wait: bool = False, poll_minutes: float = 1, state_dir: Path = EMBED_BATCHES) -> None:
    """Embed through OpenAI's Batch API: half the price, results within 24 hours. Safe to stop and
    rerun at any point: the state is in data/embed_batches/state.json, batches keep running on
    OpenAI's side, finished ones are loaded into Chroma on the next run, and chunks from failed
    batches or failed requests are resubmitted. Each account has a cap on queued tokens; if a batch
    fails on it, the limit is read from the error, and from then on batches are sized and submitted
    so the tokens in flight stay under 90% of it. OpenAI counts queued tokens as characters / 4, not
    real tokens: b006 passed at 19.7M by that count (16.1M real), b014 failed alone at 25.7M (18.0M real)."""
    import base64
    import json
    import time

    import numpy as np
    from openai import OpenAI

    client = OpenAI(timeout=300, max_retries=3)
    col = open_collection(collection)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"jobs": []})
    state.setdefault("queue_tokens", None)  # learned from the first token_limit_exceeded error
    by_id = chunks.set_index("chunk_id", drop=False)
    half_price = EMBED_PRICE_PER_M / 2

    def save():
        state_path.write_text(json.dumps(state, indent=1))

    queue_est = by_id.text.str.len() // 4  # OpenAI's queued-token count; billing uses real tokens

    def job_tokens(job):
        return int(queue_est.loc[job["ids"]].sum())

    def load(job):
        """Download a finished batch's results and add them to Chroma; failed requests stay to do."""
        out = state_dir / f"{job['name']}_out.jsonl"
        b = client.batches.retrieve(job["batch_id"])
        if b.output_file_id and not out.exists():
            client.files.content(b.output_file_id).write_to_file(out)
        ids, vecs = [], []
        for line in out.read_text().splitlines() if out.exists() else []:
            r = json.loads(line)
            if r["response"] and r["response"]["status_code"] == 200:
                i = int(r["custom_id"].rsplit("-", 1)[1])
                part = job["ids"][i * REQUEST_INPUTS:(i + 1) * REQUEST_INPUTS]
                data = sorted(r["response"]["body"]["data"], key=lambda d: d["index"])
                ids += part
                vecs += [np.frombuffer(base64.b64decode(d["embedding"]), dtype=np.float32) for d in data]
        missing = set(not_embedded(col, ids))  # skip anything already loaded by an earlier, interrupted run
        keep = [k for k, cid in enumerate(ids) if cid in missing]
        if keep:
            add_to_chroma(col, by_id.loc[[ids[k] for k in keep]], [vecs[k] for k in keep])
        job["status"], job["loaded"] = "loaded", len(ids)
        print(f"  {job['name']}: loaded {len(ids):,} of {len(job['ids']):,} chunks"
              + (f" ({b.request_counts.failed} requests failed; their chunks will be resubmitted)" if b.request_counts.failed else ""))
        out.unlink(missing_ok=True)
        save()

    while True:
        # 1. check submitted batches. A finished one frees OpenAI's queue at once, so the next batch is
        # submitted (step 2) before the slow load into Chroma (step 3)
        for job in [j for j in state["jobs"] if j["status"] == "submitted"]:
            b = client.batches.retrieve(job["batch_id"])
            if b.status == "completed":
                job["status"] = "completed"
                save()
            elif b.status in ("failed", "expired", "cancelled"):
                errors = b.errors.data if b.errors else []
                codes = [e.code for e in errors]
                job["status"] = b.status
                if "token_limit_exceeded" in codes:  # the account's queue cap; such batches aren't billed
                    limit = queue_limit([e.message or "" for e in errors])
                    # a batch can fail under the stated limit if the estimate is off: then shrink below it,
                    # rather than resubmit the same batch forever
                    state["queue_tokens"] = min(limit or job_tokens(job) // 2, job_tokens(job))
                print(f"  {job['name']}: {b.status} {codes}; its chunks go back to the queue")
                save()
        inflight = [j for j in state["jobs"] if j["status"] == "submitted"]
        finished = [j for j in state["jobs"] if j["status"] == "completed"]  # not loaded yet, maybe by an earlier run
        queued = {cid for j in inflight + finished for cid in j["ids"]}
        todo = [c for c in not_embedded(col, by_id.chunk_id.tolist()) if c not in queued]
        spent = sum(j["usd"] for j in state["jobs"] if j["status"] in ("submitted", "completed", "loaded"))
        if not todo and not inflight and not finished:
            print(f"done: {col.count():,} chunks in {collection}; Batch API spend about ${spent:.2f}")
            return
        # 2. submit more, within the budget and the queue cap
        todo_tokens = queue_est.loc[todo].tolist()
        while todo:
            room = (float("inf") if state["queue_tokens"] is None
                    else QUEUE_SHARE * state["queue_tokens"] - sum(job_tokens(j) for j in inflight))
            n = take(todo_tokens, BATCH_MAX_INPUTS, room)
            if n == 0:
                break  # the queue is full: wait for batches in flight to finish
            ids, todo, todo_tokens = todo[:n], todo[n:], todo_tokens[n:]
            rows = by_id.loc[ids]
            usd = rows.n_tokens.sum() / 1e6 * half_price
            if spent + usd > budget:
                raise SystemExit(f"cost cap: about ${spent:.2f} submitted; the next batch (about ${usd:.2f}) would pass "
                                 f"${budget:.2f}. Rerun with a higher --budget to continue.")
            name = f"b{len(state['jobs']):03d}"
            path = state_dir / f"{name}_in.jsonl"
            with path.open("w") as f:
                for i in range(0, len(ids), REQUEST_INPUTS):
                    f.write(json.dumps({"custom_id": f"{name}-{i // REQUEST_INPUTS}", "method": "POST", "url": "/v1/embeddings",
                                        "body": {"model": EMBED_MODEL, "dimensions": EMBED_DIMS, "encoding_format": "base64",
                                                 "input": rows.text.iloc[i:i + REQUEST_INPUTS].tolist()}}) + "\n")
            assert path.stat().st_size < 190e6, "batch file over OpenAI's 200 MB limit; lower BATCH_MAX_INPUTS"
            file_id = client.files.create(file=path.open("rb"), purpose="batch").id
            b = client.batches.create(input_file_id=file_id, endpoint="/v1/embeddings", completion_window="24h")
            job = {"name": name, "batch_id": b.id, "file_id": file_id, "status": "submitted", "ids": ids,
                   "usd": float(usd), "submitted": time.strftime("%Y-%m-%d %H:%M")}
            state["jobs"].append(job)
            inflight.append(job)
            spent += usd
            path.unlink()
            save()
            print(f"  submitted {name}: {len(ids):,} chunks, about ${usd:.2f} ({b.id})", flush=True)
        # 3. load finished batches; chunks of failed requests in them go to the queue on the next round
        for job in finished:
            load(job)
        print(f"{time.strftime('%H:%M')} {len(inflight)} batches in flight, {len(todo):,} chunks waiting, "
              f"{col.count():,} in {collection}, about ${spent:.2f} submitted", flush=True)
        if not wait:
            return
        time.sleep(poll_minutes * 60)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--embed", action="store_true")
    p.add_argument("--budget", type=float, default=5.0, help="USD; embedding stops before starting if over it")
    p.add_argument("--batch", action="store_true", help="with --embed: use the Batch API (half price, within 24 hours)")
    p.add_argument("--wait", action="store_true", help="with --batch: keep polling until every chunk is in")
    p.add_argument("--no-rebuild", action="store_true", help="embed the existing chunks.parquet instead of rebuilding it")
    args = p.parse_args()
    chunks = pd.read_parquet(CHUNKS) if args.no_rebuild else build_chunks(args.tickers, out=SUBSET if args.tickers else CHUNKS)
    if args.embed:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (add it to .env)")
        import json

        from sec_filings_tone.budget import run_log

        state = EMBED_BATCHES / "state.json"

        def jobs():
            return json.loads(state.read_text())["jobs"] if state.exists() else []

        before = {j["batch_id"]: j["status"] for j in jobs()}

        def usage():  # --batch: what embed_batches counts as spent, for the batches this run submitted
            new = [j for j in jobs() if j["batch_id"] not in before and j["status"] in ("submitted", "completed", "loaded")]
            loaded = [j for j in jobs() if j["status"] == "loaded" and before.get(j["batch_id"]) != "loaded"]
            return {"usd": {"openai": sum(j["usd"] for j in new)}, "batches_submitted": [j["batch_id"] for j in new],
                    "rows": sum(j["loaded"] for j in loaded)} if args.batch else {}

        with run_log([EMBED_MODEL], usage, collection=COLLECTION) as rec:
            if args.batch:
                embed_batches(chunks, budget=args.budget, wait=args.wait)
            else:
                embed_chunks(chunks, budget=args.budget, rec=rec)


if __name__ == "__main__":
    main()
