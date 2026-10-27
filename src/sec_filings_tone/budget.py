"""Cost and time caps for anything that calls a paid API, and the run log (data/runs.jsonl) every paid
command appends one line to."""

import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sec_filings_tone.config import PRICES, RERANK_MODEL, ROOT, RUNS

rerank_usage = {"input_tokens": 0, "output_tokens": 0}  # summed over retrieve.llm_rerank calls, for cost caps
rerank_lock = threading.Lock()


class CapReached(RuntimeError):
    pass


class Cap:
    """Stops a run before a paid call would push spending past `usd`, or once `minutes` have passed.
    Reranker spend comes from rerank_usage; callers add anything else with charge()."""

    def __init__(self, usd: float = float("inf"), minutes: float = float("inf")):
        self.usd, self.deadline, self.other = usd, time.monotonic() + minutes * 60, 0.0
        self._lock = threading.Lock()

    def spent(self) -> float:
        u, (pin, pout) = rerank_usage, PRICES[RERANK_MODEL]
        return self.other + (u["input_tokens"] * pin + u["output_tokens"] * pout) / 1e6

    def charge(self, usd: float) -> None:
        with self._lock:
            self.other += usd

    def check(self, next_usd: float = 0.0) -> None:
        if self.spent() + next_usd > self.usd:
            raise CapReached(f"cost cap: ${self.spent():.2f} spent, the next call (up to ${next_usd:.3f}) "
                             f"would pass ${self.usd:.2f}")
        if time.monotonic() > self.deadline:
            raise CapReached(f"time cap reached, ${self.spent():.2f} spent")


def call_usd(model: str, input_tokens: float, output_tokens: float) -> float:
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1e6


def rerank_call_usd(n_input_tokens: float) -> float:
    """Upper bound for one rerank call: its input plus the 8,000-token output limit."""
    return call_usd(RERANK_MODEL, n_input_tokens, 8000)


def git_state() -> tuple[str | None, bool | None]:
    """HEAD's SHA, and whether tracked files differ from it (what export's `git describe --dirty` calls dirty)."""
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return (sha, subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode != 0) if sha else (None, None)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def run_log(models: list[str], usage=dict, outputs=(), path: Path | None = None, **fields):
    """Appends one JSON line for a paid run to `path` (default RUNS) when the block ends, also when it raises:
    UTC start and end, the command, the git commit, `models`, `fields` (e.g. prompt hashes), what `usage()` returns at
    the end (USD per provider from the run's own accounting, counts, batch ids), the `outputs` changed during
    the run, and the status: ok, stopped (a cap, or SystemExit with a message), interrupted or failed.
    The block may set keys on the yielded dict; they override usage()'s ("status" only if nothing raised)."""
    main = sys.modules["__main__"]
    spec = getattr(main, "__spec__", None)  # python -m: the module; a script: its path
    sha, dirty = git_state()
    rec, mine, start = {"started": now(), "command": [spec.name if spec else sys.argv[0], *sys.argv[1:]],
                        "commit": sha, "dirty": dirty, "models": list(dict.fromkeys(models)), **fields}, {}, time.time()
    status, error = "failed", None
    try:
        yield mine
        status = mine.pop("status", "ok")
    except (CapReached, SystemExit) as e:
        status = "ok" if isinstance(e, SystemExit) and e.code in (None, 0) else "stopped"
        error = str(e) if status == "stopped" else None
        raise
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        mine.pop("status", None)
        try:
            used = usage()
        except Exception as e:  # the record still goes out
            used = {"usage_error": f"{type(e).__name__}: {e}"}
        rec |= {"finished": now(), "status": status, "error": error, **used, **mine,
                "outputs": [str(p) for p in map(Path, outputs) if p.exists() and p.stat().st_mtime >= start]}
        path = Path(path or RUNS)
        with path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"run record: {path}", flush=True)
