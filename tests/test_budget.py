"""Run: uv run pytest tests/test_budget.py  (offline: no API calls, no data files)"""

import json
import sys

import pytest

from sec_filings_tone import budget
from sec_filings_tone.budget import CapReached, run_log


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "git_state", lambda: ("abc123", True))
    monkeypatch.setattr("sys.argv", ["tone.py", "score", "--budget", "0.2"])
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", None, raising=False)  # as when run as a script
    monkeypatch.setattr(budget, "RUNS", tmp_path / "runs.jsonl")  # never the real data/runs.jsonl
    return budget.RUNS


def records(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_record_of_a_finished_run(log, tmp_path):
    out, untouched = tmp_path / "scores.parquet", tmp_path / "old.parquet"
    untouched.write_text("written before the run")
    spent = {"usd": 0.0}
    with run_log(["claude-haiku-5-5"], lambda: {"usd": {"anthropic": spent["usd"]}, "rows": 3}, [out, untouched],
                 scorer={"prompt_key": "6f6deef21feb"}) as rec:
        spent["usd"] = 0.0123  # usage() is read when the block ends, not when it starts
        out.write_text("x")
        rec["batches_collected"] = ["msgbatch_1"]
    [r] = records(log)
    assert r["status"] == "ok" and r["error"] is None
    assert r["command"] == ["tone.py", "score", "--budget", "0.2"] and r["commit"] == "abc123" and r["dirty"] is True
    assert r["models"] == ["claude-haiku-5-5"] and r["scorer"] == {"prompt_key": "6f6deef21feb"}
    assert r["usd"] == {"anthropic": 0.0123} and r["rows"] == 3 and r["batches_collected"] == ["msgbatch_1"]
    assert r["outputs"] == [str(out)]  # only files changed during the run
    assert r["started"].endswith("+00:00") and r["finished"] >= r["started"]


@pytest.mark.parametrize("exc, status", [(RuntimeError("API down"), "failed"), (CapReached("cost cap"), "stopped"),
                                         (SystemExit("over --budget $0.00; nothing submitted"), "stopped"),
                                         (KeyboardInterrupt(), "interrupted"), (SystemExit(0), "ok")])
def test_a_run_that_raises_is_still_recorded(log, exc, status):
    with pytest.raises(type(exc)):
        with run_log(["claude-haiku-5-5"], lambda: {"usd": {"anthropic": 0.0}}):
            raise exc
    with run_log(["m"]) as rec:  # a second run appends; the block can mark it stopped itself
        rec["status"] = "stopped"
    first, second = records(log)
    assert first["status"] == status and first["usd"] == {"anthropic": 0.0}
    if status in ("failed", "stopped"):
        assert str(exc) in first["error"]
    assert second["status"] == "stopped"


def test_a_broken_usage_does_not_lose_the_record(log):
    with pytest.raises(ValueError):
        with run_log(["m"], lambda: 1 / 0):
            raise ValueError("boom")
    [r] = records(log)
    assert r["status"] == "failed" and r["error"] == "ValueError: boom" and "ZeroDivisionError" in r["usage_error"]
