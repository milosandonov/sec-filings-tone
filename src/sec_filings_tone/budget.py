"""Cost and time caps for anything that calls a paid API."""

import threading
import time

# USD per 1M input and output tokens (claude.com pricing, checked 2026-10-12)
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}


class CapReached(RuntimeError):
    pass


class Cap:
    """Stops a run before a paid call would push spending past `usd`, or once `minutes` have passed.
    Reranker spend comes from retrieve.usage; callers add anything else with charge()."""

    def __init__(self, usd: float = float("inf"), minutes: float = float("inf")):
        self.usd, self.deadline, self.other = usd, time.monotonic() + minutes * 60, 0.0
        self._lock = threading.Lock()

    def spent(self) -> float:
        from sec_filings_tone import retrieve

        u, (pin, pout) = retrieve.usage, PRICES[retrieve.RERANK_MODEL]
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
