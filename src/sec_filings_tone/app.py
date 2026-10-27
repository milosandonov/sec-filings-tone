"""Phase 10: a local chat window over answer.answer(), the same function the CLI uses.

Usage:
    uv run python -m sec_filings_tone.app               # http://127.0.0.1:7860, $0.50 per chat session
    uv run python -m sec_filings_tone.app --budget 0.20 --effort medium

Local only: no share link, since anyone with the link could spend the API key.
"""

import argparse

import gradio as gr
from dotenv import load_dotenv

from sec_filings_tone import answer
from sec_filings_tone.budget import Cap, run_log
from sec_filings_tone.config import ANSWER_MODEL, EMBED_MODEL, RERANK_MODEL

EXAMPLES = [
    "How did NVIDIA describe the financial impact of the US license requirement for H20 exports to China, and when did it learn of it?",
    "Compare how NVIDIA and AMD described China export-control risk in fiscal 2022 versus fiscal 2024.",
    "How did the large regional banks describe deposit outflows and funding costs in their first 10-Qs after the SVB failure in March 2023?",
    "Which Energy companies reported a direct financial hit from Russia's invasion of Ukraine (impairments, exits from projects), and how large was it?",
    "Which sectors' 10-Ks treated tariffs as an effect that already happened in FY2025, rather than a hypothetical risk, and how does that compare with FY2023?",
    "Which Health Care companies added AI as a risk factor after 2022, and what specifically are they worried about?",
    "How did REITs describe the effect of higher interest rates on their financing costs and property values in FY2023?",
    "Which companies disclosed an actual cybersecurity incident, not just a hypothetical one, in their FY2024 10-K?",
    "What were Apple's main supply-chain risks in FY2023?",
    "What risk factors did JPMorgan add to its FY2023 10-K compared with FY2022?",
    "Will NVIDIA's stock outperform next quarter?",
]

HEADER = (f"Ask about the 10-K risk factors and MD&A (FY2019–FY2025) and 10-Q MD&A of the current S&P 500 companies. "
          f"Answers come from the filings only, cite the passages they use, and show the precomputed tone profile. "
          f"Model: {ANSWER_MODEL}. Follow-up questions use the conversation so far.")


def text(content) -> str:
    """Gradio 6 passes a message's content as a list of parts; the text parts, joined."""
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content if isinstance(p, dict))


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.50, help="USD per chat session")
    ap.add_argument("--model", default=ANSWER_MODEL)
    ap.add_argument("--effort", default=answer.EFFORT, choices=["low", "medium", "high"])
    a = ap.parse_args()
    # browser session -> its cap. Reranking spend is counted process-wide (budget.rerank_usage), so with
    # several sessions open each cap also counts the others' reranking: a cap stops early, never late.
    caps, asked = {}, []

    def chat(message: str, history: list[dict], request: gr.Request):
        cap = caps.setdefault(request.session_hash, Cap(a.budget))
        asked.append(request.session_hash)
        history = [{"role": m["role"], "content": text(m["content"])} for m in history]
        yield from answer.answer(message, history, cap, a.model, a.effort)

    def usage():  # every session's own calls, plus the process-wide reranking once (a fresh Cap holds only that)
        return {"usd": {"anthropic": sum(c.other for c in caps.values()) + Cap().spent()}, "sessions": len(caps),
                "rows": len(asked)}

    print("loading chunks, the BM25 index and the tone cells ...", flush=True)
    answer.warm()
    # one record per app run, written when the server stops (Ctrl-C): Gradio has no reliable end-of-session event
    with run_log([a.model, RERANK_MODEL, EMBED_MODEL], usage, effort=a.effort, budget_per_session=a.budget):
        gr.ChatInterface(chat, title="SEC Filings Tone: ask the filings", description=HEADER, examples=EXAMPLES,
                         cache_examples=False, analytics_enabled=False).launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
