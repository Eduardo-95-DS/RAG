#!/usr/bin/env python3
"""
Diagnostic: for a failing question, does the retriever return the chunk that
holds the answer?

Follows ci/inspect_chunks.py, which established that the answer-bearing chunk
EXISTS in the collection for every failing question. That leaves two candidates,
and they need different fixes:

  A. the chunk is never retrieved      -> ranking / embedding problem
  B. it is retrieved and ignored       -> prompt / attention problem

This separates them. For each question it runs the REAL retriever (same Config
width the app serves) on both the raw question and the rewriter's typical output,
then reports whether the target figure appears in the returned chunks and at what
rank.

No LLM calls for the retrieval check itself, so it is fast and free. Passing
questions are included as controls — if the target ranks poorly for them too,
rank is not what separates pass from fail.

Usage
-----
    uv run python ci/inspect_retrieval.py

Needs QDRANT_URL / QDRANT_API_KEY, and Vertex ADC for the dense query embedding.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# CrossEncoderReranker resolves its cache as: cache_dir arg > FLASHRANK_CACHE_DIR
# > /app/.flashrank_cache (baked into the container image). That last default is
# unwritable outside the container, so a bare local run dies with
# PermissionError: '/app'. The CI yamls export this explicitly; default it here
# so the diagnostic just works instead of requiring the incantation.
os.environ.setdefault(
    "FLASHRANK_CACHE_DIR", str(Path.home() / ".cache" / "flashrank")
)

from src.config.config import Config
from src.vectorstore.vectorstore import VectorStore

# (tag, expected-to-pass?, query variants, accepted value forms)
# Variants matter: the rewriter reshapes the question before retrieval, and the
# 2026-08-03 work showed phrasing changes which chunks come back. Both the raw
# question and a plausible rewrite are tried so the result isn't an artifact of
# one phrasing.
CASES = [
    ("Q6  operating cash flow", False, [
        "How much cash did NVIDIA generate from operating activities in FY2025?",
        "NVIDIA cash provided by operating activities fiscal year 2025",
    ], ["64,089", "64089"]),
    ("Q22 employee count", False, [
        "How many employees does NVIDIA have?",
        "NVIDIA total employees headcount fiscal year 2025",
    ], ["36,000", "36000"]),
    ("Q5  income tax expense", False, [
        "What was NVIDIA's income tax expense in FY2025?",
        "NVIDIA income tax expense fiscal year 2025",
    ], ["11,146", "11146"]),
    # --- controls: these pass the answer eval ---
    ("Q4  R&D expense (control)", True, [
        "How much did NVIDIA spend on research and development in FY2025?",
        "NVIDIA research and development expense fiscal year 2025",
    ], ["12,914", "12914"]),
    ("Q1  total revenue (control)", True, [
        "What was NVIDIA's total revenue in fiscal year 2025?",
        "NVIDIA total revenue fiscal year 2025",
    ], ["130,497", "130497"]),
]


def rank_of_value(docs, values: list[str]) -> int:
    """1-based rank of the first chunk containing any accepted form, else -1."""
    for i, d in enumerate(docs, start=1):
        if any(v in d.page_content for v in values):
            return i
    return -1


def main() -> int:
    if not Config.QDRANT_URL or not Config.QDRANT_API_KEY:
        print("QDRANT_URL / QDRANT_API_KEY not set (put them in .env).")
        return 2

    retriever = VectorStore().get_hybrid_retriever(
        k=Config.RETRIEVAL_K, rerank_top_k=Config.RERANK_TOP_K
    )

    print("=" * 72)
    print(f"Retriever: k={Config.RETRIEVAL_K}  rerank_top_k={Config.RERANK_TOP_K} "
          f"(same width the app serves)")
    print("=" * 72)

    retrieved_when_failing = []
    for tag, should_pass, queries, values in CASES:
        print(f"\n{tag}   [{'passes' if should_pass else 'FAILS'} the answer eval]"
              f"   looking for {values[0]!r}")
        found_any = False
        for q in queries:
            docs = retriever.invoke(q)
            rank = rank_of_value(docs, values)
            found_any |= rank > 0
            verdict = f"rank {rank} of {len(docs)}" if rank > 0 else "NOT RETRIEVED"
            print(f"     {verdict:>18}  <- {q[:62]}")
            if rank > 0:
                hit = docs[rank - 1].page_content.replace("\n", " ")
                v = next(v for v in values if v in hit)
                i = hit.find(v)
                print(f"                         ...{hit[max(0, i-70):i+50]}...")
        if not should_pass:
            retrieved_when_failing.append(found_any)

    print("\n" + "=" * 72)
    if all(retrieved_when_failing):
        print("VERDICT B — the answer chunk IS retrieved for every failing question.")
        print("Retrieval is doing its job; the model is not using what it's given.")
        print("Look at ANSWER_PROMPT and at where the chunk lands in the context")
        print("block (a figure at rank 7 of 8 sits deep in the prompt). Cheap things")
        print("to try: order chunks by rank explicitly, cut RERANK_TOP_K so the")
        print("signal isn't diluted, or state in the prompt that figures may appear")
        print("in table rows separated from their labels by whitespace.")
    elif not any(retrieved_when_failing):
        print("VERDICT A — the answer chunk is NEVER retrieved for the failing")
        print("questions, though it exists in the collection. This is a ranking or")
        print("embedding problem, not a prompt one. Compare against the controls")
        print("above: if their targets rank 1-3 and these never appear, the dense")
        print("and sparse channels are both missing a chunk they should match.")
    else:
        print("MIXED — some failing questions retrieve the chunk and some don't.")
        print("Read the per-question output; the two groups need different fixes.")
    print()
    print("Controls exist to make the comparison honest: if the passing questions")
    print("also rank their target poorly, rank is not what separates pass from fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
