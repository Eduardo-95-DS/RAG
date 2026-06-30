#!/usr/bin/env python3
"""
Retrieval evaluation for rag-production.

Measures retrieval quality independently of answer quality. The guardrail
node tells you when the final *answer* is bad; this script tells you whether
the *retriever* is returning the right chunks in the first place.

Metrics
-------
Hit rate (Recall@k)
    Percentage of queries where at least one retrieved chunk is relevant.
    A chunk is relevant if it contains at least one of the expected keywords
    (case-insensitive). Hit rate = 0 means the retriever missed entirely.

Context precision
    For each query: fraction of retrieved chunks that are relevant.
    Averaged across all queries. Low precision = the retriever is returning
    noise alongside the signal.

Usage
-----
    python eval/retrieval_eval.py

Run from the repo root. Requires faiss_index/ to exist (built by the main app
on first run). The cross-encoder reranker will download its model (~80 MB)
on first run if not already cached.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.vectorstore.vectorstore import VectorStore

FAISS_INDEX_PATH = "faiss_index"

# ---------------------------------------------------------------------------
# Test cases
#
# Each case has:
#   query     - the question as a user would ask it
#   keywords  - list of strings; a chunk is "relevant" if it contains ANY
#               of these (case-insensitive). Keep keywords specific enough
#               to identify a genuinely relevant chunk, not just any chunk
#               that mentions the broad topic.
#   note      - short label for the report
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- Financial overview (income statement) ---
    {
        "query": "What was NVIDIA's total revenue in fiscal year 2025?",
        "keywords": ["130,497", "130.5", "total revenue", "fiscal 2025"],
        "note": "Total FY2025 revenue",
    },
    {
        "query": "What was NVIDIA's net income in FY2025?",
        "keywords": ["net income", "72,880", "net earnings"],
        "note": "Net income FY2025",
    },
    {
        "query": "What were NVIDIA's earnings per share in FY2025?",
        "keywords": ["earnings per share", "diluted", "2.94", "per share"],
        "note": "EPS FY2025",
    },
    {
        "query": "What were NVIDIA's sales, general and administrative expenses in FY2025?",
        "keywords": ["sales, general and administrative", "3,491", "sg&a", "2.7"],
        "note": "SG&A expenses",
    },
    {
        "query": "What was NVIDIA's income tax expense in FY2025?",
        "keywords": ["11,146", "income tax", "13.3", "effective tax rate"],
        "note": "Income tax expense",
    },

    # --- Cash flow and balance sheet ---
    {
        "query": "How much cash did NVIDIA generate from operating activities in FY2025?",
        "keywords": ["64,089", "operating activities", "cash provided by operating"],
        "note": "Operating cash flow",
    },
    {
        "query": "What were NVIDIA's total cash, cash equivalents, and marketable securities at end of FY2025?",
        "keywords": ["43,210", "8,589", "34,621", "cash and cash equivalents", "marketable securities"],
        "note": "Cash and marketable securities",
    },

    # --- Business segments ---
    {
        "query": "What was NVIDIA's data center segment revenue in FY2025?",
        "keywords": ["data center", "115,186", "115.2", "compute and networking"],
        "note": "Data center segment revenue",
    },
    {
        "query": "What was NVIDIA's gaming revenue in FY2025?",
        "keywords": ["gaming", "11,446", "11.4", "geforce"],
        "note": "Gaming segment revenue",
    },
    {
        "query": "What was NVIDIA's professional visualization revenue?",
        "keywords": ["professional visualization", "pro viz", "1,591", "workstation"],
        "note": "Professional visualization revenue",
    },
    {
        "query": "What was NVIDIA's automotive segment revenue in FY2025?",
        "keywords": ["automotive", "1,695", "self-driving", "orin"],
        "note": "Automotive segment revenue",
    },
    {
        "query": "What was NVIDIA's Compute and Networking segment operating income in FY2025?",
        "keywords": ["82,875", "compute and networking", "segment operating income", "compute & networking"],
        "note": "Compute & Networking operating income",
    },

    # --- Products and architecture ---
    {
        "query": "What is the Blackwell GPU architecture?",
        "keywords": ["blackwell", "b100", "b200", "gb200"],
        "note": "Blackwell architecture",
    },
    {
        "query": "What products use the Hopper architecture?",
        "keywords": ["hopper", "h100", "h200"],
        "note": "Hopper architecture",
    },
    {
        "query": "What is NVLink and how does it work?",
        "keywords": ["nvlink", "nvswitch", "interconnect", "gpu-to-gpu"],
        "note": "NVLink / NVSwitch",
    },
    {
        "query": "What is CUDA and why is it important to NVIDIA?",
        "keywords": ["cuda", "parallel computing", "developer", "software platform"],
        "note": "CUDA platform",
    },
    {
        "query": "What automotive products does NVIDIA offer?",
        "keywords": ["drive", "orin", "jetson", "autonomous vehicle"],
        "note": "Automotive products",
    },

    # --- Strategy and competition ---
    {
        "query": "What is NVIDIA's strategy for accelerated computing?",
        "keywords": ["accelerated computing", "full-stack", "platform", "data center"],
        "note": "Accelerated computing strategy",
    },
    {
        "query": "What are the main risks NVIDIA faces from competition?",
        "keywords": ["competition", "competitive", "amd", "intel", "rival"],
        "note": "Competitive risks",
    },
    {
        "query": "What export controls affect NVIDIA's China business?",
        "keywords": ["china", "export", "license", "restrictions", "entity list"],
        "note": "Export controls / China",
    },

    # --- Supply chain and operations ---
    {
        "query": "Who manufactures NVIDIA chips?",
        "keywords": ["tsmc", "taiwan semiconductor", "foundry", "fabrication"],
        "note": "Chip manufacturer / TSMC",
    },
    {
        "query": "How many employees does NVIDIA have?",
        "keywords": ["employees", "headcount", "29,600", "workforce"],
        "note": "Employee count",
    },

    # --- Capital allocation ---
    {
        "query": "How much did NVIDIA return to shareholders in FY2025?",
        "keywords": ["repurchase", "34,000", "34.0", "834", "dividends", "shareholder"],
        "note": "Capital return / buybacks",
    },
    {
        "query": "What is NVIDIA's R&D spending?",
        "keywords": ["research and development", "r&d", "12,914"],
        "note": "R&D expenditure",
    },

    # --- Geographic revenue ---
    {
        "query": "What percentage of NVIDIA revenue comes from outside the United States?",
        "keywords": ["united states", "international", "geographic", "taiwan", "singapore"],
        "note": "Geographic revenue breakdown",
    },
]


def chunk_is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    """Return True if chunk_text contains at least one keyword (case-insensitive)."""
    text = chunk_text.lower()
    return any(kw.lower() in text for kw in keywords)


def run_eval():
    print("NVIDIA RAG — Retrieval Evaluation")
    print("=" * 60)

    # Load index
    vs = VectorStore()
    vs.load(FAISS_INDEX_PATH)
    retriever = vs.get_hybrid_retriever(k=8, rerank_top_k=5)

    print(f"Index : {FAISS_INDEX_PATH}")
    print(f"Retriever : HybridRetriever  RRF k=8  rerank top_k=5")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    col_w = 46  # query column width
    header = f"{'#':>2}  {'Query':<{col_w}}  {'Hit':^3}  {'Precision':^12}  Note"
    print(header)
    print("-" * len(header))

    hits = 0
    precisions = []

    for i, case in enumerate(TEST_CASES, start=1):
        docs = retriever.invoke(case["query"])

        relevant = [chunk_is_relevant(d.page_content, case["keywords"]) for d in docs]
        hit = any(relevant)
        precision = sum(relevant) / len(docs) if docs else 0.0

        hits += int(hit)
        precisions.append(precision)

        hit_sym = "✓" if hit else "✗"
        prec_str = f"{sum(relevant)}/{len(docs)} ({precision:.0%})"
        query_short = case["query"][:col_w]
        print(f"{i:>2}  {query_short:<{col_w}}  {hit_sym:^3}  {prec_str:^12}  {case['note']}")

    n = len(TEST_CASES)
    mean_precision = sum(precisions) / n

    print()
    print("=" * 60)
    print(f"Hit rate (Recall@5) : {hits}/{n}  ({hits/n:.0%})")
    print(f"Mean context precision: {mean_precision:.0%}")
    print()
    misses = [TEST_CASES[i]["note"] for i, p in enumerate(precisions) if p == 0.0]
    low_prec_hits = [
        TEST_CASES[i]["note"]
        for i, p in enumerate(precisions)
        if 0.0 < p < 0.2
    ]

    print("Interpretation")
    print("-" * 60)
    if misses:
        print(f"  Misses ({len(misses)}): retriever returned no relevant chunk.")
        for m in misses:
            print(f"    - {m}")
        print("  Investigate: chunk missing from index, or reranker dropped it.")
    else:
        print("  No misses. Retriever found at least one relevant chunk for every query.")

    if low_prec_hits:
        print()
        print(f"  Low-precision hits ({len(low_prec_hits)}, < 20%): relevant chunk found")
        print("  but most returned chunks are noise. Consider raising rerank_top_k.")
        for lp in low_prec_hits:
            print(f"    - {lp}")


if __name__ == "__main__":
    run_eval()
