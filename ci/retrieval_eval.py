#!/usr/bin/env python3
"""
Retrieval evaluation gate for Cloud Build (manual trigger).

This is the CI counterpart to eval/retrieval_eval.py (which stays local-only,
see .gitignore). Same test cases, same metrics. The difference: this script
exits non-zero when hit rate falls below a threshold, so a Cloud Build step
can use it as a pass/fail gate instead of just printing a report.

Usage
-----
    python ci/retrieval_eval.py --fail-under-hit-rate=0.90

Requires faiss_index/ to exist (pulled from GCS by the calling Cloud Build
step — see cloudbuild-eval.yaml). Not wired to any push trigger: run it
manually via `gcloud builds triggers run rag-gcp-eval-manual`.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.vectorstore.vectorstore import VectorStore
from src.config.config import Config

# (item 9) retrieval is served by Qdrant Cloud; no local FAISS index path.

# ---------------------------------------------------------------------------
# Test cases — kept in sync with eval/retrieval_eval.py by hand. If you
# change one, change both (see known tradeoff: this duplicates rather than
# shares the list, since eval/ is intentionally local-only and this file is
# intentionally repo-tracked).
# ---------------------------------------------------------------------------
TEST_CASES = [
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


def run_eval(fail_under_hit_rate: float) -> int:
    """Run the eval and return a process exit code (0 = pass, 1 = fail)."""
    print("NVIDIA RAG — Retrieval Evaluation (CI gate)")
    print("=" * 60)

    # Item 9: retrieval is served by Qdrant Cloud — no local index to load.
    vs = VectorStore()
    retriever = vs.get_hybrid_retriever(k=8, rerank_top_k=5)

    print(f"Index : Qdrant collection '{Config.QDRANT_COLLECTION}'")
    print(f"Retriever : HybridRetriever (Qdrant dense+sparse RRF)  k=8  rerank top_k=5")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Gate : hit rate must be >= {fail_under_hit_rate:.0%}")
    print()

    hits = 0
    precisions = []
    misses = []

    for case in TEST_CASES:
        docs = retriever.invoke(case["query"])
        relevant = [chunk_is_relevant(d.page_content, case["keywords"]) for d in docs]
        hit = any(relevant)
        precision = sum(relevant) / len(docs) if docs else 0.0

        hits += int(hit)
        precisions.append(precision)
        if not hit:
            misses.append(case["note"])

    n = len(TEST_CASES)
    hit_rate = hits / n
    mean_precision = sum(precisions) / n

    print(f"Hit rate (Recall@5)   : {hits}/{n}  ({hit_rate:.0%})")
    print(f"Mean context precision: {mean_precision:.0%}")

    if misses:
        print()
        print(f"Misses ({len(misses)}):")
        for m in misses:
            print(f"  - {m}")

    print()
    print("=" * 60)

    if hit_rate < fail_under_hit_rate:
        print(f"FAIL: hit rate {hit_rate:.0%} is below gate of {fail_under_hit_rate:.0%}")
        return 1

    print(f"PASS: hit rate {hit_rate:.0%} meets gate of {fail_under_hit_rate:.0%}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-under-hit-rate",
        type=float,
        default=0.90,
        help="Exit 1 if hit rate falls below this fraction (default: 0.90)",
    )
    args = parser.parse_args()
    sys.exit(run_eval(args.fail_under_hit_rate))
