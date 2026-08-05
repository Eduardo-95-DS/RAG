#!/usr/bin/env python3
"""
Diagnostic: does a chunk containing a financial row LABEL also contain its VALUE?

Tests the known_issues.md item-4 hypothesis directly instead of inferring it.
The Vertex judge repeatedly observed things like "the passages do contain a
'Consolidated Statements of Cash Flows' table that shows the line item 'Cash
provided by operating activities'... while the specific dollar amount is not
visible in the excerpt provided" — which implies CHUNK_SIZE=500 is splitting
row labels away from their numbers. That is an inference. This script checks it.

Reads the live Qdrant collection, so it measures the corpus the app actually
serves. No LLM calls, no cost.

FAILING questions and their expected figures are checked alongside PASSING ones
as controls. That matters: if the passing questions ALSO show label/value
severance, the hypothesis is wrong and the difference lies elsewhere.

Usage
-----
    uv run python ci/inspect_chunks.py

Needs QDRANT_URL and QDRANT_API_KEY (from .env locally).
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from src.config.config import Config

# (question tag, row label as it appears in the statement, accepted value forms)
# Grouped by whether the eval question currently passes, so the comparison is
# built into the output rather than left to the reader.
FAILING = [
    ("Q6  operating cash flow", "provided by operating activities", ["64,089", "64089"]),
    ("Q5  income tax expense", "income tax expense", ["11,146", "11146"]),
    ("Q22 employee count", "employees", ["36,000", "36000"]),
]

PASSING = [
    ("Q1  total revenue", "revenue", ["130,497", "130497"]),
    ("Q2  net income", "net income", ["72,880", "72880"]),
    ("Q4  R&D expense", "research and development", ["12,914", "12914"]),
]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace — PDF extraction spaces cells oddly."""
    return re.sub(r"\s+", " ", (text or "").lower())


def fetch_all_chunks(client: QdrantClient) -> list[str]:
    """Scroll the whole collection. ~1,574 chunks, so paging it all is fine."""
    texts, offset = [], None
    while True:
        points, offset = client.scroll(
            collection_name=Config.QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        texts.extend((p.payload or {}).get("text", "") for p in points)
        if offset is None:
            break
    return texts


def check(tag: str, label: str, values: list[str], chunks: list[str]) -> bool:
    """Report where the label lives, where the value lives, and whether they meet."""
    label_n = normalize(label)
    with_label = [c for c in chunks if label_n in normalize(c)]
    with_value = [c for c in chunks if any(v in c for v in values)]
    with_both = [c for c in with_label if any(v in c for v in values)]

    ok = bool(with_both)
    print(f"\n{'✅' if ok else '❌'} {tag}")
    print(f"     label  '{label}'  -> {len(with_label)} chunk(s)")
    print(f"     value  {values[0]!r}      -> {len(with_value)} chunk(s)")
    print(f"     BOTH in the same chunk  -> {len(with_both)} chunk(s)")

    if not ok and with_label:
        # The interesting case: show where the label sits and what's near it.
        sample = normalize(with_label[0])
        idx = sample.find(label_n)
        window = sample[max(0, idx - 90): idx + 160]
        print(f"     label chunk reads: ...{window}...")
    if not ok and with_value:
        sample = next(c for c in with_value)
        v = next(v for v in values if v in sample)
        idx = normalize(sample).find(v)
        window = normalize(sample)[max(0, idx - 120): idx + 90]
        print(f"     value chunk reads: ...{window}...")
    return ok


def main() -> int:
    if not Config.QDRANT_URL or not Config.QDRANT_API_KEY:
        print("QDRANT_URL / QDRANT_API_KEY not set (put them in .env).")
        return 2

    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
    chunks = fetch_all_chunks(client)

    print("=" * 66)
    print(f"Collection '{Config.QDRANT_COLLECTION}': {len(chunks)} chunks")
    print(f"CHUNK_SIZE={Config.CHUNK_SIZE}  CHUNK_OVERLAP={Config.CHUNK_OVERLAP}")
    print("=" * 66)

    print("\n--- questions that FAIL the answer eval ---")
    fail_results = [check(*case, chunks) for case in FAILING]

    print("\n--- questions that PASS (controls) ---")
    pass_results = [check(*case, chunks) for case in PASSING]

    print("\n" + "=" * 66)
    intact_failing = sum(fail_results)
    intact_passing = sum(pass_results)
    print(f"label+value in one chunk — failing questions: "
          f"{intact_failing}/{len(fail_results)}")
    print(f"label+value in one chunk — passing questions: "
          f"{intact_passing}/{len(pass_results)}")
    print()

    if intact_failing == 0 and intact_passing == len(pass_results):
        print("HYPOTHESIS CONFIRMED: every failing question has its row label and its")
        print("value in different chunks, while every passing one has them together.")
        print("Fix is chunking (raise CHUNK_SIZE or use a table-aware splitter),")
        print("then re-ingest with ci/ingest_qdrant.py --wipe.")
    elif intact_failing == len(fail_results):
        print("HYPOTHESIS REJECTED: the failing questions DO have label and value in")
        print("the same chunk. The corpus is fine and the problem is downstream —")
        print("look at retrieval ranking or the answer prompt instead.")
    else:
        print("MIXED RESULT. Chunking explains some failures but not all; read the")
        print("per-question output above before committing to a fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
