#!/usr/bin/env python3
"""
Diagnostic: is the answer chunk lost by Qdrant's fusion, or by the reranker?

Third in the sequence. ci/inspect_chunks.py established the answer-bearing chunk
EXISTS; ci/inspect_retrieval.py established it is NOT RETRIEVED for Q6 and Q22.
The retrieval pipeline has two stages that can drop it:

    dense + sparse prefetch -> server-side RRF -> RETRIEVAL_K (16) candidates
                            -> FlashRank rerank -> RERANK_TOP_K (8) returned

So either the chunk never makes the 16 (embedding / query-matching problem —
fix at ingestion or in query construction), or it makes the 16 and FlashRank
demotes it (reranker problem — fix by raising RERANK_TOP_K, changing the
reranker, or dropping it for tabular content). Completely different fixes.

This bypasses HybridRetriever and issues the same Qdrant query it does, then
reports the chunk's rank in the RAW fused candidates. It also reports each
channel separately (dense-only, sparse-only) so a miss can be attributed.

No LLM calls. One Vertex embedding call per query.

Usage
-----
    uv run python ci/inspect_fusion.py
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

os.environ.setdefault(
    "FLASHRANK_CACHE_DIR", str(Path.home() / ".cache" / "flashrank")
)

from qdrant_client import QdrantClient, models

from src.config.config import Config
from src.vectorstore.vectorstore import VectorStore

CASES = [
    ("Q6  operating cash flow", [
        "How much cash did NVIDIA generate from operating activities in FY2025?",
        "NVIDIA cash provided by operating activities fiscal year 2025",
    ], ["64,089", "64089"]),
    ("Q22 employee count", [
        "How many employees does NVIDIA have?",
        "NVIDIA total employees headcount fiscal year 2025",
    ], ["36,000", "36000"]),
    ("Q4  R&D expense (control)", [
        "How much did NVIDIA spend on research and development in FY2025?",
    ], ["12,914", "12914"]),
]

# How deep to look past RETRIEVAL_K. If the chunk sits at rank 20 of a 50-deep
# fetch, raising RETRIEVAL_K is a plausible fix; if it's absent from 50, it is not.
DEEP_LIMIT = 50


def rank_in(points, values: list[str]) -> int:
    for i, p in enumerate(points, start=1):
        if any(v in (p.payload or {}).get("text", "") for v in values):
            return i
    return -1


def fmt(rank: int, limit: int) -> str:
    return f"rank {rank}/{limit}" if rank > 0 else f"absent from {limit}"


def main() -> int:
    if not Config.QDRANT_URL or not Config.QDRANT_API_KEY:
        print("QDRANT_URL / QDRANT_API_KEY not set (put them in .env).")
        return 2

    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)

    # Build queries through the REAL retriever rather than reimplementing them.
    # An earlier version of this script duplicated the sparse-embedding call,
    # which meant it would have faithfully reproduced a bug in that call instead
    # of exposing it. Diagnostics that reimplement the thing they measure can
    # only ever agree with it.
    retriever = VectorStore().get_hybrid_retriever(
        k=Config.RETRIEVAL_K, rerank_top_k=Config.RERANK_TOP_K
    )
    dense = retriever.dense
    sparse_vec = retriever._sparse_query

    print("=" * 74)
    print(f"RETRIEVAL_K={Config.RETRIEVAL_K} (fused candidates)   "
          f"RERANK_TOP_K={Config.RERANK_TOP_K} (returned)   deep probe={DEEP_LIMIT}")
    print("=" * 74)

    verdicts = []
    for tag, queries, values in CASES:
        print(f"\n{tag}   looking for {values[0]!r}")
        for q in queries:
            dv, sv = dense.embed_query(q), sparse_vec(q)

            fused = client.query_points(
                collection_name=Config.QDRANT_COLLECTION,
                prefetch=[
                    models.Prefetch(query=dv, using="dense", limit=Config.RETRIEVAL_K),
                    models.Prefetch(query=sv, using="sparse", limit=Config.RETRIEVAL_K),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=Config.RETRIEVAL_K, with_payload=True,
            ).points

            deep = client.query_points(
                collection_name=Config.QDRANT_COLLECTION,
                prefetch=[
                    models.Prefetch(query=dv, using="dense", limit=DEEP_LIMIT),
                    models.Prefetch(query=sv, using="sparse", limit=DEEP_LIMIT),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=DEEP_LIMIT, with_payload=True,
            ).points

            d_only = client.query_points(
                collection_name=Config.QDRANT_COLLECTION, query=dv,
                using="dense", limit=DEEP_LIMIT, with_payload=True).points
            s_only = client.query_points(
                collection_name=Config.QDRANT_COLLECTION, query=sv,
                using="sparse", limit=DEEP_LIMIT, with_payload=True).points

            r_fused = rank_in(fused, values)
            r_deep = rank_in(deep, values)
            r_dense = rank_in(d_only, values)
            r_sparse = rank_in(s_only, values)

            print(f"  {q[:66]}")
            print(f"     fused@{Config.RETRIEVAL_K:<3} {fmt(r_fused, Config.RETRIEVAL_K):<18}"
                  f" fused@{DEEP_LIMIT} {fmt(r_deep, DEEP_LIMIT)}")
            print(f"     dense-only  {fmt(r_dense, DEEP_LIMIT):<18}"
                  f" sparse-only {fmt(r_sparse, DEEP_LIMIT)}")
            if "control" not in tag:
                verdicts.append((tag, r_fused, r_deep, r_dense, r_sparse))

    print("\n" + "=" * 74)
    in_fused = [v for v in verdicts if v[1] > 0]
    deep_only = [v for v in verdicts if v[1] <= 0 < v[2]]
    nowhere = [v for v in verdicts if v[2] <= 0]

    if in_fused:
        print(f"RERANKER IS THE PROBLEM for {len(in_fused)} case(s): the chunk is in the")
        print(f"fused {Config.RETRIEVAL_K} but FlashRank demotes it out of the top "
              f"{Config.RERANK_TOP_K}.")
        print("Fix candidates: raise RERANK_TOP_K, or bypass/replace the reranker for")
        print("number-dense chunks — a cross-encoder trained on prose scores a wall of")
        print("digits poorly regardless of relevance.")
    if deep_only:
        print(f"\nDEPTH IS THE PROBLEM for {len(deep_only)} case(s): absent from the fused")
        print(f"{Config.RETRIEVAL_K} but present within {DEEP_LIMIT}. Raising RETRIEVAL_K")
        print("would surface it — cheap, since prefetch depth costs no LLM calls.")
    if nowhere:
        print(f"\nMATCHING IS THE PROBLEM for {len(nowhere)} case(s): absent even from "
              f"{DEEP_LIMIT}.")
        print("Neither channel ranks it, so no retrieval-side tuning reaches it. Check")
        print("the dense-only vs sparse-only lines above to see if either is close, then")
        print("look at ingestion: number-dense chunks carry little semantic signal for")
        print("the dense channel and may tokenize badly for BM42.")
    print("\nCompare every line against the control, which should rank near the top")
    print("everywhere. Where the control also ranks poorly, the metric isn't the story.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
