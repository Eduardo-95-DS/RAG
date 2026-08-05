#!/usr/bin/env python3
"""
Diagnostic: do the unreachable chunks actually have vectors?

Fourth and (hopefully) last in the sequence:

  inspect_chunks.py     -> the answer chunk EXISTS in the collection
  inspect_retrieval.py  -> it is NOT RETRIEVED for Q6 / Q22
  inspect_fusion.py     -> it is absent from the top 50 on BOTH channels,
                           while a control ranks 1 everywhere
  this script           -> is it indexed at all?

`scroll` returns points by ID and does not touch the vector index, so a point
whose payload was written but whose vectors are missing, zero-length or stored
under the wrong name would look present to inspect_chunks.py and be invisible
to every vector query. That matches the evidence exactly: near-verbatim lexical
overlap with the query, yet absent from a 50-deep sparse search.

Compares the unreachable points against a control point that ranks 1, so
"healthy" is defined by example rather than by assumption.

Usage
-----
    uv run python ci/inspect_vectors.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from src.config.config import Config

TARGETS = [
    ("Q6  operating cash flow  (unreachable)", ["64,089", "64089"]),
    ("Q22 employee count       (unreachable)", ["36,000", "36000"]),
    ("Q4  R&D expense          (CONTROL, ranks 1)", ["12,914", "12914"]),
]


def describe_vectors(vec) -> str:
    """Qdrant returns a dict of named vectors; report shape and health of each."""
    if vec is None:
        return "NO VECTORS AT ALL (point has payload only)"
    if not isinstance(vec, dict):
        return f"unnamed vector, len={len(vec)}"

    parts = []
    for name in ("dense", "sparse"):
        if name not in vec:
            parts.append(f"{name}=MISSING")
            continue
        v = vec[name]
        if name == "dense":
            n = len(v) if v is not None else 0
            allzero = bool(v) and not any(v)
            flag = ""
            if n != Config.DENSE_DIM:
                flag = f" ⚠️ expected {Config.DENSE_DIM}"
            if allzero:
                flag += " ⚠️ ALL ZERO"
            parts.append(f"dense=dim {n}{flag}")
        else:
            idx = getattr(v, "indices", None)
            n = len(idx) if idx is not None else 0
            parts.append(f"sparse={n} nonzero terms" + (" ⚠️ EMPTY" if n == 0 else ""))
    extra = [k for k in vec if k not in ("dense", "sparse")]
    if extra:
        parts.append(f"unexpected named vectors: {extra}")
    return "  ".join(parts)


def main() -> int:
    if not Config.QDRANT_URL or not Config.QDRANT_API_KEY:
        print("QDRANT_URL / QDRANT_API_KEY not set (put them in .env).")
        return 2

    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)

    info = client.get_collection(Config.QDRANT_COLLECTION)
    print("=" * 74)
    print(f"Collection '{Config.QDRANT_COLLECTION}'")
    print(f"  points={info.points_count}  indexed_vectors={info.indexed_vectors_count}")
    if info.points_count and info.indexed_vectors_count is not None:
        if info.indexed_vectors_count < info.points_count:
            print(f"  ⚠️ {info.points_count - info.indexed_vectors_count} point(s) "
                  f"are NOT in the vector index")
    print("=" * 74)

    # Scroll everything with vectors attached; 1,574 points is small.
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=Config.QDRANT_COLLECTION, limit=256, offset=offset,
            with_payload=True, with_vectors=True,
        )
        points.extend(batch)
        if offset is None:
            break

    for tag, values in TARGETS:
        matches = [p for p in points
                   if any(v in (p.payload or {}).get("text", "") for v in values)]
        print(f"\n{tag}   {values[0]!r} -> {len(matches)} point(s)")
        for p in matches:
            text = " ".join((p.payload or {}).get("text", "").split())
            i = max(text.find(values[0]), 0)
            print(f"  id={str(p.id)[:12]}  {describe_vectors(p.vector)}")
            print(f"     text: ...{text[max(0, i-55):i+45]}...")

    print("\n" + "=" * 74)
    print("Read the CONTROL line first — that is what a healthy, retrievable point")
    print("looks like in this collection. If the unreachable points differ (missing")
    print("a named vector, zero-length dense, empty sparse), ingestion is at fault")
    print("and a `ci/ingest_qdrant.py --wipe` re-ingest is the fix. If they look")
    print("IDENTICAL to the control, the points are indexed fine and the problem is")
    print("genuinely in how the query matches them — back to embeddings, not plumbing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
