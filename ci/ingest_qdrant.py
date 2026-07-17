#!/usr/bin/env python3
"""
One-off ingestion into Qdrant Cloud (item 9).

Chunks the NVIDIA annual-report PDF (from GCS, via the existing
DocumentProcessor), dense-embeds each chunk with Vertex text-embedding-005,
sparse-embeds it with BM42 (fastembed), and upserts the points into the Qdrant
collection with payload {text, source, page}. The collection uses named
vectors: "dense" (768-dim, cosine) + "sparse" (BM42), which the retrieval path
queries together with server-side RRF fusion.

Run manually once (and again whenever the corpus or the dense embedding model
changes — always with --wipe, since mixing embedding spaces in one collection
silently corrupts retrieval).

Usage
-----
    python ci/ingest_qdrant.py --wipe          # drop + recreate, then ingest
    python ci/ingest_qdrant.py                 # ingest into existing collection

Env: QDRANT_URL, QDRANT_API_KEY (cluster), plus Vertex ADC for the dense
embeddings. Reads Config.SOURCES / CHUNK_SIZE / CHUNK_OVERLAP / QDRANT_* .
"""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from langchain_google_vertexai import VertexAIEmbeddings

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor

BATCH = 64


def _client() -> QdrantClient:
    if not Config.QDRANT_URL or not Config.QDRANT_API_KEY:
        raise SystemExit("QDRANT_URL and QDRANT_API_KEY must be set (env / .env).")
    return QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)


def _ensure_collection(client: QdrantClient, wipe: bool):
    exists = client.collection_exists(Config.QDRANT_COLLECTION)
    if exists and wipe:
        print(f"--wipe: deleting existing collection '{Config.QDRANT_COLLECTION}'")
        client.delete_collection(Config.QDRANT_COLLECTION)
        exists = False
    if not exists:
        print(f"creating collection '{Config.QDRANT_COLLECTION}' "
              f"(dense {Config.DENSE_DIM}-dim cosine + BM42 sparse)")
        client.create_collection(
            collection_name=Config.QDRANT_COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=Config.DENSE_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                # IDF modifier is what makes BM42's sparse scores behave like a
                # proper lexical (BM25-style) signal.
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )


def main(wipe: bool):
    print("Qdrant ingestion (item 9)")
    print("=" * 60)

    # 1. chunk the PDF (same processor / chunk config the app uses)
    proc = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
    )
    docs = proc.process_urls(Config.SOURCES)
    print(f"chunks: {len(docs)}")

    # 2. embedders: Vertex dense (ADC) + BM42 sparse (fastembed, local ONNX)
    dense_embedder = VertexAIEmbeddings(model_name="text-embedding-005")
    sparse_embedder = SparseTextEmbedding(model_name=Config.SPARSE_MODEL)

    client = _client()
    _ensure_collection(client, wipe)

    # 3. embed + upsert in batches
    total = 0
    for start in range(0, len(docs), BATCH):
        batch = docs[start:start + BATCH]
        texts = [d.page_content for d in batch]

        dense_vecs = dense_embedder.embed_documents(texts)
        sparse_vecs = list(sparse_embedder.embed(texts))

        points = []
        for d, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
            meta = d.metadata or {}
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload={
                    "text": d.page_content,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", ""),
                },
            ))
        client.upsert(collection_name=Config.QDRANT_COLLECTION, points=points)
        total += len(points)
        print(f"  upserted {total}/{len(docs)}")

    count = client.count(Config.QDRANT_COLLECTION).count
    print("=" * 60)
    print(f"done. collection '{Config.QDRANT_COLLECTION}' now holds {count} points.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wipe", action="store_true",
                    help="drop + recreate the collection before ingesting "
                         "(required whenever the dense embedding model changes)")
    args = ap.parse_args()
    main(args.wipe)
