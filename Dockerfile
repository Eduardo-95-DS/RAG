FROM python:3.13-slim

# Pin uv to a specific version rather than :latest — reproducible builds,
# per Astral's own best-practice note. Distroless image, just the binaries.
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/

# libgomp1 is required at runtime by faiss-cpu (OpenMP) and is not present in
# the slim base image. (torch used to need it too, but torch was removed in
# item 7 when the reranker moved to FlashRank/onnxruntime; faiss still needs it.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Don't let uv try to download/manage its own Python — use the system
# python:3.13-slim interpreter already in this image.
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_LINK_MODE=copy

# Install dependencies first, without the project itself, so this layer is
# cached across code-only changes (mirrors the old requirements.txt-first
# pattern, using uv's documented --no-install-project intermediate-layer
# technique instead). --locked fails the build if uv.lock is out of sync
# with pyproject.toml, rather than silently re-resolving.
#
# Item 7 (2026-07-15): torch and sentence-transformers are gone — the reranker
# runs on FlashRank/onnxruntime now, a much smaller runtime with no CUDA
# machinery, so the old CPU-torch index override is no longer needed.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-cache

# App code. No index is committed; the FAISS index is built at startup
# from Vertex AI embeddings (text-embedding-005).
COPY . .

# Sync the project itself now that the code is present. uv installs it in
# editable mode by default, so "src.config", "src.node", etc. resolve as
# real imports — no sys.path.append hack needed in streamlit_app.py.
RUN uv sync --locked --no-cache

# Bake the FlashRank ONNX reranker model into the image (item 7) so cold
# starts don't re-download it on the first query. Must match MODEL and the
# default cache_dir in CrossEncoderReranker (vectorstore.py).
RUN uv run python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/.flashrank_cache')"

# Cloud Run sets $PORT (defaults to 8080) and routes traffic to it.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uv run streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
