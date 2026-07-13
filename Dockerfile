FROM python:3.13-slim

# Pin uv to a specific version rather than :latest — reproducible builds,
# per Astral's own best-practice note. Distroless image, just the binaries.
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/

# libgomp1 is required at runtime by faiss-cpu and torch (OpenMP), and is not
# present in the slim base image.
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
# torch resolves to the CPU-only build (no CUDA/cuDNN/NCCL, smaller image —
# Cloud Run has no GPU to use them anyway) via the [tool.uv.sources] /
# [tool.uv.index] pytorch-cpu override in pyproject.toml. Confirmed in
# uv.lock: the sys_platform != 'darwin' torch block resolves to
# "2.13.0+cpu" from download.pytorch.org/whl/cpu, no cuda-* dependencies.
# A prior TLS handshake failure against that index (2026-07-06) no longer
# reproduces; see reference/rag-gcp/known_issues.md.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-cache

# App code. No index is committed; the FAISS index is built at startup
# from Vertex AI embeddings (text-embedding-005).
COPY . .

# Sync the project itself now that the code is present. uv installs it in
# editable mode by default, so "src.config", "src.node", etc. resolve as
# real imports — no sys.path.append hack needed in streamlit_app.py.
RUN uv sync --locked --no-cache

# Cloud Run sets $PORT (defaults to 8080) and routes traffic to it.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uv run streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
