FROM python:3.13-slim

# libgomp1 is required at runtime by faiss-cpu and torch (OpenMP), and is not
# present in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change
#
# Uses the CPU-only torch wheel index (no CUDA/cuDNN/NCCL, smaller image) —
# Cloud Run has no GPU to use them anyway. Previously reverted 2026-07-06
# because download-r2.pytorch.org rejected the TLS handshake from the dev
# machine; confirmed working again 2026-07-13.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# App code. No index is committed; the FAISS index is built at startup
# from Vertex AI embeddings (text-embedding-005).
COPY . .

# Cloud Run sets $PORT (defaults to 8080) and routes traffic to it.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
