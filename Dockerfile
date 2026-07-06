FROM python:3.13-slim

# libgomp1 is required at runtime by faiss-cpu and torch (OpenMP), and is not
# present in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, including the committed faiss_index/ (bge-small-en-v1.5 embeddings)
COPY . .

# Cloud Run sets $PORT (defaults to 8080) and routes traffic to it.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
