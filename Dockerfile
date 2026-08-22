# Multi-stage build: install deps in a builder, copy only what's needed to a slim
# runtime image. No secrets and no data/ are baked in -- the Groq key comes from
# an env var and the index/DB is mounted at /app/data at runtime.

# ---------- builder ----------
FROM python:3.12-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY requirements.txt .
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --upgrade pip \
 && pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ANONYMIZED_TELEMETRY=False

# Installed packages + console scripts (gunicorn) from the builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY templates/ ./templates/

# Bake the embedding model into the image so there's no cold download on boot.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 5000

# data/ (chroma index, app.db, clean/, chunks/) is mounted at runtime:
#   docker run -e GROQ_API_KEY=... -v $(pwd)/data:/app/data -p 5000:5000 <image>
# High worker timeout because an agent query can pause on Groq's rate limit.
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "-t", "300", "src.app:app"]
