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
    ANONYMIZED_TELEMETRY=False \
    USER_DB_PATH=/app/userdata/users.db

# Installed packages + console scripts (gunicorn) from the builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY templates/ ./templates/

# Writable home for the user DB (accounts + history). Kept OUT of the read-only
# corpus mount; back it with a named volume so it survives container restarts.
RUN mkdir -p /app/userdata
VOLUME /app/userdata

# Bake the embedding model into the image so there's no cold download on boot.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Bake the serve-time corpus (app.db, index/, clean/, chunks/; raw/parsed/users.db
# are excluded via .dockerignore) so the image is self-contained and runs on any
# Docker host with no mounted volume. Copied last so corpus changes do not
# invalidate the model-bake layer above. When a volume IS mounted (compose / EC2),
# it simply overrides this baked copy.
COPY data/ ./data/

EXPOSE 5000

# Corpus is mounted read-only at /app/data, EXCEPT the ChromaDB dir, which needs
# write access (SQLite locking) even for reads; user data persists in /app/userdata:
#   docker run -e GROQ_API_KEY=... -e FLASK_SECRET_KEY=... -e ADMIN_EMAIL=... \
#     -v $(pwd)/data:/app/data:ro \
#     -v $(pwd)/data/index/chroma:/app/data/index/chroma \
#     -v seglit_userdata:/app/userdata -p 5000:5000 <image>
# Or just: docker compose up --build   (see docker-compose.yml)
# High worker timeout because an agent query can pause on Groq's rate limit.
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "-t", "300", "src.app:app"]
