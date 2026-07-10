FROM python:3.11-slim-bookworm

# Run as non-root -- not all hosts require it, but it's a reasonable default and
# some (e.g. HF Spaces) enforce it.
RUN useradd -m -u 1000 user

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf-cache \
    ANONYMIZED_TELEMETRY=False \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    ENABLE_RERANKER=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY config.py .
COPY data/manifest.json data/manifest.json
COPY data/processed/ data/processed/
# chroma_db_deploy/ was built with the small embedder above (see
# scripts/ or the one-off build script) -- dimension-compatible with
# EMBEDDING_MODEL here. The full-size ./chroma_db (bge-base-en-v1.5) used for
# local dev is NOT what ships in this image.
COPY chroma_db_deploy/ chroma_db/

# No persistent volume on most free hosting tiers -- the embedding model cache
# re-downloads into this dir on cold start/restart.
RUN mkdir -p /app/.hf-cache && chown -R user:user /app

USER user

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
