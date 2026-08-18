FROM python:3.11-slim-bookworm

# Run as non-root -- not all hosts require it, but it's a reasonable default and
# some (e.g. HF Spaces) enforce it.
RUN useradd -m -u 1000 user

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    EMBEDDING_BACKEND=api \
    ENABLE_RERANKER=false

COPY requirements.txt .
# CPU-only torch first -- plain `pip install torch` on Linux pulls the CUDA-enabled
# build (several GB of bundled nvidia-* packages), pointless on a host with no GPU.
# sentence-transformers (needed for local dev / EMBEDDING_BACKEND=local) hard-
# requires torch to be installed on disk, so this stays even though the app never
# imports torch at runtime in this image's config -- see the note below.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
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

# With EMBEDDING_BACKEND=api and ENABLE_RERANKER=false above, src/retrieval/
# retriever.py never actually imports torch/sentence-transformers at runtime --
# embeddings come from the HF Inference API call instead (see config.py).
# Importing torch alone costs ~500MB+ RSS, most of a 512MB free-tier budget;
# skipping the import (not just the install) is what makes this fit. HF_TOKEN
# must be set as a secret at runtime, never baked into the image.
RUN chown -R user:user /app

USER user

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
