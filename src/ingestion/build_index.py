"""Embed all chunks with bge-base-en-v1.5 and persist them into a local Chroma collection."""
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL  # noqa: E402


def main():
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text())
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Rebuild clean each run so re-embedding after a chunking change doesn't
    # leave stale vectors behind.
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(CHROMA_COLLECTION)

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "report_id": c["report_id"],
            "report_title": c["report_title"],
            "section": c["section"],
            "chemical": c["chemical"],
            "incident_type": c["incident_type"],
            "industry": c["industry"],
            "year": c["year"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
        }
        for c in chunks
    ]

    print("Embedding chunks (bge-base-en-v1.5, no query-instruction prefix needed on the document side)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    batch_size = 200
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Indexed {collection.count()} chunks into Chroma collection '{CHROMA_COLLECTION}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
