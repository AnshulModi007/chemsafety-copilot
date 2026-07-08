"""Embed changed/new chunks with bge-base-en-v1.5 and persist them into a
local Chroma collection -- incrementally: a report whose PDF hash hasn't
changed since the last run is skipped entirely (no re-embedding cost), only
new/changed/removed reports touch the collection.
"""
import json
import sys
from pathlib import Path

# Must be imported before torch (pulled in below by sentence_transformers) --
# on Windows, if the CUDA-enabled torch build loads its DLLs first, pyarrow's
# own bundled Arrow runtime (pulled in transitively via
# sentence_transformers -> datasets -> pandas -> pyarrow) segfaults on import.
# Importing it here first makes it win that DLL load-order race (see the same
# guard in src/retrieval/retriever.py).
import pyarrow  # noqa: F401,E402

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL  # noqa: E402
from src.ingestion.parse_pdf import PARSE_HASHES_PATH  # noqa: E402

# Tracks the source-PDF hash each report_id was indexed from (mirrors
# parse_hashes.json) -- a report is re-embedded only when its hash here is
# stale relative to parse_hashes.json, i.e. its parsed text actually changed.
INDEX_HASHES_PATH = PROCESSED_DIR / "index_hashes.json"


def _load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _chunk_metadata(c: dict) -> dict:
    return {
        "report_id": c["report_id"],
        "report_title": c["report_title"],
        "section": c["section"],
        "chemical": c["chemical"],
        "incident_type": c["incident_type"],
        "industry": c["industry"],
        "year": c["year"],
        "page_start": c["page_start"],
        "page_end": c["page_end"],
        "parent_text": c.get("parent_text", c["text"]),
        "parent_page_start": c.get("parent_page_start", c["page_start"]),
        "parent_page_end": c.get("parent_page_end", c["page_end"]),
    }


def main():
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text())
    print(f"Loaded {len(chunks)} chunks")

    parse_hashes = _load_json(PARSE_HASHES_PATH, {})
    index_hashes = _load_json(INDEX_HASHES_PATH, {})

    chunks_by_report: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_report.setdefault(c["report_id"], []).append(c)

    changed_report_ids = [
        rid for rid in chunks_by_report if parse_hashes.get(rid) != index_hashes.get(rid)
    ]
    removed_report_ids = [rid for rid in index_hashes if rid not in chunks_by_report]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)

    if not changed_report_ids and not removed_report_ids:
        print(f"Nothing changed since last index -- {collection.count()} chunks already up to date.")
        return

    print(f"{len(changed_report_ids)} report(s) changed/new, {len(removed_report_ids)} removed "
          "-- re-indexing only those, not the whole corpus")

    for rid in changed_report_ids + removed_report_ids:
        existing = collection.get(where={"report_id": rid})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    if changed_report_ids:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

        for rid in changed_report_ids:
            report_chunks = chunks_by_report[rid]
            ids = [c["chunk_id"] for c in report_chunks]
            texts = [c["text"] for c in report_chunks]
            metadatas = [_chunk_metadata(c) for c in report_chunks]

            print(f"  embedding {rid} ({len(texts)} chunks)...")
            embeddings = model.encode(texts, show_progress_bar=False, batch_size=32).tolist()

            batch_size = 200
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i + batch_size],
                    embeddings=embeddings[i:i + batch_size],
                    documents=texts[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                )
            index_hashes[rid] = parse_hashes[rid]

    for rid in removed_report_ids:
        index_hashes.pop(rid, None)

    INDEX_HASHES_PATH.write_text(json.dumps(index_hashes, indent=2))
    print(f"Indexed {collection.count()} chunks total into Chroma collection '{CHROMA_COLLECTION}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
