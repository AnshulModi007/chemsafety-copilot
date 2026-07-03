"""Retrieval over the CSB report corpus: dense (Chroma), sparse (BM25), and
hybrid (both, fused with Reciprocal Rank Fusion).
"""
import json
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, RERANKER_MODEL,
    RERANK_POOL, TOP_K, PROCESSED_DIR,
)

# bge-base-en-v1.5's recommended instruction prefix for the query side of
# retrieval (the document/passage side needs no prefix -- see build_index.py).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RRF_K = 60  # standard RRF damping constant
CANDIDATE_K = 20  # how many candidates each retriever contributes before fusion

_model = None
_collection = None
_bm25 = None
_bm25_chunks = None
_reranker = None
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(CHROMA_COLLECTION)
    return _collection


def _get_bm25():
    global _bm25, _bm25_chunks
    if _bm25 is None:
        chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text())
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
        _bm25_chunks = chunks
    return _bm25, _bm25_chunks


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def dense_retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(QUERY_INSTRUCTION + query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            **results["metadatas"][0][i],
        })
    return hits


# Backwards-compatible alias -- existing callers used `retrieve` for
# dense-only search; hybrid_retrieve is now the recommended default.
retrieve = dense_retrieve


def bm25_retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    bm25, chunks = _get_bm25()
    scores = bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    hits = []
    for i in ranked_idx:
        c = chunks[i]
        hits.append({
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "bm25_score": float(scores[i]),
            "report_id": c["report_id"],
            "report_title": c["report_title"],
            "section": c["section"],
            "chemical": c["chemical"],
            "incident_type": c["incident_type"],
            "industry": c["industry"],
            "year": c["year"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
        })
    return hits


def hybrid_retrieve(query: str, top_k: int = TOP_K, candidate_k: int = CANDIDATE_K) -> list[dict]:
    """Dense + BM25 fused with Reciprocal Rank Fusion (RRF)."""
    dense_hits = dense_retrieve(query, top_k=candidate_k)
    sparse_hits = bm25_retrieve(query, top_k=candidate_k)

    by_id: dict[str, dict] = {}
    rrf_scores: dict[str, float] = {}

    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            by_id.setdefault(hit["chunk_id"], hit)
            rrf_scores[hit["chunk_id"]] = rrf_scores.get(hit["chunk_id"], 0.0) + 1.0 / (RRF_K + rank)

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    return [{**by_id[cid], "rrf_score": rrf_scores[cid]} for cid in ranked_ids]


def reranked_retrieve(query: str, top_k: int = TOP_K, pool_size: int = RERANK_POOL) -> list[dict]:
    """Hybrid search for a wide candidate pool, then a cross-encoder reranks
    it down to top_k -- catches cases where RRF's rank fusion under-ranks a
    genuinely relevant chunk that only one of dense/BM25 surfaced strongly.
    """
    candidates = hybrid_retrieve(query, top_k=pool_size)
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [{**c, "rerank_score": float(s)} for c, s in ranked]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What caused the ammonium nitrate explosion?"
    for hit in reranked_retrieve(query):
        print(f"[rerank={hit['rerank_score']:.4f}] {hit['report_title']} - {hit['section']} (p{hit['page_start']}-{hit['page_end']})")
        print(f"  {hit['text'][:200]}...")
        print()
