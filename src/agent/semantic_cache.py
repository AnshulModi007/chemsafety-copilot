"""In-memory semantic cache for copilot.ask(): reuse a full answer for a new
question if its embedding is still cosine-similar enough to a still-fresh
cached question, skipping the entire agentic pipeline (routing + retrieval +
one or more Groq calls) for near-duplicate questions.

Scoped per-intent on purpose: two "calculation" (PSV sizing) questions can be
textually near-identical while differing only in a number ("5000 lb/hr" vs
"6000 lb/hr"), and embeddings aren't reliably sensitive to that difference --
returning a cached numeric result for different inputs would be a silent
correctness bug, not just a staleness one. Only historical/comparative/
chemical_property answers are cached; calculation is never cached.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SEMANTIC_CACHE_SIMILARITY_THRESHOLD, SEMANTIC_CACHE_TTL_SECONDS  # noqa: E402
from src.retrieval.retriever import embed_text  # noqa: E402

CACHEABLE_INTENTS = {"historical", "comparative", "chemical_property"}

_cache: list[dict] = []


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _prune() -> None:
    now = time.time()
    _cache[:] = [e for e in _cache if e["expires_at"] > now]


def get_cached(query: str, intent: str) -> dict | None:
    if intent not in CACHEABLE_INTENTS:
        return None

    _prune()
    candidates = [e for e in _cache if e["intent"] == intent]
    if not candidates:
        return None

    embedding = embed_text(query)
    best = max(candidates, key=lambda e: _cosine(embedding, e["embedding"]))
    if _cosine(embedding, best["embedding"]) >= SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
        return best["result"]
    return None


def store(query: str, intent: str, result: dict) -> None:
    if intent not in CACHEABLE_INTENTS:
        return
    _cache.append({
        "embedding": embed_text(query),
        "intent": intent,
        "result": result,
        "expires_at": time.time() + SEMANTIC_CACHE_TTL_SECONDS,
    })
