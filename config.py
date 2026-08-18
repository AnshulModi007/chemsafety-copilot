"""Central settings for ChemSafety Copilot, loaded from .env (see .env.example)."""
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Groq decommissioned both llama-3.3-70b-versatile and llama-3.1-8b-instant,
# which returned a hard 404 ("model does not exist") on every request and took
# the whole pipeline down -- routing fails before retrieval is ever reached.
# Pinned to the gpt-oss pair now served in their place; both were verified to
# support response_format={"type": "json_object"}, which the router, CRAG
# grader, and non-streaming generation all depend on.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Smaller/faster model for simple structured-extraction tasks (intent
# routing, CRAG relevance grading, query rewriting) that don't need the main
# model's reasoning depth -- final answer generation still uses GROQ_MODEL.
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
# "local" loads the model in-process via sentence-transformers (needs torch --
# ~500MB+ RSS just to import torch+transformers+sentence-transformers, before any
# model weights). "api" calls Hugging Face's hosted Inference API instead, so the
# deployed backend never imports torch at all -- the biggest lever for fitting in
# memory-constrained free-tier hosting. Verified numerically equivalent output
# (cosine ~1.0, max abs diff ~1.8e-7) for BAAI/bge-small-en-v1.5, since the API
# already applies the same pooling/normalization sentence-transformers does.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").strip().lower()
# Needed only when EMBEDDING_BACKEND=api -- a free token from
# https://huggingface.co/settings/tokens with "Make calls to Inference Providers"
# permission enabled.
HF_TOKEN = os.getenv("HF_TOKEN")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_POOL = 20  # candidates pulled from hybrid search before reranking down to TOP_K
# Cross-encoder reranking loads a second ~278M-param model (~1.1GB in fp32) on top
# of the embedder -- too much for memory-constrained free-tier hosting. When off,
# retrieval falls back to RRF-fused hybrid order with no cross-encoder pass, and
# CRAG's high-confidence fast path never fires (no rerank_score to gate on), so
# every query takes the LLM-graded path instead -- still correct, just without
# that latency shortcut.
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").strip().lower() not in ("false", "0", "")

# Web search fallback (used only when the CSB corpus has no confident answer).
# No default on purpose -- get a free key at https://tavily.com. If unset, the
# app degrades to the old refusal-only behavior instead of erroring.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Semantic cache: reuse an answer for a new question if its embedding is at
# least this cosine-similar to a still-fresh cached question.
SEMANTIC_CACHE_TTL_SECONDS = 3600
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.95

# --- Production telemetry -----------------------------------------------------
# One structured record per query, written to its own SQLite file (never mixed
# into the Chroma store). See src/telemetry/.
METRICS_DB_PATH = Path(os.getenv("METRICS_DB_PATH", PROJECT_ROOT / "metrics.db"))

# Query text can carry sensitive process/plant detail, so persisting it is opt-out.
# Default on for a self-hosted demo; set LOG_QUERY_TEXT=false for a deployment
# handling third-party queries.
LOG_QUERY_TEXT = os.getenv("LOG_QUERY_TEXT", "true").strip().lower() not in ("false", "0", "")

# Groq re-prices and retires models on a rolling schedule, so this is a single
# table rather than literals spread across the cost math. USD per 1M tokens,
# current as of Aug 2026. A model absent from this table costs 0.0 and is
# surfaced as "unpriced" by the metrics API rather than silently ignored --
# a missing price is a reporting bug worth seeing, not a free query.
GROQ_PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}

# Cache hits run zero LLM calls and complete in milliseconds. Including them in
# latency/cost percentiles flatters the numbers, so aggregates exclude them by
# default; the metrics API takes ?include_cached=true to see the true user-facing
# distribution.
METRICS_EXCLUDE_CACHED_BY_DEFAULT = True

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_PDF_DIR = Path(os.getenv("RAW_PDF_DIR", DATA_DIR / "raw_pdfs"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", DATA_DIR / "processed"))
MANIFEST_PATH = DATA_DIR / "manifest.json"

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "csb_reports")

EVAL_DIR = PROJECT_ROOT / "src" / "eval"
GOLDEN_QA_PATH = EVAL_DIR / "golden_qa.json"
BASELINE_METRICS_PATH = PROJECT_ROOT / "baseline_metrics.json"

# User thumbs up/down feedback, appended as JSON Lines -- the seed for a
# future eval-feedback loop (see FAILURE_GALLERY.md-style review workflow).
FEEDBACK_LOG_PATH = PROJECT_ROOT / "feedback_log.jsonl"

CHUNK_TOKENS = 768
CHUNK_OVERLAP_TOKENS = 100
TOP_K = 5

# Upper bound on the characters of report text handed to a single generation
# call (and therefore to the faithfulness check, which re-reads the same
# context). Parent-child retrieval hands generation the *wide* parent window
# per chunk, which for TOP_K=5 runs to ~55k characters -- about 13.7k tokens,
# comfortably past Groq's free-tier 8,000 tokens-per-minute-per-model cap, so
# every grounded answer failed with a 413 before this existed.
#
# This budgets generation only. Retrieval still returns TOP_K=5 and the
# reported Recall@5/MRR figures are retrieval metrics, so they are unaffected.
# See _build_context in src/generation/generate.py for how the budget is spent:
# full parent windows when they fit, then the narrower retrieval windows, then
# proportional truncation.
#
# ~4 chars/token, so 20k chars is ~5k tokens, leaving headroom for the system
# prompt and the completion inside one 8k request. Raise it on a paid tier --
# wider context is strictly better for answer quality when the cap allows.
GENERATION_CONTEXT_CHAR_BUDGET = int(os.getenv("GENERATION_CONTEXT_CHAR_BUDGET", "20000"))

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"

for _dir in (RAW_PDF_DIR, PROCESSED_DIR, CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
