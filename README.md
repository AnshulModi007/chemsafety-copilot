# ChemSafety Copilot

An agentic RAG system for process-safety intelligence: retrieval over U.S. Chemical
Safety Board (CSB) investigation reports, combined with live chemical data lookups
(PubChem) and engineering calculations — built with production-grade evaluation and
self-correction as first-class citizens, not afterthoughts.

See [`chemsafety-copilot-prompt.md`](./chemsafety-copilot-prompt.md) for the full
project brief and 5-week build order.

## Status: Week 4 — Failure gallery + deployment

- Week 1: CSB ingestion pipeline, dense retrieval, grounded generation, golden eval set, baseline RAGAS/retrieval metrics
- Week 2: hybrid (dense + BM25 + RRF) retrieval, cross-encoder reranker, CRAG grade/rewrite/retry loop
- Week 3: agentic router (historical / chemical_property / calculation / comparative intents), PubChem tool, API 520 relief-valve sizing tool
- Week 4 (in progress): stress-test failure gallery (see [`FAILURE_GALLERY.md`](./FAILURE_GALLERY.md)), FastAPI + Streamlit, Dockerized deployment

## Environment

- Python venv lives under this project directory (`D:`). Model/cache storage
  (Ollama models, HuggingFace cache, pip cache) is redirected to `C:\ai-cache\`
  instead — D: turned out to have only 9GB free (older unrelated projects already
  use most of it), while C: has 42GB free.
- Local LLM: `llama3.1:8b-instruct-q4_K_M` via [Ollama](https://ollama.com), run on
  an RTX 4060 (8GB VRAM).
- Embeddings: `BAAI/bge-base-en-v1.5` via `sentence-transformers`, local/free.
- Vector store: local Chroma.

Setup:
```powershell
scripts\setup_env.ps1
```

## Metrics: Before / After

| Metric | Week 1 - Dense only | Week 2 - Hybrid (dense + BM25, RRF) | Week 2 - Hybrid + Reranker | Week 2 - + CRAG |
|---|---|---|---|---|
| Recall@5 | 0.682 | 0.818 | 0.818 | 0.818 |
| MRR | 0.511 | 0.661 | 0.674 | 0.674 |
| Faithfulness (RAGAS) | 0.824 | 0.877 | 0.840 | 0.764 |
| Answer Relevance (RAGAS) | 0.685 | 0.663 | 0.656 | 0.629 |

_Precision@5 omitted from the table above: with exactly one relevant chunk per golden question it's mathematically capped at 1/5, so it doesn't carry independent signal beyond Recall@5._

## Deployment

`Dockerfile.backend` (FastAPI) and `Dockerfile.frontend` (Streamlit) build independent,
lean images -- see `docker-compose.yml` for local reference and `render.yaml` for a
Render Blueprint. The corpus is pre-ingested and baked into `chroma_db/` at build time
rather than re-ingested on container start.

**Important constraint:** free-tier Render/Railway web services have no GPU and
typically 512MB-1GB RAM, which cannot run the 4.9GB `llama3.1:8b` model locally in
the container. `OLLAMA_HOST` must point at a reachable Ollama instance you control
(e.g. a tunnel to this dev machine, or a paid instance with the model loaded) --
Render can't provision that for you, which is why it has no default in `render.yaml`.

For a deployment that's actually self-contained on free tiers, the LLM call sites
(`src/agent/router.py`, `src/generation/crag.py`, `src/generation/generate.py` --
all currently call `ollama.chat(...)`) would need to swap to a hosted free-tier API
(e.g. Groq) behind the same interface. Not done here since it's a real behavioral
change to already-verified code (would need the failure-gallery fixes re-verified
against the new model) -- worth doing as a deliberate next step, not silently.

