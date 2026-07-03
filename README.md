# ChemSafety Copilot

An agentic RAG system for process-safety intelligence: retrieval over U.S. Chemical
Safety Board (CSB) investigation reports, combined with live chemical data lookups
(PubChem) and engineering calculations — built with production-grade evaluation and
self-correction as first-class citizens, not afterthoughts.

See [`chemsafety-copilot-prompt.md`](./chemsafety-copilot-prompt.md) for the full
project brief and 5-week build order.

## Status: Week 1 — Core RAG

Building: CSB ingestion pipeline → dense retrieval → grounded generation → golden
eval set → baseline RAGAS/retrieval metrics. Hybrid search, reranking, CRAG, and the
agentic router land in later weeks.

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

