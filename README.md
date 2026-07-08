# ChemSafety Copilot

An agentic RAG system for process-safety intelligence: retrieval over U.S. Chemical
Safety Board (CSB) investigation reports, combined with live chemical data lookups
(PubChem), engineering calculations, and general chemical-engineering knowledge —
built with production-grade evaluation and self-correction as first-class citizens,
not afterthoughts. Routes each question to one of 5 tools (historical RAG+CRAG,
comparative multi-incident RAG, live PubChem lookup, API 520 relief-valve sizing, or
an ungrounded general-knowledge answer for concept questions), and auto-generates an
SVG diagram alongside the answer when one would actually help.

See [`chemsafety-copilot-prompt.md`](./chemsafety-copilot-prompt.md) for the full
project brief and 5-week build order.

## Status: Week 7 — 5 intents, auto-generated diagrams, hardened UI

- Week 1: CSB ingestion pipeline, dense retrieval, grounded generation, golden eval set, baseline RAGAS/retrieval metrics
- Week 2: hybrid (dense + BM25 + RRF) retrieval, cross-encoder reranker, CRAG grade/rewrite/retry loop
- Week 3: agentic router (historical / chemical_property / calculation / comparative intents), PubChem tool, API 520 relief-valve sizing tool
- Week 4: stress-test failure gallery (see [`FAILURE_GALLERY.md`](./FAILURE_GALLERY.md)), FastAPI + Streamlit, Dockerized deployment
- Week 5: Groq migration (from local Ollama), web search fallback (Tavily) when the corpus has no confident answer, retrieval confidence scoring, conversational memory + follow-up reformulation, HyDE/multi-query/step-back query expansion, parent-child retrieval, semantic response caching, incremental (hash-based) re-indexing, RAGAS Context Precision/Recall, and token-streamed responses
- Week 6: CRAG debug/trace panel (per-attempt retrieval method, expansion queries, chunk-level grading verdicts), independent faithfulness verification (a second LLM call checks the generated answer against its own source context, not just a prompt instruction), user feedback logging (👍/👎 per answer)
- Week 7: a 5th router intent (`general_knowledge`) for chemical-engineering concept questions that aren't about a specific chemical/incident/calculation, an auto-generated SVG diagram pipeline across all 5 intents, a full Streamlit UI overhaul, input-validation/error-handling hardening, and a pytest suite

### What's new in Week 6

- **CRAG trace panel**: every retrieval attempt (method used, HyDE passage, expansion queries, per-chunk rerank score + grading verdict/reason) is captured and shown in a collapsible "Under the hood" panel -- the CRAG grade/rewrite/retry loop is inspectable, not a black box.
- **Faithfulness verification**: after generation, a second independent Groq call checks whether every claim in the answer is actually supported by its source context, surfacing a warning with the specific unsupported claim(s) if not. This is a real check against the model's own output, not just a "don't hallucinate" instruction in the generation prompt.
- **Feedback logging**: 👍/👎 buttons on every answer log the query, resolved query, intent, answer, and rating to `feedback_log.jsonl` (gitignored -- may contain real user queries) for later review.

### What's new in Week 7

- **5th intent — `general_knowledge`**: added after dogfooding turned up a real routing gap -- questions like "what is a tray tower" or "what is mass transfer" aren't about a named chemical, a past incident, or a calculation, so the router was forcing them into `chemical_property`, which then correctly found no chemical name and returned a confusing refusal. `general_knowledge` answers straight from the model's own knowledge, clearly disclaimed as ungrounded (not from the CSB corpus or a live data source).
- **Auto-generated diagrams (`src/visualization/`)**: a post-processing layer, not a 6th intent -- diagram generation is a presentation concern on top of whichever tool already ran. Same division of labor throughout: an LLM extracts structured content (from already-available context, no extra retrieval), a plain Python function renders it as SVG. The model never touches SVG/XML directly, so a failed extraction just means "no diagram," never malformed markup.
  - **PSV cross-section schematic** -- parametric, scales the nozzle/disc throat width with the recommended API 526 orifice area.
  - **Incident bowtie diagram** -- threats -> critical event -> consequences, with barriers as tick marks; falls back to a simpler causal-chain flowchart (precondition -> escalation -> critical event -> consequence, severity-colored) when the report doesn't cleanly support a bowtie structure.
  - **Comparative side-by-side** -- one causal-chain column per incident being compared.
  - **GHS hazard pictograms** -- deterministic (no LLM), a pure H-code -> hazard-category lookup.
  - **General-knowledge concept diagrams** -- conditional: an LLM call decides whether *this specific question* (not just the topic) is actually asking to explain a physical structure/layout before generating anything, so a tray-tower's-cost or -material follow-up correctly gets no diagram while "what is a tray tower" does.
- **UI overhaul**: intent-colored badges (purple=historical, teal=chemical_property, green=calculation, amber=comparative, blue=general_knowledge), structured result cards (PSV sizing table with the required area highlighted, PubChem property card) with copy-to-clipboard buttons, inline diagram rendering with SVG downloads, a redesigned sidebar (tool legend, last-5 recent queries, per-tool grouped examples, "New chat" at top), a disclaimer banner that collapses after the first question, per-tool loading text driven by an early `routing` SSE event, and mobile-responsive layout.
- **Hardening**: input validation in `size_psv_vapor` (physically-invalid inputs raise a clear error; edge cases like very low flow or non-ideal compressibility get an explicit warning instead of a silently-wrong answer), a distinct `PubChemUnavailable` exception for network/timeout/malformed-response failures (vs. "compound not found"), and a retry-once-on-malformed-JSON wrapper around every structured LLM call in the router.
- **Tests**: a pytest suite (`tests/`) covering PSV sizing correctness against the API 520 C-vs-k reference table and a worked example, routing logic against a mocked Groq client (no live API calls), and SVG well-formedness for every diagram generator.
- Several of the Week 7 features exist because real bugs were found through actual interactive use, not just code review -- see the newest entries in [`FAILURE_GALLERY.md`](./FAILURE_GALLERY.md).

## Environment

- Python venv lives under this project directory (`D:`). Model/cache storage
  (HuggingFace cache, pip cache) is redirected to `C:\ai-cache\`
  instead — D: turned out to have only 9GB free (older unrelated projects already
  use most of it), while C: has 42GB free.
- LLM: `llama-3.3-70b-versatile` (generation) + `llama-3.1-8b-instant` (routing/
  grading) via the hosted [Groq](https://console.groq.com) API.
- Web search fallback: [Tavily](https://tavily.com) API (optional -- `TAVILY_API_KEY`;
  if unset, the app just declines instead of searching the web).
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

_This table reflects the Week 1-2 retrieval-layer eval (`src/eval/run_baseline_eval.py`
against `src/eval/golden_qa.json`) and hasn't been re-run since the Week 5-7 additions
(query expansion, parent-child retrieval, the router, diagram generation) -- it's
evidence for the retrieval pipeline specifically, not a claim about the full current
system's end-to-end accuracy._

## Testing

```powershell
pytest tests/
```

`tests/test_calculations.py` checks PSV sizing against the API 520 C-vs-k reference
table and a hand-verified worked example, plus every input-validation/edge-case
warning. `tests/test_router.py` exercises routing/extraction logic against a mocked
Groq client (`tests/conftest.py`'s `fake_groq` fixture) -- deterministic and free, no
live API calls. `tests/test_diagrams.py` checks every SVG generator produces
well-formed XML. LLM-judgment-dependent code (CRAG grading, causal-chain/bowtie
extraction, the diagram-necessity decision) is intentionally out of scope for this
offline suite; those are covered by the failure-gallery-driven manual verification
instead.

## Running locally

Backend (FastAPI):
```powershell
uvicorn app.main:app --reload
```

Frontend (Streamlit), in a second terminal:
```powershell
streamlit run app/streamlit_app.py
```

The frontend talks to the backend over HTTP (`BACKEND_URL` env var, default
`http://localhost:8000`) rather than importing the agent in-process. LLM calls
(`src/agent/router.py`, `src/generation/crag.py`, `src/generation/generate.py`) go to
the hosted Groq API rather than a local model -- set `GROQ_API_KEY` in `.env` (get a
free key at https://console.groq.com/keys).

