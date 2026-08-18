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

## Running on Groq's free tier

The free tier allows **8,000 tokens per minute, per model**, and that ceiling
applies to each individual request. Parent-child retrieval hands generation the
wide parent window for each of `TOP_K=5` chunks -- about 55k characters, or
~13.7k tokens -- so every grounded answer used to fail outright with
`413 Request too large`. The faithfulness check re-reads the same context, so it
failed for the same reason.

`GENERATION_CONTEXT_CHAR_BUDGET` (default 20,000 chars, ~5k tokens) bounds what a
single generation call receives. It degrades in three steps, giving up quality
only when it must: full parent windows when they all fit, then the narrower
retrieval windows that were actually scored as relevant, then proportional
truncation with an explicit `[... excerpt truncated ...]` marker. **Chunks are
never dropped** -- losing one silently removes a citable source, which is worse
than shortening all of them.

This budgets *generation only*. Retrieval still returns `TOP_K=5`, so the
reported Recall@5 and MRR figures -- which are retrieval metrics -- are
unaffected. Measured after the change: the largest single request fell from
~13,970 to **4,283 tokens**, and grounded answers now return with citations, a
passing faithfulness check, and a generated bowtie diagram.

The router, CRAG grader, query expansion, faithfulness check, and diagram
extraction all run on `GROQ_FAST_MODEL`, while only final generation uses
`GROQ_MODEL` -- and the two draw on separate token buckets, which is what makes a
7-call query fit. Raise the budget on a paid tier; wider context is strictly
better for answer quality when the cap allows.

## Production observability (`src/telemetry/`)

Every query writes one structured telemetry row to a dedicated SQLite database
(`metrics.db`, never mixed into the Chroma store), exposed as read-only aggregate
endpoints and an Analytics page in the React app. The point is to be able to
answer *how fast, how expensive, how good, and where does it fail* about a system
running in production -- not to keep a log file.

<!-- ![Analytics dashboard](docs/analytics.png) -->
_Screenshot placeholder -- add `docs/analytics.png`._

### What is captured

Per query: a generated id and UTC timestamp, total latency, **per-stage latency**
(reformulation, routing, cache lookup/store, expansion, retrieval, CRAG grading,
generation, faithfulness, web search, diagram, tool call), every model used with
input/output token counts, a computed cost, the routed intent, the LLM-as-Judge
faithfulness verdict and unsupported-claim count, retrieval confidence and top
rerank score, whether the CRAG retry fired, whether the Tavily fallback fired,
whether it was a cache hit, and on failure a categorised error
(`llm_provider` / `llm_output_invalid` / `retrieval` / `external_tool` /
`validation` / `internal`) with the exception *type* only.

Per LLM call, in its own table: model, pipeline stage, tokens, cost, client-side
latency, and Groq's own server-side inference time -- so network overhead can be
separated from model time. A single grounded query fans out to 8-12 Groq calls
across seven modules, so cost is always a sum over these rows.

### Privacy

`LOG_QUERY_TEXT` (default `true`) controls whether raw query text is persisted at
all. Set `LOG_QUERY_TEXT=false` for any deployment handling third-party queries.
No API key, secret, or exception *message* is ever written -- messages are
excluded specifically because they can echo the query text this toggle exists to
protect.

### How it stays off the critical path

The FastAPI handlers are sync `def`, so Starlette runs them in a worker
threadpool and there is no event loop to await on. Instead the request path only
does a non-blocking `queue.put_nowait`, and a single daemon thread owns the one
SQLite connection and performs the inserts. A metrics failure -- a full queue, an
unopenable database, a bug in the recorder -- is counted, logged, and never
surfaced to the user. That is covered by tests, not just intent.

Instrumentation is deliberately non-invasive: `instrumented_groq()` is a drop-in
proxy for `Groq()`, so swapping one line per module captures model/token/latency
for every call without touching any business logic. Only per-stage timing appears
inside the pipeline, as `with stage(...)` wrappers around calls that already
exist. Everything is a no-op outside a request scope, so the CLI entrypoints and
the eval harness are unaffected.

### Metrics API

Read-only, aggregate-only, on their own router:

| Endpoint | Returns |
|---|---|
| `GET /metrics/latency` | p50/p95/p99 total, plus per-stage mean/p95 |
| `GET /metrics/cost` | total and per-query cost, broken down by model |
| `GET /metrics/quality` | faithfulness pass-rate, intent distribution, CRAG-retry and fallback rates |
| `GET /metrics/reliability` | error rate and breakdown by category |
| `GET /metrics/timeseries` | volume, latency, cost and faithfulness bucketed over time |
| `GET /metrics/failures` | recent incidents (no exception messages) |
| `GET /metrics/overview` | all of the above in one round-trip, for the dashboard |

All take `?hours=` and `?include_cached=`. **Cache hits are excluded by default**:
they run zero LLM calls and return in milliseconds, so including them flatters
the latency and cost percentiles. Prices live in one place --
`config.GROQ_PRICING_USD_PER_MTOK` -- because Groq re-prices and retires models on
a rolling schedule; a model missing from that table is reported as `priced:
false` rather than silently costing zero.

### Viewing the dashboard

Open the React app and switch to the **Analytics** tab (`npm run dev`, then
http://localhost:5173). It polls `/metrics/overview` every 30s and shows latency
percentiles with a per-stage breakdown, cost by model and over time, intent
distribution, faithfulness trend, error rate and a recent-incident list -- plus
the health of the telemetry writer itself, since a dashboard that has quietly
stopped recording otherwise looks identical to a system with no traffic.

## Web frontend (React + TypeScript)

`frontend/` is a React 18 + TypeScript SPA (Vite, Tailwind) that talks to the same
FastAPI backend the Streamlit app does. Its purpose is to make the pipeline's
reasoning inspectable rather than hiding it behind a chat box -- this is a
document-grounded tool, so the evidence is laid out beside the answer, not buried.

<!-- ![ChemSafety Copilot](docs/screenshot.png) -->
_Screenshot placeholder -- add `docs/screenshot.png`._

What it surfaces, all from fields the backend already returns:

- **Streamed answers** over SSE (`POST /ask/stream`), consumed with `fetch` +
  a `ReadableStream` reader rather than `EventSource`, which cannot POST a body.
  The inline `[[report:id:page]]` citation tags the streaming prompt emits are
  stripped live, with a partially-arrived tag held back rather than flashed.
- **Intent badge** -- which of the five router intents handled the question,
  plus the router's own stated reasoning.
- **Sources panel** -- the report excerpts that actually grounded the answer,
  with title, section, page range, excerpt, and cross-encoder relevance score
  (from `data.sources`, see below).
- **"How this answer was found"** -- the CRAG trace: hybrid dense+BM25 retrieval,
  HyDE/multi-query expansion, per-chunk grading verdicts, and whether the
  corrective retry loop fired, with the rewritten query.
- **Faithfulness status** -- the independent verification pass's result, shown
  only for the intents that produce one.
- **SVG diagrams** rendered inline with enlarge/download, parsed and scrubbed of
  scripts, event handlers, and `javascript:` URLs before injection.
- **Structured cards** for PubChem property lookups and API 520 sizing results.

### Two backend changes it required

Both additive; neither alters retrieval or generation behaviour.

1. **CORS** (`app/main.py`) -- an explicit origin allowlist for the Vite dev
   server. The production origin goes in the same list, marked with a comment.
2. **`data.sources`** (`src/agent/copilot.py`) -- `citations` only carries
   `(report_id, page)` and `retrieved_chunks` only chunk ids, so neither could
   back a sources panel showing which report an answer came from and how
   strongly it matched. `_sources()` projects the `used_chunks` already in
   memory into citation-grade metadata. `rerank_score` is `null` when
   `ENABLE_RERANKER=false` -- there is no cross-encoder pass to score with, and
   the UI renders "score unavailable" rather than a misleading `0.000`.

### Running the React frontend

```powershell
# Terminal 1 -- backend
uvicorn app.main:app --reload

# Terminal 2 -- frontend
cd frontend
npm install
copy .env.example .env    # VITE_API_BASE_URL=http://localhost:8000
npm run dev               # http://localhost:5173
```

`npm run build` type-checks under `tsc` strict and emits a static bundle to
`frontend/dist/`. No API keys reach the browser -- every LLM call stays
server-side behind the FastAPI backend.

`npm test` runs a 19-test Vitest suite over the logic most likely to break
silently: the SSE reader (replaying a recorded backend stream cut at hostile
chunk boundaries -- mid-JSON, and between the two newlines of an event
terminator), the inline citation-tag stripper (asserting no frame of an
incremental replay ever leaks a raw `[[`), and the diagram-field normaliser.

## Running locally (Streamlit)

The original Streamlit app is still maintained and works unchanged; it is a
separate client against the same API.

Backend (FastAPI):
```powershell
uvicorn app.main:app --reload
```

Frontend (Streamlit), in a second terminal:
```powershell
streamlit run app/streamlit_app.py
```

Both frontends talk to the backend over HTTP (`BACKEND_URL` / `VITE_API_BASE_URL`,
default `http://localhost:8000`) rather than importing the agent in-process. LLM calls
(`src/agent/router.py`, `src/generation/crag.py`, `src/generation/generate.py`) go to
the hosted Groq API rather than a local model -- set `GROQ_API_KEY` in `.env` (get a
free key at https://console.groq.com/keys).

