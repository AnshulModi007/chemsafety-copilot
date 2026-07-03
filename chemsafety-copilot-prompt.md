# ChemSafety Copilot — Build Prompt

Copy the block below into a new Claude conversation (or Claude Code) to kick off the build.

---

```
I'm a chemical engineer building a portfolio project to demonstrate production-grade 
RAG and agentic AI skills for job applications (targeting AI/ML roles that value 
domain expertise in chemical engineering). I want your help building this end-to-end, 
starting with a working skeleton and iterating up to full production-grade.

PROJECT: ChemSafety Copilot — An Agentic RAG System for Process Safety Intelligence

ONE-LINE PITCH: An agentic assistant that helps chemical engineers learn from 
decades of industry incidents by combining retrieval over CSB (Chemical Safety 
Board) investigation reports with live chemical data lookups and engineering 
calculations — with production-grade evaluation and self-correction built in.

DATA SOURCE: U.S. Chemical Safety Board (CSB) investigation reports (public PDFs) 
as the core corpus. PubChem API for live chemical property lookups.

ARCHITECTURE:
1. Ingestion: Parse CSB PDFs (text + tables separately), semantic chunking by 
   section (Root Cause, Timeline, Recommendations), metadata tagging (chemical 
   involved, incident type, industry sector, year).
2. Hybrid retrieval: Dense embeddings (bge-large or OpenAI) + BM25 sparse search, 
   fused with Reciprocal Rank Fusion (RRF).
3. Reranking: Cross-encoder reranker (bge-reranker) on top-20 → top-5 final chunks.
4. Query transformation: Query rewriting, HyDE, multi-query expansion for 
   ambiguous/technical queries.
5. Corrective RAG (CRAG): Grade each retrieved chunk's relevance (Correct/
   Ambiguous/Incorrect). If Ambiguous/Incorrect, rewrite the query and retry, 
   or fall back to web search, or explicitly say retrieval was insufficient 
   rather than hallucinate.
6. Agentic router: Classify each query's intent and route to the right tool:
   - Historical/precedent questions → RAG pipeline above
   - Live chemical property questions → PubChem API tool
   - Engineering calculations (e.g., relief valve sizing per API 520) → calc tool
   - Comparative/multi-hop questions → multi-doc RAG + synthesis
7. Generation: Strict grounded prompt, must cite report ID + page number, 
   structured JSON output so the frontend can render clickable sources.
8. Guardrails: Safety disclaimers, refuses to give definitive engineering 
   judgment calls on critical decisions ("consult a PE").
9. Evaluation: Golden set of 50-100 QA pairs built from the reports. Track 
   Precision@k, Recall@k, MRR for retrieval; Faithfulness and Answer Relevance 
   (via RAGAS) for generation. Every architecture change should be measured 
   against this baseline with a before/after metrics table.
10. Observability: Log every query → retrieved chunks → tool calls → final 
    answer → latency for debugging and demo purposes.
11. Deployment: FastAPI backend + Streamlit frontend, Dockerized, deployed live 
    (Render/Railway free tier).

WHAT MAKES THIS "PRODUCTION-GRADE" (please keep these as first-class citizens, 
not afterthoughts):
- Evaluation-first: build the eval harness in Week 1, before optimizing anything
- A documented "failure gallery": 4-5 real cases where naive RAG failed, the fix 
  applied, and before/after metrics
- The agentic layer is not optional — it's what differentiates this from a 
  standard RAG demo
- Everything should produce a metric or artifact I can put in a README/resume

BUILD ORDER I WANT TO FOLLOW:
Week 1: Core RAG (ingestion + dense retrieval + generation) + golden eval set + 
        baseline RAGAS scores
Week 2: Hybrid search + reranker + CRAG loop, each measured against baseline
Week 3: Agentic router + PubChem tool + calculation tool
Week 4: Failure gallery (stress-test, document, fix) + deployment + real user 
        feedback (5-10 practicing engineers)
Week 5: Write-up/blog post + polished README with architecture diagram and 
        metrics tables

MY BACKGROUND: I'm a chemical engineer, comfortable with Python, learning ML/AI 
engineering as I go. Please explain architectural decisions clearly (not just 
"do X"), since I want to be able to defend every choice in an interview, not just 
have working code.

MY ENVIRONMENT:
- Python 3.x (please tell me if a specific version matters for any library)
- No paid API keys — I want this built entirely on FREE / open-source components. 
  Please pick free-tier or fully local/open-source options at every layer, e.g.:
  - Embeddings: open-source local models (e.g., bge-large, sentence-transformers) 
    instead of OpenAI/Voyage embeddings
  - LLM for generation, grading (CRAG), and query rewriting: a free-tier API 
    (e.g., Groq's free tier, Google Gemini free tier) or a local open-source 
    model (e.g., via Ollama) — please recommend the best free option given the 
    task and explain the tradeoff vs. a paid model
  - Reranker: open-source cross-encoder (e.g., bge-reranker) run locally, not 
    Cohere's paid rerank API
  - Vector DB: local Chroma (free, no hosting cost) to start; can discuss 
    migrating to a free tier of Qdrant Cloud later if needed
  - Chemical data lookups: PubChem API (free, no key required)
  - Deployment: free tiers only (Render/Railway/Streamlit Cloud free tier)
  - If any component genuinely has no viable free option, flag it clearly and 
    suggest the closest free/low-cost alternative rather than assuming I'll pay
- Starting with ~15-20 CSB reports for fast iteration; will scale up later
- Will version-control this in a GitHub repo from day one

WHAT I NEED FROM YOU RIGHT NOW:
Start with Week 1: scaffold the repo structure, then walk me through building the 
ingestion pipeline for CSB reports, a basic dense-retrieval RAG pipeline, and the 
golden evaluation set + baseline RAGAS scoring. Ask me clarifying questions if 
anything about my environment or setup is still unclear before scaffolding.
```

---

### Usage notes
- **Claude Code**: pastes this in and scaffolds actual files, can run/test code as it builds — recommended for this multi-week project.
- **Regular chat**: you'll get guided code + explanations but will need to copy files yourself.
- The prompt is scoped to "Week 1 only" on purpose — building incrementally with eval checkpoints at each step is the core of the production-grade story. Trim that line if you'd rather get everything scaffolded up front.
- **Free-tier reality check**: free LLM API tiers (Groq, Gemini, etc.) usually have rate limits — fine for building and testing with 15-20 documents and a small eval set, but worth knowing if things feel slow during heavier eval runs. If you hit limits, running a small model locally via Ollama is the fallback with zero rate limits, at the cost of needing decent local hardware.
