"""Query expansion for retrieval recall: HyDE, multi-query rephrasing, and
step-back prompting, fused into a single retrieval call via
src/retrieval/retriever.py's expanded_retrieve.

Each technique attacks a different recall gap:
- HyDE: a hypothetical *answer passage* embeds closer to real report passages
  than a short question does, since it's written in the same register.
- Multi-query: rephrasing in a few different ways surfaces chunks that only
  match one particular phrasing of the question.
- Step-back: a more general question surfaces foundational/background
  context a narrowly-phrased question misses entirely.
"""
import sys
from pathlib import Path

from groq import Groq
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GROQ_FAST_MODEL  # noqa: E402
from src.retrieval.retriever import expanded_retrieve  # noqa: E402

_client = Groq()

HYDE_SYSTEM_PROMPT = """You write a short hypothetical excerpt from a U.S. Chemical Safety Board \
(CSB) incident investigation report that would answer the user's question, in the same factual, \
technical register as a real report (root cause, timeline, or recommendations language). It does not \
need to be accurate -- it exists only to improve semantic search, not to be shown to the user.

Respond with ONLY a JSON object: {"passage": "<hypothetical report excerpt, 2-4 sentences>"}
"""

MULTI_QUERY_SYSTEM_PROMPT = """Generate 3 alternative phrasings of the user's question for a search \
system over CSB chemical incident investigation reports. Vary vocabulary and specificity (e.g. \
technical terms, equipment names, synonyms for the same hazard) so different phrasings can surface \
different relevant chunks. Do not change what's being asked.

Respond with ONLY a JSON object: {"queries": ["<rephrasing 1>", "<rephrasing 2>", "<rephrasing 3>"]}
"""

STEP_BACK_SYSTEM_PROMPT = """Given a specific question about a chemical incident, write one more \
general "step-back" question that surfaces foundational/background context useful for answering it \
(e.g. asking about the general hazard class or equipment type instead of only the specific incident).

Respond with ONLY a JSON object: {"step_back_query": "<more general question>"}
"""


class HydePassage(BaseModel):
    passage: str


class MultiQueries(BaseModel):
    queries: list[str]


class StepBackQuery(BaseModel):
    step_back_query: str


def _ask(system_prompt: str, user_content: str) -> str:
    response = _client.chat.completions.create(
        model=GROQ_FAST_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def generate_hyde_passage(query: str) -> str:
    return HydePassage.model_validate_json(_ask(HYDE_SYSTEM_PROMPT, query)).passage


def generate_multi_queries(query: str) -> list[str]:
    return MultiQueries.model_validate_json(_ask(MULTI_QUERY_SYSTEM_PROMPT, query)).queries


def generate_step_back_query(query: str) -> str:
    return StepBackQuery.model_validate_json(_ask(STEP_BACK_SYSTEM_PROMPT, query)).step_back_query


def retrieve_with_expansion(query: str, top_k: int = 5, trace: dict | None = None) -> list[dict]:
    """Best-effort: any expansion call that fails (rare malformed JSON from
    the fast model) is simply dropped rather than failing the whole retrieval
    -- expansions are a recall booster, not a correctness requirement.

    If `trace` is passed, it's filled in-place with the expansion queries and
    HyDE passage actually used, for the CRAG debug trace (see crag.py).
    """
    expansion_queries: list[str] = []
    try:
        expansion_queries.extend(generate_multi_queries(query))
    except Exception:
        pass
    try:
        expansion_queries.append(generate_step_back_query(query))
    except Exception:
        pass

    hyde_passage = None
    try:
        hyde_passage = generate_hyde_passage(query)
    except Exception:
        pass

    if trace is not None:
        trace["expansion_queries"] = expansion_queries
        trace["hyde_passage"] = hyde_passage

    return expanded_retrieve(query, expansion_queries, hyde_passage, top_k=top_k)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What caused the ammonium nitrate explosion?"
    for hit in retrieve_with_expansion(query):
        print(f"[rerank={hit['rerank_score']:.4f}] {hit['report_title']} - {hit['section']}")
