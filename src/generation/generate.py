"""Grounded generation over retrieved CSB report chunks, via a local Ollama model."""
import json
import sys
from pathlib import Path

import ollama
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OLLAMA_MODEL, TOP_K  # noqa: E402
from src.retrieval.retriever import reranked_retrieve  # noqa: E402
from src.generation.crag import retrieve_with_crag  # noqa: E402

INSUFFICIENT_RETRIEVAL_MESSAGE = (
    "The retrieved CSB report excerpts do not contain enough information to answer this "
    "question confidently, even after retrying with a rewritten query. Rather than guess, "
    "ChemSafety Copilot is declining to answer -- try rephrasing the question or narrowing "
    "it to a specific incident."
)

SYSTEM_PROMPT = """You are ChemSafety Copilot, an assistant that helps chemical engineers learn from \
past industry incidents by grounding every answer in retrieved excerpts from U.S. Chemical Safety \
Board (CSB) investigation reports.

Rules:
- Answer ONLY using the provided report excerpts. Do not use outside knowledge.
- Every factual claim must cite the report_id and page number it came from.
- If the excerpts do not contain enough information to answer, say so explicitly instead of guessing.
- This is historical incident information, not engineering advice. Never give a definitive \
engineering judgment call on a critical safety decision (e.g. "this design is safe"). Point to \
what the reports found and recommend the user consult a licensed Professional Engineer (PE) for \
any real design decision.
- Respond with ONLY a JSON object matching this schema, no other text:
{"answer": "<grounded answer text>", "citations": [{"report_id": "<id>", "page": <int>}]}
"""


class Citation(BaseModel):
    report_id: str
    page: int


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]


def _build_context(hits: list[dict]) -> str:
    blocks = [
        f'[report_id={hit["report_id"]} title="{hit["report_title"]}" '
        f'section="{hit["section"]}" page={hit["page_start"]}-{hit["page_end"]}]\n{hit["text"]}'
        for hit in hits
    ]
    return "\n\n---\n\n".join(blocks)


def generate_from_hits(query: str, hits: list[dict], max_retries: int = 2) -> dict:
    context = _build_context(hits)
    user_prompt = f"Report excerpts:\n\n{context}\n\nQuestion: {query}"

    last_error = None
    for _ in range(max_retries + 1):
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # Schema-constrained decoding: far more reliable than describing
            # the JSON shape in the prompt and hoping the model complies.
            format=GroundedAnswer.model_json_schema(),
            options={"num_ctx": 8192},
        )
        raw = response["message"]["content"]
        try:
            parsed = GroundedAnswer.model_validate_json(raw)
            return {
                "answer": parsed.answer,
                "citations": [c.model_dump() for c in parsed.citations],
                "retrieved_chunks": [h["chunk_id"] for h in hits],
            }
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = e
            continue

    raise RuntimeError(f"Model failed to produce valid grounded JSON after {max_retries + 1} attempts: {last_error}")


def generate(query: str, top_k: int = TOP_K, max_retries: int = 2) -> dict:
    hits = reranked_retrieve(query, top_k=top_k)
    return generate_from_hits(query, hits, max_retries)


def generate_with_crag(query: str, top_k: int = TOP_K, max_retries: int = 2) -> dict:
    crag_result = retrieve_with_crag(query, top_k=top_k)

    if crag_result["insufficient"]:
        return {
            "answer": INSUFFICIENT_RETRIEVAL_MESSAGE,
            "citations": [],
            "retrieved_chunks": [c["chunk_id"] for c in crag_result["chunks"]],
            "crag_insufficient": True,
            "crag_rewritten_query": crag_result["rewritten_query"],
        }

    result = generate_from_hits(query, crag_result["used_chunks"], max_retries)
    result["crag_insufficient"] = False
    result["crag_rewritten_query"] = crag_result["rewritten_query"]
    return result


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What caused the ammonium nitrate explosion at West Fertilizer?"
    result = generate_with_crag(query)
    print(json.dumps(result, indent=2))
