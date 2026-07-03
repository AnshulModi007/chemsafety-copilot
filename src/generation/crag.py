"""Corrective RAG (CRAG): grade retrieved chunks for relevance to the query,
rewrite-and-retry the query once on weak retrieval, and signal when even a
retry can't find correct evidence so the caller can refuse rather than
hallucinate an answer.
"""
import sys
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OLLAMA_MODEL, TOP_K  # noqa: E402
from src.retrieval.retriever import reranked_retrieve  # noqa: E402

GRADER_SYSTEM_PROMPT = """You are a strict relevance grader for a RAG system over CSB chemical \
incident investigation reports. Given a question and a numbered list of retrieved excerpts, grade \
each excerpt's relevance to answering the question:
- "correct": directly contains information needed to answer the question
- "ambiguous": topically related but doesn't clearly answer the question
- "incorrect": not relevant to the question

Important: if the question names or clearly implies a specific incident, facility, or company, an \
excerpt about a DIFFERENT incident/facility/company is "incorrect" even if it discusses similar \
hazards in general (explosions, chemical releases, root-cause findings, etc.) -- superficial topical \
similarity (both are "explosions", both mention "root cause") is not relevance. Only grade "correct" \
or "ambiguous" if the excerpt is actually about the incident/facility/company/chemical the question asks about.

Respond with ONLY a JSON object matching this schema:
{"grades": [{"chunk_index": <int>, "verdict": "correct"|"ambiguous"|"incorrect", "reason": "<brief reason>"}]}
"""

REWRITE_SYSTEM_PROMPT = """You rewrite search queries for a retrieval system over CSB chemical \
incident investigation reports. The original query retrieved poor results. Rewrite it to be more \
specific and use terminology likely to appear in formal incident report text (named chemicals, \
equipment, or CSB-style phrasing). Respond with ONLY a JSON object: {"rewritten_query": "<query>"}
"""


class Grade(BaseModel):
    chunk_index: int
    verdict: Literal["correct", "ambiguous", "incorrect"]
    reason: str


class GradingResult(BaseModel):
    grades: list[Grade]


class RewrittenQuery(BaseModel):
    rewritten_query: str


def grade_chunks(query: str, chunks: list[dict]) -> list[Grade]:
    # Showing report_title/chemical up front turns "is this the same incident?"
    # into a structural comparison instead of something the grader has to infer
    # from prose alone -- the latter is where the small local model was prone
    # to false-positive on superficial topical similarity (see failure gallery).
    listing = "\n\n".join(
        f'[{i}] (from report "{c["report_title"]}", chemical: {c["chemical"]}) {c["text"][:600]}'
        for i, c in enumerate(chunks)
    )
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nExcerpts:\n\n{listing}"},
        ],
        format=GradingResult.model_json_schema(),
        options={"num_ctx": 8192},
    )
    return GradingResult.model_validate_json(response["message"]["content"]).grades


def rewrite_query(query: str) -> str:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        format=RewrittenQuery.model_json_schema(),
        options={"num_ctx": 4096},
    )
    return RewrittenQuery.model_validate_json(response["message"]["content"]).rewritten_query


def retrieve_with_crag(query: str, top_k: int = TOP_K, max_retries: int = 1) -> dict:
    """Returns:
    {
      "chunks": the last attempt's raw retrieved chunks (for retrieval metrics),
      "used_chunks": Correct/Ambiguous-graded chunks to actually generate from,
      "insufficient": True if no "correct" chunk was found even after retry,
      "rewritten_query": the retry query, or None if no retry happened,
      "attempts": how many retrieval attempts were made,
    }
    """
    current_query = query
    rewritten_query = None
    chunks = []

    for attempt in range(max_retries + 1):
        chunks = reranked_retrieve(current_query, top_k=top_k)
        # Always grade against the original question -- a rewritten query is
        # a better *search* string, but relevance is judged against user intent.
        grades = {g.chunk_index: g for g in grade_chunks(query, chunks)}

        correct = [c for i, c in enumerate(chunks) if grades.get(i) and grades[i].verdict == "correct"]
        ambiguous = [c for i, c in enumerate(chunks) if grades.get(i) and grades[i].verdict == "ambiguous"]

        if correct:
            return {
                "chunks": chunks,
                "used_chunks": correct + ambiguous,
                "insufficient": False,
                "rewritten_query": rewritten_query,
                "attempts": attempt + 1,
            }

        if attempt < max_retries:
            current_query = rewrite_query(query)
            rewritten_query = current_query

    return {
        "chunks": chunks,
        "used_chunks": [],
        "insufficient": True,
        "rewritten_query": rewritten_query,
        "attempts": max_retries + 1,
    }
