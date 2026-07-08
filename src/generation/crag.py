"""Corrective RAG (CRAG): grade retrieved chunks for relevance to the query,
rewrite-and-retry the query once on weak retrieval, and signal when even a
retry can't find correct evidence so the caller can refuse rather than
hallucinate an answer.
"""
import json
import re
import sys
from pathlib import Path
from typing import Literal

from groq import Groq
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GROQ_FAST_MODEL, PROCESSED_DIR, TOP_K  # noqa: E402
from src.retrieval.retriever import reranked_retrieve  # noqa: E402
from src.retrieval.query_expansion import retrieve_with_expansion  # noqa: E402

_client = Groq()

# Deterministic backstop for failure gallery case #3: the LLM grader, even
# shown report_title/chemical metadata, still rates a wrong-incident chunk
# "correct"/"ambiguous" on superficial topical similarity (both "explosions",
# both "root cause") when a query names a specific incident that's actually
# out of corpus. A cross-encoder score threshold doesn't fix this either --
# tested empirically against the documented failure query and the reranker
# gave the single most-wrong chunk (West Fertilizer, unrelated ammonium
# nitrate explosion) a HIGHER score (0.82) than several genuinely-different-
# incident chunks -- so score alone can't separate "same incident" from
# "different incident, similar language". Instead, enforce the grader
# prompt's own stated rule in code: if the question names a specific
# incident/facility, an excerpt from a report that shares no real vocabulary
# with that name is incorrect, no matter what the LLM says.
_GENERIC_ENTITY_WORDS = {
    "explosion", "explosions", "fire", "fires", "release", "releases",
    "toxic", "chemical", "chemicals", "company", "co", "inc", "llc", "corp",
    "corporation", "facility", "facilities", "refinery", "plant", "tank",
    "vessel", "vessels", "dust", "fatal", "pressure", "industries",
    "services", "processing", "manufacturing", "waste", "and", "of", "the",
    "a", "an", "in", "at", "for", "on",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z&.-]*")


def _entity_words(text: str) -> set[str]:
    words = {w.lower().strip("&.-") for w in _WORD_RE.findall(text)}
    return {w for w in words if w and w not in _GENERIC_ENTITY_WORDS and len(w) > 1}


def _query_entity_phrases(query: str) -> list[set[str]]:
    """Capitalized word-runs in the query, each reduced to its entity words.
    Skips the first word since nearly every question here starts with a
    capitalized interrogative ("What", "How", "Who", ...) that's capitalized
    purely from sentence position, not because it names anything. An empty
    return means the query doesn't appear to name a specific incident/
    facility/company at all, so the backstop shouldn't apply -- a generic
    conceptual question is fair game for ordinary topical grading.
    """
    tokens = query.split()
    phrases = []
    current = []
    for tok in tokens[1:]:
        core = tok.strip(".,?!;:'\"()")
        if core[:1].isupper():
            current.append(core)
        else:
            if current:
                phrases.append(_entity_words(" ".join(current)))
            current = []
    if current:
        phrases.append(_entity_words(" ".join(current)))
    return [p for p in phrases if p]


# A word from a chunk's own EXCERPT TEXT (as opposed to its short, curated
# report_title/chemical fields) only counts as grounding evidence if it's
# rare across the corpus. Long excerpts contain enough incidental vocabulary
# that common words collide by chance -- e.g. "Texas" turns up in most CSB
# reports' regulatory/location asides (measured: 17 of 20 reports), so its
# presence in some unrelated report's text says nothing about whether that
# report is the one actually named in the query. Distinctive terms like a
# specific product code ("XL 10") or chemical ("KOH") appear in far fewer
# reports (measured: 2 of 20) and are a real signal.
_TEXT_DISTINCTIVENESS_DF_MAX = 3
_report_word_doc_freq: dict[str, int] | None = None


def _get_doc_freq() -> dict[str, int]:
    global _report_word_doc_freq
    if _report_word_doc_freq is None:
        chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text())
        words_by_report: dict[str, set[str]] = {}
        for c in chunks:
            words_by_report.setdefault(c["report_id"], set()).update(_entity_words(c["text"]))
        counts: dict[str, int] = {}
        for words in words_by_report.values():
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        _report_word_doc_freq = counts
    return _report_word_doc_freq


def _distinctive_text_words(text: str) -> set[str]:
    doc_freq = _get_doc_freq()
    return {w for w in _entity_words(text) if doc_freq.get(w, 0) <= _TEXT_DISTINCTIVENESS_DF_MAX}


def _grounds_to_chunk(query: str, chunk: dict) -> bool:
    """False only when the query names a specific incident/facility/product
    AND neither this chunk's own report metadata nor its actual excerpt text
    shares any of that name's (corpus-rare) vocabulary -- i.e. the chunk is
    from a report about a different named incident than the one actually
    asked about. Checking excerpt text too (not just report_title/chemical)
    matters because some golden questions name a specific chemical/product
    (e.g. "XL 10", "KOH") rather than the facility -- that vocabulary won't
    be in the title, but it will be in a genuinely relevant chunk's own text.
    """
    query_phrases = _query_entity_phrases(query)
    if not query_phrases:
        return True
    report_words = (
        _entity_words(chunk["report_title"])
        | _entity_words(chunk.get("chemical", ""))
        | _distinctive_text_words(chunk.get("text", ""))
    )
    return any(phrase & report_words for phrase in query_phrases)

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
    response = _client.chat.completions.create(
        model=GROQ_FAST_MODEL,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nExcerpts:\n\n{listing}"},
        ],
        response_format={"type": "json_object"},
    )
    return GradingResult.model_validate_json(response.choices[0].message.content).grades


def rewrite_query(query: str) -> str:
    response = _client.chat.completions.create(
        model=GROQ_FAST_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
    )
    return RewrittenQuery.model_validate_json(response.choices[0].message.content).rewritten_query


# Skip the LLM grading call entirely when a chunk is unambiguously relevant:
# cross-encoder rerank score at or above this AND it passes the same
# deterministic entity-grounding check used to override the grader. Measured
# against the golden set: correct-answer chunks score 0.836-1.0 in all but
# one retrieval miss; this sits above that whole cluster except the single
# lowest case (which just falls through to LLM grading as before, no
# correctness cost -- it only forgoes the latency win for that one query).
# Score alone isn't a safe signal on its own -- the documented false-positive
# chunk from the failure gallery's out-of-corpus query scored 0.82, comfortably
# under this threshold, but the real backstop is _grounds_to_chunk: a
# wrong-incident chunk is blocked from the fast path regardless of score.
_HIGH_CONFIDENCE_RERANK_SCORE = 0.85


def _confidence(chunks: list[dict]) -> float:
    """Mean cross-encoder rerank score of the chunks actually used for
    generation -- a simple, honest proxy for how strongly retrieval matched
    the question, independent of which CRAG path (fast/graded/insufficient)
    produced them. 0.0 when nothing was used.
    """
    scores = [c.get("rerank_score", 0.0) for c in chunks]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _chunk_trace(c: dict, grade=None) -> dict:
    return {
        "chunk_id": c["chunk_id"],
        "report_title": c["report_title"],
        "section": c["section"],
        "rerank_score": c.get("rerank_score"),
        "verdict": grade.verdict if grade else None,
        "reason": grade.reason if grade else None,
    }


def retrieve_with_crag(query: str, top_k: int = TOP_K, max_retries: int = 1) -> dict:
    """Returns:
    {
      "chunks": the last attempt's raw retrieved chunks (for retrieval metrics),
      "used_chunks": Correct/Ambiguous-graded chunks to actually generate from,
      "insufficient": True if no "correct" chunk was found even after retry,
      "rewritten_query": the retry query, or None if no retry happened,
      "attempts": how many retrieval attempts were made,
      "confidence": mean rerank score of used_chunks, 0.0 if insufficient,
      "trace": [{"attempt", "query_used", "retrieval_method", "expansion_queries",
                 "hyde_passage", "path", "chunks": [...]}, ...] -- full debug
                trace of every attempt, for the UI's "under the hood" view.
    }
    """
    current_query = query
    rewritten_query = None
    chunks = []
    trace = []

    for attempt in range(max_retries + 1):
        record = {"attempt": attempt + 1, "query_used": current_query}

        # Query expansion (HyDE + multi-query + step-back) only on the first
        # attempt -- it's a recall booster for the original phrasing. A retry
        # already targets better vocabulary via rewrite_query, so it uses
        # plain reranked_retrieve to keep the retry bounded in latency/cost.
        if attempt == 0:
            expansion_meta = {}
            chunks = retrieve_with_expansion(current_query, top_k=top_k, trace=expansion_meta)
            record["retrieval_method"] = "expansion"
            record["expansion_queries"] = expansion_meta.get("expansion_queries", [])
            record["hyde_passage"] = expansion_meta.get("hyde_passage")
        else:
            chunks = reranked_retrieve(current_query, top_k=top_k)
            record["retrieval_method"] = "plain_rerank"

        confident = [
            c for c in chunks
            if c.get("rerank_score", 0) >= _HIGH_CONFIDENCE_RERANK_SCORE and _grounds_to_chunk(query, c)
        ]
        if confident:
            record["path"] = "fast_path"
            record["chunks"] = [_chunk_trace(c) for c in chunks]
            trace.append(record)
            return {
                "chunks": chunks,
                "used_chunks": confident,
                "insufficient": False,
                "rewritten_query": rewritten_query,
                "attempts": attempt + 1,
                "confidence": _confidence(confident),
                "trace": trace,
            }

        # Always grade against the original question -- a rewritten query is
        # a better *search* string, but relevance is judged against user intent.
        grades = {g.chunk_index: g for g in grade_chunks(query, chunks)}

        for i, c in enumerate(chunks):
            g = grades.get(i)
            if g and g.verdict != "incorrect" and not _grounds_to_chunk(query, c):
                g.reason = f"deterministic override ({g.verdict} -> incorrect): no shared entity with report '{c['report_title']}' -- {g.reason}"
                g.verdict = "incorrect"

        correct = [c for i, c in enumerate(chunks) if grades.get(i) and grades[i].verdict == "correct"]
        ambiguous = [c for i, c in enumerate(chunks) if grades.get(i) and grades[i].verdict == "ambiguous"]

        record["path"] = "graded"
        record["chunks"] = [_chunk_trace(c, grades.get(i)) for i, c in enumerate(chunks)]
        trace.append(record)

        if correct:
            return {
                "chunks": chunks,
                "used_chunks": correct + ambiguous,
                "insufficient": False,
                "rewritten_query": rewritten_query,
                "attempts": attempt + 1,
                "confidence": _confidence(correct),
                "trace": trace,
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
        "confidence": 0.0,
        "trace": trace,
    }
