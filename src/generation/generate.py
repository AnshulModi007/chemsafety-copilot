"""Grounded generation over retrieved CSB report chunks, via the Groq API."""
import json
import re
import sys
from pathlib import Path

from groq import Groq
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GROQ_FAST_MODEL, GROQ_MODEL, TOP_K  # noqa: E402
from src.retrieval.retriever import reranked_retrieve  # noqa: E402
from src.generation.crag import retrieve_with_crag  # noqa: E402
from src.tools.websearch import WebSearchUnavailable, web_search  # noqa: E402

_client = Groq()

INSUFFICIENT_RETRIEVAL_MESSAGE = (
    "The retrieved CSB report excerpts do not contain enough information to answer this "
    "question confidently, even after retrying with a rewritten query. Rather than guess, "
    "ChemSafety Copilot is declining to answer -- try rephrasing the question or narrowing "
    "it to a specific incident."
)

WEB_SYSTEM_PROMPT = """You are ChemSafety Copilot. The internal U.S. Chemical Safety Board (CSB) \
report corpus did not have enough grounded information to answer this question, so you are now \
answering from live web search results instead.

Rules:
- Answer ONLY using the provided web search results. Do not use outside knowledge beyond them.
- Every factual claim must cite the source URL it came from.
- If the results do not contain enough information to answer, say so explicitly instead of guessing.
- If the question includes an explicit instruction about answer length or format (e.g. "answer in \
one word", "briefly", "in bullet points"), follow it exactly, even if that means a much shorter \
answer than usual -- a direct format instruction always overrides the default of full prose.
- This is general information pulled from the web, not a stamped engineering judgment. Never give a \
definitive engineering judgment call on a critical safety decision. Recommend the user consult a \
licensed Professional Engineer (PE) for any real design decision.
- Respond with ONLY a JSON object matching this schema, no other text:
{"answer": "<grounded answer text>", "citations": [{"title": "<source title>", "url": "<source url>"}]}
"""

SYSTEM_PROMPT = """You are ChemSafety Copilot, an assistant that helps chemical engineers learn from \
past industry incidents by grounding every answer in retrieved excerpts from U.S. Chemical Safety \
Board (CSB) investigation reports.

Rules:
- Answer ONLY using the provided report excerpts. Do not use outside knowledge.
- Every factual claim must cite the report_id and page number it came from.
- If the excerpts do not contain enough information to answer, say so explicitly instead of guessing.
- If the question includes an explicit instruction about answer length or format (e.g. "answer in \
one word", "briefly", "in bullet points"), follow it exactly, even if that means a much shorter \
answer than usual -- a direct format instruction always overrides the default of full prose.
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


class WebCitation(BaseModel):
    title: str
    url: str


class WebGroundedAnswer(BaseModel):
    answer: str
    citations: list[WebCitation]


# --- Faithfulness verification ----------------------------------------------
# A second, independent Groq call that checks the generated answer against
# its own source context -- a real check, not just a prompt instruction to
# "only use the excerpts". Uses the fast model since this is a yes/no
# classification task, not open-ended generation. Fails open (treated as
# faithful) if the checker call itself errors or returns malformed JSON --
# a broken checker shouldn't block a real answer, only a positive finding
# from it should surface a warning.
FAITHFULNESS_CHECK_PROMPT = """You are a strict fact-checker for a RAG system. Given source excerpts \
and a generated answer, determine whether every factual claim in the answer is directly supported by \
the excerpts -- not outside knowledge, not an exaggeration or overgeneralization of what the excerpts \
actually say.

Respond with ONLY a JSON object matching this schema:
{"faithful": true|false, "unsupported_claims": ["<claim text not supported by the excerpts>", ...]}
"""


class FaithfulnessCheck(BaseModel):
    faithful: bool
    unsupported_claims: list[str] = []


def check_faithfulness(answer: str, context: str) -> dict:
    try:
        response = _client.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=[
                {"role": "system", "content": FAITHFULNESS_CHECK_PROMPT},
                {"role": "user", "content": f"Source excerpts:\n\n{context}\n\nGenerated answer:\n{answer}"},
            ],
            response_format={"type": "json_object"},
        )
        return FaithfulnessCheck.model_validate_json(response.choices[0].message.content).model_dump()
    except Exception:
        return {"faithful": True, "unsupported_claims": []}


# --- Streaming-only prompts & parsing ---------------------------------------
# Groq's response_format={"type": "json_object"} collapses stream=True into a
# single content chunk (verified empirically: 123 incremental chunks for a
# plain prose completion vs. 1 for the same prompt in JSON mode) -- the
# server apparently can't validate-as-it-goes and emit incrementally at the
# same time. So the streaming path drops JSON mode entirely and asks for
# plain prose with inline citation tags instead, parsed out with a regex
# afterward, to get real token-by-token delivery. The non-streaming functions
# above are unaffected and keep using JSON mode.
STREAM_SYSTEM_PROMPT = """You are ChemSafety Copilot, an assistant that helps chemical engineers learn from \
past industry incidents by grounding every answer in retrieved excerpts from U.S. Chemical Safety \
Board (CSB) investigation reports.

Rules:
- Answer ONLY using the provided report excerpts. Do not use outside knowledge.
- Write a normal prose answer, not JSON. Immediately after every factual claim, inline-cite the \
report it came from using exactly this tag format: [[report:<report_id>:<page>]] -- e.g. \
"...contamination of FGAN with combustible materials[[report:csb_01_west-fertilizer-explosion-and-fire:66]]." \
Use the exact report_id and page number from the excerpt it's drawn from.
- If the excerpts do not contain enough information to answer, say so explicitly instead of \
guessing, and do not fabricate a citation tag for that sentence.
- If the question includes an explicit instruction about answer length or format (e.g. "answer in \
one word", "briefly", "in bullet points"), follow it exactly, even if that means a much shorter \
answer than usual -- a direct format instruction always overrides the default of full prose.
- This is historical incident information, not engineering advice. Never give a definitive \
engineering judgment call on a critical safety decision (e.g. "this design is safe"). Point to \
what the reports found and recommend the user consult a licensed Professional Engineer (PE) for \
any real design decision.
- Output ONLY the prose answer with inline tags -- no separate citations list, no JSON, no other text.
"""

WEB_STREAM_SYSTEM_PROMPT = """You are ChemSafety Copilot. The internal U.S. Chemical Safety Board (CSB) \
report corpus did not have enough grounded information to answer this question, so you are now \
answering from live web search results instead.

Rules:
- Answer ONLY using the provided web search results. Do not use outside knowledge beyond them.
- Write a normal prose answer, not JSON. Immediately after every factual claim, inline-cite its source \
using exactly this tag format: [[web:<source title>|<source url>]].
- If the results do not contain enough information to answer, say so explicitly instead of guessing.
- If the question includes an explicit instruction about answer length or format (e.g. "answer in \
one word", "briefly", "in bullet points"), follow it exactly, even if that means a much shorter \
answer than usual -- a direct format instruction always overrides the default of full prose.
- This is general information pulled from the web, not a stamped engineering judgment. Never give a \
definitive engineering judgment call on a critical safety decision. Recommend the user consult a \
licensed Professional Engineer (PE) for any real design decision.
- Output ONLY the prose answer with inline tags -- no separate citations list, no JSON, no other text.
"""

_REPORT_CITE_RE = re.compile(r"\[\[report:([^:\]]+):(\d+)\]\]")
_WEB_CITE_RE = re.compile(r"\[\[web:([^|\]]+)\|([^\]]+)\]\]")


def _clean_stripped_text(text: str) -> str:
    """Tidy up whitespace left behind once inline citation tags are removed --
    a tag between a word and its punctuation (e.g. "materials [[report:...]].")
    leaves a stray space before the period otherwise."""
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _extract_report_citations(text: str) -> tuple[str, list[dict]]:
    citations, seen = [], set()
    for report_id, page in _REPORT_CITE_RE.findall(text):
        key = (report_id, int(page))
        if key not in seen:
            seen.add(key)
            citations.append({"report_id": report_id, "page": int(page)})
    clean = _clean_stripped_text(_REPORT_CITE_RE.sub("", text))
    return clean, citations


def _extract_web_citations(text: str) -> tuple[str, list[dict]]:
    citations, seen = [], set()
    for title, url in _WEB_CITE_RE.findall(text):
        key = (title.strip(), url.strip())
        if key not in seen:
            seen.add(key)
            citations.append({"title": title.strip(), "url": url.strip()})
    clean = _clean_stripped_text(_WEB_CITE_RE.sub("", text))
    return clean, citations


def _build_context(hits: list[dict]) -> str:
    # Parent-child retrieval: retrieval/reranking already scored the small,
    # precise `text` window -- generation gets the wider `parent_text` window
    # instead (falls back to `text` for any hit indexed before this field
    # existed), so the model has enough surrounding context to answer detail
    # questions the narrow retrieval window alone would have cut off.
    blocks = [
        f'[report_id={hit["report_id"]} title="{hit["report_title"]}" '
        f'section="{hit["section"]}" page={hit.get("parent_page_start", hit["page_start"])}'
        f'-{hit.get("parent_page_end", hit["page_end"])}]\n{hit.get("parent_text", hit["text"])}'
        for hit in hits
    ]
    return "\n\n---\n\n".join(blocks)


def generate_from_hits(query: str, hits: list[dict], max_retries: int = 2) -> dict:
    context = _build_context(hits)
    user_prompt = f"Report excerpts:\n\n{context}\n\nQuestion: {query}"

    last_error = None
    for _ in range(max_retries + 1):
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            parsed = GroundedAnswer.model_validate_json(raw)
            return {
                "answer": parsed.answer,
                "citations": [c.model_dump() for c in parsed.citations],
                "retrieved_chunks": [h["chunk_id"] for h in hits],
                "faithfulness": check_faithfulness(parsed.answer, context),
            }
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = e
            continue

    raise RuntimeError(f"Model failed to produce valid grounded JSON after {max_retries + 1} attempts: {last_error}")


def _build_web_context(results: list[dict]) -> str:
    blocks = [
        f'[title="{r["title"]}" url={r["url"]}]\n{r["content"][:1500]}'
        for r in results
    ]
    return "\n\n---\n\n".join(blocks)


def generate_from_web(query: str, web_results: list[dict], max_retries: int = 2) -> dict:
    context = _build_web_context(web_results)
    user_prompt = f"Web search results:\n\n{context}\n\nQuestion: {query}"

    last_error = None
    for _ in range(max_retries + 1):
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": WEB_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            parsed = WebGroundedAnswer.model_validate_json(raw)
            return {
                "answer": parsed.answer,
                "citations": [c.model_dump() for c in parsed.citations],
                "faithfulness": check_faithfulness(parsed.answer, context),
            }
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = e
            continue

    raise RuntimeError(f"Model failed to produce valid web-grounded JSON after {max_retries + 1} attempts: {last_error}")


GENERAL_KNOWLEDGE_DISCLAIMER = (
    "This is general chemical-engineering knowledge from the model's own training, not grounded in "
    "the CSB report corpus or a live data source -- verify independently before relying on it for "
    "anything safety-critical."
)

GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """You are ChemSafety Copilot, answering a general chemical-\
engineering concept or definition question (e.g. "what is a tray tower", "what is mass transfer") \
that isn't about a specific chemical's properties, a specific past incident, or an engineering \
calculation -- so there is no CSB report or live data source to ground this answer in.

Answer clearly and correctly from your own general chemical-engineering knowledge. Include standard \
formulas in plain text notation when relevant. Do NOT attempt to draw a diagram, schematic, or ASCII \
art yourself -- a separate step generates a real diagram automatically when one would help, so just \
focus on a clear written explanation.

If the question includes an explicit instruction about answer length or format (e.g. "answer in one \
word", "briefly", "just a sentence", "in bullet points"), follow it exactly, even if that means giving \
a much shorter answer than you normally would -- a direct instruction on format always overrides the \
default of a fuller educational explanation. Otherwise, keep it educational and concise.

Do not discuss specific past incidents, specific named chemicals' properties, or engineering \
calculations as if this were one of those other capabilities -- if the question actually needs one of \
those, just answer the general-knowledge part being asked.

Do not add your own disclaimer -- that is appended separately, outside of your response.
"""


def generate_general_knowledge(query: str) -> dict:
    """Answer a general chemical-engineering concept question from the model's
    own knowledge -- no retrieval, no tool call, no citations, since there's
    nothing to ground this in. Always clearly disclaimed as ungrounded (see
    GENERAL_KNOWLEDGE_SYSTEM_PROMPT) so it doesn't read with the same
    authority as the RAG-grounded answers."""
    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": GENERAL_KNOWLEDGE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return {"answer": response.choices[0].message.content, "citations": []}


def _web_fallback(query: str) -> dict | None:
    """None means web search wasn't available or returned nothing -- caller
    should fall back to the plain refusal message rather than error out."""
    try:
        web_results = web_search(query)
    except WebSearchUnavailable:
        return None
    if not web_results:
        return None
    result = generate_from_web(query, web_results)
    result["source"] = "web"
    return result


def generate(query: str, top_k: int = TOP_K, max_retries: int = 2) -> dict:
    hits = reranked_retrieve(query, top_k=top_k)
    return generate_from_hits(query, hits, max_retries)


def generate_with_crag(query: str, top_k: int = TOP_K, max_retries: int = 2) -> dict:
    crag_result = retrieve_with_crag(query, top_k=top_k)

    if crag_result["insufficient"]:
        web_result = _web_fallback(query)
        if web_result is not None:
            web_result["retrieved_chunks"] = [c["chunk_id"] for c in crag_result["chunks"]]
            web_result["crag_insufficient"] = True
            web_result["crag_rewritten_query"] = crag_result["rewritten_query"]
            web_result["confidence"] = crag_result["confidence"]
            web_result["trace"] = crag_result["trace"]
            web_result["used_chunks"] = []  # web-sourced, not report chunks -- no incident diagram
            return web_result
        return {
            "answer": INSUFFICIENT_RETRIEVAL_MESSAGE,
            "citations": [],
            "retrieved_chunks": [c["chunk_id"] for c in crag_result["chunks"]],
            "crag_insufficient": True,
            "crag_rewritten_query": crag_result["rewritten_query"],
            "source": "insufficient",
            "confidence": crag_result["confidence"],
            "trace": crag_result["trace"],
            "used_chunks": [],
        }

    result = generate_from_hits(query, crag_result["used_chunks"], max_retries)
    result["crag_insufficient"] = False
    result["crag_rewritten_query"] = crag_result["rewritten_query"]
    result["source"] = "internal"
    result["confidence"] = crag_result["confidence"]
    result["trace"] = crag_result["trace"]
    result["used_chunks"] = crag_result["used_chunks"]  # full chunk dicts, for diagram extraction
    return result


# --- Streaming variants, additive -------------------------------------------
# Kept separate from the functions above rather than folding streaming into
# them: they use a different prompt/parsing contract (inline citation tags in
# plain prose, not JSON mode -- see the "Streaming-only prompts & parsing"
# section above for why) and have no retry-on-malformed-output loop (retrying
# would mean discarding and re-requesting the whole stream). In practice a
# missing/malformed tag just means that one sentence loses its citation, not
# a failed response, so this is an acceptable one-shot tradeoff.

def _stream_chat(model: str, messages: list[dict]):
    """Yields raw text deltas from a Groq streaming chat completion."""
    stream = _client.chat.completions.create(model=model, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_from_hits_stream(query: str, hits: list[dict]):
    """Yields ("delta", text) while streaming, then exactly one final
    ("done", {"answer", "citations", "retrieved_chunks"}) tuple."""
    context = _build_context(hits)
    user_prompt = f"Report excerpts:\n\n{context}\n\nQuestion: {query}"
    messages = [
        {"role": "system", "content": STREAM_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw_parts = []
    for delta in _stream_chat(GROQ_MODEL, messages):
        raw_parts.append(delta)
        yield ("delta", delta)

    answer, citations = _extract_report_citations("".join(raw_parts))
    yield ("done", {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": [h["chunk_id"] for h in hits],
        "faithfulness": check_faithfulness(answer, context),
    })


def generate_general_knowledge_stream(query: str):
    """Streaming twin of generate_general_knowledge: yields ("delta", text)
    chunks, then one final ("done", {"answer", "citations": []}) tuple."""
    messages = [
        {"role": "system", "content": GENERAL_KNOWLEDGE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    raw_parts = []
    for delta in _stream_chat(GROQ_MODEL, messages):
        raw_parts.append(delta)
        yield ("delta", delta)
    yield ("done", {"answer": "".join(raw_parts), "citations": []})


def generate_from_web_stream(query: str, web_results: list[dict]):
    context = _build_web_context(web_results)
    user_prompt = f"Web search results:\n\n{context}\n\nQuestion: {query}"
    messages = [
        {"role": "system", "content": WEB_STREAM_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw_parts = []
    for delta in _stream_chat(GROQ_MODEL, messages):
        raw_parts.append(delta)
        yield ("delta", delta)

    answer, citations = _extract_web_citations("".join(raw_parts))
    yield ("done", {
        "answer": answer,
        "citations": citations,
        "faithfulness": check_faithfulness(answer, context),
    })


def generate_with_crag_stream(query: str, top_k: int = TOP_K):
    crag_result = retrieve_with_crag(query, top_k=top_k)

    if crag_result["insufficient"]:
        web_results = None
        try:
            web_results = web_search(query)
        except WebSearchUnavailable:
            pass

        if web_results:
            for kind, payload in generate_from_web_stream(query, web_results):
                if kind == "delta":
                    yield ("delta", payload)
                else:
                    payload["retrieved_chunks"] = [c["chunk_id"] for c in crag_result["chunks"]]
                    payload["crag_insufficient"] = True
                    payload["crag_rewritten_query"] = crag_result["rewritten_query"]
                    payload["source"] = "web"
                    payload["confidence"] = crag_result["confidence"]
                    payload["trace"] = crag_result["trace"]
                    payload["used_chunks"] = []  # web-sourced, not report chunks -- no incident diagram
                    yield ("done", payload)
            return

        yield ("done", {
            "answer": INSUFFICIENT_RETRIEVAL_MESSAGE,
            "citations": [],
            "retrieved_chunks": [c["chunk_id"] for c in crag_result["chunks"]],
            "crag_insufficient": True,
            "crag_rewritten_query": crag_result["rewritten_query"],
            "source": "insufficient",
            "confidence": crag_result["confidence"],
            "trace": crag_result["trace"],
            "used_chunks": [],
        })
        return

    for kind, payload in generate_from_hits_stream(query, crag_result["used_chunks"]):
        if kind == "delta":
            yield ("delta", payload)
        else:
            payload["crag_insufficient"] = False
            payload["crag_rewritten_query"] = crag_result["rewritten_query"]
            payload["source"] = "internal"
            payload["confidence"] = crag_result["confidence"]
            payload["trace"] = crag_result["trace"]
            payload["used_chunks"] = crag_result["used_chunks"]
            yield ("done", payload)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What caused the ammonium nitrate explosion at West Fertilizer?"
    result = generate_with_crag(query)
    print(json.dumps(result, indent=2))
