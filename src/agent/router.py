"""Agentic router: classify a user query's intent and extract whatever
structured arguments the downstream tool needs, via the same Groq
structured-decoding pattern used by src/generation/crag.py.

Intents:
  historical         -- precedent/root-cause questions -> RAG (+ CRAG)
  chemical_property   -- live property lookup for a named chemical -> PubChem
  calculation         -- engineering calc (PSV sizing) -> calculations tool
  comparative         -- multi-incident/multi-hop questions -> per-entity RAG
                         retrieval (see sub_queries below), merged before
                         generation
  general_knowledge   -- general chemical-engineering concept/definition
                         questions not tied to a specific incident, named
                         chemical, or calculation -> ungrounded LLM answer,
                         clearly labeled as such (see src/generation/generate.py's
                         generate_general_knowledge)

Note on "general_knowledge": added after observing that questions like "what
is a tray tower" or "what is mass transfer" have no good home in the other
four intents -- they're not about a specific chemical's properties, so the
router was forcing them into "chemical_property", which then correctly
found no chemical name to look up and returned a confusing "couldn't tell
which chemical" refusal instead of actually answering the question.

Note on "comparative": a single retrieval call over a multi-entity question can
starve one entity's chunks out of the top-k pool entirely (observed failure:
asking to compare two incidents retrieved chunks from only one of them, and the
model filled in the other from outside knowledge instead of declining -- see
failure gallery). So comparative queries get decomposed into one focused
sub-query per entity and retrieved separately; see sub_queries on RouteDecision.
"""
import sys
from pathlib import Path
from typing import Literal, TypeVar

from groq import Groq
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GROQ_FAST_MODEL  # noqa: E402

_client = Groq()

_T = TypeVar("_T", bound=BaseModel)

REFORMULATE_SYSTEM_PROMPT = """You resolve follow-up questions into standalone questions for \
ChemSafety Copilot, an assistant over U.S. Chemical Safety Board (CSB) incident reports, live \
chemical-property lookups, and engineering calculations.

Given recent conversation history and a new user message, rewrite the new message into a fully \
self-contained question that makes sense with NO prior context -- resolve pronouns ("it", "that", \
"this incident") and implicit references ("what about its pricing", "and the other one") into the \
actual incident/chemical/entity named earlier in the conversation.

If the new message ALSO includes an instruction about how to format or how long the answer should be \
(e.g. "give answer in one word", "briefly", "just a sentence", "in bullet points"), keep that \
instruction attached to the rewritten standalone question -- do not drop it. E.g. "give answer in one \
word or sentence" following a question about a tray tower's application should resolve to something \
like "What is the application of a tray tower? Answer in one word or a single sentence." not just \
"What is the application of a tray tower?".

If the new message is already self-contained and doesn't depend on anything said earlier, return it \
completely unchanged.

Respond with ONLY a JSON object matching this schema:
{"resolved_query": "<standalone question>"}
"""

ROUTER_SYSTEM_PROMPT = """You are the intent router for ChemSafety Copilot, an assistant over U.S. \
Chemical Safety Board (CSB) incident reports plus live chemical-data and engineering-calculation tools.

Classify the user's question into exactly one intent:
- "historical": asks about a specific past incident, its root cause, timeline, or recommendations
- "chemical_property": asks for a physical/chemical property (formula, molecular weight, hazard \
classification, etc.) of a named chemical, not tied to a specific incident
- "calculation": asks to size or calculate an engineering quantity (e.g. relief valve / PSV sizing)
- "comparative": asks to compare, contrast, or find commonalities across multiple incidents
- "general_knowledge": asks about a general chemical-engineering concept, term, or definition (e.g. \
"what is a tray tower", "what is mass transfer", "explain distillation") that is NOT about a specific \
named chemical's properties, a specific past incident, a calculation, or a comparison -- use this \
whenever none of the other four intents actually fit, rather than forcing it into "chemical_property" \
just because the topic is chemistry-related.

If the intent is "chemical_property", extract the chemical's name into chemical_name.

If the intent is "comparative", break the question into one focused, self-contained sub-question per \
incident/entity being compared, into sub_queries (e.g. "Compare X and Y's root causes" -> \
["What was the root cause of X?", "What was the root cause of Y?"]). Otherwise leave sub_queries empty.

Respond with ONLY a JSON object matching this schema:
{"intent": "historical"|"chemical_property"|"calculation"|"comparative"|"general_knowledge", \
"chemical_name": "<name or null>", "sub_queries": [<strings, only for comparative>], \
"reasoning": "<brief reason>"}
"""

PSV_EXTRACTION_SYSTEM_PROMPT = """Extract pressure relief valve (PSV) sizing parameters for the API 520 \
vapor/gas relief equation from the user's question. Required fields (no sensible default -- must come \
from the question or process data): mass_flow_lb_hr, molecular_weight, relieving_temp_value + \
relieving_temp_unit, set_pressure_value + set_pressure_unit. Optional, with standard defaults if not \
stated: k (ideal-gas specific heat ratio, default 1.4), compressibility_z (default 1.0).

Do NOT convert units yourself -- just report the number exactly as stated and tag which unit it's in. \
Unit conversion is done in code afterward, not by you.
- relieving_temp_unit must be one of: "R" (Rankine), "F" (Fahrenheit), "C" (Celsius).
- set_pressure_unit must be one of: "psig" (gauge, the common case) or "psia" (absolute).
If the question doesn't state a unit for temperature or pressure, assume "R" and "psig" respectively.

List any required field you could not find in missing_required_fields (use the exact field names: \
mass_flow_lb_hr, molecular_weight, relieving_temp_value, set_pressure_value).

Respond with ONLY a JSON object matching this schema:
{"mass_flow_lb_hr": <float or null>, "molecular_weight": <float or null>, \
"relieving_temp_value": <float or null>, "relieving_temp_unit": "R"|"F"|"C", \
"set_pressure_value": <float or null>, "set_pressure_unit": "psig"|"psia", \
"k": <float or null>, "compressibility_z": <float or null>, "missing_required_fields": [<field names>]}
"""


class ResolvedQuery(BaseModel):
    resolved_query: str


class RouteDecision(BaseModel):
    intent: Literal["historical", "chemical_property", "calculation", "comparative", "general_knowledge"]
    chemical_name: str | None = None
    sub_queries: list[str] = []
    # Defaults to "" rather than being required: observed in practice that the
    # fast model occasionally omits this free-text field even though intent
    # classification itself succeeded -- losing the explanation isn't worth
    # crashing the whole request over.
    reasoning: str = ""


class PSVExtraction(BaseModel):
    mass_flow_lb_hr: float | None = None
    molecular_weight: float | None = None
    relieving_temp_value: float | None = None
    relieving_temp_unit: Literal["R", "F", "C"] = "R"
    set_pressure_value: float | None = None
    set_pressure_unit: Literal["psig", "psia"] = "psig"
    k: float | None = None
    compressibility_z: float | None = None
    missing_required_fields: list[str] = []

    @property
    def relieving_temp_rankine(self) -> float | None:
        if self.relieving_temp_value is None:
            return None
        if self.relieving_temp_unit == "R":
            return self.relieving_temp_value
        if self.relieving_temp_unit == "F":
            return self.relieving_temp_value + 459.67
        return (self.relieving_temp_value + 273.15) * 1.8  # C

    @property
    def set_pressure_psig(self) -> float | None:
        if self.set_pressure_value is None:
            return None
        if self.set_pressure_unit == "psig":
            return self.set_pressure_value
        return self.set_pressure_value - 14.7  # psia -> psig

    @property
    def actually_missing_fields(self) -> list[str]:
        """The model's own `missing_required_fields` self-report is unreliable
        (observed: it omitted set_pressure_value when the question gave no
        numbers at all). We already have the real extracted values, so check
        completeness deterministically instead of trusting the model's list.
        """
        required = {
            "mass_flow_lb_hr": self.mass_flow_lb_hr,
            "molecular_weight": self.molecular_weight,
            "relieving_temp_value": self.relieving_temp_value,
            "set_pressure_value": self.set_pressure_value,
        }
        return [name for name, value in required.items() if value is None]


def _structured_call(system_prompt: str, user_content: str, schema: type[_T]) -> _T:
    """Call GROQ_FAST_MODEL in JSON mode and validate the response against
    `schema`, retrying once on a malformed/incomplete response before giving
    up -- observed in practice that this small, fast model occasionally
    returns JSON missing a required field on an otherwise-valid call, and a
    single retry reliably recovers.

    Raises:
        pydantic.ValidationError: if both attempts return an invalid shape.
    """
    last_error: ValidationError | None = None
    for _ in range(2):
        response = _client.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        try:
            return schema.model_validate_json(response.choices[0].message.content)
        except ValidationError as e:
            last_error = e
    raise last_error


def reformulate_query(query: str, history: list[dict]) -> str:
    """Resolve a follow-up ("what about its pricing?") into a standalone
    question using the last few conversation turns, before routing/retrieval
    ever see it. A no-op (returns query unchanged) when there's no history --
    skips the extra LLM call for the common single-turn case.
    """
    if not history:
        return query

    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
    try:
        result = _structured_call(
            REFORMULATE_SYSTEM_PROMPT,
            f"Conversation history:\n{history_text}\n\nNew message: {query}",
            ResolvedQuery,
        )
    except ValidationError:
        return query  # fail soft -- treat the raw message as already self-contained
    return result.resolved_query


def classify_intent(query: str) -> RouteDecision:
    """Classify a (resolved) query's intent via the fast model.

    Returns:
        A validated RouteDecision. Raises pydantic.ValidationError if the
        model's response is malformed on both the initial call and the retry
        (see _structured_call) -- there's no safe default intent to fall
        back to, so this propagates to the caller.
    """
    return _structured_call(ROUTER_SYSTEM_PROMPT, query, RouteDecision)


def extract_psv_params(query: str) -> PSVExtraction:
    """Extract PSV sizing parameters from a calculation query via the fast model."""
    return _structured_call(PSV_EXTRACTION_SYSTEM_PROMPT, query, PSVExtraction)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What caused the ammonium nitrate explosion at West Fertilizer?"
    print(classify_intent(query).model_dump_json(indent=2))
