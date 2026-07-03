"""Agentic router: classify a user query's intent and extract whatever
structured arguments the downstream tool needs, via the same local-LLM
structured-decoding pattern used by src/generation/crag.py.

Intents:
  historical         -- precedent/root-cause questions -> RAG (+ CRAG)
  chemical_property   -- live property lookup for a named chemical -> PubChem
  calculation         -- engineering calc (PSV sizing) -> calculations tool
  comparative         -- multi-incident/multi-hop questions -> per-entity RAG
                         retrieval (see sub_queries below), merged before
                         generation

Note on "comparative": a single retrieval call over a multi-entity question can
starve one entity's chunks out of the top-k pool entirely (observed failure:
asking to compare two incidents retrieved chunks from only one of them, and the
model filled in the other from outside knowledge instead of declining -- see
failure gallery). So comparative queries get decomposed into one focused
sub-query per entity and retrieved separately; see sub_queries on RouteDecision.
"""
import sys
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OLLAMA_MODEL  # noqa: E402

ROUTER_SYSTEM_PROMPT = """You are the intent router for ChemSafety Copilot, an assistant over U.S. \
Chemical Safety Board (CSB) incident reports plus live chemical-data and engineering-calculation tools.

Classify the user's question into exactly one intent:
- "historical": asks about a specific past incident, its root cause, timeline, or recommendations
- "chemical_property": asks for a physical/chemical property (formula, molecular weight, hazard \
classification, etc.) of a named chemical, not tied to a specific incident
- "calculation": asks to size or calculate an engineering quantity (e.g. relief valve / PSV sizing)
- "comparative": asks to compare, contrast, or find commonalities across multiple incidents

If the intent is "chemical_property", extract the chemical's name into chemical_name.

If the intent is "comparative", break the question into one focused, self-contained sub-question per \
incident/entity being compared, into sub_queries (e.g. "Compare X and Y's root causes" -> \
["What was the root cause of X?", "What was the root cause of Y?"]). Otherwise leave sub_queries empty.

Respond with ONLY a JSON object matching this schema:
{"intent": "historical"|"chemical_property"|"calculation"|"comparative", "chemical_name": "<name or null>", \
"sub_queries": [<strings, only for comparative>], "reasoning": "<brief reason>"}
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


class RouteDecision(BaseModel):
    intent: Literal["historical", "chemical_property", "calculation", "comparative"]
    chemical_name: str | None = None
    sub_queries: list[str] = []
    reasoning: str


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


def classify_intent(query: str) -> RouteDecision:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        format=RouteDecision.model_json_schema(),
        options={"num_ctx": 4096},
    )
    return RouteDecision.model_validate_json(response["message"]["content"])


def extract_psv_params(query: str) -> PSVExtraction:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": PSV_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        format=PSVExtraction.model_json_schema(),
        options={"num_ctx": 4096},
    )
    return PSVExtraction.model_validate_json(response["message"]["content"])


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What caused the ammonium nitrate explosion at West Fertilizer?"
    print(classify_intent(query).model_dump_json(indent=2))
