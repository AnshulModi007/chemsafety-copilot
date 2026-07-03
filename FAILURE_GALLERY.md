# Failure Gallery

Real failures found by stress-testing ChemSafety Copilot, the fix applied, and
verification evidence. Each case was reproduced before the fix and re-run
against the identical input after, to confirm the fix actually closed the gap
rather than just looking plausible.

All four cases share a theme: **the local 8B model (llama3.1:8b-instruct) is
unreliable at small conditional-logic and self-bookkeeping tasks** (arithmetic
with a conditional rule, enumerating its own gaps, judging topical relevance
under superficial similarity). The fixes follow one design principle in
response: **push anything checkable into deterministic code, and only ask the
LLM for the part that genuinely requires language understanding.**

---

## 1. Silent unit-conversion errors in calculation parameter extraction

**Query:** "Size a relief valve for a vapor stream at 5000 lb/hr, molecular
weight 44, relieving temperature 600 R, **set pressure 250 psig**"

**Failure:** The extraction step was asked to report `set_pressure_psig`
directly, with a conditional rule ("if the value is given in psia, subtract
14.7 to get psig; if already psig, use as-is"). The model applied the
subtraction anyway, turning 250 psig into ~235-236 psig -- reproduced twice,
including after the prompt was rewritten to be more explicit and add an
"assume no conversion needed" fallback.

**Fix:** Stopped asking the LLM to do conditional arithmetic at all. It now
extracts only a raw `(value, unit)` pair per field (`set_pressure_value: 250,
set_pressure_unit: "psig"`), and the psig/psia conversion is a plain Python
property (`PSVExtraction.set_pressure_psig` in `src/agent/router.py`).
Extraction is now a pure-language task (read the number, read the unit word);
arithmetic is code.

**Before:** `set_pressure_psig: 235.3` (wrong) / `236.0` (wrong, after prompt fix)
**After:** `set_pressure_psig: 250.0` (correct, reproduced on identical input)

---

## 2. Unreliable self-reported "missing fields" list

**Query:** "Size a relief valve for my reactor" (no numeric values at all)

**Failure:** The model correctly left all four required fields `null`, but
its own `missing_required_fields` list omitted `set_pressure_value` --
reporting only 3 of the 4 actually-missing fields. Harmless in this exact
case (the handler short-circuits on any non-empty missing list), but it's a
latent crash risk: if a future caller trusted the list's *completeness*
rather than just its non-emptiness, a silently-omitted null would reach
`calculations.size_psv_vapor()` and blow up in the `sqrt()` arithmetic.

**Fix:** Don't ask the model to self-report what it didn't extract -- we
already have the actual extracted values, so completeness is checked
deterministically (`PSVExtraction.actually_missing_fields` in
`src/agent/router.py` checks each required field for `None` in Python).

**Before:** `missing_required_fields: [mass_flow_lb_hr, molecular_weight, relieving_temp_value]` (3 of 4)
**After:** `[mass_flow_lb_hr, molecular_weight, relieving_temp_value, set_pressure_value]` (correct, all 4)

---

## 3. CRAG grading false-negative on out-of-corpus queries (partially mitigated, not fully resolved)

**Query:** "What caused the Texas A&M bonfire collapse in 1999?" -- not a CSB
chemical incident, not in the corpus at all.

**Failure:** Hybrid retrieval, as always, returned *something* (there's no
"no results" case for top-k similarity search) -- chunks from the Concept
Sciences hydroxylamine explosion and the BP Texas City refinery explosion.
The CRAG grader was expected to mark these "incorrect" (wrong incident
entirely) and trigger the `insufficient` refusal path. Instead it rated at
least one chunk "correct" or "ambiguous" on superficial similarity (both are
"explosions", both discuss "root cause"), so `crag_insufficient` came back
`false` when it should have been `true`.

**Attempted fixes, in order:**
1. Strengthened the grader prompt to explicitly warn against superficial
   topical similarity across different incidents -- **no change**.
2. Added report title + chemical metadata to each candidate chunk shown to
   the grader, turning "is this the same incident?" into a structural
   comparison instead of a prose-reading-comprehension task -- **no change**.

**Current state:** Not resolved at the CRAG-grading level with this model.
However, the *user-facing* behavior is still correct in every trial, because
the generation step has its own independent grounding instruction ("if the
excerpts don't contain enough information, say so explicitly") which caught
what CRAG's grader missed, both before and after the prompt attempts. This is
a real defense-in-depth result, not a coincidence -- but it means
`crag_insufficient` is not a trustworthy signal for anything that reads it
programmatically (observability dashboards, automated eval scoring) until
fixed at the grading layer.

**Recommended next fix (not yet implemented):** add a deterministic backstop
independent of the LLM grader -- e.g. a minimum cross-encoder rerank-score
threshold below which a chunk is forced to "incorrect" regardless of the
grader's verdict, calibrated against the score distribution on known
relevant vs. irrelevant pairs from the golden eval set.

---

## 4. Multi-hop retrieval starves one entity out of a comparative query

**Query:** "Compare the root causes of the Chevron Richmond refinery fire and
the Tesoro Anacortes explosion"

**Failure:** A single retrieval call over a two-entity question returned
chunks from *only one* of the two reports (Tesoro Anacortes) -- Chevron
Richmond was completely absent from `retrieved_chunks`. The generation step
still produced specific, plausible-sounding technical detail about Chevron
("sulfidation corrosion in Chevron's piping circuit") with **no citation** --
i.e. it answered from outside/parametric knowledge rather than declining,
a direct violation of the "answer ONLY from provided excerpts" grounding
rule, and one that citation-checking alone wouldn't catch (the Tesoro half
of the answer *was* validly cited, masking the ungrounded half).

**Root cause:** Reciprocal Rank Fusion ranks a flat pool of candidates by
aggregate score across dense + BM25; nothing about it guarantees per-entity
coverage in a multi-entity query. One incident's chunks scored high enough
to fill the entire top-k pool, crowding out the other's genuinely relevant
(but comparatively lower-scoring) chunks.

**Fix:** Comparative queries are now decomposed at the router step into one
self-contained sub-question per entity (`RouteDecision.sub_queries`, e.g.
"Compare X and Y" -> ["What was the root cause of X?", "What was the root
cause of Y?"]). Each sub-question gets its own full CRAG retrieval pass, and
the results are merged (deduplicated by chunk ID) before a single generation
call synthesizes across both. See `_handle_comparative` in `src/agent/copilot.py`.

**Before:** `retrieved_chunks` = 3 chunks, all from `csb_09_tesoro-...`; unsourced Chevron claims in the answer
**After:** `retrieved_chunks` = 7 chunks spanning both `csb_03_chevron-richmond-...` (3) and `csb_09_tesoro-...` (4); both halves of the answer independently cited to their own report
