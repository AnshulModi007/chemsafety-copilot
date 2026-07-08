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

## 3. CRAG grading false-negative on out-of-corpus queries

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
3. A minimum cross-encoder rerank-score threshold, the originally planned
   next fix -- **tested and rejected before implementing**. Empirically,
   the wrong-incident West Fertilizer chunk scored 0.82 on the reranker for
   this exact query, higher than several genuinely-different-incident
   chunks -- the reranker doesn't reliably separate "same incident" from
   "different incident, similar language", so a global score cutoff can't
   fix this without also rejecting legitimately correct chunks elsewhere.

**Fix:** A deterministic entity-grounding backstop, applied after grading,
in `_grounds_to_chunk` (`src/generation/crag.py`). It operationalizes the
grader prompt's own stated rule in code instead of trusting the LLM to
follow it: if the question names a specific incident/facility/product
(detected via capitalized word-runs in the query, e.g. "Texas A&M", "XL 10"),
and a graded chunk's own report shares none of that name's vocabulary, its
verdict is forced to "incorrect" regardless of what the grader said.
Vocabulary is checked against the chunk's report_title/chemical metadata
(unfiltered -- short and curated, so collisions are rare) plus its own
excerpt text (filtered to corpus-rare words only, doc-frequency <= 3 of 20
reports -- otherwise incidental background vocabulary like "Texas", which
turns up in 17 of 20 reports' regulatory/location asides, would collide by
chance in any sufficiently long excerpt).

**Verification:** Re-ran the exact failing query end-to-end through
`retrieve_with_crag`: `insufficient` now comes back `True` (previously
`False`) after 2 attempts, `used_chunks: []`. Regression-checked against all
22 golden-set questions at the entity-grounding layer (0 regressions) and
re-ran 3 of them (including the "XL 10"/"KOH" case, which needed the
excerpt-text fallback since those terms aren't in the report's title) end to
end through the full CRAG + LLM-grading pipeline -- all still resolve
correctly with `insufficient: False` and the right report's chunks used.

**Known residual limitation:** the two BP Texas City chunks still pass the
entity-grounding check on this query, because "Texas" is a genuine substring
of that report's own title ("Bp America **Texas** City Refinery Explosion")
-- an honest token collision no vocabulary-overlap heuristic can fully
resolve without deeper semantic disambiguation. In this run the LLM grader
itself rated both "incorrect" anyway, so the pipeline still returns
`insufficient: True` overall -- but a grader run where it doesn't would let
those two through. `crag_insufficient` is meaningfully more trustworthy now,
not provably airtight against every entity-name collision.

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

---

## Week 7: failures found through real interactive use

The four cases above came from deliberate stress-testing. These four came from
actually using the deployed app and reading real screenshots of real
responses -- a different failure mode than the Week 4 batch: less "the LLM
reasons unreliably about a specific task" and more "structured-output
brittleness, a scope gap in the router, and a framework-level rendering quirk
that no amount of prompt engineering would have surfaced." None of these
would have been caught by the pytest suite either, since they only show up
against a live model or a live browser render -- which is itself the reason
to keep dogfooding after the test suite is green, not instead of it.

## 5. Router crash on a missing (but non-essential) structured-output field

**Query:** any `chemical_property`-routed question -- not input-dependent,
this was a structured-output reliability issue, reproduced with "what is the
molecular weight of chlorine?"

**Failure:** `RouteDecision.reasoning` (a human-readable "why" string, used
only for a UI debug caption) had no default value in the Pydantic schema.
The fast model occasionally returned otherwise-correct JSON --
`intent: "chemical_property"`, `chemical_name: "chlorine"` both right -- but
omitted just the `reasoning` field. Pydantic validation raised uncaught, and
nothing distinguished "routing logic failed" from "the model forgot an
optional-seeming field," so the whole request surfaced as a raw 500.

**Fix:** gave `reasoning` a default (`""`), and wrapped every structured-decode
call (`classify_intent`, `reformulate_query`, `extract_psv_params`) in a
shared `_structured_call` helper (`src/agent/router.py`) that retries once on
a `ValidationError` before giving up -- a dropped field on an otherwise-good
response reliably recovers on retry, and the default is a backstop for the
cases it doesn't.

**Before:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for RouteDecision / reasoning / Field required`, request fails with a 500.
**After:** re-ran the same query pattern repeatedly; the rare field omission now either recovers on retry or falls back to `reasoning: ""` instead of crashing.

## 6. Misrouting general concept questions into `chemical_property`

**Query:** "can you tell me about tray towers with diagrams and preferred
formulas" / "can you tell me what mass transfer is"

**Failure:** neither question is about a named chemical, a past incident, or
a calculation -- the only four intents that existed at the time. Forced to
pick the least-wrong bucket, the router chose `chemical_property`, which then
correctly found no chemical name to look up and returned "I couldn't tell
which chemical you're asking about" -- technically correct given its inputs,
but a confusing dead end for a legitimate question the app should be able to
answer.

**Fix:** added a 5th intent, `general_knowledge` (`src/agent/router.py`,
`src/generation/generate.py`, `src/agent/copilot.py`), that answers
straight from the model's own knowledge with a clear "not grounded in the
CSB corpus" disclaimer, for exactly the case where none of the other four
intents fit.

**Before:** "I couldn't tell which chemical you're asking about -- could you name it explicitly?"
**After:** both questions route to `general_knowledge` and get a real, substantive answer.

## 7. Auto-generated SVG diagram body silently dropped, only the title rendered

**Query:** any PSV sizing question (diagram always attempted on success).

**Failure:** the PSV cross-section SVG (`src/visualization/psv_schematic.py`)
was built as one triple-quoted f-string with blank lines between commented
sections (bonnet/spring/body/disc/...), purely for source readability. When
rendered via `st.markdown(f'<div>{svg}</div>', unsafe_allow_html=True)`,
Streamlit runs raw HTML through a CommonMark HTML-block parser first -- and a
blank line **terminates** an HTML block. Everything after the first blank
line (the title text was before it; the entire valve drawing was after it)
was silently dropped. `xml.etree.ElementTree.fromstring()` still validated
the raw SVG string as well-formed XML in isolation, which is why this wasn't
caught by `tests/test_diagrams.py` -- the bug only exists in how Streamlit's
Markdown pipeline re-processes the string, not in the SVG itself.

**Fix:** removed the blank lines from the PSV template, and added a
defensive blank-line strip inside `render_diagram()`
(`app/streamlit_app.py`) so no future diagram generator can reintroduce this
by choosing to format its own f-string with blank lines for readability.

**Before:** only "PSV Cross-Section (illustrative)" / "Orifice G (0.503 in²)" visible; the rest of the diagram was blank space.
**After:** the full valve cross-section (bonnet, spring, body, disc, nozzle, inlet/outlet arrows) renders.

## 8. Explicit format/length instructions silently dropped on follow-up questions

**Query:** (as a follow-up, after asking about a tray tower) "give answer in one word or sentence"

**Failure:** two compounding bugs. First, `reformulate_query`'s prompt
(`src/agent/router.py`) only resolved *what* was being asked into a
standalone question -- it dropped the user's instruction on *how* to answer,
so "give answer in one word or sentence" collapsed into a plain content
question with no trace of the formatting request. Second, even had that
survived, `GENERAL_KNOWLEDGE_SYSTEM_PROMPT` (`src/generation/generate.py`)
hardcoded a mandatory verbatim disclaimer sentence the model had to append to
every answer -- structurally making a true one-word/one-sentence response
impossible regardless of prompt tuning.

**Fix:** updated `REFORMULATE_SYSTEM_PROMPT` to explicitly preserve any
format/length instruction when rewriting a follow-up into a standalone
question. Moved the disclaimer out of the model's own required output
entirely -- it's now appended by code after generation (`ask()`/
`stream_ask()` in `src/agent/copilot.py`), the same pattern already used for
the PSV disclaimer, so the model is free to actually honor a brevity
request. Applied the same "honor explicit format instructions" rule to the
shared RAG-grounded prompts too (`SYSTEM_PROMPT`, `STREAM_SYSTEM_PROMPT`,
`WEB_SYSTEM_PROMPT`, `WEB_STREAM_SYSTEM_PROMPT`), after finding the identical
bug class in a comparative-query answer that ignored an explicit "in bullet
points" request.

**Before:** a 3-paragraph explanation despite an explicit "one word or sentence" request; a bullet-point request answered as a plain paragraph.
**After:** `resolved_query` correctly preserves the format instruction, and the model's answer honors it -- verified for both the one-sentence case and the bullet-point case.
