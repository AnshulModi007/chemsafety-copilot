/**
 * Types mirroring the ChemSafety Copilot backend envelope.
 *
 * Every field here was read off the backend's actual return statements --
 * `src/agent/copilot.py` (the envelope + per-intent `data`), `src/generation/
 * crag.py` (the trace), and `app/main.py` (the SSE event shapes). Nothing is
 * assumed: fields the backend only sets on some paths are optional here, and
 * the UI degrades rather than inventing them.
 */

export const INTENTS = [
  'historical',
  'comparative',
  'chemical_property',
  'calculation',
  'general_knowledge',
] as const;

export type Intent = (typeof INTENTS)[number];

/** Where an answer's grounding came from. `insufficient` means CRAG declined. */
export type AnswerSource = 'internal' | 'web' | 'insufficient' | 'general_knowledge';

export interface ReportCitation {
  report_id: string;
  page: number;
}

export interface WebCitation {
  title: string;
  url: string;
}

export type Citation = ReportCitation | WebCitation;

export function isWebCitation(citation: Citation): citation is WebCitation {
  return 'url' in citation;
}

/**
 * A chunk that actually grounded the answer, with citation-grade metadata.
 * `rerank_score` is null whenever the backend runs with ENABLE_RERANKER=false
 * -- there is no cross-encoder pass to score with, so the UI shows no score
 * rather than a misleading zero.
 */
export interface Source {
  chunk_id: string;
  report_id: string;
  report_title: string;
  section: string;
  chemical: string | null;
  // Chroma metadata round-trips these as numbers, but chunks indexed before a
  // reingest can still carry the original strings -- accept both.
  year: string | number | null;
  page_start: string | number;
  page_end: string | number;
  rerank_score: number | null;
  snippet: string;
}

export type GradeVerdict = 'correct' | 'ambiguous' | 'incorrect';

export interface TraceChunk {
  chunk_id: string;
  report_title: string;
  section: string;
  rerank_score: number | null;
  verdict: GradeVerdict | null;
  reason: string | null;
}

export interface TraceAttempt {
  attempt: number;
  query_used: string;
  retrieval_method: 'expansion' | 'plain_rerank';
  expansion_queries?: string[];
  hyde_passage?: string | null;
  path: 'fast_path' | 'graded';
  chunks: TraceChunk[];
}

/** Comparative queries nest one attempt list per decomposed sub-question. */
export interface SubQueryTrace {
  sub_query: string;
  attempts: TraceAttempt[];
}

export type Trace = TraceAttempt[] | SubQueryTrace[];

export function isSubQueryTrace(trace: Trace): trace is SubQueryTrace[] {
  return trace.length > 0 && 'sub_query' in trace[0]!;
}

export interface Faithfulness {
  faithful: boolean;
  unsupported_claims: string[];
}

export type DiagramKind = 'bowtie' | 'causal_chain' | 'side_by_side';

export interface Diagram {
  kind: DiagramKind;
  svg: string;
}

export interface OrificeRecommendation {
  designation: string;
  area_in2: number;
}

export interface PsvInputs {
  mass_flow_lb_hr: number;
  molecular_weight: number;
  relieving_temp_rankine: number;
  set_pressure_psig: number;
  k: number;
  compressibility_z: number;
  overpressure_fraction: number;
  kd: number;
  kb: number;
  kc: number;
}

/**
 * The `data` object is intent-dependent. Rather than a discriminated union
 * (the backend does not tag `data` itself, and several fields are shared
 * across intents), this is a single optional-field shape: every consumer
 * checks for what it needs. Each field is annotated with which intents set it.
 */
export interface AnswerData {
  // historical | comparative
  citations?: Citation[];
  sources?: Source[];
  retrieved_chunks?: string[];
  crag_insufficient?: boolean;
  crag_rewritten_query?: string | null;
  sub_queries?: string[];
  source?: AnswerSource;
  confidence?: number;
  trace?: Trace;
  faithfulness?: Faithfulness | null;
  diagram?: Diagram | null;

  // chemical_property (PubChem)
  cid?: number;
  pubchem_url?: string;
  iupac_name?: string | null;
  molecular_formula?: string | null;
  molecular_weight?: string | number | null;
  canonical_smiles?: string | null;
  xlogp?: number | null;
  tpsa?: number | null;
  h_bond_donor_count?: number | null;
  h_bond_acceptor_count?: number | null;
  ghs_hazard_statements?: string[];
  ghs_diagram_svg?: string | null;
  pubchem_unavailable?: boolean;

  // calculation (API 520 PSV sizing)
  inputs?: PsvInputs;
  intermediate?: Record<string, number>;
  required_area_in2?: number;
  recommended_orifice?: OrificeRecommendation | null;
  warnings?: string[];
  disclaimer?: string;
  missing_required_fields?: string[];
  invalid_input?: string;

  // calculation | general_knowledge both use this flat field name
  diagram_svg?: string | null;
}

/** The envelope returned by POST /ask and carried on the SSE `done` event. */
export interface AskResponse {
  query: string;
  resolved_query: string | null;
  intent: Intent;
  routing_reasoning: string;
  from_cache: boolean;
  answer: string;
  data: AnswerData;
}

// --- SSE events (app/main.py::ask_stream) -----------------------------------

export interface RoutingEvent {
  type: 'routing';
  intent: Intent;
  reasoning: string;
}

export interface DeltaEvent {
  type: 'delta';
  text: string;
}

export type DoneEvent = { type: 'done' } & AskResponse;

export interface ErrorEvent {
  type: 'error';
  detail: string;
}

export type StreamEvent = RoutingEvent | DeltaEvent | DoneEvent | ErrorEvent;

// --- Requests ---------------------------------------------------------------

export interface HistoryTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface AskRequest {
  query: string;
  history: HistoryTurn[];
}

export interface FeedbackRequest {
  query: string;
  resolved_query: string | null;
  intent: Intent | null;
  answer: string;
  rating: 'up' | 'down';
  comment?: string;
}
