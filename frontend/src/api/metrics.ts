/**
 * Types and fetcher for the read-only metrics API (app/metrics.py).
 *
 * Every field mirrors a Pydantic response model on the backend. `null` is
 * meaningful in several places and is never collapsed to zero: a null
 * faithfulness pass-rate means "nothing was checked in this window", which the
 * dashboard must not render as a 0% failure.
 */
import { ApiError, apiUrl, readErrorDetail } from './client';

export interface StageLatency {
  stage: string;
  mean_ms: number;
  p95_ms: number;
  /** Fraction of queries that entered this stage at all. */
  share_of_queries: number;
}

export interface LatencySummary {
  sample_size: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  mean_ms: number;
  max_ms: number;
  stages: StageLatency[];
}

export interface ModelCost {
  model: string;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  /** False when the model is missing from the config price table. */
  priced: boolean;
}

export interface CostSummary {
  sample_size: number;
  total_cost_usd: number;
  mean_cost_usd: number;
  total_tokens: number;
  mean_tokens: number;
  mean_llm_calls: number;
  by_model: ModelCost[];
}

export interface QualitySummary {
  sample_size: number;
  faithfulness_checked: number;
  faithfulness_pass_rate: number | null;
  crag_retry_rate: number;
  crag_insufficient_rate: number;
  web_fallback_rate: number;
  cache_hit_rate: number;
  mean_retrieval_confidence: number | null;
  intent_distribution: { intent: string; count: number }[];
}

export interface ReliabilitySummary {
  sample_size: number;
  failures: number;
  error_rate: number;
  by_category: { category: string; error_type: string | null; count: number }[];
}

export interface TimeseriesPoint {
  t: string;
  queries: number;
  failures: number;
  mean_latency_ms: number;
  cost_usd: number;
  faithfulness_pass_rate: number | null;
}

export interface Timeseries {
  bucket_minutes: number;
  points: TimeseriesPoint[];
}

/** Health of the telemetry writer itself — a dashboard that has silently
 *  stopped recording is worse than no dashboard. */
export interface WriterHealth {
  written: number;
  dropped: number;
  failed: number;
  queue_depth: number;
  thread_alive: boolean;
}

export interface MetricsOverview {
  window_hours: number;
  include_cached: boolean;
  latency: LatencySummary;
  cost: CostSummary;
  quality: QualitySummary;
  reliability: ReliabilitySummary;
  timeseries: Timeseries;
  writer: WriterHealth;
}

export interface FailureRow {
  query_id: string;
  timestamp: string;
  intent: string | null;
  error_category: string | null;
  error_type: string | null;
  total_latency_ms: number;
  query_text: string | null;
}

export interface MetricsParams {
  hours: number;
  includeCached: boolean;
  bucketMinutes: number;
}

export async function fetchOverview(params: MetricsParams): Promise<MetricsOverview> {
  const query = new URLSearchParams({
    hours: String(params.hours),
    include_cached: String(params.includeCached),
    bucket_minutes: String(params.bucketMinutes),
  });
  const response = await fetch(apiUrl(`/metrics/overview?${query}`));
  if (!response.ok) throw new ApiError(await readErrorDetail(response), response.status);
  return (await response.json()) as MetricsOverview;
}

export async function fetchFailures(hours: number): Promise<FailureRow[]> {
  const response = await fetch(apiUrl(`/metrics/failures?hours=${hours}&limit=20`));
  if (!response.ok) throw new ApiError(await readErrorDetail(response), response.status);
  return (await response.json()) as FailureRow[];
}
