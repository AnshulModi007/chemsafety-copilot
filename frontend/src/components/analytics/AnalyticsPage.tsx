import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { friendlyNetworkError } from '../../api/client';
import { fetchFailures, fetchOverview, type MetricsOverview } from '../../api/metrics';
import { INTENT_META } from '../../lib/intents';
import type { Intent } from '../../api/types';
import { Notice } from '../primitives';
import { BarRow, ProportionBar, Sparkline } from './charts';

const WINDOWS = [
  { label: '1h', hours: 1, bucketMinutes: 5 },
  { label: '24h', hours: 24, bucketMinutes: 60 },
  { label: '7d', hours: 24 * 7, bucketMinutes: 360 },
  { label: '30d', hours: 24 * 30, bucketMinutes: 1440 },
] as const;

/**
 * The ops view. Deliberately dense and unglamorous: an on-call engineer should
 * be able to answer "how fast, how expensive, how good, where does it fail"
 * without scrolling or interpreting a legend.
 */
export function AnalyticsPage() {
  const [windowIndex, setWindowIndex] = useState(1);
  const [includeCached, setIncludeCached] = useState(false);
  const selected = WINDOWS[windowIndex]!;

  const params = {
    hours: selected.hours,
    includeCached,
    bucketMinutes: selected.bucketMinutes,
  };

  const overview = useQuery({
    queryKey: ['metrics', 'overview', params],
    queryFn: () => fetchOverview(params),
    refetchInterval: 30_000,
  });

  const failures = useQuery({
    queryKey: ['metrics', 'failures', selected.hours],
    queryFn: () => fetchFailures(selected.hours),
    refetchInterval: 30_000,
  });

  return (
    <div className="pb-10">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b hairline py-5">
        <div>
          <h2 className="font-sans text-lg font-semibold tracking-tight text-ink">
            Production analytics
          </h2>
          <p className="mt-0.5 text-[0.8125rem] text-ink-muted">
            One telemetry record per query — latency by pipeline stage, notional LLM cost,
            answer quality, and failures.
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex rounded-sm border hairline">
            {WINDOWS.map((w, i) => (
              <button
                key={w.label}
                type="button"
                onClick={() => setWindowIndex(i)}
                aria-pressed={i === windowIndex}
                className={`px-3 py-1.5 text-[0.8125rem] transition-colors ${
                  i === windowIndex
                    ? 'bg-ink text-paper'
                    : 'text-ink-muted hover:bg-paper-sunken hover:text-ink'
                } ${i === 0 ? 'rounded-l-sm' : ''} ${i === WINDOWS.length - 1 ? 'rounded-r-sm' : ''}`}
              >
                {w.label}
              </button>
            ))}
          </div>

          <label className="flex cursor-pointer items-center gap-2 text-[0.8125rem] text-ink-muted">
            <input
              type="checkbox"
              checked={includeCached}
              onChange={(e) => setIncludeCached(e.target.checked)}
              className="h-3.5 w-3.5 accent-signal"
            />
            Include cache hits
          </label>
        </div>
      </header>

      {overview.isError ? (
        <div className="mt-6">
          <Notice tone="alert" title="Could not load metrics">
            {friendlyNetworkError(overview.error)}
          </Notice>
        </div>
      ) : null}

      {overview.isPending ? (
        <LoadingGrid />
      ) : overview.data ? (
        <Dashboard data={overview.data} failures={failures.data ?? []} />
      ) : null}
    </div>
  );
}

function Dashboard({
  data,
  failures,
}: {
  data: MetricsOverview;
  failures: Awaited<ReturnType<typeof fetchFailures>>;
}) {
  const { latency, cost, quality, reliability, timeseries, writer } = data;
  const points = timeseries.points;

  if (latency.sample_size === 0) {
    return (
      <div className="mt-6">
        <Notice tone="signal" title="No queries recorded in this window">
          Ask something on the main page, then come back — every query writes one telemetry row.
          {data.include_cached ? null : ' Cache hits are excluded; tick the box above to include them.'}
        </Notice>
      </div>
    );
  }

  const slowestStage = latency.stages[0];

  return (
    <>
      {/* Headline figures ------------------------------------------------- */}
      <section className="mt-6 grid gap-px overflow-hidden rounded-sm border hairline bg-rule sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="p95 latency"
          value={formatMs(latency.p95_ms)}
          note={`p50 ${formatMs(latency.p50_ms)} · p99 ${formatMs(latency.p99_ms)}`}
        />
        <Stat
          label="Cost / query"
          value={`$${cost.mean_cost_usd.toFixed(5)}`}
          note={`$${cost.total_cost_usd.toFixed(4)} total · ${cost.mean_llm_calls} LLM calls avg`}
        />
        <Stat
          label="Faithfulness"
          value={
            quality.faithfulness_pass_rate === null
              ? '—'
              : `${Math.round(quality.faithfulness_pass_rate * 100)}%`
          }
          note={
            quality.faithfulness_checked === 0
              ? 'no answers were verifiable'
              : `${quality.faithfulness_checked} of ${quality.sample_size} answers checked`
          }
          tone={
            quality.faithfulness_pass_rate !== null && quality.faithfulness_pass_rate < 0.9
              ? 'caution'
              : undefined
          }
        />
        <Stat
          label="Error rate"
          value={`${(reliability.error_rate * 100).toFixed(1)}%`}
          note={`${reliability.failures} of ${reliability.sample_size} queries`}
          tone={reliability.error_rate > 0 ? 'alert' : undefined}
        />
      </section>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        {/* Latency by stage ---------------------------------------------- */}
        <Panel
          title="Latency by pipeline stage"
          note={
            slowestStage
              ? `${slowestStage.stage} dominates at ${formatMs(slowestStage.mean_ms)} mean`
              : undefined
          }
        >
          {latency.stages.map((s) => (
            <BarRow
              key={s.stage}
              label={s.stage.replace(/_/g, ' ')}
              value={s.mean_ms}
              max={latency.stages[0]?.mean_ms ?? 1}
              display={formatMs(s.mean_ms)}
              note={
                // A stage only some queries enter is easy to misread as fast.
                s.share_of_queries < 0.999
                  ? `p95 ${formatMs(s.p95_ms)} · ran on ${Math.round(s.share_of_queries * 100)}% of queries`
                  : `p95 ${formatMs(s.p95_ms)}`
              }
            />
          ))}
        </Panel>

        {/* Volume + latency trend ---------------------------------------- */}
        <Panel title="Volume & latency" note={`${bucketLabel(timeseries.bucket_minutes)} buckets`}>
          <MiniTrend
            label="Queries"
            value={String(points.reduce((n, p) => n + p.queries, 0))}
            points={points.map((p) => ({ t: p.t, value: p.queries }))}
            unit="queries"
            color="#14608F"
          />
          <MiniTrend
            label="Mean latency"
            value={formatMs(latency.mean_ms)}
            points={points.map((p) => ({ t: p.t, value: p.mean_latency_ms }))}
            unit="ms"
            color="#A8700F"
          />
          <MiniTrend
            label="Faithfulness pass rate"
            value={
              quality.faithfulness_pass_rate === null
                ? '—'
                : `${Math.round(quality.faithfulness_pass_rate * 100)}%`
            }
            points={points.map((p) => ({
              t: p.t,
              value: p.faithfulness_pass_rate === null ? null : p.faithfulness_pass_rate * 100,
            }))}
            unit="%"
            color="#17715A"
          />
        </Panel>

        {/* Cost ----------------------------------------------------------- */}
        <Panel
          title="Cost by model"
          note={`${cost.total_tokens.toLocaleString()} tokens · notional, at configured list prices`}
        >
          {cost.by_model.map((m) => (
            <BarRow
              key={m.model}
              label={m.model}
              value={m.cost_usd}
              max={cost.by_model[0]?.cost_usd || 1}
              display={`$${m.cost_usd.toFixed(5)}`}
              color={m.priced ? '#14608F' : '#A8700F'}
              note={
                m.priced
                  ? `${m.calls} calls · ${m.prompt_tokens.toLocaleString()} in / ${m.completion_tokens.toLocaleString()} out`
                  : // Loud on purpose: an unpriced model silently understates the total.
                    `${m.calls} calls · no price configured — cost understated`
              }
            />
          ))}
          <div className="mt-3 border-t hairline pt-3">
            <Sparkline
              points={points.map((p) => ({ t: p.t, value: p.cost_usd }))}
              unit="USD"
              color="#14608F"
            />
            <p className="mt-1 text-[0.6875rem] text-ink-faint">Spend per bucket</p>
          </div>
        </Panel>

        {/* Routing + pipeline behaviour ----------------------------------- */}
        <Panel title="Routing & pipeline behaviour">
          <ProportionBar
            segments={quality.intent_distribution.map((d) => ({
              label: d.intent.replace(/_/g, ' '),
              value: d.count,
              color: INTENT_META[d.intent as Intent]?.marker ?? '#7B8E9C',
            }))}
          />
          <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 border-t hairline pt-3">
            <Rate label="CRAG retry fired" value={quality.crag_retry_rate} />
            <Rate label="Declined (insufficient)" value={quality.crag_insufficient_rate} />
            <Rate label="Web fallback fired" value={quality.web_fallback_rate} />
            <Rate label="Cache hit rate" value={quality.cache_hit_rate} />
          </dl>
          {quality.mean_retrieval_confidence !== null ? (
            <p className="mt-3 text-[0.75rem] text-ink-faint">
              Mean retrieval confidence {(quality.mean_retrieval_confidence * 100).toFixed(1)}%
            </p>
          ) : null}
        </Panel>
      </div>

      {/* Incidents --------------------------------------------------------- */}
      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Panel title="Failures by category">
          {reliability.by_category.length === 0 ? (
            <p className="py-2 text-[0.8125rem] text-ink-muted">
              No failures in this window.
            </p>
          ) : (
            reliability.by_category.map((c) => (
              <BarRow
                key={`${c.category}-${c.error_type}`}
                label={c.category.replace(/_/g, ' ')}
                value={c.count}
                max={reliability.by_category[0]?.count ?? 1}
                display={String(c.count)}
                color="#B93A2B"
                note={c.error_type ?? undefined}
              />
            ))
          )}
        </Panel>

        <Panel title="Recent incidents" note={`${failures.length} shown`}>
          {failures.length === 0 ? (
            <p className="py-2 text-[0.8125rem] text-ink-muted">Nothing to investigate.</p>
          ) : (
            <ul className="divide-y divide-rule">
              {failures.map((f) => (
                <li key={f.query_id} className="py-2">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-[0.8125rem] font-medium text-alert">
                      {f.error_category ?? 'unknown'}
                    </span>
                    <span className="tabular shrink-0 text-[0.6875rem] text-ink-faint">
                      {new Date(f.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <p className="mt-0.5 text-[0.75rem] text-ink-muted">
                    {f.error_type ?? '—'}
                    {f.intent ? ` · routed as ${f.intent}` : ' · failed before routing'}
                    {` · ${formatMs(f.total_latency_ms)}`}
                  </p>
                  {f.query_text ? (
                    <p className="mt-1 truncate font-serif text-[0.75rem] text-ink-faint">
                      {f.query_text}
                    </p>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <WriterStatus writer={writer} />
    </>
  );
}

/** The dashboard's own health. A telemetry pipeline that has quietly stopped
 *  recording looks identical to a system with no traffic. */
function WriterStatus({ writer }: { writer: MetricsOverview['writer'] }) {
  const degraded = !writer.thread_alive || writer.dropped > 0 || writer.failed > 0;
  return (
    <p
      className={`mt-5 text-[0.6875rem] ${degraded ? 'text-caution' : 'text-ink-faint'}`}
      role={degraded ? 'status' : undefined}
    >
      Telemetry writer {writer.thread_alive ? 'running' : 'STOPPED'} · {writer.written} written
      {writer.dropped > 0 ? ` · ${writer.dropped} dropped (queue full)` : ''}
      {writer.failed > 0 ? ` · ${writer.failed} write failures` : ''}
      {writer.queue_depth > 0 ? ` · ${writer.queue_depth} queued` : ''}
    </p>
  );
}

// --- Small pieces -------------------------------------------------------------

function Stat({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: string;
  note?: string;
  tone?: 'caution' | 'alert';
}) {
  const color =
    tone === 'alert' ? 'text-alert' : tone === 'caution' ? 'text-caution' : 'text-ink';
  return (
    <div className="bg-paper-raised px-4 py-3">
      <p className="text-[0.6875rem] font-semibold uppercase tracking-[0.09em] text-ink-faint">
        {label}
      </p>
      <p className={`tabular mt-1 text-2xl font-semibold ${color}`}>{value}</p>
      {note ? <p className="mt-0.5 text-[0.6875rem] text-ink-faint">{note}</p> : null}
    </div>
  );
}

function Panel({
  title,
  note,
  children,
}: {
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-sm border hairline bg-paper-raised p-4">
      <header className="mb-2.5">
        <h3 className="text-[0.8125rem] font-semibold text-ink">{title}</h3>
        {note ? <p className="mt-0.5 text-[0.6875rem] text-ink-faint">{note}</p> : null}
      </header>
      {children}
    </section>
  );
}

function MiniTrend({
  label,
  value,
  points,
  unit,
  color,
}: {
  label: string;
  value: string;
  points: { t: string; value: number | null }[];
  unit: string;
  color: string;
}) {
  return (
    <div className="border-t hairline py-2.5 first:border-t-0 first:pt-0">
      <div className="flex items-baseline justify-between">
        <span className="text-[0.8125rem] text-ink-muted">{label}</span>
        <span className="tabular text-[0.8125rem] font-medium text-ink">{value}</span>
      </div>
      <div className="mt-1">
        <Sparkline points={points} unit={unit} color={color} />
      </div>
    </div>
  );
}

function Rate({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="truncate text-[0.75rem] text-ink-muted">{label}</dt>
      <dd className="tabular text-[0.8125rem] font-medium text-ink">
        {(value * 100).toFixed(1)}%
      </dd>
    </div>
  );
}

function LoadingGrid() {
  return (
    <div className="mt-6 space-y-5" aria-busy="true">
      <div className="grid gap-px overflow-hidden rounded-sm border hairline bg-rule sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="bg-paper-raised px-4 py-3">
            <div className="h-2.5 w-20 animate-pulse rounded-sm bg-paper-sunken" />
            <div className="mt-2 h-6 w-24 animate-pulse rounded-sm bg-paper-sunken" />
          </div>
        ))}
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        {[0, 1].map((i) => (
          <div key={i} className="h-48 animate-pulse rounded-sm border hairline bg-paper-raised" />
        ))}
      </div>
    </div>
  );
}

function formatMs(ms: number): string {
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms >= 1_000) return `${(ms / 1000).toFixed(2)}s`;
  return `${Math.round(ms)}ms`;
}

function bucketLabel(minutes: number): string {
  if (minutes >= 1440) return `${minutes / 1440}d`;
  if (minutes >= 60) return `${minutes / 60}h`;
  return `${minutes}m`;
}
