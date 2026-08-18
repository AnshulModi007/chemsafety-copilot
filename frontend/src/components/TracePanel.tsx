import type { AnswerData, GradeVerdict, Trace, TraceAttempt } from '../api/types';
import { isSubQueryTrace } from '../api/types';
import { formatScore } from '../lib/format';
import { Disclosure, FieldLabel } from './primitives';

interface TracePanelProps {
  data: AnswerData;
}

const METHOD_LABEL: Record<TraceAttempt['retrieval_method'], string> = {
  expansion: 'Hybrid dense + BM25, RRF-fused, with query expansion',
  plain_rerank: 'Hybrid dense + BM25, RRF-fused',
};

const PATH_LABEL: Record<TraceAttempt['path'], string> = {
  fast_path: 'High-confidence match — skipped LLM grading',
  graded: 'Each excerpt graded for relevance by the LLM',
};

const VERDICT_STYLE: Record<GradeVerdict, string> = {
  correct: 'text-signal',
  ambiguous: 'text-caution',
  incorrect: 'text-ink-faint line-through decoration-1',
};

/**
 * "How this answer was found" -- the CRAG loop made inspectable.
 *
 * Every field here is optional in practice: the trace only exists for the two
 * retrieval intents, expansion queries only on the first attempt, grading
 * verdicts only when the fast path did not fire, and rerank scores only when
 * the cross-encoder is enabled. Each is rendered only when present.
 */
export function TracePanel({ data }: TracePanelProps) {
  const trace: Trace | undefined = data.trace;
  if (!trace || trace.length === 0) return null;

  const attemptCount = isSubQueryTrace(trace)
    ? trace.reduce((total, entry) => total + entry.attempts.length, 0)
    : trace.length;

  const retried = attemptCount > (isSubQueryTrace(trace) ? trace.length : 1);

  return (
    <Disclosure
      title="How this answer was found"
      meta={`${attemptCount} retrieval pass${attemptCount === 1 ? '' : 'es'}`}
    >
      <div className="space-y-4">
        {retried || data.crag_rewritten_query ? (
          <div className="rounded-sm bg-caution-soft px-3 py-2">
            <p className="text-[0.75rem] leading-relaxed text-caution">
              <span className="font-semibold">Corrective retry fired.</span> The first pass did not
              return excerpts that graded as relevant, so the query was rewritten and retrieval ran
              again.
            </p>
            {data.crag_rewritten_query ? (
              <p className="mt-1 font-mono text-[0.75rem] text-caution/90">
                → {data.crag_rewritten_query}
              </p>
            ) : null}
          </div>
        ) : null}

        {data.sub_queries && data.sub_queries.length > 0 ? (
          <div>
            <FieldLabel>Decomposed into</FieldLabel>
            <ol className="mt-1.5 space-y-1">
              {data.sub_queries.map((subQuery) => (
                <li key={subQuery} className="text-[0.8125rem] leading-snug text-ink-muted">
                  — {subQuery}
                </li>
              ))}
            </ol>
          </div>
        ) : null}

        {isSubQueryTrace(trace)
          ? trace.map((entry) => (
              <div key={entry.sub_query} className="space-y-3">
                <p className="text-[0.8125rem] font-medium text-ink">{entry.sub_query}</p>
                {entry.attempts.map((attempt) => (
                  <AttemptCard key={attempt.attempt} attempt={attempt} />
                ))}
              </div>
            ))
          : trace.map((attempt) => <AttemptCard key={attempt.attempt} attempt={attempt} />)}
      </div>
    </Disclosure>
  );
}

function AttemptCard({ attempt }: { attempt: TraceAttempt }) {
  return (
    <article className="border-l-2 border-rule pl-3.5">
      <header>
        <p className="text-[0.8125rem] font-medium text-ink">
          Pass {attempt.attempt}
          <span className="ml-2 font-normal text-ink-faint">
            {METHOD_LABEL[attempt.retrieval_method]}
          </span>
        </p>
        <p className="mt-0.5 text-[0.75rem] text-ink-muted">{PATH_LABEL[attempt.path]}</p>
      </header>

      {attempt.query_used ? (
        <p className="mt-1.5 font-mono text-[0.75rem] leading-relaxed text-ink-faint">
          query: {attempt.query_used}
        </p>
      ) : null}

      {attempt.expansion_queries && attempt.expansion_queries.length > 0 ? (
        <div className="mt-2">
          <FieldLabel>Expansion queries</FieldLabel>
          <ul className="mt-1 space-y-0.5">
            {attempt.expansion_queries.map((query) => (
              <li key={query} className="text-[0.75rem] leading-snug text-ink-muted">
                — {query}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {attempt.hyde_passage ? (
        <div className="mt-2">
          <FieldLabel>HyDE hypothetical passage</FieldLabel>
          <p className="mt-1 font-serif text-[0.75rem] leading-relaxed text-ink-muted">
            {attempt.hyde_passage}
          </p>
        </div>
      ) : null}

      {attempt.chunks.length > 0 ? (
        <div className="mt-2.5 overflow-x-auto">
          <table className="w-full min-w-[30rem] border-collapse text-left">
            <thead>
              <tr className="border-b hairline">
                <th className="py-1 pr-3 text-[0.6875rem] font-semibold uppercase tracking-wide text-ink-faint">
                  Excerpt
                </th>
                <th className="py-1 pr-3 text-[0.6875rem] font-semibold uppercase tracking-wide text-ink-faint">
                  Score
                </th>
                <th className="py-1 text-[0.6875rem] font-semibold uppercase tracking-wide text-ink-faint">
                  Grade
                </th>
              </tr>
            </thead>
            <tbody>
              {attempt.chunks.map((chunk) => (
                <tr key={chunk.chunk_id} className="border-b border-rule/50 align-top">
                  <td className="py-1.5 pr-3 text-[0.75rem] leading-snug text-ink-muted">
                    <span className="text-ink">{chunk.report_title}</span>
                    <span className="mx-1.5 text-rule-strong">·</span>
                    {chunk.section}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-[0.75rem] text-ink-muted">
                    {formatScore(chunk.rerank_score) ?? '—'}
                  </td>
                  <td className="py-1.5 text-[0.75rem]">
                    {chunk.verdict ? (
                      <span className={VERDICT_STYLE[chunk.verdict]}>{chunk.verdict}</span>
                    ) : (
                      <span className="text-ink-faint">—</span>
                    )}
                    {chunk.reason ? (
                      <span className="mt-0.5 block leading-snug text-ink-faint">
                        {chunk.reason}
                      </span>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </article>
  );
}
