import type { Turn as TurnState } from '../hooks/useConversation';
import { stripStreamingCitations } from '../lib/citations';
import { intentMeta, normalizeDiagram } from '../lib/intents';
import { AnswerBody } from './AnswerBody';
import { DiagramFigure } from './DiagramFigure';
import { FaithfulnessStatus } from './FaithfulnessStatus';
import { FeedbackButtons } from './FeedbackButtons';
import { IntentBadge } from './IntentBadge';
import { PsvSizingCard, PubChemCard } from './IntentCards';
import { Notice } from './primitives';
import { SourcesPanel } from './SourcesPanel';
import { TracePanel } from './TracePanel';

/**
 * One question and its answer.
 *
 * Laid out as a two-column grid on wide viewports -- the answer holds a
 * comfortable reading measure on the left while its evidence sits alongside
 * on the right, so sources are never something you have to go looking for.
 * Below `xl` the evidence stacks underneath, still open by default.
 */
export function Turn({ turn }: { turn: TurnState }) {
  const { result } = turn;
  const data = result?.data;

  const annotations: string[] = [];
  if (result?.from_cache) annotations.push('cached');
  if (data?.source === 'web') annotations.push('web fallback — corpus had no confident answer');
  if (data?.source === 'insufficient') annotations.push('declined — insufficient evidence');

  const diagram = result ? normalizeDiagram(result.intent, result.data) : null;
  const hasEvidence =
    Boolean(data) &&
    ((data?.sources?.length ?? 0) > 0 ||
      (data?.citations?.length ?? 0) > 0 ||
      (data?.trace?.length ?? 0) > 0);

  const answerText = result ? result.answer : stripStreamingCitations(turn.streamedText);
  const isStreaming = !turn.settled && answerText.length > 0;

  return (
    <article className="border-t hairline py-8 first:border-t-0">
      <h2 className="max-w-measure font-sans text-lg font-semibold leading-snug tracking-tight text-ink">
        {turn.question}
      </h2>

      {result?.resolved_query ? (
        <p className="mt-1.5 max-w-measure text-[0.8125rem] text-ink-faint">
          Read as: <span className="italic">{result.resolved_query}</span>
        </p>
      ) : null}

      <div className="mt-5 grid gap-x-10 gap-y-6 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0">
          <IntentBadge intent={turn.intent} annotations={annotations} />

          <div className="mt-3 max-w-measure">
            {turn.error ? (
              <Notice tone="alert" title="Could not answer this question">
                {turn.error}
              </Notice>
            ) : answerText ? (
              <>
                {data?.source === 'insufficient' ? (
                  <Notice tone="caution" title="Declined rather than guessed">
                    {answerText}
                  </Notice>
                ) : (
                  <AnswerBody text={answerText} streaming={isStreaming} />
                )}
              </>
            ) : (
              <PendingState intent={turn.intent} />
            )}

            {data ? (
              <>
                {result?.intent === 'chemical_property' ? <PubChemCard data={data} /> : null}
                {result?.intent === 'calculation' ? <PsvSizingCard data={data} /> : null}
                {diagram ? <DiagramFigure diagram={diagram} /> : null}

                <div className="mt-4 space-y-3">
                  <FaithfulnessStatus faithfulness={data.faithfulness} />
                </div>
              </>
            ) : null}
          </div>

          {result ? (
            <footer className="mt-5 max-w-measure space-y-2 border-t hairline pt-3">
              {result.routing_reasoning ? (
                <p className="text-[0.75rem] leading-relaxed text-ink-faint">
                  <span className="font-medium">Why this route:</span> {result.routing_reasoning}
                </p>
              ) : null}
              <FeedbackButtons result={result} />
            </footer>
          ) : null}
        </div>

        {hasEvidence && data ? (
          <aside className="min-w-0 border-rule xl:border-l xl:pl-8">
            <p className="pb-1 text-[0.6875rem] font-semibold uppercase tracking-[0.09em] text-ink-faint">
              Evidence
            </p>
            {typeof data.confidence === 'number' && data.source === 'internal' ? (
              <p className="tabular pb-2 text-[0.75rem] text-ink-muted">
                Retrieval confidence {Math.round(data.confidence * 100)}%
              </p>
            ) : null}
            <SourcesPanel sources={data.sources ?? []} citations={data.citations ?? []} />
            <TracePanel data={data} />
          </aside>
        ) : null}
      </div>
    </article>
  );
}

/**
 * Between submitting and the first token. The `routing` SSE event names the
 * intent before any answer text exists, so this can say what the pipeline is
 * actually doing rather than showing a generic spinner -- which matters, since
 * a retrieval pass with a CRAG retry can run for a while.
 */
function PendingState({ intent }: { intent: TurnState['intent'] }) {
  const meta = intentMeta(intent);

  return (
    <div>
      <p className="flex items-center gap-2 text-[0.8125rem] text-ink-muted">
        <span className="relative flex h-1.5 w-1.5" aria-hidden="true">
          <span
            className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
            style={{ backgroundColor: meta.marker }}
          />
          <span
            className="relative inline-flex h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: meta.marker }}
          />
        </span>
        {meta.loading}
      </p>

      <div className="mt-4 space-y-2" aria-hidden="true">
        {['100%', '96%', '88%'].map((width, index) => (
          <div
            key={width}
            className="h-3 animate-pulse rounded-sm bg-paper-sunken"
            style={{ width, animationDelay: `${index * 120}ms` }}
          />
        ))}
      </div>
      <span className="sr-only" role="status">
        {meta.loading}
      </span>
    </div>
  );
}
