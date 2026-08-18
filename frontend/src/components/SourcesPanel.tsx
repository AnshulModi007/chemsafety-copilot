import type { Citation, Source } from '../api/types';
import { isWebCitation } from '../api/types';
import { formatScore, humanizeReportId, pageLabel, relativeStrength } from '../lib/format';
import { Disclosure, FieldLabel } from './primitives';

interface SourcesPanelProps {
  sources: Source[];
  citations: Citation[];
}

/**
 * The trust anchor: which report excerpts actually grounded this answer.
 *
 * Laid out as a numbered ledger with a hanging index and a page reference,
 * closer to a citation apparatus than to chat-app source chips -- this is a
 * document-grounded tool, and a safety officer checking an answer is doing
 * the same thing they would do with a printed report.
 */
export function SourcesPanel({ sources, citations }: SourcesPanelProps) {
  const webCitations = citations.filter(isWebCitation);

  // Web-fallback answers carry title/url citations and no report chunks.
  if (sources.length === 0 && webCitations.length > 0) {
    return (
      <Disclosure title="Web sources" meta={String(webCitations.length)} defaultOpen>
        <ol className="space-y-2">
          {webCitations.map((citation, index) => (
            <li key={citation.url} className="flex gap-3 text-[0.8125rem]">
              <span className="tabular mt-px w-4 shrink-0 text-right text-ink-faint">
                {index + 1}
              </span>
              <a
                href={citation.url}
                target="_blank"
                rel="noreferrer noopener"
                className="text-signal underline underline-offset-2 hover:no-underline"
              >
                {citation.title}
              </a>
            </li>
          ))}
        </ol>
      </Disclosure>
    );
  }

  if (sources.length === 0) return null;

  const citedPages = pagesByReport(citations);

  return (
    <Disclosure title="Sources" meta={`${sources.length} excerpt${sources.length === 1 ? '' : 's'}`} defaultOpen>
      <ol className="space-y-3.5">
        {sources.map((source, index) => (
          <SourceEntry
            key={source.chunk_id}
            index={index + 1}
            source={source}
            all={sources}
            citedPages={citedPages.get(source.report_id)}
          />
        ))}
      </ol>
    </Disclosure>
  );
}

interface SourceEntryProps {
  index: number;
  source: Source;
  all: Source[];
  citedPages: number[] | undefined;
}

function SourceEntry({ index, source, all, citedPages }: SourceEntryProps) {
  const strength = relativeStrength(source, all);
  const score = formatScore(source.rerank_score);
  const title = source.report_title || humanizeReportId(source.report_id);

  return (
    <li className="flex gap-3">
      <span className="tabular mt-0.5 w-4 shrink-0 text-right text-[0.75rem] text-ink-faint">
        {index}
      </span>

      <div className="min-w-0 flex-1 border-l hairline pl-3">
        <p className="text-[0.8125rem] font-medium leading-snug text-ink">{title}</p>

        <p className="tabular mt-0.5 text-[0.75rem] text-ink-muted">
          {source.section}
          <span className="mx-1.5 text-rule-strong">·</span>
          {pageLabel(source.page_start, source.page_end)}
          {source.year ? (
            <>
              <span className="mx-1.5 text-rule-strong">·</span>
              {source.year}
            </>
          ) : null}
        </p>

        <p className="mt-1.5 line-clamp-3 font-serif text-[0.8125rem] leading-relaxed text-ink-muted">
          {source.snippet}…
        </p>

        <div className="mt-2 flex items-center gap-2">
          {strength !== null && score !== null ? (
            <>
              <span
                className="h-[3px] w-16 shrink-0 overflow-hidden rounded-full bg-paper-sunken"
                role="img"
                aria-label={`Relevance score ${score}`}
              >
                <span
                  className="block h-full rounded-full bg-signal"
                  style={{ width: `${strength * 100}%` }}
                />
              </span>
              <span className="tabular text-[0.6875rem] text-ink-faint">{score}</span>
            </>
          ) : (
            // ENABLE_RERANKER=false means there is no cross-encoder pass, so
            // there is genuinely no score -- say so rather than showing 0.000.
            <span className="text-[0.6875rem] text-ink-faint">relevance score unavailable</span>
          )}

          {citedPages && citedPages.length > 0 ? (
            <span className="tabular ml-auto text-[0.6875rem] text-ink-faint">
              cited p. {citedPages.join(', ')}
            </span>
          ) : null}
        </div>
      </div>
    </li>
  );
}

/** Which pages the model explicitly cited, grouped by report. */
function pagesByReport(citations: Citation[]): Map<string, number[]> {
  const byReport = new Map<string, number[]>();
  for (const citation of citations) {
    if (isWebCitation(citation)) continue;
    const pages = byReport.get(citation.report_id) ?? [];
    if (!pages.includes(citation.page)) pages.push(citation.page);
    byReport.set(citation.report_id, pages);
  }
  for (const pages of byReport.values()) pages.sort((a, b) => a - b);
  return byReport;
}

export function SourcesSummary({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  const reports = new Set(sources.map((source) => source.report_id));
  return (
    <p className="text-[0.75rem] text-ink-faint">
      <FieldLabel>Grounding</FieldLabel>{' '}
      {sources.length} excerpt{sources.length === 1 ? '' : 's'} from {reports.size} report
      {reports.size === 1 ? '' : 's'}
    </p>
  );
}
