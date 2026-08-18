import type { Source } from '../api/types';

/** "p. 14" or "pp. 14–17", matching how the reports themselves are cited. */
export function pageLabel(start: string | number, end: string | number): string {
  const from = String(start);
  const to = String(end);
  return from === to ? `p. ${from}` : `pp. ${from}–${to}`;
}

/**
 * Rerank scores are cross-encoder outputs, not probabilities, but within a
 * single answer they are directly comparable -- so they are shown as a
 * relative bar against the strongest source rather than as a bare number
 * implying calibrated confidence.
 */
export function relativeStrength(source: Source, all: Source[]): number | null {
  if (source.rerank_score === null) return null;
  const scores = all.map((s) => s.rerank_score).filter((s): s is number => s !== null);
  if (scores.length === 0) return null;
  const max = Math.max(...scores);
  if (max <= 0) return null;
  return Math.max(0.04, Math.min(1, source.rerank_score / max));
}

export function formatScore(score: number | null): string | null {
  return score === null ? null : score.toFixed(3);
}

/** Report ids are slugs (`csb_01_west-fertilizer-explosion-and-fire`). */
export function humanizeReportId(reportId: string): string {
  const withoutPrefix = reportId.replace(/^csb_\d+_/, '');
  return withoutPrefix
    .split('-')
    .map((word) => (word.length > 2 ? word[0]!.toUpperCase() + word.slice(1) : word))
    .join(' ');
}

export function formatNumber(value: number, digits = 4): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}
