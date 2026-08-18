import type { Intent } from '../api/types';
import { intentMeta } from '../lib/intents';
import { Placard } from './Placard';

interface IntentBadgeProps {
  intent: Intent | null;
  /** Rendered alongside the badge: cached, web-fallback, etc. */
  annotations?: string[];
}

/**
 * Which of the five router intents handled this question, marked with the
 * placard diamond. Still a byline rather than a tag cloud -- the colour and
 * the silhouette carry the meaning, so the label can stay quiet.
 */
export function IntentBadge({ intent, annotations = [] }: IntentBadgeProps) {
  const meta = intentMeta(intent);

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <span className="inline-flex items-center gap-2">
        <Placard color={meta.marker} hollow={intent === null} />
        <span
          className="text-[0.6875rem] font-semibold uppercase tracking-[0.11em]"
          style={{ color: meta.textColor }}
        >
          {meta.label}
        </span>
      </span>
      {annotations.map((annotation) => (
        <span
          key={annotation}
          className="text-[0.75rem] text-ink-faint before:mr-3 before:text-rule-strong before:content-['/']"
        >
          {annotation}
        </span>
      ))}
    </div>
  );
}
