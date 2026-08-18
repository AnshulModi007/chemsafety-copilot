import type { Intent } from '../api/types';
import { INTENT_META } from '../lib/intents';

const EXAMPLES: Array<{ intent: Intent; question: string }> = [
  {
    intent: 'historical',
    question: 'What caused the ammonium nitrate explosion at West Fertilizer?',
  },
  {
    intent: 'comparative',
    question:
      'Compare the root causes of the West Fertilizer explosion and another ammonium nitrate incident.',
  },
  { intent: 'chemical_property', question: 'What is the molecular weight of chlorine?' },
  {
    intent: 'calculation',
    question:
      'Size a relief valve for a mass flow of 5000 lb/hr, molecular weight 44, relieving temperature 200F, set pressure 150 psig.',
  },
  { intent: 'general_knowledge', question: 'What is a tray tower?' },
];

/**
 * The opening screen doubles as the router's legend: each example is labelled
 * with the intent it will be classified as, so the five-way routing is legible
 * before the first question rather than only after one.
 */
export function EmptyState({ onPick }: { onPick: (question: string) => void }) {
  return (
    <div className="py-10">
      <h2 className="max-w-measure font-serif text-2xl leading-snug tracking-tight text-ink">
        Ask about process safety, and see exactly where the answer came from.
      </h2>
      <p className="mt-3 max-w-measure text-[0.9375rem] leading-relaxed text-ink-muted">
        Every question is routed to one of five tools. Grounded answers cite the CSB report
        excerpts behind them, show how those excerpts were retrieved, and are checked against
        their own sources before you see them.
      </p>

      <ul className="mt-8 space-y-px overflow-hidden rounded-sm border hairline bg-rule">
        {EXAMPLES.map(({ intent, question }) => {
          const meta = INTENT_META[intent];
          return (
            <li key={question} className="bg-paper-raised">
              <button
                type="button"
                onClick={() => onPick(question)}
                className="group flex w-full items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-paper-sunken"
              >
                <span
                  aria-hidden="true"
                  className="mt-[0.4rem] h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ backgroundColor: meta.marker }}
                />
                <span className="min-w-0 flex-1">
                  <span className="block text-[0.875rem] leading-snug text-ink">{question}</span>
                  <span className="mt-0.5 block text-[0.75rem] text-ink-faint">{meta.label}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
