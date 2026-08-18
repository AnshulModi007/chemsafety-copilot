import type { Faithfulness } from '../api/types';
import { Notice } from './primitives';

interface FaithfulnessStatusProps {
  faithfulness: Faithfulness | null | undefined;
}

/**
 * The backend runs an independent second LLM call that checks the generated
 * answer against its own source context. Only the retrieval-grounded intents
 * produce this -- a PubChem lookup or a PSV calculation has no free-text
 * context to verify against.
 *
 * When it is absent, this renders nothing. Showing "unverified" for an intent
 * that has nothing to verify would read as a warning about the answer rather
 * than a statement about the pipeline.
 */
export function FaithfulnessStatus({ faithfulness }: FaithfulnessStatusProps) {
  if (!faithfulness) return null;

  if (faithfulness.faithful) {
    // Verified green, not blueprint blue: this is a status the pipeline is
    // asserting about itself, not a provenance marker -- the two never share
    // a colour family in this palette.
    return (
      <p className="inline-flex items-center gap-1.5 text-[0.75rem] font-medium text-verified">
        <svg viewBox="0 0 12 12" aria-hidden="true" className="h-3 w-3">
          <path
            d="M2.5 6.2 4.8 8.5 9.5 3.8"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        Every claim checked against the retrieved excerpts
      </p>
    );
  }

  const claims = faithfulness.unsupported_claims;

  return (
    <Notice tone="caution" title="Possible unsupported claim detected">
      <p>
        A verification pass flagged part of this answer as not directly supported by the retrieved
        excerpts. Check it against the sources below before relying on it.
      </p>
      {claims.length > 0 ? (
        <ul className="mt-1.5 space-y-1">
          {claims.map((claim) => (
            <li key={claim} className="leading-snug">
              — {claim}
            </li>
          ))}
        </ul>
      ) : null}
    </Notice>
  );
}
