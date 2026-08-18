/**
 * The streaming generation prompt asks the model for plain prose with inline
 * citation tags -- `[[report:<report_id>:<page>]]` or `[[web:<title>|<url>]]`
 * -- because Groq's JSON mode collapses a stream into a single chunk (see the
 * "Streaming-only prompts & parsing" note in src/generation/generate.py). The
 * backend strips those tags before the final `done` event, but they are
 * visible in the raw deltas, so the live preview has to strip them too.
 */
const CITATION_TAG = /\[\[(?:report|web):[^\]]*\]\]/g;

/** Tidy whitespace a removed tag leaves behind (" ." -> "."). */
function tidy(text: string): string {
  return text.replace(/\s+([.,!?;:])/g, '$1').replace(/[ \t]{2,}/g, ' ');
}

/**
 * Strip citation tags from a partially-streamed answer.
 *
 * A tag that is still arriving (opened but not yet closed) is held back
 * entirely rather than rendered as raw brackets for a frame -- otherwise the
 * reader watches "[[report:csb_01..." type itself out and then vanish.
 */
export function stripStreamingCitations(accumulated: string): string {
  const lastOpen = accumulated.lastIndexOf('[[');
  const lastClose = accumulated.lastIndexOf(']]');
  const safe = lastOpen > lastClose ? accumulated.slice(0, lastOpen) : accumulated;
  return tidy(safe.replace(CITATION_TAG, ''));
}
