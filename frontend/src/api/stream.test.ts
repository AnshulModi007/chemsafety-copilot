import { describe, expect, it, vi, afterEach } from 'vitest';
import { streamAsk } from './stream';
import { ApiError } from './client';
import type { StreamEvent } from './types';

/**
 * The SSE reader's real hazard is chunk boundaries: the network splits the byte
 * stream wherever it likes, including mid-JSON and mid-multibyte-character, and
 * a parser that assumes one chunk equals one event drops or corrupts answers.
 * These tests replay a recorded backend stream at deliberately hostile split
 * points.
 */

/** Serve `body` as a fetch Response whose stream is cut at the given offsets. */
function mockFetch(body: string, splitAt: number[], status = 200): void {
  const bytes = new TextEncoder().encode(body);
  const bounds = [0, ...splitAt, bytes.length];
  const chunks: Uint8Array[] = [];
  for (let i = 0; i < bounds.length - 1; i++) {
    chunks.push(bytes.slice(bounds[i]!, bounds[i + 1]!));
  }

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });

  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(status === 200 ? stream : body, { status })),
  );
}

async function collect(body: string, splitAt: number[]): Promise<StreamEvent[]> {
  mockFetch(body, splitAt);
  const events: StreamEvent[] = [];
  for await (const event of streamAsk(
    { query: 'q', history: [] },
    new AbortController().signal,
  )) {
    events.push(event);
  }
  return events;
}

// Shaped exactly like app/main.py::ask_stream emits.
const SSE = [
  `data: ${JSON.stringify({ type: 'routing', intent: 'historical', reasoning: 'names an incident' })}\n\n`,
  `data: ${JSON.stringify({ type: 'delta', text: 'The blast ' })}\n\n`,
  `data: ${JSON.stringify({ type: 'delta', text: 'killed 12 responders' })}\n\n`,
  `data: ${JSON.stringify({ type: 'done', intent: 'historical', answer: 'The blast killed 12 responders', data: {} })}\n\n`,
].join('');

afterEach(() => vi.unstubAllGlobals());

describe('streamAsk', () => {
  it('yields every event when the whole body arrives in one chunk', async () => {
    const events = await collect(SSE, []);
    expect(events.map((e) => e.type)).toEqual(['routing', 'delta', 'delta', 'done']);
  });

  it('reassembles events split mid-JSON across chunk boundaries', async () => {
    // Cut inside the first event's JSON, and again inside the last event's.
    const events = await collect(SSE, [30, 120, SSE.length - 25]);
    expect(events.map((e) => e.type)).toEqual(['routing', 'delta', 'delta', 'done']);
    expect(events.filter((e) => e.type === 'delta').map((e) => (e as { text: string }).text)).toEqual([
      'The blast ',
      'killed 12 responders',
    ]);
  });

  it('survives a split landing between the two newlines of the terminator', async () => {
    const boundary = SSE.indexOf('\n\n') + 1;
    const events = await collect(SSE, [boundary]);
    expect(events.map((e) => e.type)).toEqual(['routing', 'delta', 'delta', 'done']);
  });

  it('emits a trailing event even without a final blank line', async () => {
    const events = await collect(SSE.trimEnd(), []);
    expect(events.at(-1)?.type).toBe('done');
  });

  it('surfaces an in-band pipeline error as an error event, not a throw', async () => {
    const body = `data: ${JSON.stringify({ type: 'error', detail: 'Groq 429' })}\n\n`;
    const events = await collect(body, []);
    expect(events).toEqual([{ type: 'error', detail: 'Groq 429' }]);
  });

  it('drops a malformed frame without losing the rest of the stream', async () => {
    const body = `data: {not json\n\n${SSE}`;
    const events = await collect(body, []);
    expect(events.map((e) => e.type)).toEqual(['routing', 'delta', 'delta', 'done']);
  });

  it('throws a readable ApiError when the request is rejected before streaming', async () => {
    mockFetch(JSON.stringify({ detail: 'query must not be empty' }), [], 400);
    await expect(async () => {
      for await (const _ of streamAsk({ query: '', history: [] }, new AbortController().signal)) {
        // no events expected
      }
    }).rejects.toThrow(ApiError);
  });
});
