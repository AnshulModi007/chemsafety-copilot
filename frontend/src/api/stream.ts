import { ApiError, apiUrl, readErrorDetail } from './client';
import type { AskRequest, StreamEvent } from './types';

/**
 * Consumes POST /ask/stream as Server-Sent Events.
 *
 * Written against `fetch` + a ReadableStream reader rather than `EventSource`
 * for one hard reason: EventSource can only issue GET requests, and this
 * endpoint needs a JSON body (the query plus conversation history). The
 * framing is otherwise ordinary SSE -- `data: <json>\n\n` per event -- so this
 * buffers across chunk boundaries and only parses on a complete event.
 *
 * Note the endpoint returns 200 as soon as the stream opens, so a pipeline
 * failure arrives as an in-band `{type: "error"}` event rather than an HTTP
 * status. Only pre-stream validation (an empty query) fails with a real 4xx.
 */
export async function* streamAsk(
  request: AskRequest,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(apiUrl('/ask/stream'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status);
  }
  if (!response.body) {
    throw new ApiError('The server returned an empty response stream.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Events are separated by a blank line; keep the trailing partial event
      // in the buffer until its terminator arrives.
      const rawEvents = buffer.split('\n\n');
      buffer = rawEvents.pop() ?? '';

      for (const raw of rawEvents) {
        const event = parseEvent(raw);
        if (event) yield event;
      }
    }

    const trailing = parseEvent(buffer);
    if (trailing) yield trailing;
  } finally {
    reader.releaseLock();
  }
}

function parseEvent(raw: string): StreamEvent | null {
  const payload = raw
    .split('\n')
    .filter((line) => line.startsWith('data: '))
    .map((line) => line.slice('data: '.length))
    .join('');

  if (!payload) return null;

  try {
    return JSON.parse(payload) as StreamEvent;
  } catch {
    // A malformed frame costs one event, not the whole answer -- the stream
    // keeps going and the terminal `done` event still carries the full result.
    return null;
  }
}
