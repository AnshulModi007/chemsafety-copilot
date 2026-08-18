import type { FeedbackRequest } from './types';

const BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export function apiUrl(path: string): string {
  return `${BASE_URL.replace(/\/$/, '')}${path}`;
}

/**
 * An API failure with a message already fit for a human to read. Every network
 * or backend error is funnelled through this so no component ever has to
 * render a raw exception or an empty state.
 */
export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** FastAPI puts its message in `detail`; fall back sensibly when it doesn't. */
export async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === 'string') return detail;
    }
  } catch {
    // Non-JSON error body -- fall through to the status-based message.
  }
  return `The server responded with ${response.status} ${response.statusText}.`;
}

export function friendlyNetworkError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof DOMException && error.name === 'AbortError') {
    return 'Request cancelled.';
  }
  return (
    `Could not reach the ChemSafety Copilot backend at ${BASE_URL}. ` +
    'Check that it is running (uvicorn app.main:app) and that VITE_API_BASE_URL points at it.'
  );
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(apiUrl('/health'));
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Feedback is deliberately best-effort: it logs to the backend's feedback
 * JSONL for later eval review, and a failure to record it should never
 * interrupt the user's session.
 */
export async function sendFeedback(body: FeedbackRequest): Promise<void> {
  const response = await fetch(apiUrl('/feedback'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status);
  }
}
