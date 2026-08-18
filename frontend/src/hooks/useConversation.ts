import { useCallback, useRef, useState } from 'react';
import { friendlyNetworkError } from '../api/client';
import { streamAsk } from '../api/stream';
import type { AskResponse, HistoryTurn, Intent } from '../api/types';

/** How many prior turns to send back for follow-up resolution. Matches the
 *  window the backend's reformulate_query actually reads (last 6). */
const HISTORY_WINDOW = 6;

export interface Turn {
  id: string;
  question: string;
  /** Populated as the answer streams; replaced by the final envelope on done. */
  streamedText: string;
  intent: Intent | null;
  routingReasoning: string | null;
  result: AskResponse | null;
  error: string | null;
  /** Set once a `done` or `error` event lands, or the request is cancelled. */
  settled: boolean;
}

export interface ConversationState {
  turns: Turn[];
  isBusy: boolean;
  ask: (question: string) => void;
  cancel: () => void;
  reset: () => void;
}

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function useConversation(): ConversationState {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [isBusy, setIsBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patchTurn = useCallback((id: string, patch: Partial<Turn>) => {
    setTurns((current) =>
      current.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)),
    );
  }, []);

  const ask = useCallback(
    (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || abortRef.current) return;

      const id = newId();
      const controller = new AbortController();
      abortRef.current = controller;
      setIsBusy(true);

      // Snapshot history *before* appending this turn, so the backend sees the
      // conversation as it stood when the question was asked.
      let history: HistoryTurn[] = [];
      setTurns((current) => {
        history = toHistory(current);
        return [
          ...current,
          {
            id,
            question: trimmed,
            streamedText: '',
            intent: null,
            routingReasoning: null,
            result: null,
            error: null,
            settled: false,
          },
        ];
      });

      void (async () => {
        // Deltas arrive far faster than React needs to paint; batching them
        // into one state write per frame keeps a long answer from queueing
        // hundreds of renders.
        let pending = '';
        let flushHandle: number | null = null;

        const flush = () => {
          flushHandle = null;
          if (!pending) return;
          const chunk = pending;
          pending = '';
          setTurns((current) =>
            current.map((turn) =>
              turn.id === id ? { ...turn, streamedText: turn.streamedText + chunk } : turn,
            ),
          );
        };

        try {
          for await (const event of streamAsk({ query: trimmed, history }, controller.signal)) {
            switch (event.type) {
              case 'routing':
                patchTurn(id, { intent: event.intent, routingReasoning: event.reasoning });
                break;

              case 'delta':
                pending += event.text;
                if (flushHandle === null) {
                  flushHandle = window.requestAnimationFrame(flush);
                }
                break;

              case 'done': {
                if (flushHandle !== null) window.cancelAnimationFrame(flushHandle);
                flush();
                const { type: _type, ...result } = event;
                patchTurn(id, { result, intent: result.intent, settled: true });
                break;
              }

              case 'error':
                if (flushHandle !== null) window.cancelAnimationFrame(flushHandle);
                patchTurn(id, { error: event.detail, settled: true });
                break;
            }
          }
        } catch (error) {
          if (flushHandle !== null) window.cancelAnimationFrame(flushHandle);
          if (!controller.signal.aborted) {
            patchTurn(id, { error: friendlyNetworkError(error), settled: true });
          }
        } finally {
          // A stream that ends without a `done` event would otherwise leave the
          // turn spinning forever.
          setTurns((current) =>
            current.map((turn) =>
              turn.id === id && !turn.settled
                ? {
                    ...turn,
                    settled: true,
                    error: turn.error ?? (controller.signal.aborted ? null : 'The answer stream ended before completing.'),
                  }
                : turn,
            ),
          );
          abortRef.current = null;
          setIsBusy(false);
        }
      })();
    },
    [patchTurn],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setTurns([]);
  }, []);

  return { turns, isBusy, ask, cancel, reset };
}

function toHistory(turns: Turn[]): HistoryTurn[] {
  const history: HistoryTurn[] = [];
  for (const turn of turns) {
    history.push({ role: 'user', content: turn.question });
    if (turn.result) history.push({ role: 'assistant', content: turn.result.answer });
  }
  return history.slice(-HISTORY_WINDOW);
}
