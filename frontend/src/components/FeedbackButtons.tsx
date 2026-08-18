import { useMutation } from '@tanstack/react-query';
import { sendFeedback } from '../api/client';
import type { AskResponse } from '../api/types';

/**
 * Thumbs up/down, logged to the backend's feedback JSONL for later eval
 * review. Best-effort by design: a logging failure is reported quietly and
 * never blocks the conversation.
 */
export function FeedbackButtons({ result }: { result: AskResponse }) {
  const mutation = useMutation({
    mutationFn: (rating: 'up' | 'down') =>
      sendFeedback({
        query: result.query,
        resolved_query: result.resolved_query,
        intent: result.intent,
        answer: result.answer,
        rating,
      }),
  });

  if (mutation.isSuccess) {
    return <span className="text-[0.75rem] text-ink-faint">Thanks — feedback recorded.</span>;
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-[0.75rem] text-ink-faint">Was this useful?</span>
      {(['up', 'down'] as const).map((rating) => (
        <button
          key={rating}
          type="button"
          disabled={mutation.isPending}
          onClick={() => mutation.mutate(rating)}
          aria-label={rating === 'up' ? 'Mark answer as useful' : 'Mark answer as not useful'}
          className="rounded-sm border hairline px-2 py-0.5 text-[0.75rem] text-ink-muted transition-colors hover:border-rule-strong hover:text-ink disabled:opacity-50"
        >
          {rating === 'up' ? 'Yes' : 'No'}
        </button>
      ))}
      {mutation.isError ? (
        <span className="text-[0.75rem] text-ink-faint">Could not record that.</span>
      ) : null}
    </div>
  );
}
