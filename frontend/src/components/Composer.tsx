import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';

interface ComposerProps {
  onSubmit: (question: string) => void;
  onCancel: () => void;
  isBusy: boolean;
  /** Set when an example is picked, so the box fills and focuses. */
  prefill: string | null;
}

export function Composer({ onSubmit, onCancel, isBusy, prefill }: ComposerProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (prefill === null) return;
    setValue(prefill);
    textareaRef.current?.focus();
  }, [prefill]);

  // Grow with the question instead of scrolling inside a fixed two-line box.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || isBusy) return;
    onSubmit(trimmed);
    setValue('');
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    submit();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter breaks the line. Questions here are often long
    // (a PSV sizing question carries four or five process values).
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t hairline bg-paper/95 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-end gap-3 px-5 py-3.5 sm:px-8">
        <label htmlFor="composer" className="sr-only">
          Ask a chemical-safety question
        </label>
        <textarea
          id="composer"
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about an incident, a chemical, or a relief-valve size…"
          className="min-h-[2.5rem] flex-1 resize-none rounded-sm border hairline bg-paper-raised px-3.5 py-2 text-[0.9375rem] leading-relaxed text-ink placeholder:text-ink-faint focus:border-signal"
        />

        {isBusy ? (
          <button
            type="button"
            onClick={onCancel}
            className="h-10 shrink-0 rounded-sm border hairline px-4 text-[0.875rem] font-medium text-ink-muted transition-colors hover:border-rule-strong hover:text-ink"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!value.trim()}
            className="h-10 shrink-0 rounded-sm bg-signal px-5 text-[0.875rem] font-medium text-paper transition-opacity hover:opacity-90 disabled:opacity-35"
          >
            Ask
          </button>
        )}
      </div>
    </form>
  );
}
