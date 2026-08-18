import { useId, useState, type ReactNode } from 'react';

interface DisclosureProps {
  title: ReactNode;
  /** Right-aligned count or status, kept out of the title's type hierarchy. */
  meta?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}

/**
 * A keyboard-accessible expandable section. Built on a real <button> with
 * aria-expanded/aria-controls rather than <details>, so the open state can be
 * driven from React and styled consistently across browsers.
 */
export function Disclosure({ title, meta, defaultOpen = false, children }: DisclosureProps) {
  const [open, setOpen] = useState(defaultOpen);
  const panelId = useId();

  return (
    <section className="border-t hairline">
      <h3>
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          aria-controls={panelId}
          className="group flex w-full items-center gap-2 py-2.5 text-left"
        >
          <svg
            viewBox="0 0 12 12"
            aria-hidden="true"
            className={`h-2.5 w-2.5 shrink-0 text-ink-faint transition-transform duration-150 ${
              open ? 'rotate-90' : ''
            }`}
          >
            <path d="M3 1.5 8.5 6 3 10.5z" fill="currentColor" />
          </svg>
          <span className="text-[0.8125rem] font-medium tracking-wide text-ink-muted group-hover:text-ink">
            {title}
          </span>
          {meta ? (
            <span className="ml-auto text-[0.75rem] tabular text-ink-faint">{meta}</span>
          ) : null}
        </button>
      </h3>
      {open ? (
        <div id={panelId} className="pb-4 pt-1">
          {children}
        </div>
      ) : null}
    </section>
  );
}

/** Small uppercase label used to head a block of metadata. */
export function FieldLabel({ children }: { children: ReactNode }) {
  return (
    <span className="text-[0.6875rem] font-semibold uppercase tracking-[0.09em] text-ink-faint">
      {children}
    </span>
  );
}

interface NoticeProps {
  tone: 'caution' | 'alert' | 'signal';
  title: string;
  children?: ReactNode;
}

const NOTICE_TONE: Record<NoticeProps['tone'], string> = {
  caution: 'border-caution/30 bg-caution-soft text-caution',
  alert: 'border-alert/30 bg-alert-soft text-alert',
  signal: 'border-signal/25 bg-signal-soft text-signal',
};

export function Notice({ tone, title, children }: NoticeProps) {
  return (
    <div className={`rounded-sm border-l-2 px-3.5 py-2.5 ${NOTICE_TONE[tone]}`}>
      <p className="text-[0.8125rem] font-semibold">{title}</p>
      {children ? <div className="mt-1 text-[0.8125rem] leading-relaxed">{children}</div> : null}
    </div>
  );
}
