import { useCallback, useEffect, useRef, useState } from 'react';
import { checkHealth } from './api/client';
import { AnalyticsPage } from './components/analytics/AnalyticsPage';
import { Composer } from './components/Composer';
import { EmptyState } from './components/EmptyState';
import { Placard } from './components/Placard';
import { Notice } from './components/primitives';
import { Turn } from './components/Turn';
import { useConversation } from './hooks/useConversation';
import { INTENT_META } from './lib/intents';

type View = 'ask' | 'analytics';

const DISCLAIMER =
  'Historical incident findings and reference data — not a stamped engineering judgment. ' +
  'Consult a licensed Professional Engineer for any real design or safety decision.';

export default function App() {
  const { turns, isBusy, ask, cancel, reset } = useConversation();
  const [prefill, setPrefill] = useState<string | null>(null);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [view, setView] = useState<View>('ask');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void checkHealth().then(setBackendUp);
  }, []);

  // Follow the answer as it streams, but never fight a user who has scrolled
  // up to read a source.
  useEffect(() => {
    const scroller = document.scrollingElement;
    if (!scroller) return;
    const nearBottom =
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 240;
    if (nearBottom) endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [turns]);

  const handlePick = useCallback((question: string) => setPrefill(question), []);

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="sticky top-0 z-30 border-b hairline bg-paper/92 backdrop-blur-sm">
        {/* Hazard-stripe hairline: the five intent colours laid edge to edge,
            so the router's palette is declared once at the top of the page and
            every badge below reads against it. */}
        <div className="flex h-[3px]">
          {(['historical', 'comparative', 'chemical_property', 'calculation', 'general_knowledge'] as const).map(
            (i) => (
              <span key={i} className="flex-1" style={{ backgroundColor: INTENT_META[i].marker }} />
            ),
          )}
        </div>

        <div className="mx-auto flex max-w-5xl items-center gap-4 px-5 py-3 sm:px-8">
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <Placard color="#14608F" size={13} />
            <div className="min-w-0">
              <h1 className="text-[0.9375rem] font-semibold tracking-tight text-ink">
                ChemSafety Copilot
              </h1>
              <p className="truncate text-[0.75rem] text-ink-faint">
                Grounded in U.S. Chemical Safety Board investigation reports
              </p>
            </div>
          </div>

          <nav className="flex shrink-0 items-center gap-1" aria-label="Views">
            {(['ask', 'analytics'] as const).map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setView(v)}
                aria-current={view === v ? 'page' : undefined}
                className={`rounded-sm px-3 py-1.5 text-[0.8125rem] transition-colors ${
                  view === v
                    ? 'bg-paper-sunken font-medium text-ink'
                    : 'text-ink-muted hover:text-ink'
                }`}
              >
                {v === 'ask' ? 'Ask' : 'Analytics'}
              </button>
            ))}
          </nav>

          {view === 'ask' && turns.length > 0 ? (
            <button
              type="button"
              onClick={reset}
              className="shrink-0 rounded-sm border hairline px-3 py-1.5 text-[0.8125rem] text-ink-muted transition-colors hover:border-rule-strong hover:text-ink"
            >
              New session
            </button>
          ) : null}
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-5 sm:px-8">
        {backendUp === false ? (
          <div className="mt-6">
            <Notice tone="alert" title="Backend not reachable">
              Start it with <code className="font-mono">uvicorn app.main:app</code> from the repo
              root, then reload. If it is running elsewhere, point{' '}
              <code className="font-mono">VITE_API_BASE_URL</code> at it.
            </Notice>
          </div>
        ) : null}

        {view === 'analytics' ? (
          <AnalyticsPage />
        ) : turns.length === 0 ? (
          <EmptyState onPick={handlePick} />
        ) : (
          <div className="pb-6">
            {turns.map((turn) => (
              <Turn key={turn.id} turn={turn} />
            ))}
          </div>
        )}
        <div ref={endRef} />
      </main>

      {/* The composer belongs to the Ask view only -- a question box pinned
          under an ops dashboard reads as a search field for the metrics. */}
      {view === 'ask' ? (
        <div className="sticky bottom-0 z-20">
          <Composer onSubmit={ask} onCancel={cancel} isBusy={isBusy} prefill={prefill} />
          <div className="border-t hairline bg-paper-sunken">
            <p className="mx-auto max-w-5xl px-5 py-2 text-[0.6875rem] leading-relaxed text-ink-faint sm:px-8">
              {DISCLAIMER}
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
