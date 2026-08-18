import { useEffect, useMemo, useState } from 'react';
import { downloadSvg, sanitizeSvg } from '../lib/svg';
import type { NormalizedDiagram } from '../lib/intents';

interface DiagramFigureProps {
  diagram: NormalizedDiagram;
}

/**
 * Renders a backend-generated SVG safety diagram inline, with enlarge and
 * download. The markup is parsed and scrubbed before injection (see
 * lib/svg.ts); if it will not parse, the figure is dropped silently --
 * diagram generation fails soft server-side too, so "no diagram" is an
 * expected outcome rather than an error worth showing the user.
 */
export function DiagramFigure({ diagram }: DiagramFigureProps) {
  const [expanded, setExpanded] = useState(false);
  const sanitized = useMemo(() => sanitizeSvg(diagram.svg), [diagram.svg]);

  useEffect(() => {
    if (!expanded) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setExpanded(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [expanded]);

  if (!sanitized.ok) return null;

  return (
    <figure className="mt-4">
      <div className="overflow-x-auto rounded-sm border hairline bg-paper-raised p-3">
        <div
          className="[&_svg]:h-auto [&_svg]:w-full"
          // Sanitized above: parsed as XML, with script/foreignObject-class
          // elements, every on* handler, and javascript: URLs removed.
          dangerouslySetInnerHTML={{ __html: sanitized.markup }}
        />
      </div>

      <figcaption className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-[0.75rem] text-ink-faint">{diagram.label}</span>
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="text-[0.75rem] text-signal underline underline-offset-2 hover:no-underline"
        >
          Enlarge
        </button>
        <button
          type="button"
          onClick={() => downloadSvg(sanitized.markup, diagram.filename)}
          className="text-[0.75rem] text-signal underline underline-offset-2 hover:no-underline"
        >
          Download SVG
        </button>
      </figcaption>

      {expanded ? (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={diagram.label}
          className="fixed inset-0 z-50 flex flex-col bg-ink/60 p-4 sm:p-8"
          onClick={() => setExpanded(false)}
        >
          <div
            className="mx-auto flex max-h-full w-full max-w-5xl flex-col overflow-hidden rounded-sm bg-paper-raised"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b hairline px-4 py-2.5">
              <span className="text-[0.8125rem] font-medium text-ink">{diagram.label}</span>
              <button
                type="button"
                onClick={() => setExpanded(false)}
                autoFocus
                className="text-[0.8125rem] text-ink-muted hover:text-ink"
              >
                Close
              </button>
            </div>
            <div className="overflow-auto p-5">
              <div
                className="[&_svg]:h-auto [&_svg]:w-full"
                dangerouslySetInnerHTML={{ __html: sanitized.markup }}
              />
            </div>
          </div>
        </div>
      ) : null}
    </figure>
  );
}
