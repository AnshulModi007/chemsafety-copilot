import { useId } from 'react';

/**
 * Hand-drawn SVG charts rather than a charting library.
 *
 * The dashboard needs four simple marks (a sparkline, a bar row, a stacked
 * proportion bar, a horizontal ranking), and pulling in a chart library for
 * that would add ~100kB gzipped to a bundle that is currently 113kB. These also
 * inherit the app's own palette instead of fighting a library's defaults.
 */

const AXIS = 'var(--chart-axis, #B0C2CF)';

interface SparklineProps {
  points: { t: string; value: number | null }[];
  /** Rendered in the accessible description, e.g. "ms" or "queries". */
  unit: string;
  color?: string;
  /** Pin the y-axis floor at 0 rather than the minimum sample. */
  zeroBased?: boolean;
  height?: number;
}

export function Sparkline({
  points,
  unit,
  color = '#14608F',
  zeroBased = true,
  height = 56,
}: SparklineProps) {
  const gradientId = useId();
  const defined = points.filter((p): p is { t: string; value: number } => p.value !== null);

  if (defined.length === 0) {
    return <EmptyChart height={height} label="No data in this window" />;
  }

  const width = 100; // viewBox units; the SVG scales to its container
  const values = defined.map((p) => p.value);
  const max = Math.max(...values);
  const min = zeroBased ? 0 : Math.min(...values);
  const span = max - min || 1;

  const x = (i: number) => (defined.length === 1 ? width / 2 : (i / (defined.length - 1)) * width);
  const y = (v: number) => height - ((v - min) / span) * (height - 6) - 3;

  const line = defined.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(p.value)}`).join(' ');
  const area = `${line} L ${x(defined.length - 1)} ${height} L ${x(0)} ${height} Z`;

  const last = values[values.length - 1]!;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="h-14 w-full"
      role="img"
      aria-label={`Trend over time, latest ${last.toFixed(1)} ${unit}, peak ${max.toFixed(1)} ${unit}`}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.18" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gradientId})`} />
      <path
        d={line}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

interface BarRowProps {
  label: string;
  value: number;
  max: number;
  /** Right-aligned formatted figure. */
  display: string;
  color?: string;
  /** Secondary line under the label. */
  note?: string;
}

export function BarRow({ label, value, max, display, color = '#14608F', note }: BarRowProps) {
  const width = max > 0 ? Math.max(1.5, (value / max) * 100) : 0;
  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="truncate text-[0.8125rem] text-ink">{label}</span>
        <span className="tabular shrink-0 text-[0.8125rem] font-medium text-ink">{display}</span>
      </div>
      <div className="mt-1 h-[5px] w-full overflow-hidden rounded-full bg-paper-sunken">
        <div
          className="h-full rounded-full"
          style={{ width: `${width}%`, backgroundColor: color }}
        />
      </div>
      {note ? <p className="mt-0.5 text-[0.6875rem] text-ink-faint">{note}</p> : null}
    </div>
  );
}

interface Segment {
  label: string;
  value: number;
  color: string;
}

/** A single stacked bar — used for intent distribution, where the whole is
 *  meaningful and the parts should be read as shares of it. */
export function ProportionBar({ segments }: { segments: Segment[] }) {
  const total = segments.reduce((sum, s) => sum + s.value, 0);
  if (total === 0) return <EmptyChart height={40} label="No queries in this window" />;

  return (
    <div>
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-paper-sunken">
        {segments.map((segment) => (
          <div
            key={segment.label}
            style={{ width: `${(segment.value / total) * 100}%`, backgroundColor: segment.color }}
            title={`${segment.label}: ${segment.value}`}
          />
        ))}
      </div>
      <ul className="mt-3 space-y-1.5">
        {segments.map((segment) => (
          <li key={segment.label} className="flex items-center gap-2 text-[0.8125rem]">
            <span
              aria-hidden="true"
              className="h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ backgroundColor: segment.color }}
            />
            <span className="truncate text-ink-muted">{segment.label}</span>
            <span className="tabular ml-auto text-ink">
              {segment.value}
              <span className="ml-1.5 text-ink-faint">
                {Math.round((segment.value / total) * 100)}%
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function EmptyChart({ height, label }: { height: number; label: string }) {
  return (
    <div
      className="flex items-center justify-center rounded-sm border border-dashed"
      style={{ height, borderColor: AXIS }}
    >
      <span className="text-[0.75rem] text-ink-faint">{label}</span>
    </div>
  );
}
