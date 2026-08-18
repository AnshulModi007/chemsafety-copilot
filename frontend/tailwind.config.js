/** @type {import('tailwindcss').Config} */

/**
 * Palette: the drafting room, not the cream-paper default.
 *
 * Two rules govern every colour in this app, and nothing is coloured outside
 * them:
 *
 *   1. COOL hues encode provenance -- where an answer came from.
 *   2. WARM hues are reserved for status -- alarm, caution, verified.
 *
 * That split is borrowed from the domain itself: in process safety, colour is
 * already load-bearing. Alarm priorities (ISA-18.2) and hazard placards use
 * red/amber for severity, so spending those hues on decoration would misread
 * as an alert. Keeping status warm and provenance cool means a red on this
 * screen always means something is wrong.
 *
 * The ground is a cool blue-grey drafting vellum rather than warm cream --
 * cream reads as literary, and this is an instrument, not a book.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: {
          DEFAULT: '#EDF1F4', // drafting vellum
          raised: '#FFFFFF', // sheet
          sunken: '#DFE7ED', // recessed wells, track backgrounds
        },
        ink: {
          DEFAULT: '#14212B', // drafting ink -- blue-black, never neutral
          muted: '#4E6273',
          // WCAG AA requires 4.5:1 for body-sized text, and captions/labels
          // across this app are 12-13px -- normal text, not "large text" at the
          // 3:1 threshold. #7B8E9C measured 2.99:1 against the paper ground and
          // failed; this clears 4.64:1 while staying the palest ink step.
          faint: '#5D6E7C',
        },
        rule: {
          DEFAULT: '#D3DEE6',
          strong: '#B0C2CF',
        },
        // Primary. P&ID linework blue: structural, interactive, never a status.
        signal: {
          DEFAULT: '#14608F',
          deep: '#0E4668', // hover/active
          soft: '#E1EDF5',
        },
        // --- Status triad. Warm, standardised, used nowhere decorative. ---
        verified: {
          DEFAULT: '#17715A',
          soft: '#E2EFEA',
        },
        caution: {
          DEFAULT: '#A8700F',
          soft: '#FAF0DC',
        },
        alert: {
          DEFAULT: '#B93A2B',
          soft: '#FBE9E6',
        },
      },
      fontFamily: {
        // IBM Plex, used as a superfamily across all three roles. Plex was
        // commissioned for technical and industrial documentation, so the
        // interface reads as one engineering document system rather than a
        // chat app wearing a serif. Loaded as real webfonts in index.html --
        // the previous Charter/Inter stack silently fell back to Cambria and
        // Segoe UI on Windows, which is most of why this looked generic.
        sans: ['IBM Plex Sans', 'Segoe UI', 'system-ui', 'sans-serif'],
        serif: ['IBM Plex Serif', 'Georgia', 'serif'],
        mono: ['IBM Plex Mono', 'Consolas', 'monospace'],
      },
      maxWidth: {
        measure: '68ch',
      },
      boxShadow: {
        // Barely-there lift for raised sheets. Cool-tinted so it reads as
        // depth on the vellum rather than grey haze.
        sheet: '0 1px 2px rgba(20, 33, 43, 0.05), 0 1px 1px rgba(20, 33, 43, 0.03)',
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
