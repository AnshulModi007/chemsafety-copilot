interface PlacardProps {
  color: string;
  /** Edge length in px. Kept small — this is a mark, not an illustration. */
  size?: number;
  /** Hollow reads as "pending"; filled as "decided". */
  hollow?: boolean;
  className?: string;
}

/**
 * The app's signature mark: a square stood on its corner.
 *
 * Every hazard marking in this domain is a diamond — the NFPA 704 fire
 * diamond, GHS pictograms, DOT transport placards. Using that silhouette as
 * the provenance marker means the interface is speaking the reader's own
 * visual vocabulary before a single word is read, where a coloured dot would
 * have said nothing at all.
 *
 * Drawn as a rotated square rather than a glyph so the stroke stays crisp at
 * these sizes and the fill can be driven straight from the intent colour.
 */
export function Placard({ color, size = 9, hollow = false, className = '' }: PlacardProps) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block shrink-0 rotate-45 ${className}`}
      style={{
        width: size,
        height: size,
        backgroundColor: hollow ? 'transparent' : color,
        border: hollow ? `1.5px solid ${color}` : undefined,
        borderRadius: 1,
      }}
    />
  );
}
