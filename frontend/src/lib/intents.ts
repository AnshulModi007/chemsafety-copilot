import type { AnswerData, Diagram, Intent } from '../api/types';

export interface IntentMeta {
  label: string;
  /** What the router decided this question needs, in the user's language. */
  description: string;
  /** Stage text shown between the `routing` event and the first token. */
  loading: string;
  /** Diamond fill. A graphical mark only needs WCAG's 3:1 non-text threshold,
   *  so the palest markers (general_knowledge, unrouted) stay at their
   *  intended near-grey here. */
  marker: string;
  /** Label text colour: the same hue, darkened to clear text's stricter 4.5:1
   *  contrast requirement. Equal to `marker` wherever the marker already
   *  passes on its own. */
  textColor: string;
}

/**
 * Intent hues are all cool, because in this palette warm hues mean status and
 * a warm intent marker would read as an alert.
 *
 * Within that, saturation is not arbitrary — it encodes how grounded the
 * answer is. Report-backed and live-data answers carry full chroma; a
 * deterministic calculation is neutral steel because nothing was retrieved to
 * be right or wrong about; an ungrounded general-knowledge answer is
 * desaturated almost to grey, so the weakest provenance is also visibly the
 * palest mark on the page.
 */
export const INTENT_META: Record<Intent, IntentMeta> = {
  historical: {
    label: 'Incident report retrieval',
    description: 'Answered from CSB investigation reports',
    loading: 'Searching CSB incident reports…',
    marker: '#14608F', // blueprint blue — grounded in the corpus
    textColor: '#14608F',
  },
  comparative: {
    label: 'Multi-incident comparison',
    description: 'Retrieved each incident separately, then compared',
    loading: 'Retrieving each incident, then comparing…',
    marker: '#7B4B8E', // plum — grounded across several reports
    textColor: '#7B4B8E',
  },
  chemical_property: {
    label: 'PubChem lookup',
    description: 'Live chemical property data from PubChem',
    loading: 'Looking up chemical properties on PubChem…',
    marker: '#0D7C7C', // teal — grounded in live external data
    textColor: '#0D7A7A',
  },
  calculation: {
    label: 'API 520 relief sizing',
    description: 'Deterministic engineering calculation',
    loading: 'Sizing the relief valve (API 520)…',
    marker: '#4A5D6E', // steel — computed, not retrieved
    textColor: '#4A5D6E',
  },
  general_knowledge: {
    label: 'General knowledge',
    description: 'Model knowledge — not grounded in the report corpus',
    loading: 'Answering from general chemical-engineering knowledge…',
    marker: '#828A9F', // near-grey — ungrounded, and looks it (3:1 as a mark)
    textColor: '#646D83', // darkened further to clear 4.5:1 as label text
  },
};

export function intentMeta(intent: Intent | null): IntentMeta {
  if (intent && intent in INTENT_META) return INTENT_META[intent];
  return {
    label: 'Routing…',
    description: 'Deciding which tool this question needs',
    loading: 'Reading the question…',
    marker: '#798D9B',
    textColor: '#5D6F7C',
  };
}

export const DIAGRAM_LABEL: Record<string, string> = {
  bowtie: 'Bowtie risk diagram',
  causal_chain: 'Causal chain',
  side_by_side: 'Incident comparison',
  ghs: 'GHS hazard pictograms',
  psv: 'PSV cross-section schematic',
  concept: 'Concept diagram',
};

export interface NormalizedDiagram {
  svg: string;
  label: string;
  filename: string;
}

/**
 * The backend names its diagram field three different ways depending on which
 * intent produced it -- `diagram: {kind, svg}` for incident diagrams,
 * `ghs_diagram_svg` for pictograms, and a bare `diagram_svg` shared by PSV
 * schematics and concept diagrams. Normalising here keeps that historical
 * inconsistency in one place instead of spread across the components.
 */
export function normalizeDiagram(intent: Intent, data: AnswerData): NormalizedDiagram | null {
  const structured: Diagram | null | undefined = data.diagram;
  if (structured?.svg) {
    return {
      svg: structured.svg,
      label: DIAGRAM_LABEL[structured.kind] ?? 'Incident diagram',
      filename: `${structured.kind}-diagram`,
    };
  }

  if (data.ghs_diagram_svg) {
    return {
      svg: data.ghs_diagram_svg,
      label: DIAGRAM_LABEL.ghs!,
      filename: 'ghs-pictograms',
    };
  }

  if (data.diagram_svg) {
    const isPsv = intent === 'calculation';
    return {
      svg: data.diagram_svg,
      label: isPsv ? DIAGRAM_LABEL.psv! : DIAGRAM_LABEL.concept!,
      filename: isPsv ? 'psv-schematic' : 'concept-diagram',
    };
  }

  return null;
}
