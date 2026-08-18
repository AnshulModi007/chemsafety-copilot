import { describe, expect, it } from 'vitest';
import { stripStreamingCitations } from './citations';
import { normalizeDiagram } from './intents';
import { pageLabel, relativeStrength, formatScore } from './format';
import type { AnswerData, Source } from '../api/types';

describe('stripStreamingCitations', () => {
  it('removes a complete tag and the space it strands before punctuation', () => {
    expect(stripStreamingCitations('contamination of FGAN [[report:csb_01_west:66]].')).toBe(
      'contamination of FGAN.',
    );
  });

  it('holds back a tag that is still arriving rather than flashing raw brackets', () => {
    for (const partial of ['[[', '[[repo', '[[report:csb_0', '[[report:csb_01_west:6']) {
      expect(stripStreamingCitations(`12 responders${partial}`)).toBe('12 responders');
    }
  });

  it('handles the web-citation tag form', () => {
    expect(stripStreamingCitations('per the CSB [[web:CSB Report|https://csb.gov/x]] finding.')).toBe(
      'per the CSB finding.',
    );
  });

  it('never leaks a bracket at any point of an incremental replay', () => {
    const full = 'The blast[[report:csb_01_west:66]] killed 12[[report:csb_01_west:17]].';
    for (let i = 1; i <= full.length; i++) {
      expect(stripStreamingCitations(full.slice(0, i))).not.toContain('[[');
    }
  });

  it('leaves prose without citations untouched', () => {
    expect(stripStreamingCitations('plain prose')).toBe('plain prose');
  });
});

describe('normalizeDiagram', () => {
  // The backend names its diagram field three different ways by intent; this is
  // the single place that difference is absorbed.
  it('reads the structured incident diagram', () => {
    const data: AnswerData = { diagram: { kind: 'bowtie', svg: '<svg/>' } };
    expect(normalizeDiagram('historical', data)).toEqual({
      svg: '<svg/>',
      label: 'Bowtie risk diagram',
      filename: 'bowtie-diagram',
    });
  });

  it('reads the GHS pictogram field on chemical_property', () => {
    const result = normalizeDiagram('chemical_property', { ghs_diagram_svg: '<svg/>' });
    expect(result?.label).toBe('GHS hazard pictograms');
  });

  it('disambiguates the shared diagram_svg field by intent', () => {
    expect(normalizeDiagram('calculation', { diagram_svg: '<svg/>' })?.label).toBe(
      'PSV cross-section schematic',
    );
    expect(normalizeDiagram('general_knowledge', { diagram_svg: '<svg/>' })?.label).toBe(
      'Concept diagram',
    );
  });

  it('returns null when the answer carries no diagram', () => {
    expect(normalizeDiagram('historical', {})).toBeNull();
    expect(normalizeDiagram('historical', { diagram: null })).toBeNull();
  });
});

describe('source scoring', () => {
  const withScore = (score: number | null): Source => ({
    chunk_id: `c${score}`,
    report_id: 'r',
    report_title: 'Report',
    section: 'S',
    chemical: null,
    year: null,
    page_start: 1,
    page_end: 1,
    rerank_score: score,
    snippet: '',
  });

  it('scales each source against the strongest one in the same answer', () => {
    const all = [withScore(0.98), withScore(0.49)];
    expect(relativeStrength(all[0]!, all)).toBe(1);
    expect(relativeStrength(all[1]!, all)).toBeCloseTo(0.5, 5);
  });

  it('reports no strength when reranking is disabled', () => {
    // ENABLE_RERANKER=false means there is genuinely no score to show.
    const all = [withScore(null), withScore(null)];
    expect(relativeStrength(all[0]!, all)).toBeNull();
    expect(formatScore(null)).toBeNull();
  });

  it('formats page ranges the way the reports are cited', () => {
    expect(pageLabel(16, 16)).toBe('p. 16');
    expect(pageLabel('16', '17')).toBe('pp. 16–17');
  });
});
