import type { AnswerData } from '../api/types';
import { formatNumber } from '../lib/format';
import { FieldLabel, Notice } from './primitives';

/**
 * Structured renderings for the two intents whose answers are tool output
 * rather than prose: a PubChem property lookup and an API 520 sizing result.
 * Both render only from fields the backend actually returned -- a missing
 * property is omitted, never shown as a placeholder value.
 */

export function PubChemCard({ data }: { data: AnswerData }) {
  if (data.pubchem_unavailable || !data.molecular_formula) return null;

  const properties: Array<[string, string | null]> = [
    ['Formula', data.molecular_formula ?? null],
    ['Molecular weight', data.molecular_weight != null ? String(data.molecular_weight) : null],
    ['XLogP', data.xlogp != null ? String(data.xlogp) : null],
    ['TPSA', data.tpsa != null ? String(data.tpsa) : null],
  ];
  const present = properties.filter((entry): entry is [string, string] => entry[1] !== null);

  return (
    <section className="mt-4 rounded-sm border hairline bg-paper-raised">
      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-sm bg-rule sm:grid-cols-4">
        {present.map(([label, value]) => (
          <div key={label} className="bg-paper-raised px-3.5 py-2.5">
            <dt>
              <FieldLabel>{label}</FieldLabel>
            </dt>
            <dd className="tabular mt-0.5 text-[0.9375rem] font-medium text-ink">{value}</dd>
          </div>
        ))}
      </dl>

      {data.ghs_hazard_statements && data.ghs_hazard_statements.length > 0 ? (
        <div className="border-t hairline px-3.5 py-2.5">
          <FieldLabel>GHS hazard statements</FieldLabel>
          <ul className="mt-1.5 space-y-1">
            {data.ghs_hazard_statements.map((statement) => (
              <li key={statement} className="text-[0.8125rem] leading-snug text-ink-muted">
                — {statement}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {data.canonical_smiles ? (
        <div className="border-t hairline px-3.5 py-2.5">
          <FieldLabel>SMILES</FieldLabel>
          <p className="mt-0.5 break-all font-mono text-[0.75rem] text-ink-muted">
            {data.canonical_smiles}
          </p>
        </div>
      ) : null}

      {data.pubchem_url ? (
        <div className="border-t hairline px-3.5 py-2">
          <a
            href={data.pubchem_url}
            target="_blank"
            rel="noreferrer noopener"
            className="text-[0.75rem] text-signal underline underline-offset-2 hover:no-underline"
          >
            PubChem CID {data.cid}
          </a>
        </div>
      ) : null}
    </section>
  );
}

export function PsvSizingCard({ data }: { data: AnswerData }) {
  if (data.invalid_input) return null;

  if (data.missing_required_fields && data.missing_required_fields.length > 0) {
    return (
      <div className="mt-4">
        <Notice tone="caution" title="More process data needed">
          <ul className="mt-1 space-y-0.5">
            {data.missing_required_fields.map((field) => (
              <li key={field} className="font-mono text-[0.75rem]">
                — {field}
              </li>
            ))}
          </ul>
        </Notice>
      </div>
    );
  }

  if (data.required_area_in2 === undefined || !data.inputs) return null;

  const { inputs } = data;
  const orifice = data.recommended_orifice;

  const rows: Array<[string, string]> = [
    ['Mass flow', `${formatNumber(inputs.mass_flow_lb_hr, 1)} lb/hr`],
    ['Molecular weight', `${formatNumber(inputs.molecular_weight, 2)} lb/lbmol`],
    ['Relieving temperature', `${formatNumber(inputs.relieving_temp_rankine, 1)} °R`],
    ['Set pressure', `${formatNumber(inputs.set_pressure_psig, 1)} psig`],
    ['k (Cp/Cv)', formatNumber(inputs.k, 3)],
    ['Compressibility Z', formatNumber(inputs.compressibility_z, 3)],
  ];

  return (
    <section className="mt-4 space-y-3">
      <div className="grid gap-px overflow-hidden rounded-sm border hairline bg-rule sm:grid-cols-2">
        <div className="bg-signal-soft px-3.5 py-3">
          <FieldLabel>Required effective area</FieldLabel>
          <p className="tabular mt-0.5 text-xl font-semibold text-signal">
            {data.required_area_in2.toFixed(4)} in²
          </p>
        </div>
        <div className="bg-paper-raised px-3.5 py-3">
          <FieldLabel>Recommended API 526 orifice</FieldLabel>
          <p className="tabular mt-0.5 text-xl font-semibold text-ink">
            {orifice ? `${orifice.designation} (${orifice.area_in2} in²)` : 'None standard'}
          </p>
          {!orifice ? (
            <p className="mt-0.5 text-[0.75rem] text-ink-muted">
              Larger than the largest standard orifice — consider multiple valves.
            </p>
          ) : null}
        </div>
      </div>

      <div className="overflow-x-auto rounded-sm border hairline bg-paper-raised">
        <table className="w-full border-collapse text-left">
          <caption className="sr-only">Sizing inputs</caption>
          <tbody>
            {rows.map(([label, value], index) => (
              <tr key={label} className={index > 0 ? 'border-t hairline' : ''}>
                <th
                  scope="row"
                  className="px-3.5 py-2 text-[0.8125rem] font-normal text-ink-muted"
                >
                  {label}
                </th>
                <td className="tabular px-3.5 py-2 text-right text-[0.8125rem] font-medium text-ink">
                  {value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.warnings && data.warnings.length > 0
        ? data.warnings.map((warning) => (
            <Notice key={warning} tone="caution" title="Sizing caveat">
              {warning}
            </Notice>
          ))
        : null}
    </section>
  );
}
