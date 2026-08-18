/**
 * Diagram SVGs are assembled server-side by plain Python templates (see
 * src/visualization/) -- the model never emits markup, only the text that gets
 * substituted into it. So the realistic risk is not a hostile document but
 * model-authored *text* reaching an attribute or element that executes.
 *
 * Rather than trust that, parse the SVG and drop anything executable before it
 * is injected. Parsing (not regex) is what makes this trustworthy: it sees the
 * same tree the browser will build, so obfuscated markup cannot slip past.
 */
const FORBIDDEN_ELEMENTS = new Set([
  'script',
  'foreignobject',
  'iframe',
  'embed',
  'object',
  'audio',
  'video',
  'animate',
  'set',
  'handler',
]);

const URL_ATTRIBUTES = new Set(['href', 'xlink:href', 'src']);

export interface SanitizedSvg {
  markup: string;
  ok: boolean;
}

export function sanitizeSvg(raw: string): SanitizedSvg {
  if (typeof window === 'undefined') return { markup: '', ok: false };

  const parsed = new DOMParser().parseFromString(raw, 'image/svg+xml');
  if (parsed.querySelector('parsererror')) return { markup: '', ok: false };

  const root = parsed.documentElement;
  if (!root || root.nodeName.toLowerCase() !== 'svg') return { markup: '', ok: false };

  scrub(root);

  // Let the diagram scale to its container instead of its authored pixel size,
  // while keeping viewBox so the aspect ratio survives.
  root.removeAttribute('width');
  root.removeAttribute('height');
  root.setAttribute('width', '100%');

  return { markup: new XMLSerializer().serializeToString(root), ok: true };
}

function scrub(element: Element): void {
  for (const child of Array.from(element.children)) {
    if (FORBIDDEN_ELEMENTS.has(child.nodeName.toLowerCase())) {
      child.remove();
      continue;
    }
    scrub(child);
  }

  for (const attr of Array.from(element.attributes)) {
    const name = attr.name.toLowerCase();

    // Every inline event handler, whatever the element.
    if (name.startsWith('on')) {
      element.removeAttribute(attr.name);
      continue;
    }

    // javascript: / data: payloads hidden in a link or image reference.
    if (URL_ATTRIBUTES.has(name)) {
      const value = attr.value.trim().toLowerCase();
      if (value.startsWith('javascript:') || value.startsWith('data:text/html')) {
        element.removeAttribute(attr.name);
      }
    }
  }
}

/**
 * Hand the viewer a downloadable copy of a diagram. Built as an object URL
 * from the sanitized markup, so what they save is what they saw.
 */
export function downloadSvg(markup: string, filename: string): void {
  const blob = new Blob([markup], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename.endsWith('.svg') ? filename : `${filename}.svg`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
