const display: Record<string, string> = { Altair: 'Altaïr' };

/** Card keys in the export are CamelCase without diacritics */
export const cardName = (key: string) => display[key] ?? key.replace(/([a-z])([A-Z])/g, '$1 $2');
/** A scan table angle as text, turned into 0 up to 360 */
export const deg = (a: number) => `${+(((a % 360) + 360) % 360).toFixed(1)}°`;
/** What a glyph is called in text, with its face when it is not the regular one */
export const glyphLabel = (ch: string, style = 'regular') => `${ch === '.' ? 'full stop' : ch}${style === 'regular' ? '' : `, ${style}`}`;

const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
/** A count as a word up to twelve, as digits above, for prose that follows the data */
export const countWord = (n: number) => WORDS[n] ?? String(n);
export const capital = (s: string) => s[0].toUpperCase() + s.slice(1);

export interface FitBox { x0: number; x1: number; y0: number; y1: number }

/** Bounding box of an absolute SVG path, taken over its control points */
export function bbox(d: string): FitBox {
  const nums = (d.match(/-?\d+\.?\d*/g) ?? []).map(Number);
  const xs = nums.filter((_, i) => i % 2 === 0), ys = nums.filter((_, i) => i % 2 === 1);
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}

export const viewBox = (b: FitBox, pad: number) => `${b.x0 - pad} ${b.y0 - pad} ${b.x1 - b.x0 + 2 * pad} ${b.y1 - b.y0 + 2 * pad}`;
/** Padding in font units around a glyph, enough for the blur skirt of the 0.1 contour */
export const PAD = 60;

export interface Spread { w: number; h: number; full_scale: number; to_fu: number[]; data: number[] }

/** Paths of the spread grid quantised into bands of opacity, one path per band, in grid cell units */
export function spreadBands(sp: Spread, bands = 6) {
  const out: string[] = Array(bands).fill('');
  sp.data.forEach((v, k) => {
    if (v < 6) return;
    out[Math.min(bands - 1, Math.floor((v / 255) * bands))] += `M${(k % sp.w) - 0.5},${Math.floor(k / sp.w) - 0.5}h1v1h-1z`;
  });
  return out;
}

export interface Bezier { nodes: [number, number][]; handles: [number, number][] }

/** Segment end points and control points of an absolute M, L, C, Z path as written by the trace export */
export function bezierParts(d: string): Bezier {
  const nodes: [number, number][] = [], handles: [number, number][] = [];
  for (const m of d.matchAll(/([MLC])([^MLCZ]*)/g)) {
    const n = (m[2].match(/-?\d+\.?\d*/g) ?? []).map(Number);
    if (m[1] === 'C') {
      handles.push([n[0], n[1]], [n[2], n[3]]);
      nodes.push([n[4], n[5]]);
    } else if (m[1] === 'L') nodes.push([n[0], n[1]]);
  }
  return { nodes, handles };
}
