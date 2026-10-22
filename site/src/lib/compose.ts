export interface Spacing {
  metrics: { upm: number; ascent: number; descent: number; line_gap: number; x_height: number; cap_height: number; space_adv: number };
  /** the italic face's metrics, only its space width differs from the regular face in use */
  italic_metrics?: Spacing['metrics'];
  /** glyphs by key, a regular glyph is its character and an italic glyph is `i:` and its character */
  glyphs: Record<string, { name: string; adv: number }>;
  /** pair as the two characters, then samples, mean observed deviation from the designed gap without kerning, designed kern value. Italic pairs have `i:` in front */
  pairs: Record<string, [number, number, number]>;
}

export type Mode = 'observed' | 'designed';
export type Face = 'regular' | 'italic';
export interface Placed { ch: string; x: number; adv: number; known: boolean; space: boolean }
export interface Layout { lines: Placed[][]; width: number; observed: number; defaulted: number; missing: string[]; average: number; averages: Record<Face, number>; italic: boolean }

/** Pairs measured on at least this many cards positions use their own offset, the others use the average */
export const MIN_SAMPLES = 3;

export const ITALIC = 'i:';
/** A cell is one character of one face, a regular cell is the character and an italic cell has `i:` in front */
export const faceOf = (cell: string): Face => ([...cell].length === 1 ? 'regular' : 'italic');
export const charOf = (cell: string) => [...cell].pop()!;
export const isSpace = (cell: string) => charOf(cell) === ' ';
/** What a cell is called in text */
export const cellLabel = (cell: string) => (faceOf(cell) === 'italic' ? `${charOf(cell)} (italic)` : cell);
const pairKey = (a: string, b: string) => (faceOf(a) === 'regular' ? a + b : ITALIC + charOf(a) + charOf(b));

/**
 * Splits text into paragraphs of cells. Text between asterisks is italic and the asterisks are not set, an
 * asterisk left open runs to the end of the text
 */
export function parse(text: string): string[][] {
  const paras: string[][] = [[]];
  let italic = false;
  for (const ch of text) {
    if (ch === '*') italic = !italic;
    else if (ch === '\n') paras.push([]);
    else paras[paras.length - 1].push(italic ? ITALIC + ch : ch);
  }
  return paras;
}

/** Average observed deviation of the pairs of one face the font does not kern, weighted by their samples */
export function averageOffset(sp: Spacing, face: Face = 'regular') {
  let w = 0, s = 0;
  for (const [k, [n, dev, kern]] of Object.entries(sp.pairs)) if (kern === 0 && ([...k].length === 2) === (face === 'regular')) { w += n; s += n * dev; }
  return w ? s / w : 0;
}

/** Mean advance of the scanned glyphs of one face, the width of a ghost cell */
export function ghostAdvance(sp: Spacing, face: Face = 'regular') {
  const a = Object.entries(sp.glyphs).filter(([k]) => faceOf(k) === face).map(([, g]) => g.adv);
  return a.length ? a.reduce((x, y) => x + y, 0) / a.length : ghostAdvance(sp);
}

const spaceAdvance = (sp: Spacing, cell: string) => (faceOf(cell) === 'italic' && sp.italic_metrics ? sp.italic_metrics : sp.metrics).space_adv;

/** Offset of a glyph's origin from the previous glyph's advance edge, in font units. Between faces there is no measured pair, so observed spacing uses the average of the second glyph's face and designed spacing none */
export function pairOffset(sp: Spacing, mode: Mode, a: string | null, b: string, avg: Record<Face, number>) {
  if (a === null || isSpace(a) || isSpace(b)) return 0;
  if (faceOf(a) !== faceOf(b)) return mode === 'designed' ? 0 : avg[faceOf(b)];
  const pair = sp.pairs[pairKey(a, b)];
  if (mode === 'designed') return pair ? pair[2] : 0;
  return pair && pair[0] >= MIN_SAMPLES ? pair[1] : avg[faceOf(b)];
}

/**
 * Sets text on lines no wider than maxWidth font units, wrapping at spaces. A glyph's origin follows the previous
 * origin by its advance plus the pair's observed offset, or its designed kern in designed mode
 */
export function layout(text: string, sp: Spacing, mode: Mode, maxWidth: number): Layout {
  const avg: Record<Face, number> = { regular: averageOffset(sp, 'regular'), italic: averageOffset(sp, 'italic') };
  const ghost: Record<Face, number> = { regular: ghostAdvance(sp, 'regular'), italic: ghostAdvance(sp, 'italic') };
  const lines: Placed[][] = [];
  const missing: string[] = [];
  let observed = 0, defaulted = 0, widest = 0, italic = false;
  const advance = (cell: string) => (isSpace(cell) ? spaceAdvance(sp, cell) : cell in sp.glyphs ? sp.glyphs[cell].adv : ghost[faceOf(cell)]);

  for (const para of parse(text)) {
    let line: Placed[] = [];
    let edge = 0;
    const flush = () => {
      while (line.length && line[line.length - 1].space) line.pop();
      widest = Math.max(widest, line.length ? line[line.length - 1].x + line[line.length - 1].adv : 0);
      lines.push(line);
      line = [];
      edge = 0;
    };
    // places a run of cells after the current edge, returns them without committing
    const run = (cells: string[], prev: string | null, from: number) => {
      const out: Placed[] = [];
      let x = from, p = prev;
      for (const ch of cells) {
        const o = pairOffset(sp, mode, p, ch, avg);
        out.push({ ch, x: x + o, adv: advance(ch), known: ch in sp.glyphs, space: isSpace(ch) });
        x = x + o + advance(ch);
        p = ch;
      }
      return out;
    };
    // runs of spaces and runs of letters, in order
    const tokens: string[][] = [];
    for (const cell of para) {
      const t = tokens[tokens.length - 1];
      if (t && isSpace(t[0]) === isSpace(cell)) t.push(cell); else tokens.push([cell]);
    }
    for (const chars of tokens) {
      const spaces = isSpace(chars[0]);
      if (spaces && !line.length) continue;
      const last = () => line[line.length - 1];
      let placed = run(chars, line.length ? last().ch : null, edge);
      const endOf = (p: Placed[]) => p[p.length - 1].x + p[p.length - 1].adv;
      if (!spaces && line.length && endOf(placed) > maxWidth) {
        flush();
        placed = run(chars, null, 0);
      }
      if (!spaces && endOf(placed) > maxWidth) {
        // a word wider than a whole line breaks between letters
        for (const ch of chars) {
          const one = run([ch], line.length ? last().ch : null, edge);
          if (line.length && endOf(one) > maxWidth) {
            flush();
            line.push(...run([ch], null, 0));
          } else line.push(...one);
          edge = last().x + last().adv;
        }
        continue;
      }
      line.push(...placed);
      edge = last().x + last().adv;
    }
    flush();
  }
  const flat = lines.flat();
  for (const c of flat) {
    if (!c.space && faceOf(c.ch) === 'italic') italic = true;
    if (!c.known && !c.space && !missing.includes(c.ch)) missing.push(c.ch);
  }
  for (const ln of lines) for (let i = 1; i < ln.length; i++) {
    if (!ln[i].known || !ln[i - 1].known || faceOf(ln[i].ch) !== faceOf(ln[i - 1].ch)) continue;
    const pair = sp.pairs[pairKey(ln[i - 1].ch, ln[i].ch)];
    if (pair && pair[0] >= MIN_SAMPLES) observed++; else defaulted++;
  }
  return { lines, width: widest, observed, defaulted, missing, average: avg.regular, averages: avg, italic };
}

/** Line pitch in font units, the ascent and descent plus the line gap */
export const lineHeight = (sp: Spacing) => sp.metrics.ascent - sp.metrics.descent + sp.metrics.line_gap;
