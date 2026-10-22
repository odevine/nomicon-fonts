// Noise Lab geometry: every crop is drawn at one scale on one baseline, so the letters keep their real relative size
import type { FitBox } from './format';

/** The window the lab shows, in font units: wide enough for the widest crop, tall enough for ascender to descender plus padding */
export const WIN = { w: 700, y0: -760, y1: 250 };

export interface Crop { w: number; h: number }
export interface Fit { k: number; pad: number }

/** Crops are the glyph's box plus a fixed margin, all on one pixel grid, so a straight line through (font units, crop pixels) gives px per font unit and the margin per side */
export function fitCrops(rows: { crop: Crop; box: FitBox }[]): Fit {
  const pts = rows.flatMap((r) => [[r.box.x1 - r.box.x0, r.crop.w], [r.box.y1 - r.box.y0, r.crop.h]]);
  const n = pts.length, sx = pts.reduce((a, p) => a + p[0], 0), sy = pts.reduce((a, p) => a + p[1], 0);
  const sxx = pts.reduce((a, p) => a + p[0] * p[0], 0), sxy = pts.reduce((a, p) => a + p[0] * p[1], 0);
  const k = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  return { k, pad: (sy - k * sx) / n / 2 };
}

/** Where a crop sits in the window, as percentages. Centred across, and the glyph's top edge (y0, font units, y down) places it down. */
export function place(crop: Crop, y0: number, fit: Fit) {
  const cw = crop.w / fit.k, ch = crop.h / fit.k, top = y0 - fit.pad / fit.k;
  const H = WIN.y1 - WIN.y0;
  const pc = (v: number) => `${(v * 100).toFixed(3)}%`;
  return { left: pc((WIN.w / 2 - cw / 2) / WIN.w), top: pc((top - WIN.y0) / H), width: pc(cw / WIN.w), height: pc(ch / H) };
}
export const placeStyle = (p: ReturnType<typeof place>) => `left:${p.left};top:${p.top};width:${p.width};height:${p.height}`;
export const baselinePct = () => `${((-WIN.y0 / (WIN.y1 - WIN.y0)) * 100).toFixed(3)}%`;
