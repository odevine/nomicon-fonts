import { readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(process.cwd(), 'data');
const read = (f: string) => JSON.parse(readFileSync(resolve(root, f), 'utf8'));

export const fileSize = (f: string) => statSync(resolve(root, f)).size;
export const glyphs = (): GlyphRow[] => read('glyphs.json');
export const glyph = (name: string): GlyphDoc => read(`glyphs/${name}.json`);
export const sweepOf = (name: string): Sweep => read(`sweep/${name}.json`);
export const pairInfo = (name: string): PairInfo => read(`noise/${name}.json`);
export const noise = (): Noise => read('noise.json');
export const calibration = (): Calibration => read('calibration.json');
export const cards = (): Record<string, Card> => read('cards.json');
export const steps = (g = 'e'): Steps => read(`steps/${g}.json`);
export const meta = (): Meta => read('meta.json');
export const kerning = (): Record<string, Kern> => read('kerning.json');

/** Thresholds the pipeline used, the site quotes them rather than repeating the numbers */
export interface Meta { min_instances: number; confident_instances: number; confident_cards: number }
export interface GlyphRow {
  name: string; char: string; unicode: string; style: string; source: string;
  n: number; cards: number; nodes: number; edge_rms_um: number; iou: number;
  recommended_n: number; low_confidence: boolean;
}
export interface GlyphDoc extends Omit<GlyphRow, 'cards' | 'source'> {
  adv: number; cards: string[]; trace: string; instances_total: number; instances_exported: number;
  trace_by_n: { n: number; cards: number; trace: string; nodes: number; edge_rms_um: number | null }[];
  mean: Record<string, string>; cutaways: { path: string; radius: number; deviation_um: number }[];
}
export interface Sweep { rows: { n: number; rms_vs_ref_um: number; p10: number; p90: number; noise_model_um: number }[]; recommended_n: number }
interface NoisePair { hist: Hist; n_pairs: number; pair_rms_um: number; single_scan_um: number; pair_mean_um: number; tied: boolean; note: string }
export interface Noise { headline: NoisePair; alternative: NoisePair }
export interface Ruler { scan_axis: string; ticks: number; tilt_deg: number; px_per_mm: number; resid_px: number }
export interface Calibration {
  summary: { px_per_mm: number; scan_anisotropy: number; print_x_over_y: number; edge_w1090_um_median: number };
  rulers: { a: { x: Ruler; y: Ruler }; b: { x: Ruler; y: Ruler } };
  edge_w1090_um: number[];
  tone: { a: number[]; b: number[] };
  tone_percent_black: number[];
}
export interface Hist { bin_um: number; counts: number[]; n: number; over: number; median_um: number; rms_um: number }
export interface PairInfo { glyph: string; id: string; card: string; upright: string; turned: string; upright_angle: number; turned_angle: number; pair_dev_um: number; median_pair_dev_um: number; w: number; h: number; um_per_grid_px: number }
export interface Card {
  scans: { scan: string; angle: number; edge_level: number; rot_deg: number }[];
  print_scale: number; units: number;
}
export interface Kern { n: number; designed_gap: number; kern: number; mean_dev_nokern: number; mean_dev_kern: number; sd: number }

/** Noise for the site: the untied fit (each scan's affine on its own) leads, the tied fit sits beside it */
export function noisePair() {
  const n = noise();
  return { lead: n.alternative, tied: n.headline };
}

export function edgeLevels() {
  const lv = Object.values(cards()).flatMap((c) => c.scans.map((s) => s.edge_level));
  return { min: Math.min(...lv), max: Math.max(...lv), scans: lv.length };
}

export const fmt = (v: number, d = 1) => v.toFixed(d);

export interface Steps {
  glyph: string; id: string; card: string; upright: string; turned: string; upright_angle: number; turned_angle: number; um_per_px: number; um_per_grid_px: number; um_per_fu: number; ink_spread_um: number;
  paper: number; ink: number; rot_deg: number; print_scale: number; edge_levels: number[]; pair_dev_um: number; deviation_um: number;
  s1: { file: string; w: number; h: number; glyph_box: number[] };
  s2: { file: string; halo_px: number };
  s3: { linear: string; clean: string; w: number; h: number };
  s4: { file: string; w: number; h: number };
  s5: { w: number; h: number; cards: string[]; before: string[]; after: string[] };
  s6: { files: string[]; w: number; h: number };
}
