// Geometry of the data page's sweep overview, shared by the server render and the script that redraws the picked glyph

export interface SweepRow { n: number; rms_vs_ref_um: number; p10: number; p90: number; noise_model_um: number }

export const SM = { W: 640, H: 300, l: 34, r: 10, t: 10, b: 26 };

/** N runs on a log axis from 1 to nmax, edge RMS on a linear one from 0 to ymax */
export function sweepAxes(nmax: number, ymax: number) {
  const X = (n: number) => SM.l + (Math.log(n) / Math.log(nmax)) * (SM.W - SM.l - SM.r);
  const Y = (v: number) => SM.H - SM.b - (v / ymax) * (SM.H - SM.t - SM.b);
  const pt = (n: number, v: number) => `${X(n).toFixed(1)},${Y(v).toFixed(1)}`;
  const line = (rows: SweepRow[], k: 'rms_vs_ref_um' | 'noise_model_um') => rows.map((r, i) => `${i ? 'L' : 'M'}${pt(r.n, r[k])}`).join('');
  const band = (rows: SweepRow[]) => rows.map((r, i) => `${i ? 'L' : 'M'}${pt(r.n, r.p90)}`).join('') + [...rows].reverse().map((r) => `L${pt(r.n, r.p10)}`).join('') + 'Z';
  const xticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].filter((t) => t <= nmax);
  const yticks = Array.from({ length: Math.floor(ymax / 4) + 1 }, (_, i) => i * 4);
  return { X, Y, line, band, xticks, yticks };
}
