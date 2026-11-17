import { glyph } from './data';
import { bbox } from './format';

// the stacked n, traced from its own printed copies, centred in a square with a little room around it
export function iconSvg({ tile = false } = {}) {
  const d = glyph('n').trace;
  const b = bbox(d);
  const side = Math.max(b.x1 - b.x0, b.y1 - b.y0) * 1.12;
  const x = (b.x0 + b.x1 - side) / 2, y = (b.y0 + b.y1 - side) / 2;
  const f = (v: number) => +v.toFixed(1);
  const vb = `${f(x)} ${f(y)} ${f(side)} ${f(side)}`;
  // a home screen icon needs an opaque square, the tab icon follows the browser's light or dark theme
  const body = tile
    ? `<rect x="${f(x)}" y="${f(y)}" width="${f(side)}" height="${f(side)}" fill="#fbfaf7"/><path d="${d}" fill="#161616" fill-rule="evenodd"/>`
    : `<style>path{fill:#161616}@media (prefers-color-scheme:dark){path{fill:#fbfaf7}}</style><path d="${d}" fill-rule="evenodd"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vb}">${body}</svg>`;
}
