// Serves a slimmed copy of site/data to the browser: no cutaways, instance paths split into their own files and re-simplified as relative integers at half a font unit
import { readFileSync, readdirSync, mkdirSync, writeFileSync, cpSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { spacingTable } from './spacing.mjs';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../data');
export const INSTANCE_TOL = 0.75;
const STEP = 0.5;

function rdp(pts, tol) {
  if (pts.length < 3) return pts;
  const [a, b] = [pts[0], pts[pts.length - 1]];
  const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1e-9;
  let imax = 0, dmax = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const d = Math.abs(dy * (pts[i][0] - a[0]) - dx * (pts[i][1] - a[1])) / len;
    if (d > dmax) { dmax = d; imax = i; }
  }
  if (dmax <= tol) return [a, b];
  return [...rdp(pts.slice(0, imax + 1), tol).slice(0, -1), ...rdp(pts.slice(imax), tol)];
}

function closed(pts, tol) {
  let far = 0, dm = 0;
  for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[0][0], pts[i][1] - pts[0][1]); if (d > dm) { dm = d; far = i; } }
  const h1 = rdp(pts.slice(0, far + 1), tol), h2 = rdp([...pts.slice(far), pts[0]], tol);
  return [...h1.slice(0, -1), ...h2.slice(0, -1)];
}

/** Re-encodes absolute M/L polygons as relative integer-step paths (the unit is STEP font units) */
export function compact(d, tol) {
  return d.split('M').slice(1).map((s) => {
    const pts = s.replace('Z', '').split('L').map((p) => p.split(',').map(Number));
    const q = closed(pts, tol).map(([x, y]) => [Math.round(x / STEP), Math.round(y / STEP)]);
    if (q.length < 3) return '';
    let o = `M${q[0][0] * STEP},${q[0][1] * STEP}l`;
    o += q.slice(1).map((p, i) => `${(p[0] - q[i][0]) * STEP},${(p[1] - q[i][1]) * STEP}`).join(' ');
    return o + 'z';
  }).join('');
}

/** What the glyph page needs: the trace, the traces at each N and the glyph's numbers. Instances and the heavier layers are not sent */
export function servedGlyph(doc) {
  const { cutaways, instances, mean, spread, ...rest } = doc;
  return rest;
}

/** What the Compositor needs from one glyph: its advance, mean trace and the sampled instance outlines */
export function servedCompose(doc) {
  return { char: doc.char, adv: doc.adv, trace: doc.trace, instances: doc.instances.map((i) => ({ id: i.id, card: i.card, path: compact(i.path, INSTANCE_TOL) })) };
}

export function servedInstances(doc) {
  const { instances, instances_total, instances_exported, name, char } = doc;
  return { name, char, instances_total, instances_exported, instances: instances.map((i) => ({ ...i, path: compact(i.path, INSTANCE_TOL) })) };
}

const json = (f) => JSON.parse(readFileSync(resolve(root, f), 'utf8'));
const names = () => readdirSync(resolve(root, 'glyphs')).map((f) => f.replace('.json', ''));

function body(rel) {
  let m;
  if ((m = rel.match(/^glyphs\/([\w-]+)\.json$/))) return JSON.stringify(servedGlyph(json(`glyphs/${m[1]}.json`)));
  if (rel === 'spacing.json') return JSON.stringify(spacingTable());
  if ((m = rel.match(/^compose\/([\w-]+)\.json$/))) return JSON.stringify(servedCompose(json(`glyphs/${m[1]}.json`)));
  if ((m = rel.match(/^instances\/([\w-]+)\.json$/))) return JSON.stringify(servedInstances(json(`glyphs/${m[1]}.json`)));
  if (/^(sweep\/[\w-]+|cards)\.json$/.test(rel)) return readFileSync(resolve(root, rel), 'utf8');
  return null;
}

// raw crops, step images and their transforms are served as they are
const STATIC_DIRS = ['steps', 'pixels', 'noise'];
const staticFile = (rel) => (/^(steps|pixels|noise)\/[\w.-]+\.(json|webp)$/.test(rel) && existsSync(resolve(root, rel)) ? resolve(root, rel) : null);

export default function servedData() {
  let base = '/';
  return {
    name: 'served-data',
    hooks: {
      'astro:config:done': ({ config }) => { base = config.base.replace(/\/?$/, '/'); },
      'astro:server:setup': ({ server }) => {
        server.middlewares.use((req, res, next) => {
          const p = (req.url ?? '').split('?')[0];
          const pre = `${base}data/`;
          const rel = p.startsWith(pre) ? p.slice(pre.length) : '';
          const file = rel ? staticFile(rel) : null;
          if (file) {
            res.setHeader('content-type', file.endsWith('.webp') ? 'image/webp' : 'application/json');
            return res.end(readFileSync(file));
          }
          const out = rel ? body(rel) : null;
          if (out === null) return next();
          res.setHeader('content-type', 'application/json');
          res.end(out);
        });
      },
      'astro:build:done': ({ dir }) => {
        const out = resolve(fileURLToPath(dir), 'data');
        for (const sub of ['glyphs', 'sweep', 'instances', 'compose']) mkdirSync(resolve(out, sub), { recursive: true });
        for (const n of names()) for (const sub of ['glyphs', 'sweep', 'instances', 'compose']) writeFileSync(resolve(out, sub, `${n}.json`), body(`${sub}/${n}.json`));
        writeFileSync(resolve(out, 'cards.json'), body('cards.json'));
        writeFileSync(resolve(out, 'spacing.json'), body('spacing.json'));
        for (const sub of STATIC_DIRS) if (existsSync(resolve(root, sub))) cpSync(resolve(root, sub), resolve(out, sub), { recursive: true });
      },
    },
  };
}
