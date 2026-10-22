// Builds the compact spacing table the Compositor needs: advance widths, font metrics and the observed pair offsets of both faces
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// built from the site folder, as the other data readers are
const root = resolve(process.cwd(), 'data');
const json = (f) => JSON.parse(readFileSync(resolve(root, f), 'utf8'));

/**
 * Glyphs are keyed by character for the regular face and by `i:` and the character for italic, pairs likewise
 * @returns {{metrics: Record<string, number>, italic_metrics?: Record<string, number>, glyphs: Record<string, {name: string, adv: number}>, pairs: Record<string, [number, number, number]>}}
 */
export function spacingTable() {
  const glyphs = {};
  for (const g of json('glyphs.json')) glyphs[g.style === 'italic' ? `i:${g.char}` : g.char] = { name: g.name, adv: json(`glyphs/${g.name}.json`).adv };
  const pairs = {};
  const add = (file, prefix) => {
    if (!existsSync(resolve(root, file))) return;
    for (const [k, v] of Object.entries(json(file))) pairs[prefix + k] = [v.n, Number(v.mean_dev_nokern.toFixed(2)), v.kern];
  };
  add('kerning.json', '');
  add('kerning.italic.json', 'i:');
  const metrics = json('metrics.json');
  const table = { metrics, glyphs, pairs };
  if (existsSync(resolve(root, 'metrics.italic.json'))) {
    const italic = json('metrics.italic.json');
    // glyphs are placed in one em, so a face on another grid would need scaling the Compositor does not do
    if (italic.upm !== metrics.upm) throw new Error(`the italic face has ${italic.upm} units per em, the regular face ${metrics.upm}`);
    table.italic_metrics = italic;
  }
  return table;
}
