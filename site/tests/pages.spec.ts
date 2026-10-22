import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, readdirSync } from 'node:fs';

const read = (f: string) => JSON.parse(readFileSync(`data/${f}`, 'utf8'));
const glyphs = read('glyphs.json');
const paths = ['/', '/capture/', '/cleaning/', '/printing-leaves-a-mark/', '/glyphs/', '/typeset/', '/data/', '/about/', ...glyphs.map((g: { name: string }) => `/glyphs/${g.name}/`)];

for (const p of paths) {
  test(`${p} builds, has a title and the Fan Content notice, and passes axe`, async ({ page }) => {
    // reveal fades make contrast depend on timing, so the audit runs with motion reduced
    await page.emulateMedia({ reducedMotion: 'reduce' });
    const res = await page.goto(p);
    expect(res?.status()).toBe(200);
    await expect(page).toHaveTitle(/\S/);
    await expect(page.locator('footer.site')).toContainText('Fan Content Policy');
    const out = await new AxeBuilder({ page }).analyze();
    expect(out.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
  });
}

test('no page calls the font Plantin in title or description', async ({ page }) => {
  for (const p of paths) {
    await page.goto(p);
    expect(await page.title()).not.toMatch(/plantin/i);
    const d = await page.locator('meta[name=description]').getAttribute('content');
    expect(d).not.toMatch(/plantin/i);
  }
});

test('the figures on the data page come from the exported data, not the home page', async ({ page }) => {
  const e = read('glyphs/e.json'), n = read('noise.json'), cal = read('calibration.json'), cards = read('cards.json'), meta = read('meta.json');
  const levels = Object.values(cards).flatMap((c: any) => c.scans.map((s: { edge_level: number }) => s.edge_level));
  const enough = glyphs.filter((g: { n: number }) => g.n >= meta.min_instances).length;
  const figures = [
    String(e.n), String(enough), `${n.alternative.single_scan_um.toFixed(1)} µm`, `${n.alternative.pair_mean_um.toFixed(1)} µm`,
    cal.summary.px_per_mm.toFixed(2), `${Math.min(...levels).toFixed(2)}–${Math.max(...levels).toFixed(2)}`,
  ];
  await page.goto('/data/');
  const body = await page.locator('main').innerText();
  for (const f of figures) expect(body).toContain(f);
  await page.goto('/');
  const home = await page.locator('main').innerText();
  for (const f of [figures[2], figures[4], figures[5]]) expect(home).not.toContain(f);
});

test('data page shows the measured edge width range and the tied noise pair', async ({ page }) => {
  await page.goto('/data/');
  const w = read('calibration.json').edge_w1090_um as number[];
  const body = await page.locator('main').innerText();
  expect(body).toContain(`${Math.min(...w).toFixed(0)} to ${Math.max(...w).toFixed(0)} µm`);
  expect(body).toContain(`${read('noise.json').headline.single_scan_um.toFixed(1)} µm`);
});

test('every exported glyph has a page and per-glyph files stay in the repo data', () => {
  const files = readdirSync('data/glyphs').map((f) => f.replace('.json', '')).sort();
  expect(files).toEqual(glyphs.map((g: { name: string }) => g.name).sort());
});

test('the data page carries the glyph, stacking and spacing tables', async ({ page }) => {
  await page.goto('/data/');
  await expect(page.locator('div[aria-label="Glyph statistics table"] tbody tr')).toHaveCount(glyphs.length);
  await expect(page.locator('details[id^="glyph-"]')).toHaveCount(glyphs.length);
  const pairs = Object.keys(read('kerning.json')).length;
  await expect(page.locator('div[aria-label="Observed spacing table"] tbody tr')).toHaveCount(pairs);
});

test('notes are plain text with no proofreader label', async ({ page }) => {
  await page.goto('/');
  const content = await page.locator('.proof').first().evaluate((e) => getComputedStyle(e, '::before').content);
  expect(['none', 'normal']).toContain(content);
  await expect(page.locator('main')).not.toContainText('stet');
});
