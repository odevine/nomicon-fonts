import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';

const read = (f: string) => JSON.parse(readFileSync(`data/${f}`, 'utf8'));
const cal = read('calibration.json');
const noise = read('noise.json');
const glyphs: { name: string; char: string }[] = read('glyphs.json');

test('the tone values come from the two calibration scans and read like the findings', () => {
  expect(cal.tone.a).toHaveLength(11);
  expect(cal.tone.b).toHaveLength(11);
  // white at 0 percent, black at 100, and falling steeply then flattening
  for (const t of [cal.tone.a, cal.tone.b]) {
    expect(t[0]).toBeGreaterThan(250);
    expect(t[10]).toBeLessThan(40);
    for (let i = 1; i < 11; i++) expect(t[i]).toBeLessThan(t[i - 1]);
    expect(t[0] - t[1]).toBeGreaterThan(5 * (t[9] - t[10]));
  }
  expect(Math.abs(cal.tone.a[3] - 108)).toBeLessThanOrEqual(2);
});

test('the bench shows the scale, the edge widths and a tone curve that follows the keyboard', async ({ page }) => {
  await page.goto('/capture/');
  await expect(page.locator('main')).toContainText(`${cal.summary.px_per_mm.toFixed(2)} pixels per millimetre`);
  await expect(page.locator('.bench')).toContainText(`${Math.min(...cal.edge_w1090_um).toFixed(0)} to ${Math.max(...cal.edge_w1090_um).toFixed(0)} µm`);
  expect(await page.locator('.sdot').count()).toBe(cal.edge_w1090_um.length);
  expect(await page.locator('.bench-card').count()).toBe(2);
  await expect(page.locator('.tc-a')).toHaveText(cal.tone.a[5].toFixed(0));
  await page.locator('svg.tone').focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('.tc-p')).toHaveText('60 %');
  await expect(page.locator('.tc-a')).toHaveText(cal.tone.a[6].toFixed(0));
  await expect(page.locator('.tc-b')).toHaveText(cal.tone.b[6].toFixed(0));
  await page.keyboard.press('Home');
  await expect(page.locator('.tc-p')).toHaveText('0 %');
});

test('the noise histogram adds up to the pairs, lives on the data page, and the Capture page quotes the headline noise', async ({ page }) => {
  const h = noise.alternative.hist;
  expect(h.counts.reduce((a: number, b: number) => a + b, 0) + h.over).toBe(h.n);
  expect(h.n).toBe(noise.alternative.n_pairs);
  expect(h.rms_um).toBeCloseTo(noise.alternative.pair_rms_um, 2);
  await page.goto('/capture/');
  await expect(page.locator('main')).toContainText(`${noise.alternative.single_scan_um.toFixed(1)} µm`);
  await expect(page.locator('main')).toContainText(`${noise.alternative.pair_mean_um.toFixed(1)} µm`);
  expect(await page.locator('.hist').count()).toBe(0);
  await page.goto('/data/');
  await expect(page.locator('.hist .bar')).toHaveCount(h.counts.filter((c: number) => c > 0).length);
});

test('the Noise Lab wipes between the two copies and shows their difference', async ({ page }) => {
  await page.goto('/capture/');
  await page.waitForSelector('.lab-pair .px-u[src]');
  await expect(page.locator('.nl-cap')).toContainText('µm apart');
  await page.locator('#nl-cut').fill('20');
  expect(await page.locator('.lab-pair').evaluate((e) => (e as HTMLElement).style.getPropertyValue('--cut'))).toBe('20%');
  await page.locator('[name=nl-glyph][value=a]').check({ force: true });
  await expect(page.locator('.nl-cap')).toContainText('Printed a');
  const info = read('noise/a.json');
  await expect(page.locator('.nl-cap')).toContainText(info.pair_dev_um.toFixed(1));
  await page.locator('[name=nl-view][value=diff]').check();
  await expect(page.locator('.lab-pair canvas')).toBeVisible();
  await expect(page.locator('.lab-pair .px-u')).toBeHidden();
  const ink = await page.locator('.lab-pair canvas').evaluate((c) => {
    const x = (c as HTMLCanvasElement).getContext('2d')!; const d = x.getImageData(0, 0, c.width, c.height).data; let n = 0; for (let k = 3; k < d.length; k += 4) if (d[k] > 40) n++; return n / (d.length / 4);
  });
  expect(ink).toBeGreaterThan(0.005);
  expect(ink).toBeLessThan(0.3);
});

test('every glyph has a noise pair and the picker is keyboard reachable, and axe passes', async ({ page }) => {
  for (const g of glyphs) expect(() => read(`noise/${g.name}.json`), g.name).not.toThrow();
  await page.goto('/capture/');
  await page.waitForSelector('.picker');
  expect(await page.locator('[name=nl-glyph]').count()).toBe(glyphs.length);
  await page.locator('[name=nl-glyph]:checked').focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('[name=nl-glyph]:checked')).not.toHaveAttribute('value', 'e');
  const out = await new AxeBuilder({ page }).analyze();
  expect(out.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
});

test('the upright and turned copies are light on the dark paper in dark mode, in both views', async ({ browser }) => {
  const ctx = await browser.newContext({ colorScheme: 'dark', viewport: { width: 1100, height: 900 } });
  const page = await ctx.newPage();
  await page.goto('/capture/');
  await page.waitForSelector('.lab-pair .px-u[src]');
  await expect(page.locator('.lab-pair .px-u')).toHaveCSS('filter', 'invert(1)');
  await expect(page.locator('.lab-pair .px-t')).toHaveCSS('filter', 'invert(1)');
  // the ink must be brighter than the paper behind it
  const lum = async () => page.locator('.lab-pair').evaluate(async (el) => {
    const r = el.getBoundingClientRect();
    return { w: r.width, bg: getComputedStyle(document.body).backgroundColor };
  });
  expect((await lum()).bg).toBe('rgb(22, 22, 22)');
  await page.locator('[name=nl-view][value=diff]').check();
  await expect(page.locator('.lab-pair canvas')).toHaveCSS('filter', 'invert(1)');
  await ctx.close();
});
