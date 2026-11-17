import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';

const read = (f: string) => JSON.parse(readFileSync(`data/${f}`, 'utf8'));
const cal = read('calibration.json');
const noise = read('noise.json');

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
  expect(Math.abs(cal.tone.a[3] - 114)).toBeLessThanOrEqual(2);
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
  const h = noise.headline.hist;
  expect(h.counts.reduce((a: number, b: number) => a + b, 0) + h.over).toBe(h.n);
  expect(h.n).toBe(noise.headline.n_pairs);
  expect(h.rms_um).toBeCloseTo(noise.headline.pair_rms_um, 2);
  await page.goto('/capture/');
  await expect(page.locator('main')).toContainText(`${noise.headline.single_scan_um.toFixed(1)} µm`);
  await expect(page.locator('main')).toContainText(`${noise.headline.pair_mean_um.toFixed(1)} µm`);
  expect(await page.locator('.hist').count()).toBe(0);
  await page.goto('/data/');
  await expect(page.locator('.hist .bar')).toHaveCount(h.counts.filter((c: number) => c > 0).length);
});

test('the Capture page passes axe', async ({ page }) => {
  await page.goto('/capture/');
  const out = await new AxeBuilder({ page }).analyze();
  expect(out.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
});
