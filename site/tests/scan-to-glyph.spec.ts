import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, readdirSync } from 'node:fs';

const read = (f: string) => JSON.parse(readFileSync(`data/${f}`, 'utf8'));
const s = read('steps/e.json');
const noise = read('noise.json').alternative;

test('the Scan to Glyph sequence has ten steps with working images and passes axe', async ({ page }) => {
  const bad: string[] = [];
  page.on('response', (r) => { if (r.status() >= 400) bad.push(r.url()); });
  await page.goto('/cleaning/');
  await expect(page.locator('.step')).toHaveCount(10);
  await page.waitForLoadState('networkidle');
  expect(bad).toEqual([]);
  const broken = await page.$$eval('.panel image', (els) => els.filter((e) => !(e as SVGImageElement).getBoundingClientRect().width).length);
  expect(broken).toBe(0);
  const out = await new AxeBuilder({ page }).analyze();
  expect(out.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
});

test('step texts read their numbers from the exported data', async ({ page }) => {
  await page.goto('/cleaning/');
  const t = (n: number) => page.locator(`#step-${n}`).innerText();
  expect(await t(3)).toContain(`paper ${s.paper.toFixed(0)}, ink ${s.ink.toFixed(0)}`);
  expect(await t(6)).toContain(`${s.pair_dev_um.toFixed(1)} µm`);
  expect(await t(6)).toContain(`${noise.single_scan_um.toFixed(1)} µm`);
  expect(await t(6)).toContain(`${noise.pair_mean_um.toFixed(1)} µm`);
  expect(await t(8)).toContain(`${read('glyphs/e.json').n} printed`);
  expect(await t(1)).toContain('Reef Worm');
});

test('step 9 draws the exported number of control and node markers', async ({ page }) => {
  await page.goto('/cleaning/');
  const e = read('glyphs/e.json');
  const nodes = await page.locator('#step-9 .nd').count();
  const handles = await page.locator('#step-9 .hnd').count();
  expect(handles).toBe(2 * (e.trace.match(/C/g) ?? []).length);
  expect(nodes).toBeGreaterThanOrEqual(e.nodes);
});

test('every published crop and step image is small, never a whole card', async ({ page }) => {
  const files = [...readdirSync('data/steps'), ...readdirSync('data/pixels')].filter((f) => f.endsWith('.webp'));
  expect(files.length).toBeGreaterThan(20);
  const sizes: number[] = [];
  for (const dir of ['steps', 'pixels']) {
    for (const f of readdirSync(`data/${dir}`).filter((x) => x.endsWith('.webp'))) {
      const res = await page.request.get(`/data/${dir}/${f}`);
      expect(res.status(), f).toBe(200);
      const buf = await res.body();
      // VP8L lossless header: 14 bit width and height minus one after the signature byte
      const i = buf.indexOf('VP8L');
      const b = buf.subarray(i + 9, i + 13);
      const w = 1 + (((b[1] & 0x3f) << 8) | b[0]);
      sizes.push(w);
    }
  }
  expect(Math.max(...sizes)).toBeLessThanOrEqual(480);
});

test('with reduced motion every step is visible without scrolling into view', async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: 'reduce' });
  const page = await ctx.newPage();
  await page.goto('/cleaning/');
  await expect(page.locator('#step-10 .panel')).toHaveCSS('opacity', '1');
  await ctx.close();
});

test('the step index highlights the step in view', async ({ page }) => {
  await page.goto('/cleaning/');
  await page.locator('.seq-nav a[href="#step-7"]').click();
  await expect(page.locator('.seq-nav a[aria-current]')).toHaveAttribute('href', '#step-7');
});

test('each glyph has a pixel crop file that matches the glyph index', () => {
  const names = read('glyphs.json').map((g: { name: string }) => g.name);
  for (const n of names) expect(() => read(`pixels/${n}.json`), n).not.toThrow();
});
