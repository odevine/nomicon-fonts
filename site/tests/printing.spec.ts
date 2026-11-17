import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';

const read = (f: string) => JSON.parse(readFileSync(`data/${f}`, 'utf8'));
const e = read('glyphs/e.json');
const glyphs = read('glyphs.json');

const corners = read('corners.json');

test('the Bleed figure draws the font and its slider sets the strength', async ({ page }) => {
  await page.goto('/bleed/');
  await expect(page.locator('.bleed-fig text')).toHaveText('x');
  await expect(page.locator('.bleed-fig .bf-layers')).toHaveCount(0);
  const slider = page.locator('#bf-bled');
  await expect(slider).toHaveAttribute('min', '25');
  await expect(slider).toHaveAttribute('max', '125');
  await slider.fill('40');
  expect(await page.evaluate(() => getComputedStyle(document.querySelector('.bf-live')!).fontVariationSettings)).toContain('40');
});

test('the corner figures and numbers come from corners.json', async ({ page }) => {
  await page.goto('/bleed/');
  const figs = corners.regular.figures;
  await expect(page.locator('.corner-figs figure')).toHaveCount(figs.length);
  await expect(page.locator('.corner-figs .cf-stick')).toHaveCount(figs.length);
  await expect(page.locator('.corner-figs figcaption').first()).toContainText(`${figs[0].corners[0].angle}°`);
  await expect(page.locator('.corner-figs figcaption').last()).toContainText(`${figs[1].corners[1].angle}°`);
  const fill = corners.regular.bins.find((b: { kind: string; lo: number }) => b.kind === 'inner' && b.lo === 0);
  await expect(page.locator('section[aria-labelledby=corners]')).toContainText(`about ${fill.stack_um.toFixed(0)} µm`);
  await page.goto('/data/');
  const rows = Object.values(corners as Record<string, { bins: unknown[] }>).reduce((n, c) => n + c.bins.length, 0);
  await expect(page.locator('#corner-table tbody tr')).toHaveCount(rows);
});

test('every exported e is drawn once in the overlay, under the trace', async ({ page }) => {
  await page.goto('/bleed/');
  await expect(page.locator('.eo-copy')).toHaveCount(e.instances.length);
  await expect(page.locator('.e-over path').last()).toHaveClass('eo-trace');
});

test('the closeness numbers use the exported numbers', async ({ page }) => {
  await page.goto('/bleed/');
  const scanned = glyphs.filter((g: { source: string }) => g.source === 'scanned');
  const worst = Math.max(...scanned.map((g: { edge_rms_um: number }) => g.edge_rms_um));
  expect(worst).toBeLessThan(25400 / 1200);
  await expect(page.locator('main')).toContainText('Every letter is closer than one scanner pixel');
  await expect(page.locator('.stat-line')).toContainText(`${scanned.length} letters`);
});

test('the face is never named Plantin', async ({ page }) => {
  await page.goto('/bleed/');
  const text = await page.locator('main').innerText();
  expect(text).not.toMatch(/Plantin/);
  expect(await page.title()).not.toMatch(/Plantin/);
});

test('the Printing page passes axe', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/bleed/');
  const out = await new AxeBuilder({ page }).analyze();
  expect(out.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
});
