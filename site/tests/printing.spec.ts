import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';

const read = (f: string) => JSON.parse(readFileSync(`data/${f}`, 'utf8'));
const e = read('glyphs/e.json');
const cards = read('cards.json');
const glyphs = read('glyphs.json');

test('the Printing page draws different printed e\'s for each card from the data', async ({ page }) => {
  await page.goto('/printing-leaves-a-mark/');
  const cardsInSample = new Set(e.instances.map((i: { card: string }) => i.card));
  await expect(page.locator('.erow')).toHaveCount(cardsInSample.size);
  for (const row of await page.locator('.erow').all()) expect(await row.locator('svg').count()).toBeLessThanOrEqual(6);
  const ds = await page.$$eval('.eset path', (els) => els.map((p) => p.getAttribute('d')));
  expect(new Set(ds).size).toBe(ds.length);
  expect(ds.length).toBeGreaterThan(cardsInSample.size);
});

test('the size chart and the closeness numbers use the exported numbers', async ({ page }) => {
  await page.goto('/printing-leaves-a-mark/');
  await expect(page.locator('.size-row')).toHaveCount(Object.keys(cards).length);
  const top = Math.max(...Object.values(cards as Record<string, { print_scale: number }>).map((c) => c.print_scale));
  await expect(page.locator('.size-val').first()).toHaveText(top.toFixed(2));
  const worst = Math.max(...glyphs.map((g: { edge_rms_um: number }) => g.edge_rms_um));
  expect(worst).toBeLessThan(25400 / 1200);
  await expect(page.locator('main')).toContainText('Every letter is closer than one scanner pixel');
  await expect(page.locator('.stat-line')).toContainText(`${glyphs.length} letters`);
});

test('the face is never named Plantin', async ({ page }) => {
  await page.goto('/printing-leaves-a-mark/');
  const text = await page.locator('main').innerText();
  expect(text).not.toMatch(/Plantin/);
  expect(await page.title()).not.toMatch(/Plantin/);
});

test('the Printing page passes axe', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/printing-leaves-a-mark/');
  const out = await new AxeBuilder({ page }).analyze();
  expect(out.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
});
