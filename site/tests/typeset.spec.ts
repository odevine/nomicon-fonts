import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { existsSync, readFileSync } from 'node:fs';

const read = (f: string) => JSON.parse(readFileSync(`data/${f}`, 'utf8'));
// the Compositor sets the regular face, an italic row shares its character with a regular one
const glyphs: { name: string; char: string; style: string }[] = read('glyphs.json').filter((g: { style: string }) => g.style === 'regular');
const adv: Record<string, number> = Object.fromEntries(glyphs.map((g) => [g.char, read(`glyphs/${g.name}.json`).adv]));
const kern: Record<string, { n: number; mean_dev_nokern: number; kern: number }> = read('kerning.json');
const metrics = read('metrics.json');
const unkerned = Object.values(kern).filter((v) => v.kern === 0);
const avg = unkerned.reduce((s, v) => s + v.n * v.mean_dev_nokern, 0) / unkerned.reduce((s, v) => s + v.n, 0);
const PHRASE = 'yet a proud sailor can hunt.';

const offset = (a: string, b: string, mode: 'observed' | 'designed') => {
  if (a === ' ' || b === ' ') return 0;
  const p = kern[a + b];
  if (mode === 'designed') return p ? p.kern : 0;
  return p && p.n >= 3 ? p.mean_dev_nokern : avg;
};
const expectedOrigins = (text: string, mode: 'observed' | 'designed') => {
  const xs: number[] = [];
  let x = 0, prev: string | null = null;
  for (const ch of text) {
    const a = ch === ' ' ? metrics.space_adv : adv[ch];
    const o = prev === null ? 0 : offset(prev, ch, mode);
    if (ch !== ' ') xs.push(x + o);
    x = x + o + a;
    prev = ch;
  }
  return xs;
};
const origins = (page: import('@playwright/test').Page) => page.$$eval('.set-svg .gl', (els) => els.map((e) => +(e as HTMLElement).dataset.x!));

test.beforeEach(async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/typeset/');
  await page.waitForSelector('.set-svg .gl');
  await page.locator('#ts-size').fill('40');
  // let the re-render that the size change starts finish before a test reads the letters
  await page.waitForTimeout(150);
});

test('the default phrase is set with the spacing observed on the cards', async ({ page }) => {
  const got = await origins(page);
  const want = expectedOrigins(PHRASE, 'observed');
  expect(got).toHaveLength(want.length);
  got.forEach((x, i) => expect(x).toBeCloseTo(want[i], 1));
  expect(await page.locator('.ts-spacing').innerText()).toContain('letter pairs were measured');
});

test('the spacing note counts the measured and the averaged pairs from the data', async ({ page }) => {
  let measured = 0, averaged = 0;
  const known = (c: string) => c in adv;
  for (let i = 1; i < PHRASE.length; i++) {
    const a = PHRASE[i - 1], b = PHRASE[i];
    if (!known(a) || !known(b)) continue;
    const p = kern[a + b];
    if (p && p.n >= 3) measured++; else averaged++;
  }
  await expect(page.locator('.ts-spacing')).toContainText(`${measured} of ${measured + averaged} letter pairs`);
});

test('designed kerning moves the pairs the font kerns', async ({ page }) => {
  const observed = await origins(page);
  await page.locator('[name=ts-mode][value=designed]').check();
  const designed = await origins(page);
  const want = expectedOrigins(PHRASE, 'designed');
  designed.forEach((x, i) => expect(x).toBeCloseTo(want[i], 1));
  expect(designed).not.toEqual(observed);
});

test('a repeated letter uses a different printed copy each time, and the mean removes the variation', async ({ page }) => {
  await page.locator('#ts-text').fill('eeee');
  await expect(page.locator('.set-svg .gl')).toHaveCount(4);
  const ids = await page.$$eval('.set-svg .gl', (els) => els.map((e) => (e as HTMLElement).dataset.id));
  expect(new Set(ids).size).toBe(4);
  const d1 = await page.$$eval('.set-svg .gl path', (els) => els.map((e) => e.getAttribute('d')));
  expect(new Set(d1).size).toBe(4);
  await page.locator('[data-again]').click();
  const ids2 = await page.$$eval('.set-svg .gl', (els) => els.map((e) => (e as HTMLElement).dataset.id));
  expect(ids2).not.toEqual(ids);
  await page.locator('[name=ts-var][value=mean]').check();
  const d2 = await page.$$eval('.set-svg .gl path', (els) => els.map((e) => e.getAttribute('d')));
  expect(new Set(d2).size).toBe(1);
  expect(d2[0]).toBe(read('glyphs/e.json').trace);
  await expect(page.locator('[data-again]')).toBeDisabled();
});

test('letters that are not scanned appear as dashed cells that say so', async ({ page }) => {
  await page.locator('#ts-text').fill('Qa 3');
  await expect(page.locator('.set-svg .ghost')).toHaveCount(2);
  await expect(page.locator('.set-svg .ghost title').first()).toHaveText('Q: not yet scanned');
  await expect(page.locator('.ts-status')).toContainText('Not yet scanned: Q 3');
});

test('long text wraps inside the figure', async ({ page }) => {
  await page.locator('#ts-size').fill('120');
  await page.locator('#ts-text').fill('learn to read the lines until the line has to turn over and the next one begins');
  const lines = await page.evaluate(() => new Set([...document.querySelectorAll('.set-svg .gl')].map((g) => g.getAttribute('transform')!.split(' ')[1])).size);
  expect(lines).toBeGreaterThan(1);
});

test('the image download works and the page passes axe once mounted', async ({ page }) => {
  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('[data-download]').click()]);
  expect(dl.suggestedFilename()).toBe('nomicon-rules-specimen.png');
  const out = await new AxeBuilder({ page }).analyze();
  expect(out.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
});

test('the sample buttons and keyboard path work', async ({ page }) => {
  await page.getByRole('button', { name: 'senseless' }).click();
  await expect(page.locator('#ts-text')).toHaveValue('senseless');
  await expect(page.locator('.set-svg .gl')).toHaveCount(9);
  await page.locator('#ts-text').focus();
  await page.keyboard.press('Tab');
  expect(await page.evaluate(() => document.activeElement?.tagName)).toBe('BUTTON');
});

test('without scripts the phrase is still set', async ({ browser }) => {
  const ctx = await browser.newContext({ javaScriptEnabled: false });
  const page = await ctx.newPage();
  await page.goto('/typeset/');
  await expect(page.locator('type-setter svg title')).toHaveText(PHRASE);
  expect(await page.locator('type-setter svg path').count()).toBe(PHRASE.replace(/[ ]/g, '').length);
  await ctx.close();
});

test('the set letters are plain, with no hover readout, tooltip or hover colour', async ({ page }) => {
  expect(await page.locator('.ts-hover').count()).toBe(0);
  expect(await page.locator('.set-svg .gl title').count()).toBe(0);
  const first = page.locator('.set-svg .gl path').first();
  await expect(first).toHaveCSS('fill', 'rgb(22, 22, 22)');
  const box = (await first.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await expect(first).toHaveCSS('fill', 'rgb(22, 22, 22)');
});

test('a long string with no spaces wraps inside the figure', async ({ page }) => {
  await page.locator('#ts-size').fill('120');
  await page.locator('#ts-text').fill('unreadunreadunreadunreadunreadunreadunreadunreadunreadunread');
  await page.waitForTimeout(150);
  const r = await page.evaluate(() => {
    const w = +document.querySelector('.set-svg')!.getAttribute('viewBox')!.split(' ')[2];
    const gs = [...document.querySelectorAll('.set-svg .gl')];
    return { w, ys: new Set(gs.map((g) => g.getAttribute('transform')!.split(' ')[1])).size, xs: gs.map((g) => +(g as HTMLElement).dataset.x!) };
  });
  expect(r.ys).toBeGreaterThan(1);
  expect(r.xs.every((x) => x >= 0)).toBe(true);
  const rect = await page.locator('.set-svg').boundingBox();
  const right = await page.$$eval('.set-svg .gl path', (els) => Math.max(...els.map((e) => e.getBoundingClientRect().right)));
  expect(right).toBeLessThanOrEqual(rect!.x + rect!.width + 1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('radio buttons use the site palette', async ({ page }) => {
  const on = page.locator('[name=ts-mode]:checked');
  await expect(on).toHaveCSS('border-top-color', 'rgb(214, 35, 42)');
  expect(await on.evaluate((e) => getComputedStyle(e).appearance)).toBe('none');
  await expect(page.locator('[name=ts-mode]:not(:checked)').first()).toHaveCSS('border-top-color', 'rgb(22, 22, 22)');
});

test('each radio button sits on the first line of its label', async ({ page }) => {
  const rows = await page.$$eval('.toggles label', (ls) => ls.map((l) => {
    const i = l.querySelector('input')!.getBoundingClientRect(), r = l.getBoundingClientRect();
    return Math.abs((i.top + i.height / 2) - (r.top + parseFloat(getComputedStyle(l).lineHeight) / 2));
  }));
  expect(rows.length).toBe(4);
  for (const dy of rows) expect(dy).toBeLessThan(1.5);
});


// the italic face, set between asterisks
const italic: { name: string; char: string; style: string }[] = read('glyphs.json').filter((g: { style: string }) => g.style === 'italic');
const advIt: Record<string, number> = Object.fromEntries(italic.map((g) => [g.char, read(`glyphs/${g.name}.json`).adv]));
const kernIt: Record<string, { n: number; mean_dev_nokern: number; kern: number }> = existsSync('data/kerning.italic.json') ? read('kerning.italic.json') : {};
const unkernedIt = Object.values(kernIt).filter((v) => v.kern === 0);
const avgIt = unkernedIt.reduce((s, v) => s + v.n * v.mean_dev_nokern, 0) / unkernedIt.reduce((s, v) => s + v.n, 0);
const both = (a: string, b: string) => a in advIt && b in advIt;

test.describe('italic', () => {
  test.skip(italic.length === 0, 'no italic glyphs in the data');
  const [a, b] = Object.keys(advIt).filter((c) => /[a-z]/.test(c));

  test('text between asterisks is set in italic glyphs and the asterisks are not set', async ({ page }) => {
    await page.locator('#ts-text').fill(`*${a}${b}*${a}`);
    await expect(page.locator('.set-svg .gl')).toHaveCount(3);
    expect(await page.$$eval('.set-svg .gl', (els) => els.map((e) => (e as HTMLElement).dataset.ch))).toEqual([`i:${a}`, `i:${b}`, a]);
    await expect(page.locator('.set-svg .ghost')).toHaveCount(0);
    expect(await page.$$eval('.set-svg .gl', (els) => els.map((e) => (e as HTMLElement).dataset.id!.includes('-it-')))).toEqual([true, true, false]);
  });

  test('italic letters use the italic advances and pairs', async ({ page }) => {
    test.skip(!both(a, b));
    await page.locator('#ts-text').fill(`*${a}${b}*`);
    await expect(page.locator('.set-svg .gl')).toHaveCount(2);
    const p = kernIt[a + b];
    const off = p && p.n >= 3 ? p.mean_dev_nokern : avgIt;
    const xs = await origins(page);
    expect(xs[0]).toBeCloseTo(0, 1);
    expect(xs[1]).toBeCloseTo(advIt[a] + off, 1);
  });

  test('where the face changes the second letter takes the average offset of its own face', async ({ page }) => {
    await page.locator('#ts-text').fill(`${a}*${b}*`);
    await expect(page.locator('.set-svg .gl')).toHaveCount(2);
    const xs = await origins(page);
    expect(xs[1]).toBeCloseTo(adv[a] + avgIt, 1);
  });

  test('an italic letter that is not scanned says so', async ({ page }) => {
    await page.locator('#ts-text').fill('*Q*');
    await expect(page.locator('.set-svg .ghost title')).toHaveText('Q (italic): not yet scanned');
    await expect(page.locator('.ts-status')).toContainText('Not yet scanned: Q (italic)');
  });

  test('the italic sample button fills the text and the hint explains the asterisks', async ({ page }) => {
    await expect(page.locator('.ts-hint')).toContainText('asterisks');
    await page.getByRole('button', { name: /read the stars/ }).click();
    await expect(page.locator('#ts-text')).toHaveValue('learn to *read the stars*');
    await expect(page.locator('.set-svg .gl')).toHaveCount('learn to read the stars'.replace(/ /g, '').length);
    expect(await page.$$eval('.set-svg .gl', (els) => els.filter((e) => (e as HTMLElement).dataset.ch!.startsWith('i:')).length)).toBe('readthestars'.length);
  });
});
