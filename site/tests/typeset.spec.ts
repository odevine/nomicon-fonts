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
const PHRASE = 'Whenever you draw a card, Niv-Mizzet, the Firemind deals 1 damage to target creature or player.';
// a phrase short enough to stay on one line, for the tests that read letter origins along it
const LINE = 'yet a proud sailor can hunt.';

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

test('a line is set with the designed kerning', async ({ page }) => {
  await page.locator('#ts-text').fill(LINE);
  await expect(page.locator('.set-svg .gl')).toHaveCount(LINE.replace(/ /g, '').length);
  const got = await origins(page);
  const want = expectedOrigins(LINE, 'designed');
  expect(got).toHaveLength(want.length);
  got.forEach((x, i) => expect(x).toBeCloseTo(want[i], 1));
  await expect(page.locator('.ts-spacing')).toContainText('the designed kern value for every pair');
  await expect(page.locator('[name=ts-mode]')).toHaveCount(0);
});

test('the size starts at 48 px and stops at 256 px', async ({ page }) => {
  await page.goto('/typeset/');
  await expect(page.locator('#ts-size')).toHaveValue('48');
  await expect(page.locator('.ts-sout')).toHaveText('48 px');
  await expect(page.locator('#ts-size')).toHaveAttribute('max', '256');
});

test('every letter is drawn from Nomicon Rules and the letter options are hidden', async ({ page }) => {
  await page.locator('#ts-text').fill('eeee');
  await expect(page.locator('.set-svg .gl.font')).toHaveCount(4);
  await expect(page.locator('.set-svg .gl path')).toHaveCount(0);
  await expect(page.locator('.setter .toggles')).toBeHidden();
  await expect(page.locator('[data-again]')).toBeHidden();
  await expect(page.locator('.ts-status')).toHaveText('Every letter here is from Nomicon Rules.');
});

test('a character the font lacks appears as a dashed cell', async ({ page }) => {
  await page.locator('#ts-text').fill('Ka @ Ж');
  await expect(page.locator('.set-svg .font')).toHaveCount(3);
  await expect(page.locator('.set-svg .ghost')).toHaveCount(1);
  await expect(page.locator('.set-svg .ghost title')).toHaveText('Ж: not yet scanned');
  await expect(page.locator('.ts-status')).toContainText('except Ж');
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
  await page.getByRole('button', { name: 'Uglúk of the White Hand' }).click();
  await expect(page.locator('#ts-text')).toHaveValue('If that creature was a Goblin or Orc, put two +1/+1 counters on Uglúk instead.');
  await expect(page.locator('.set-svg .gl, .set-svg .font')).toHaveCount('If that creature was a Goblin or Orc, put two +1/+1 counters on Uglúk instead.'.replace(/ /g, '').length);
  await page.locator('#ts-text').focus();
  await page.keyboard.press('Tab');
  expect(await page.evaluate(() => document.activeElement?.tagName)).toBe('BUTTON');
});

test('without scripts the phrase is still set in the font', async ({ browser }) => {
  const ctx = await browser.newContext({ javaScriptEnabled: false });
  const page = await ctx.newPage();
  await page.goto('/typeset/');
  await expect(page.locator('type-setter .set-static')).toHaveText(PHRASE);
  await expect(page.locator('type-setter .set-static')).toHaveCSS('font-family', /Nomicon Rules/);
  await ctx.close();
});

test('the set letters are plain, with no hover readout, tooltip or hover colour', async ({ page }) => {
  expect(await page.locator('.ts-hover').count()).toBe(0);
  expect(await page.locator('.set-svg .gl title').count()).toBe(0);
  const first = page.locator('.set-svg .gl text').first();
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
  const right = await page.$$eval('.set-svg .gl', (els) => Math.max(...els.map((e) => e.getBoundingClientRect().right)));
  expect(right).toBeLessThanOrEqual(rect!.x + rect!.width + 1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});


// the italic face, set between asterisks
const italic: { name: string; char: string; style: string }[] = read('glyphs.json').filter((g: { style: string }) => g.style === 'italic');
const advIt: Record<string, number> = Object.fromEntries(italic.map((g) => [g.char, read(`glyphs/${g.name}.json`).adv]));
const kernIt: Record<string, { n: number; mean_dev_nokern: number; kern: number }> = existsSync('data/kerning.italic.json') ? read('kerning.italic.json') : {};
const both = (a: string, b: string) => a in advIt && b in advIt;

test.describe('italic', () => {
  test.skip(italic.length === 0, 'no italic glyphs in the data');
  const [a, b] = Object.keys(advIt).filter((c) => /[a-z]/.test(c));

  test('text between asterisks is set in italic glyphs and the asterisks are not set', async ({ page }) => {
    await page.locator('#ts-text').fill(`*${a}${b}*${a}`);
    await expect(page.locator('.set-svg .gl')).toHaveCount(3);
    expect(await page.$$eval('.set-svg .gl', (els) => els.map((e) => (e as HTMLElement).dataset.ch))).toEqual([`i:${a}`, `i:${b}`, a]);
    await expect(page.locator('.set-svg .ghost')).toHaveCount(0);
  });

  test('italic letters use the italic advances and designed kerning', async ({ page }) => {
    const pair = Object.keys(kernIt).find((k) => [...k].length === 2 && both(k[0], k[1]) && kernIt[k].kern !== 0);
    test.skip(!pair);
    await page.locator('#ts-text').fill(`*${pair}*`);
    await expect(page.locator('.set-svg .gl')).toHaveCount(2);
    const xs = await origins(page);
    expect(xs[0]).toBeCloseTo(0, 1);
    expect(xs[1]).toBeCloseTo(advIt[pair![0]] + kernIt[pair!].kern, 1);
  });

  test('where the face changes there is no kern', async ({ page }) => {
    await page.locator('#ts-text').fill(`${a}*${b}*`);
    await expect(page.locator('.set-svg .gl')).toHaveCount(2);
    const xs = await origins(page);
    expect(xs[1]).toBeCloseTo(adv[a], 1);
  });

  test('an italic letter the scans do not have is drawn from the font', async ({ page }) => {
    await page.locator('#ts-text').fill('*7*');
    await expect(page.locator('.set-svg .font')).toHaveCount(1);
    expect(await page.locator('.set-svg .font').getAttribute('data-ch')).toBe('i:7');
  });

  test('the italic sample button fills the text and the hint explains the asterisks', async ({ page }) => {
    await expect(page.locator('.ts-hint')).toContainText('asterisks');
    const text = 'Enforcing absolute quiet with extreme prejudice since 3285 AR.';
    await page.getByRole('button', { name: 'Automatic Librarian' }).click();
    await expect(page.locator('#ts-text')).toHaveValue(`*${text}*`);
    await expect(page.locator('.set-svg .gl, .set-svg .font')).toHaveCount(text.replace(/ /g, '').length);
    expect(await page.$$eval('.set-svg .gl', (els) => els.every((e) => (e as HTMLElement).dataset.ch!.startsWith('i:')))).toBe(true);
  });
});

test('the kern of a pair the cards did not measure comes from the font', async ({ page }) => {
  await page.locator('#ts-text').fill('AV');
  const second = () => page.$$eval('.set-svg .gl', (gs) => parseFloat(gs[1].getAttribute('transform')!.split(/[( ]/)[1]));
  const advance = await page.evaluate(() => {
    const ctx = document.createElement('canvas').getContext('2d')!;
    ctx.font = '1000px "Nomicon Rules Regular"';
    return ctx.measureText('A').width;
  });
  // the kern is measured on a canvas once the font has loaded, so the letters may be set again after the first pass
  await expect.poll(second).toBeLessThan(advance - 50);
  await expect(page.locator('.ts-spacing')).toContainText('the designed kern value for every pair');
});

test('the optical size follows the size until it is moved, and the set letters are drawn at it', async ({ page }) => {
  const opsz = page.locator('#ts-opsz');
  const drawn = () => page.evaluate(() => getComputedStyle(document.querySelector('.set-svg .font text')!).fontVariationSettings);
  await expect(opsz).toHaveAttribute('min', '6');
  await expect(opsz).toHaveAttribute('max', '72');
  await expect(opsz).toHaveValue('40');
  await expect.poll(drawn).toContain('"opsz" 40');
  await page.locator('#ts-size').fill('120');
  await expect(opsz).toHaveValue('72');
  await opsz.fill('8');
  await expect(page.locator('#ts-opsz-auto')).not.toBeChecked();
  await expect(page.locator('.ts-oout')).toHaveText('8');
  await expect.poll(drawn).toContain('"opsz" 8');
  await page.locator('#ts-size').fill('60');
  await expect(opsz).toHaveValue('8');
  await page.locator('#ts-opsz-auto').check();
  await expect(opsz).toHaveValue('60');
});
