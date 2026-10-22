import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';

const e = JSON.parse(readFileSync('data/glyphs/e.json', 'utf8'));

test('the Stack mounts as a slider over traces and passes axe', async ({ page }) => {
  await page.goto('/glyphs/e/');
  await page.waitForSelector('.stack-svg');
  await expect(page.locator('.nout')).toHaveText(`${e.n} of ${e.n}, all`);
  expect(await page.locator('.stack-svg .st-now').getAttribute('d')).toBe(e.trace);
  const out = await new AxeBuilder({ page }).analyze();
  expect(out.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
});

test('moving the slider traces the glyph from fewer instances and updates the readout', async ({ page }) => {
  await page.goto('/glyphs/e/');
  await page.waitForSelector('.stack-svg');
  const series = e.trace_by_n;
  await page.locator('.n').fill('0');
  await expect(page.locator('.nout')).toHaveText(new RegExp(`^${series[0].n} of `));
  expect(await page.locator('.stack-svg .st-now').getAttribute('d')).toBe(series[0].trace);
  await page.locator('.n').fill('8');
  await expect(page.locator('.nout')).toHaveText(new RegExp(`^${series[8].n} of `));
  await expect(page.locator('.r-cards')).toHaveText(String(series[8].cards));
  expect(await page.locator('.stack-svg .st-now').getAttribute('d')).not.toBe(e.trace);
});

test('the slider works with the keyboard and names its value', async ({ page }) => {
  await page.goto('/glyphs/e/');
  await page.waitForSelector('.stack-svg');
  await page.locator('.n').focus();
  await page.keyboard.press('Home');
  await expect(page.locator('.n')).toHaveAttribute('aria-valuetext', /N equals 1 of/);
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('.nout')).toHaveText(new RegExp(`^${e.trace_by_n[1].n} of `));
});

test('the series ends at the full stack and every trace is a closed path', () => {
  const s = e.trace_by_n;
  expect(s[s.length - 1].n).toBe(e.n);
  expect(s[s.length - 1].trace).toBe(e.trace);
  for (const t of s) expect(t.trace.endsWith('Z'), String(t.n)).toBe(true);
});

test('without scripts the static figure remains', async ({ browser }) => {
  const ctx = await browser.newContext({ javaScriptEnabled: false });
  const page = await ctx.newPage();
  await page.goto('/glyphs/e/');
  await expect(page.locator('stack-view svg title')).toHaveText(/Traced outline/);
  await expect(page.locator('stack-view figcaption')).toContainText(String(e.n));
  await ctx.close();
});

const rows: { style: string }[] = JSON.parse(readFileSync('data/glyphs.json', 'utf8'));

test('the glyph map links every glyph and groups the italic face under its own heading', async ({ page }) => {
  await page.goto('/glyphs/');
  expect(await page.locator('.glyphmap a').count()).toBe(rows.length);
  const italic = rows.filter((r) => r.style === 'italic').length;
  await expect(page.getByRole('heading', { name: 'Italic', level: 2 })).toHaveCount(italic ? 1 : 0);
  await expect(page.locator('section.face').last().locator('.glyphmap a')).toHaveCount(italic || rows.length);
});

// the glyphs page is a character sheet of GF Latin Core, less the spaces and combining accents
const core: { codepoints: number[] } = JSON.parse(readFileSync('src/lib/latin-core.json', 'utf8'));
const sheetRows: { char: string; style: string; low_confidence: boolean }[] = JSON.parse(readFileSync('data/glyphs.json', 'utf8'));
const sheetSize = new Set([...core.codepoints.map((c) => String.fromCodePoint(c)).filter((ch) => !/^(\p{Zs}|\p{Mn})$/u.test(ch)), ...sheetRows.map((r) => r.char)]).size;

test('the glyphs page is a character sheet with an empty box for each character not scanned yet', async ({ page }) => {
  await page.goto('/glyphs/');
  const faces = ['regular', 'italic'].filter((f) => sheetRows.some((r) => r.style === f));
  const sections = page.locator('section.face');
  await expect(sections).toHaveCount(faces.length);
  for (let i = 0; i < faces.length; i++) {
    const scanned = sheetRows.filter((r) => r.style === faces[i]).length;
    await expect(sections.nth(i).locator('.glyphmap li')).toHaveCount(sheetSize);
    await expect(sections.nth(i).locator('.glyphmap a')).toHaveCount(scanned);
    await expect(sections.nth(i).locator('.glyphmap li.blank')).toHaveCount(sheetSize - scanned);
  }
  await expect(page.locator('.glyphmap a.low')).toHaveCount(sheetRows.filter((r) => r.low_confidence).length);
  // a tile carries only its count, so none can wrap out of line with the rest
  const heights = await page.$$eval('.glyphmap li', (els) => [...new Set(els.map((e) => Math.round(e.getBoundingClientRect().height)))]);
  expect(heights).toHaveLength(1);
});

test('per-glyph data stays under the gzip budget', async ({ request }) => {
  const { gzipSync } = await import('node:zlib');
  for (const n of JSON.parse(readFileSync('data/glyphs.json', 'utf8')).map((g: { name: string }) => g.name)) {
    for (const dir of ['glyphs', 'instances']) {
      const body = await (await request.get(`/data/${dir}/${n}.json`)).body();
      expect(gzipSync(body).length, `${dir}/${n}`).toBeLessThan(40 * 1024);
    }
  }
});

test('the pixel layers load on demand, sit under an outline and name their source', async ({ page }) => {
  await page.goto('/glyphs/a/');
  await page.waitForSelector('.stack-svg');
  const l = JSON.parse(readFileSync('data/pixels/a.json', 'utf8'));
  expect(await page.locator('.st-px image').count()).toBe(0);
  await page.locator('[data-layer=raw]').check();
  await expect(page.locator('.px-raw')).toBeVisible();
  await expect(page.locator('.px-cov')).toBeHidden();
  await expect(page.locator('.stack-svg')).toHaveClass(/px-on/);
  await expect(page.locator('.px-src')).toContainText(l.id);
  await expect(page.locator('.px-src')).toContainText(`${l.raw.w} × ${l.raw.h} px`);
  await page.locator('[data-layer=cov]').check();
  await expect(page.locator('.px-cov')).toBeVisible();
  await page.locator('[data-layer=raw]').uncheck();
  await page.locator('[data-layer=cov]').uncheck();
  await expect(page.locator('.stack-svg')).not.toHaveClass(/px-on/);
  await page.locator('[data-layer=final]').uncheck();
  await expect(page.locator('.st-final')).toBeHidden();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});

test('the background letter fades instances in, stacks them, fades them out and dims, all tied to scroll', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/');
  await page.waitForSelector('.hero-inst');
  const span = await page.locator('.hero-run').evaluate((el) => el.getBoundingClientRect().height - innerHeight);
  const at = async (y: number) => {
    await page.evaluate((v) => scrollTo({ top: v, behavior: 'instant' }), y);
    await page.waitForTimeout(100);
    return page.evaluate(() => {
      const first = document.querySelector('.hero-inst path') as SVGPathElement;
      const [dx, dy] = (first.getAttribute('transform') ?? '').match(/-?\d+\.?\d*/g)!.map(Number);
      return {
        off: Math.hypot(dx, dy), pathOpacity: +first.style.opacity, group: +(document.querySelector('.hero-inst') as SVGGElement).style.opacity,
        ink: +(document.querySelector('.hero-bg svg') as SVGSVGElement).style.getPropertyValue('--ink-o'), bg: +(document.querySelector('.hero-bg') as HTMLElement).style.opacity,
      };
    });
  };
  const start = await at(0), mid = await at(span * 0.35), settled = await at(span * 0.72), end = await at(span), after = await at(span + 800);
  expect(start.ink).toBe(0);
  expect(start.pathOpacity).toBeLessThan(0.2);
  expect(mid.off).toBeLessThan(start.off);
  expect(mid.pathOpacity).toBeGreaterThan(start.pathOpacity);
  expect(settled.off).toBeLessThan(1);
  expect(settled.group).toBeGreaterThan(0.9);
  expect(end.group).toBeLessThan(0.01);
  expect(end.ink).toBe(1);
  expect(end.bg).toBe(1);
  expect(after.bg).toBeLessThan(0.2);
  expect((await at(0)).off).toBeCloseTo(start.off, 1);
});

test('without motion the letter is a still, faint trace behind the page', async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: 'reduce' });
  const page = await ctx.newPage();
  await page.goto('/');
  await page.waitForTimeout(500);
  expect(await page.locator('.hero-inst').count()).toBe(0);
  await expect(page.locator('.hero-bg .ink')).toHaveCSS('fill-opacity', '0.12');
  await ctx.close();
});

test('the home page is a title, an arrow and the why, with nothing else above the title', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toHaveCount(1);
  await expect(page.locator('main')).not.toContainText('A text face revival');
  await expect(page.locator('main')).not.toContainText('Scroll to watch');
  const arrow = page.locator('a.down');
  await expect(arrow).toHaveAttribute('href', '#why');
  const box = await arrow.boundingBox();
  const vw = page.viewportSize()!;
  expect(Math.abs(box!.x + box!.width / 2 - vw.width / 2)).toBeLessThan(2);
  expect(box!.y + box!.height).toBeLessThanOrEqual(vw.height);
  expect(box!.y + box!.height).toBeGreaterThan(vw.height - 120);
  expect(await page.locator('.figures').count()).toBe(0);
  await expect(page.locator('[data-hero-note]')).toContainText(`${e.instances_exported} of the ${e.instances_total}`);
});

test('layers higher in the list sit in front, the outline on top, and the page has no number tables', async ({ page }) => {
  await page.goto('/glyphs/e/');
  await page.waitForSelector('.stack-svg');
  const listed = await page.$$eval('[data-layer]', (els) => els.map((e) => (e as HTMLElement).dataset.layer));
  expect(listed).toEqual(['final', 'now', 'cov', 'raw']);
  const drawn = await page.$$eval('.stack-svg > :is(g.st-px, path)', (els) => els.map((e) => e.getAttribute('class')));
  expect(drawn).toEqual(['st-px', 'st-now', 'st-final']);
  expect(await page.locator('table').count()).toBe(0);
  await expect(page.locator('main')).not.toContainText('Observed spacing');
  await expect(page.locator('main')).not.toContainText('numbers behind these figures');
  await expect(page.locator('a[href$="#glyph-e"]').first()).toBeVisible();
});

test('the page text sits in a centred container and the arrow stays at the bottom through the animation', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 800 });
  await page.goto('/');
  await page.waitForSelector('.hero-inst');
  const why = await page.locator('section.why').boundingBox();
  expect(Math.abs(why!.x + why!.width / 2 - 720)).toBeLessThan(2);
  const title = await page.locator('h1').boundingBox();
  expect(Math.abs(title!.x + title!.width / 2 - 720)).toBeLessThan(2);
  const span = await page.locator('.hero-run').evaluate((el) => el.getBoundingClientRect().height - innerHeight);
  for (const f of [0, 0.3, 0.6, 0.95]) {
    await page.evaluate((y) => scrollTo({ top: y, behavior: 'instant' }), span * f);
    await page.waitForTimeout(80);
    const a = await page.locator('a.down').boundingBox();
    expect(a!.y + a!.height, `at ${f}`).toBeGreaterThan(800 - 40);
    expect(a!.y + a!.height).toBeLessThanOrEqual(800);
  }
  await page.evaluate(() => document.querySelector('#why')!.scrollIntoView({ behavior: 'instant' }));
  await page.waitForTimeout(100);
  const gone = await page.locator('a.down').boundingBox();
  expect(gone!.y + gone!.height).toBeLessThan(0);
});

test('each checkbox sits on the line of its label', async ({ page }) => {
  await page.goto('/glyphs/e/');
  await page.waitForSelector('.stack-svg');
  const rows = await page.$$eval('.toggles label', (ls) => ls.map((l) => {
    const i = l.querySelector('input')!.getBoundingClientRect(), r = l.getBoundingClientRect();
    const line = parseFloat(getComputedStyle(l).lineHeight);
    return { dy: Math.abs((i.top + i.height / 2) - (r.top + line / 2)), single: r.height < line * 1.5 };
  }));
  expect(rows.length).toBe(4);
  for (const r of rows) expect(r.dy).toBeLessThan(1.5);
});
