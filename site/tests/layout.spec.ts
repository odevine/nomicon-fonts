import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';

const glyphs: { name: string }[] = JSON.parse(readFileSync('data/glyphs.json', 'utf8'));
const paths = ['/', '/capture/', '/cleaning/', '/printing-leaves-a-mark/', '/glyphs/', '/typeset/', '/data/', '/about/', `/glyphs/${glyphs[0].name}/`];

// Every block in main is centred and no wider than its tier: text 38rem, .mid 46rem, .wide 62rem, .full unlimited. See DESIGN.md
test.use({ viewport: { width: 1600, height: 900 } });

for (const p of paths) {
  test(`${p} keeps every block centred inside its tier`, async ({ page }) => {
    await page.goto(p);
    await page.waitForLoadState('networkidle');
    const r = await page.evaluate(() => {
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize), cw = document.documentElement.clientWidth;
      const tier = (e: Element) => (e.classList.contains('full') ? Infinity : e.classList.contains('wide') ? 62 : e.classList.contains('mid') ? 46 : 38);
      const blocks: Element[] = [];
      const walk = (parent: Element) => { for (const c of parent.children) { if (c.matches('script, style')) continue; if (c.classList.contains('flow')) walk(c); else blocks.push(c); } };
      walk(document.querySelector('main')!);
      const bad: string[] = [];
      let wideLeft = Infinity;
      for (const b of blocks) {
        const box = b.getBoundingClientRect();
        if (!box.width || !box.height) continue;
        const name = `${b.tagName.toLowerCase()}.${b.className}`;
        if (!b.classList.contains('full')) {
          if (Math.abs((box.left + box.right) / 2 - cw / 2) > 3) bad.push(`${name} is off centre`);
          if (box.width > tier(b) * rem + 2) bad.push(`${name} is ${(box.width / rem).toFixed(1)}rem wide, its tier allows ${tier(b)}`);
        }
        if (b.classList.contains('wide')) wideLeft = Math.min(wideLeft, box.left);
      }
      return { bad, wideLeft, mark: document.querySelector('header .mark')!.getBoundingClientRect().left };
    });
    expect(r.bad).toEqual([]);
    // wide figures start where the wordmark starts
    if (Number.isFinite(r.wideLeft)) expect(Math.abs(r.wideLeft - r.mark)).toBeLessThan(2);
  });
}

test('the header and footer share the container with the page', async ({ page }) => {
  await page.goto('/data/');
  const x = await page.evaluate(() => ({
    mark: document.querySelector('header .mark')!.getBoundingClientRect().left,
    foot: document.querySelector('footer p')!.getBoundingClientRect().left,
    rem: parseFloat(getComputedStyle(document.documentElement).fontSize),
    vw: document.documentElement.clientWidth,
  }));
  expect(Math.abs(x.mark - x.foot)).toBeLessThan(2);
  expect(Math.abs(x.mark - (x.vw - 62 * x.rem) / 2)).toBeLessThan(2);
});

// the header and footer fill the same container, so their content is centred too and not packed to one side
for (const vw of [1600, 1100]) {
  test(`the header and footer content is centred in the container at ${vw}px`, async ({ page }) => {
    await page.setViewportSize({ width: vw, height: 900 });
    await page.goto('/data/');
    const m = await page.evaluate(() => {
      const span = (sel: string) => {
        const r = [...document.querySelectorAll(sel)].map((e) => e.getBoundingClientRect()).filter((b) => b.width > 0);
        return { left: Math.min(...r.map((b) => b.left)), right: Math.max(...r.map((b) => b.right)) };
      };
      return { head: span('header.site > *'), foot: span('footer.site > *'), rem: parseFloat(getComputedStyle(document.documentElement).fontSize), vw: innerWidth };
    });
    const box = Math.min(62 * m.rem, m.vw - 2 * Math.max(16, 0.05 * m.vw));
    for (const [name, s] of [['header', m.head], ['footer', m.foot]] as const) {
      expect(Math.abs((s.left + s.right) / 2 - m.vw / 2), `${name} centre`).toBeLessThan(3);
      expect(s.right - s.left, `${name} fills its container`).toBeGreaterThan(box - 4);
      expect(s.right - s.left, `${name} within the container`).toBeLessThanOrEqual(62 * m.rem + 2);
    }
  });
}
