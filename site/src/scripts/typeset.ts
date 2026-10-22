import { cellLabel, charOf, layout, lineHeight, parse, type Mode, type Spacing } from '../lib/compose';

interface Inst { id: string; card: string; path: string }
interface ComposeGlyph { char: string; adv: number; trace: string; instances: Inst[] }

const SAMPLES = ['yet a proud sailor can hunt.', 'senseless', 'learn to read', 'learn to *read the stars*'];
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

function shuffled(n: number, seed: number) {
  let s = (seed * 2654435761) >>> 0 || 1;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

class TypeSetter extends HTMLElement {
  private sp!: Spacing;
  private glyphs = new Map<string, ComposeGlyph>();
  private loading = new Map<string, Promise<void>>();
  private seed = 1;
  private svg!: SVGSVGElement;
  private last?: { lay: ReturnType<typeof layout>; scale: number; paths: { d: string; x: number; line: number; ghost: boolean; adv: number }[]; w: number; h: number };

  async connectedCallback() {
    const base = this.dataset.base!;
    try {
      this.sp = await fetch(`${base}spacing.json`).then((r) => r.json());
    } catch (err) {
      console.warn('type-setter', err);
      return;
    }
    this.build();
    await this.render();
    // only a change of width needs a new layout, a change of height is the result of one
    const box = this.querySelector<HTMLElement>('.set-out')!;
    let w = box.clientWidth;
    new ResizeObserver(() => { if (box.clientWidth !== w) { w = box.clientWidth; this.render(); } }).observe(box);
  }

  ensure(chars: string[]) {
    const jobs: Promise<void>[] = [];
    for (const ch of new Set(chars)) {
      const info = this.sp.glyphs[ch];
      if (!info || this.glyphs.has(ch)) continue;
      if (!this.loading.has(ch)) {
        this.loading.set(ch, fetch(`${this.dataset.base}compose/${info.name}.json`).then((r) => r.json()).then((g: ComposeGlyph) => { this.glyphs.set(ch, g); }));
      }
      jobs.push(this.loading.get(ch)!);
    }
    return Promise.all(jobs);
  }

  build() {
    const hasItalic = Object.keys(this.sp.glyphs).some((k) => k.startsWith('i:'));
    this.innerHTML = `
<div class="setter">
  <label for="ts-text" class="kicker">Your text</label>
  <textarea id="ts-text" rows="2" maxlength="300" spellcheck="false">${esc(this.dataset.default!)}</textarea>
  ${hasItalic ? '<p class="src ts-hint">Put asterisks around text to set it in italic, as in *flavor text*.</p>' : ''}
  <p class="samples">Try: ${SAMPLES.filter((s) => hasItalic || !s.includes('*')).map((s) => `<button class="btn" type="button" data-sample="${esc(s)}">${esc(s)}</button>`).join(' ')}</p>
  <div class="setter-controls">
    <fieldset class="toggles"><legend>Spacing</legend>
      <label><input type="radio" name="ts-mode" value="observed" checked> Observed on the cards</label>
      <label><input type="radio" name="ts-mode" value="designed"> Designed kerning</label>
    </fieldset>
    <fieldset class="toggles"><legend>Letters</legend>
      <label><input type="radio" name="ts-var" value="alt" checked> A different printed letter each time</label>
      <label><input type="radio" name="ts-var" value="mean"> The mean of the stack</label>
    </fieldset>
    <div>
      <label class="nrow" for="ts-size">Size <input id="ts-size" type="range" min="28" max="160" step="2" value="72"> <output class="mono ts-sout">72 px</output></label>
      <p class="actions"><button class="btn" type="button" data-again>Pick other letters</button> <button class="btn" type="button" data-download>Download image</button></p>
    </div>
  </div>
  <figure class="set-out"><svg class="set-svg" role="img" aria-labelledby="ts-t"><title id="ts-t">Your text set in the scanned letters</title></svg></figure>
  <p class="src ts-status" aria-live="polite"></p>
  <p class="src ts-spacing"></p>
</div>`;
    this.svg = this.querySelector('.set-svg')!;
    const text = this.querySelector<HTMLTextAreaElement>('#ts-text')!;
    const size = this.querySelector<HTMLInputElement>('#ts-size')!;
    text.addEventListener('input', () => this.render());
    size.addEventListener('input', () => { this.querySelector('.ts-sout')!.textContent = `${size.value} px`; this.render(); });
    this.querySelectorAll('[name=ts-mode],[name=ts-var]').forEach((r) => r.addEventListener('change', () => this.render()));
    this.querySelectorAll<HTMLButtonElement>('[data-sample]').forEach((b) => b.addEventListener('click', () => { text.value = b.dataset.sample!; this.render(); }));
    this.querySelector('[data-again]')!.addEventListener('click', () => { this.seed++; this.render(); });
    this.querySelector('[data-download]')!.addEventListener('click', () => this.download());
  }

  async render() {
    const text = this.querySelector<HTMLTextAreaElement>('#ts-text')!.value;
    const mode = this.querySelector<HTMLInputElement>('[name=ts-mode]:checked')!.value as Mode;
    const alt = this.querySelector<HTMLInputElement>('[name=ts-var]:checked')!.value === 'alt';
    const px = +this.querySelector<HTMLInputElement>('#ts-size')!.value;
    await this.ensure(parse(text).flat());
    const m = this.sp.metrics;
    const scale = px / m.upm;
    const box = this.querySelector<HTMLElement>('.set-out')!;
    const wPx = Math.max(200, this.svg.clientWidth || box.clientWidth);
    const lay = layout(text, this.sp, mode, wPx / scale);
    const lh = lineHeight(this.sp);
    const wFu = wPx / scale, hFu = Math.max(1, lay.lines.length) * lh;
    this.svg.setAttribute('viewBox', `0 0 ${wFu.toFixed(1)} ${hFu.toFixed(1)}`);
    this.svg.style.height = `${(hFu * scale).toFixed(1)}px`;

    const order = new Map<string, number[]>();
    const used = new Map<string, number>();
    const paths: NonNullable<TypeSetter['last']>['paths'] = [];
    let out = '';
    lay.lines.forEach((ln, li) => {
      const y = li * lh + m.ascent;
      for (const c of ln) {
        if (c.space) continue;
        const t = `translate(${c.x.toFixed(1)} ${y})`;
        if (!c.known) {
          const h = m.cap_height;
          out += `<g class="ghost" transform="${t}" ><title>${esc(cellLabel(c.ch))}: not yet scanned</title><rect x="${(c.adv * 0.08).toFixed(1)}" y="${-h}" width="${(c.adv * 0.84).toFixed(1)}" height="${h}" rx="30"/></g>`;
          paths.push({ d: '', x: c.x, line: li, ghost: true, adv: c.adv });
          continue;
        }
        const g = this.glyphs.get(c.ch)!;
        let d = g.trace, id = '';
        if (alt && g.instances.length) {
          if (!order.has(c.ch)) order.set(c.ch, shuffled(g.instances.length, this.seed * 131 + charOf(c.ch).charCodeAt(0) + (c.ch.length > 1 ? 7 : 0)));
          const k = used.get(c.ch) ?? 0;
          used.set(c.ch, k + 1);
          const it = g.instances[order.get(c.ch)![k % g.instances.length]];
          d = it.path; id = it.id;
        }
        out += `<g class="gl" transform="${t}" data-ch="${esc(c.ch)}" data-id="${id}" data-x="${c.x.toFixed(2)}"><path d="${d}" fill-rule="evenodd"/></g>`;
        paths.push({ d, x: c.x, line: li, ghost: false, adv: c.adv });
      }
    });
    this.svg.querySelectorAll('g').forEach((g) => g.remove());
    this.svg.insertAdjacentHTML('beforeend', out);
    this.last = { lay, scale, paths, w: wPx, h: hFu * scale };

    const q = this.querySelector('.ts-status')!;
    q.textContent = lay.missing.length
      ? `Not yet scanned: ${lay.missing.map(cellLabel).join(' ')}. These appear as dashed cells.`
      : 'Every letter here is from the scanned set.';
    const avg = lay.average, avgIt = lay.averages.italic;
    this.querySelector('.ts-spacing')!.textContent = mode === 'observed'
      ? `Spacing: ${lay.observed} of ${lay.observed + lay.defaulted} letter pairs were measured on at least three cards. The rest use the average offset of ${avg >= 0 ? '+' : ''}${avg.toFixed(1)} font units${lay.italic ? ` (${avgIt >= 0 ? '+' : ''}${avgIt.toFixed(1)} within italic)` : ''}. Word spaces use the designed width, word spacing on the cards is not measured yet.`
      : 'Spacing: the font’s own kern value for the pairs measured on the cards, nothing for the rest. Word spaces use the designed width.';
    this.querySelector<HTMLButtonElement>('[data-again]')!.disabled = !alt;
  }

  download() {
    const l = this.last;
    if (!l) return;
    const cs = getComputedStyle(this);
    const ink = cs.getPropertyValue('--ink').trim() || '#161616', paper = cs.getPropertyValue('--paper').trim() || '#fbfaf7';
    const k = 2, c = document.createElement('canvas');
    c.width = Math.round(l.w * k); c.height = Math.round(l.h * k);
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = paper; ctx.fillRect(0, 0, c.width, c.height);
    const m = this.sp.metrics, lh = lineHeight(this.sp);
    for (const p of l.paths) {
      ctx.save();
      ctx.translate(p.x * l.scale * k, (p.line * lh + m.ascent) * l.scale * k);
      ctx.scale(l.scale * k, l.scale * k);
      if (p.ghost) {
        ctx.setLineDash([26, 18]); ctx.lineWidth = 9; ctx.strokeStyle = ink; ctx.globalAlpha = 0.65;
        ctx.strokeRect(p.adv * 0.08, -m.cap_height, p.adv * 0.84, m.cap_height);
      } else {
        ctx.fillStyle = ink; ctx.fill(new Path2D(p.d), 'evenodd');
      }
      ctx.restore();
    }
    c.toBlob((b) => {
      if (!b) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = 'nomicon-rules-specimen.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, 'image/png');
  }
}

customElements.define('type-setter', TypeSetter);
