import { cellLabel, charOf, faceOf, layout, lineHeight, parse, type Mode, type Spacing } from '../lib/compose';
import { bleed, bleedSlider, draws, fontDataUrl, previewFonts, weight, weightSlider, type Preview } from './preview-font';

interface Inst { id: string; card: string; path: string }
interface ComposeGlyph { char: string; adv: number; trace: string; instances: Inst[] }

// rules and flavor text from the scanned cards, flavor and reminder text in italic as they print
const SAMPLES: [string, string][] = [
  ['Niv-Mizzet, the Firemind', 'Whenever you draw a card, Niv-Mizzet, the Firemind deals 1 damage to target creature or player.'],
  ['Uglúk of the White Hand', 'If that creature was a Goblin or Orc, put two +1/+1 counters on Uglúk instead.'],
  ['Clay Golem', 'Berserk — When Clay Golem becomes monstrous, destroy target permanent. *(If this creature isn’t monstrous, put X +1/+1 counters on it and it becomes monstrous.)*'],
  ['Automatic Librarian', '*Enforcing absolute quiet with extreme prejudice since 3285 AR.*'],
  ['Barad-dûr', '*“Those who pass the gates of Barad-dûr do not return.”*'],
  ['Swiftfoot Boots', '*“There is great wisdom in rushing headlong into combat . . . if you’re prepared.”*'],
  ['Bloodletter', '*—Hœbrus Væm, Ærathi librarian, last words*'],
];
// the slider steps through the sizes a type scale offers rather than every pixel
const SIZES = [12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 160, 192, 224, 256];
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
  private preview: Preview | null = null;
  private advances = new Map<string, number>();
  private kerns = new Map<string, number>();
  private widths = new Map<string, number>();
  private probe?: HTMLSpanElement;
  private last?: { lay: ReturnType<typeof layout>; scale: number; paths: { d: string; x: number; line: number; ghost: boolean; adv: number; text?: string }[]; w: number; h: number };

  async connectedCallback() {
    const base = this.dataset.base!;
    try {
      this.sp = await fetch(`${base}spacing.json`).then((r) => r.json());
    } catch (err) {
      console.warn('type-setter', err);
      return;
    }
    if (this.dataset.preview) this.preview = await previewFonts(this.dataset.preview);
    this.build();
    await this.render();
    // only a change of width needs a new layout, a change of height is the result of one
    const box = this.querySelector<HTMLElement>('.set-out')!;
    let w = box.clientWidth;
    new ResizeObserver(() => { if (box.clientWidth !== w) { w = box.clientWidth; this.render(); } }).observe(box);
  }

  /** Whether the preview font draws a cell, in the cell's face */
  fontDraws(cell: string) {
    return !!this.preview && !!this.preview.family[faceOf(cell)] && draws(this.preview, faceOf(cell), charOf(cell));
  }

  /** The font's kern value for a pair of cells of one face that it draws, in the spacing table's font units */
  fontKern(a: string, b: string) {
    if (faceOf(a) !== faceOf(b) || !this.fontDraws(a) || !this.fontDraws(b)) return undefined;
    const key = `${a}\u0000${b}`;
    if (!this.kerns.has(key)) {
      const ctx = document.createElement('canvas').getContext('2d')!;
      ctx.font = `${this.sp.metrics.upm}px "${this.preview!.family[faceOf(a)]}"`;
      ctx.fontKerning = 'normal';
      const w = (t: string) => ctx.measureText(t).width;
      this.kerns.set(key, Math.round(w(charOf(a) + charOf(b)) - w(charOf(a)) - w(charOf(b))));
    }
    return this.kerns.get(key);
  }

  /** A cell's advance in the preview font at a weight, in the spacing table's font units, measured on a hidden span since a canvas cannot set an axis */
  widthAt(cell: string, w: number) {
    const key = `${w}\u0000${cell}`;
    if (!this.widths.has(key)) {
      if (!this.probe) {
        this.probe = document.createElement('span');
        this.probe.setAttribute('aria-hidden', 'true');
        this.probe.style.cssText = `position:absolute;visibility:hidden;white-space:pre;font-size:${this.sp.metrics.upm}px;font-kerning:none`;
        document.body.append(this.probe);
      }
      this.probe.style.fontFamily = `"${this.preview!.family[faceOf(cell)]}"`;
      this.probe.style.fontVariationSettings = `'wght' ${w}`;
      this.probe.textContent = charOf(cell);
      this.widths.set(key, this.probe.getBoundingClientRect().width);
    }
    return this.widths.get(key)!;
  }

  /** A cell's advance in the preview font, in the spacing table's font units */
  fontAdvance(cell: string) {
    if (!this.fontDraws(cell)) return undefined;
    if (!this.advances.has(cell)) {
      const ctx = document.createElement('canvas').getContext('2d')!;
      ctx.font = `${this.sp.metrics.upm}px "${this.preview!.family[faceOf(cell)]}"`;
      this.advances.set(cell, ctx.measureText(charOf(cell)).width);
    }
    return this.advances.get(cell);
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
  <p class="samples">Try: ${SAMPLES.filter(([, s]) => hasItalic || !s.includes('*')).map(([name, s]) => `<button class="btn" type="button" data-sample="${esc(s)}" title="${esc(s.replace(/\*/g, ''))}">${esc(name)}</button>`).join(' ')}</p>
  <div class="setter-controls">
    <fieldset class="toggles" hidden><legend>Letters</legend>
      <label><input type="radio" name="ts-var" value="font"${this.preview ? ' checked' : ' disabled'}> The Nomicon Rules font</label>
      <label><input type="radio" name="ts-var" value="mean"${this.preview ? '' : ' checked'}> The mean of the stack</label>
      <label hidden><input type="radio" name="ts-var" value="alt"> A different printed letter each time</label>
    </fieldset>
    <fieldset class="axes"><legend>Axes</legend>
      <label class="nrow" for="ts-size">Size <input id="ts-size" type="range" min="0" max="${SIZES.indexOf(this.preview ? 256 : 160)}" step="1" value="${SIZES.indexOf(48)}" aria-valuetext="48 px"> <output class="mono ts-sout">48 px</output></label>
    </fieldset>
    <p class="actions"><button class="btn" type="button" data-again hidden>Pick other letters</button> <button class="btn" type="button" data-download>Download image</button></p>
  </div>
  <figure class="set-out"><svg class="set-svg" role="img" aria-labelledby="ts-t"><title id="ts-t">Your text set in Nomicon Rules</title></svg></figure>
  <p class="src ts-status" aria-live="polite"></p>
  ${this.preview ? '<div class="preview-note ts-preview"><p>Bleed sets how strong the print look is, and 100 is what the cards show.</p></div>' : ''}
  <p class="src ts-spacing"></p>
</div>`;
    this.svg = this.querySelector('.set-svg')!;
    const note = this.querySelector('.ts-preview');
    if (note && this.preview) {
      const bled = bleedSlider(this.preview, 'ts-bled');
      this.querySelector('.setter-controls label[for=ts-size]')!.after(bled);
      if (this.preview.weight) bled.after(weightSlider(this.preview.weight, 'ts-wght', () => this.render()));
    }
    const text = this.querySelector<HTMLTextAreaElement>('#ts-text')!;
    const size = this.querySelector<HTMLInputElement>('#ts-size')!;
    text.addEventListener('input', () => this.render());
    size.addEventListener('input', () => {
      const t = `${SIZES[+size.value]} px`;
      this.querySelector('.ts-sout')!.textContent = t;
      size.setAttribute('aria-valuetext', t);
      this.render();
    });
    this.querySelectorAll('[name=ts-var]').forEach((r) => r.addEventListener('change', () => this.render()));
    this.querySelectorAll<HTMLButtonElement>('[data-sample]').forEach((b) => b.addEventListener('click', () => { text.value = b.dataset.sample!; this.render(); }));
    this.querySelector('[data-again]')!.addEventListener('click', () => { this.seed++; this.render(); });
    this.querySelector('[data-download]')!.addEventListener('click', () => this.download());
  }

  async render() {
    const text = this.querySelector<HTMLTextAreaElement>('#ts-text')!.value;
    const mode: Mode = 'designed';
    const letters = this.querySelector<HTMLInputElement>('[name=ts-var]:checked')!.value;
    const alt = letters === 'alt';
    const px = SIZES[+this.querySelector<HTMLInputElement>('#ts-size')!.value];
    await this.ensure(parse(text).flat());
    const m = this.sp.metrics;
    const scale = px / m.upm;
    const box = this.querySelector<HTMLElement>('.set-out')!;
    const wPx = Math.max(200, this.svg.clientWidth || box.clientWidth);
    // away from the default weight each glyph the font draws is wider by what the bold adds to its advance
    const w = this.preview?.weight ? weight() : 0, w0 = this.preview?.weight?.[1] ?? 0;
    const sp = w && w !== w0 ? { ...this.sp, glyphs: Object.fromEntries(Object.entries(this.sp.glyphs).map(([c, g]) => [c, this.fontDraws(c) ? { ...g, adv: g.adv + this.widthAt(c, w) - this.widthAt(c, w0) } : g])) } : this.sp;
    const lay = this.preview
      ? layout(text, sp, mode, wPx / scale, (cell) => (w && w !== w0 ? this.widthAt(cell, w) : this.fontAdvance(cell)), (a, b) => this.fontKern(a, b))
      : layout(text, this.sp, mode, wPx / scale);
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
        if ((!c.known || letters === 'font') && this.fontDraws(c.ch)) {
          const family = this.preview!.family[faceOf(c.ch)]!;
          out += `<g class="gl font" transform="${t}" data-ch="${esc(c.ch)}" data-x="${c.x.toFixed(2)}"><text font-family="&quot;${family}&quot;" font-size="${m.upm}">${esc(charOf(c.ch))}</text></g>`;
          paths.push({ d: '', x: c.x, line: li, ghost: false, adv: c.adv, text: c.ch });
          continue;
        }
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
    const fromFont = lay.missing.filter((c) => this.fontDraws(c)), ghosts = lay.missing.filter((c) => !this.fontDraws(c));
    q.textContent = letters === 'font'
      ? `Every letter here is from Nomicon Rules${ghosts.length ? `, except ${ghosts.map(cellLabel).join(' ')}, which it does not draw and appear as dashed cells` : ''}.`
      : [fromFont.length ? `Not yet scanned, drawn from Nomicon Rules: ${fromFont.map(cellLabel).join(' ')}.` : '',
        ghosts.length ? `Not yet scanned: ${ghosts.map(cellLabel).join(' ')}. These appear as dashed cells.` : '',
        lay.missing.length ? '' : 'Every letter here is from the scanned set.'].filter(Boolean).join(' ');
    this.querySelector('.ts-spacing')!.textContent = `Spacing: the designed kern value for every pair${this.preview ? '' : ' measured on the cards, nothing for the rest'}. Word spaces use the designed width.`;
    this.querySelector<HTMLButtonElement>('[data-again]')!.disabled = !alt;
  }

  download() {
    const l = this.last;
    if (!l) return;
    if (l.paths.some((p) => p.text)) return this.downloadWithFont();
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

  /** The set text as an image when some of it is drawn in the preview font, which a canvas cannot vary along its axis, so the SVG is drawn with the font embedded */
  async downloadWithFont() {
    const l = this.last!;
    const cs = getComputedStyle(this);
    const ink = cs.getPropertyValue('--ink').trim() || '#161616', paper = cs.getPropertyValue('--paper').trim() || '#fbfaf7';
    const k = 2, svg = this.svg.cloneNode(true) as SVGSVGElement;
    const vb = svg.viewBox.baseVal;
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svg.setAttribute('width', String(Math.round(l.w * k)));
    svg.setAttribute('height', String(Math.round(l.h * k)));
    const faces = Object.entries(this.preview!.urls);
    const css = (await Promise.all(faces.map(async ([face, url]) => `@font-face{font-family:"${this.preview!.family[face as 'regular' | 'italic']}";src:url(${await fontDataUrl(url!)})}`))).join('');
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = `${css}path{fill:${ink}}text{fill:${ink};font-variation-settings:'BLED' ${bleed()},'wght' ${weight()}}.ghost rect{fill:none;stroke:${ink};stroke-opacity:.65;stroke-width:9;stroke-dasharray:26 18}`;
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    Object.entries({ x: vb.x, y: vb.y, width: vb.width, height: vb.height, fill: paper }).forEach(([a, v]) => bg.setAttribute(a, String(v)));
    svg.prepend(style, bg);
    const img = new Image();
    img.src = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: 'image/svg+xml' }));
    await img.decode();
    const c = document.createElement('canvas');
    c.width = Math.round(l.w * k); c.height = Math.round(l.h * k);
    c.getContext('2d')!.drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(img.src);
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
