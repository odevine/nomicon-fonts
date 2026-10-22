import { place, type Fit } from '../lib/labfit';

// Calibration Bench tone readout and the Noise Lab, both small islands over server-rendered figures

class ToneCurve extends HTMLElement {
  connectedCallback() {
    const p: number[] = JSON.parse(this.dataset.percent!), a: number[] = JSON.parse(this.dataset.a!), b: number[] = JSON.parse(this.dataset.b!);
    const svg = this.querySelector<SVGSVGElement>('svg')!;
    const cursor = svg.querySelector<SVGLineElement>('.cursor')!;
    const pts = [...svg.querySelectorAll<SVGCircleElement>('.pa')];
    let i = p.indexOf(50);
    const show = (k: number) => {
      i = Math.max(0, Math.min(p.length - 1, k));
      const x = pts[i].getAttribute('cx')!;
      cursor.setAttribute('x1', x); cursor.setAttribute('x2', x);
      this.querySelector('.tc-p')!.textContent = `${p[i]} %`;
      this.querySelector('.tc-a')!.textContent = a[i].toFixed(0);
      this.querySelector('.tc-b')!.textContent = b[i].toFixed(0);
    };
    svg.addEventListener('pointermove', (e) => {
      const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM()!.inverse());
      let best = 0, d = Infinity;
      pts.forEach((c, k) => { const dx = Math.abs(+c.getAttribute('cx')! - pt.x); if (dx < d) { d = dx; best = k; } });
      show(best);
    });
    svg.addEventListener('keydown', (e) => {
      const m: Record<string, number> = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 };
      if (e.key === 'Home') show(0); else if (e.key === 'End') show(p.length - 1); else if (e.key in m) show(i + m[e.key]); else return;
      e.preventDefault();
    });
  }
}
customElements.define('tone-curve', ToneCurve);

interface Pair { id: string; glyph: string; card: string; upright_angle: number; turned_angle: number; pair_dev_um: number; w: number; h: number }
const cardName = (k: string) => (k === 'Altair' ? 'Altaïr' : k.replace(/([a-z])([A-Z])/g, '$1 $2'));
const glyphLabel = (c: string, style = 'regular') => `${c === '.' ? 'full stop' : c}${style === 'regular' ? '' : `, ${style}`}`;
/** A noise pair's instance id carries -it- for the italic face */
const deg = (a: number) => `${+(((a % 360) + 360) % 360).toFixed(1)}°`;
const styleOf = (id: string) => (id.includes('-it-') ? 'italic' : 'regular');

class NoiseLab extends HTMLElement {
  private canvas?: HTMLCanvasElement;
  private layer?: HTMLElement;
  private cache = new Map<string, { a: ImageData; b: ImageData; w: number; h: number }>();

  connectedCallback() {
    const glyphs: { name: string; char: string; style: string; y0: number }[] = JSON.parse(this.dataset.glyphs!);
    const fit: Fit = JSON.parse(this.dataset.fit!);
    const base = this.dataset.base!;
    const base0 = this.querySelector<HTMLElement>('.lab-pair')?.style.getPropertyValue('--base') || '75%';
    const stage = this.querySelector<HTMLElement>('.lab-stage')!;
    stage.innerHTML = `
      <fieldset class="picker"><legend>Glyph</legend>${glyphs.map((g, i) => `<label><input type="radio" name="nl-glyph" value="${g.name}" aria-label="${glyphLabel(g.char, g.style)}"${g.name === 'e' ? ' checked' : ''}><span aria-hidden="true"${g.style === 'italic' ? ' class="it"' : ''}>${g.char === '.' ? '·' : g.char}</span></label>`).join('')}</fieldset>
      <fieldset class="toggles"><legend>View</legend>
        <label><input type="radio" name="nl-view" value="wipe" checked> Upright against turned</label>
        <label><input type="radio" name="nl-view" value="diff"> Difference, scaled up 2.5 times</label>
      </fieldset>
      <div class="lab-pair" style="--cut:50%;--base:${base0}"><div class="layer"><img class="ink-img px-u" alt="" /></div><div class="layer cut"><img class="ink-img px-t" alt="" /></div><div class="divider"></div></div>
      <label class="nrow nl-wipe" for="nl-cut">Upright <input id="nl-cut" type="range" min="0" max="100" value="50"> turned</label>
      <p class="cap nl-cap" aria-live="polite"></p>`;
    const pair = stage.querySelector<HTMLElement>('.lab-pair')!;
    const imgP = stage.querySelector<HTMLImageElement>('.px-u')!, imgL = stage.querySelector<HTMLImageElement>('.px-t')!;
    const cut = stage.querySelector<HTMLInputElement>('#nl-cut')!;
    const cap = stage.querySelector<HTMLElement>('.nl-cap')!;
    let view: 'wipe' | 'diff' = 'wipe';
    let info: Pair | null = null;

    const load = async (name: string) => {
      info = await fetch(`${base}noise/${name}.json`).then((r) => r.json());
      const i = info!;
      const at = place(i, glyphs.find((g) => g.name === name)!.y0, fit);
      for (const el of [imgP, imgL, ...(this.canvas ? [this.canvas] : [])]) Object.assign(el.style, at);
      imgP.src = `${base}noise/${name}_u.webp`; imgL.src = `${base}noise/${name}_t.webp`;
      imgP.alt = `${glyphLabel(i.glyph, styleOf(i.id))}, scanned at ${deg(i.upright_angle)}`; imgL.alt = `${glyphLabel(i.glyph, styleOf(i.id))}, scanned at ${deg(i.turned_angle)}`;
      cap.textContent = `Printed ${glyphLabel(i.glyph, styleOf(i.id))} from ${cardName(i.card)}, scanned at ${deg(i.upright_angle)} and ${deg(i.turned_angle)}. These two copies are ${i.pair_dev_um.toFixed(1)} µm apart at the edge, RMS.`;
      await apply();
    };
    const diff = async () => {
      const name = (stage.querySelector<HTMLInputElement>('[name=nl-glyph]:checked')!).value;
      if (!this.cache.has(name)) {
        const grab = async (img: HTMLImageElement) => {
          await img.decode();
          const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
          const x = c.getContext('2d')!; x.drawImage(img, 0, 0);
          return x.getImageData(0, 0, c.width, c.height);
        };
        const [a, b] = await Promise.all([grab(imgP), grab(imgL)]);
        this.cache.set(name, { a, b, w: a.width, h: a.height });
      }
      const d = this.cache.get(name)!;
      const out = new ImageData(d.w, d.h);
      for (let k = 3; k < out.data.length; k += 4) out.data[k] = Math.min(255, Math.abs(d.a.data[k] - d.b.data[k]) * 2.5);
      if (!this.canvas) { this.canvas = document.createElement('canvas'); this.canvas.className = 'ink-img'; this.canvas.setAttribute('role', 'img'); this.layer = document.createElement('div'); this.layer.className = 'layer'; this.layer.append(this.canvas); pair.insertBefore(this.layer, pair.querySelector('.divider')); }
      this.canvas.width = d.w; this.canvas.height = d.h;
      Object.assign(this.canvas.style, place(info!, glyphs.find((g) => g.name === name)!.y0, fit));
      this.canvas.getContext('2d')!.putImageData(out, 0, 0);
      this.canvas.setAttribute('aria-label', `Difference between the two copies of ${glyphLabel(info!.glyph, styleOf(info!.id))}`);
    };
    const apply = async () => {
      const isDiff = view === 'diff';
      imgP.hidden = isDiff; imgL.hidden = isDiff;
      pair.querySelector<HTMLElement>('.divider')!.hidden = isDiff;
      cut.closest('label')!.hidden = isDiff;
      if (isDiff) await diff(); else if (this.layer) this.layer.hidden = true;
      if (this.layer && isDiff) this.layer.hidden = false;
    };
    stage.querySelectorAll<HTMLInputElement>('[name=nl-glyph]').forEach((r) => r.addEventListener('change', () => load(r.value)));
    stage.querySelectorAll<HTMLInputElement>('[name=nl-view]').forEach((r) => r.addEventListener('change', () => { view = r.value as 'wipe' | 'diff'; apply(); }));
    cut.addEventListener('input', () => pair.style.setProperty('--cut', `${cut.value}%`));
    load('e');
  }
}
customElements.define('noise-lab', NoiseLab);
