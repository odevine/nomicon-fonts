import { bbox, cardName, deg, glyphLabel, viewBox, PAD } from '../lib/format';

interface Step { n: number; cards: number; trace: string; nodes: number; edge_rms_um: number | null }
interface Glyph { char: string; style: string; n: number; trace: string; trace_by_n: Step[]; recommended_n: number }
interface Img { file: string; w: number; h: number; matrix: number[]; um_per_px: number }
interface Pixels { id: string; card: string; scan: string; angle: number; raw: Img; cov4: Img }
interface SweepRow { n: number; rms_vs_ref_um: number; p10: number; p90: number; noise_model_um: number }

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

class StackView extends HTMLElement {
  async connectedCallback() {
    const name = this.dataset.glyph!, base = this.dataset.base!;
    try {
      const [g, sweep] = await Promise.all([
        fetch(`${base}glyphs/${name}.json`).then((r) => r.json() as Promise<Glyph>),
        fetch(`${base}sweep/${name}.json`).then((r) => r.json() as Promise<{ rows: SweepRow[]; recommended_n: number; instances_swept?: number }>),
      ]);
      this.build(g, sweep);
    } catch (err) {
      // the static figure stays when the data cannot be loaded
      console.warn('stack-view', err);
    }
  }

  build(g: Glyph, sweep: { rows: SweepRow[]; recommended_n: number; instances_swept?: number }) {
    // the band between the pipeline's minimum instance count and its confident count
    const KNEE = this.dataset.knee!.split(',').map(Number) as [number, number];
    const label = glyphLabel(g.char, g.style);
    const steps = g.trace_by_n;
    const last = steps.length - 1;
    const vb = viewBox(bbox(g.trace), PAD);

    this.innerHTML = `
<div class="stack">
  <div class="stack-main">
    <svg class="stack-svg" viewBox="${vb}" role="img" aria-labelledby="st-t st-d">
      <title id="st-t">The traced ${esc(label)} as more instances are stacked</title>
      <desc id="st-d">The outline traced from the first N printed instances of ${esc(label)}, in black, over the outline of the full stack of ${g.n}, in red.</desc>
      <g class="st-px"></g>
      <path class="st-now" fill-rule="evenodd"/>
      <path class="st-final" d="${g.trace}" fill-rule="evenodd"/>
    </svg>
  </div>
  <div class="stack-side">
    <label class="nrow" for="st-n">N <input id="st-n" type="range" min="0" max="${last}" step="1" value="${last}" class="n"> <output class="mono nout" for="st-n"></output></label>
    <fieldset class="toggles"><legend>Layers, front to back</legend>
      <label><input type="checkbox" data-layer="final" checked> Outline of all ${g.n}, red</label>
      <label><input type="checkbox" data-layer="now" checked> Trace from the first N</label>
      <label><input type="checkbox" data-layer="cov"> Cleaned coverage of one instance, 4× grid</label>
      <label><input type="checkbox" data-layer="raw"> Raw scan pixels of that instance</label>
    </fieldset>
    <p class="src px-src" aria-live="polite" hidden></p>
    <p class="mono stack-help">Drag the slider to trace the glyph from the first N instances of a fixed random order. Layers higher in the list sit in front. Turn on the pixel layers to see how the outline sits on what was scanned.</p>
    <dl class="readout mono" aria-live="polite">
      <dt>Cards among them</dt><dd class="r-cards"></dd>
    </dl>
    <figure class="sweep-fig">
      <svg class="sweep" viewBox="0 0 340 220" role="img" aria-labelledby="sw-t sw-d"><title id="sw-t">Edge error against the number of stacked instances</title><desc id="sw-d">Edge RMS against the measuring stick falls as more instances are stacked. The numbers are on the data page.</desc></svg>
      <figcaption>Edge error against the measuring stick as N grows, from a sweep over random subsets of ${sweep.instances_swept && sweep.instances_swept < g.n ? `${sweep.instances_swept} instances drawn at random from the ${g.n}` : `all ${g.n}`}. Solid line: edge error. Dashed: modelled noise. Grey band: p10 to p90. Red tick: recommended N. Pink band: the project’s sweet spot. The dot is the trace shown on the left, which uses one particular order, so it can sit off the curve. The numbers are on the <a href="${this.dataset.dataPage}#glyph-${this.dataset.glyph}">data page</a>.</figcaption>
    </figure>
  </div>
</div>`;

    const $ = <T extends Element>(s: string) => this.querySelector<T>(s)!;
    const slider = $<HTMLInputElement>('.n'), nout = $<HTMLElement>('.nout'), now = $<SVGPathElement>('.st-now');

    // sweep chart
    const rows = sweep.rows, W = 340, H = 220, m = { l: 40, r: 10, t: 12, b: 34 };
    const nmax = Math.max(...rows.map((r) => r.n));
    const ymax = Math.ceil(Math.max(...rows.map((r) => Math.max(r.p90, r.noise_model_um)), ...steps.map((s) => s.edge_rms_um ?? 0)));
    const X = (n: number) => m.l + ((n - 1) / (nmax - 1)) * (W - m.l - m.r);
    const Y = (v: number) => H - m.b - (v / ymax) * (H - m.t - m.b);
    const line = (k: 'rms_vs_ref_um' | 'noise_model_um') => rows.map((r, i) => `${i ? 'L' : 'M'}${X(r.n).toFixed(1)},${Y(r[k]).toFixed(1)}`).join('');
    const band = rows.map((r, i) => `${i ? 'L' : 'M'}${X(r.n).toFixed(1)},${Y(r.p90).toFixed(1)}`).join('') + [...rows].reverse().map((r) => `L${X(r.n).toFixed(1)},${Y(r.p10).toFixed(1)}`).join('') + 'Z';
    const xt = [1, 10, 20, 40, 60, 80, 100].filter((t) => t <= nmax).concat([nmax]).filter((v, i, a) => a.indexOf(v) === i);
    const sw = $<SVGSVGElement>('.sweep');
    sw.insertAdjacentHTML('beforeend', `
<rect class="knee" x="${X(KNEE[0])}" y="${m.t}" width="${Math.max(0, X(Math.min(KNEE[1], nmax)) - X(KNEE[0]))}" height="${H - m.t - m.b}"/>
<path class="band" d="${band}"/><path class="noise" d="${line('noise_model_um')}"/><path class="rms" d="${line('rms_vs_ref_um')}"/>
<line class="rec" x1="${X(sweep.recommended_n)}" x2="${X(sweep.recommended_n)}" y1="${m.t}" y2="${H - m.b}"/>
<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/><line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}"/>
${xt.map((t) => `<text x="${X(t)}" y="${H - m.b + 14}" text-anchor="middle">${t}</text>`).join('')}
${Array.from({ length: ymax + 1 }, (_, v) => v).filter((v) => v % 2 === 0).map((v) => `<text x="${m.l - 5}" y="${Y(v) + 3}" text-anchor="end">${v}</text>`).join('')}
<text x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle">N instances stacked</text>
<text transform="translate(10 ${(H - m.b) / 2}) rotate(-90)" text-anchor="middle">µm</text>
<circle class="mk" r="4"/>`);
    const mk = sw.querySelector<SVGCircleElement>('.mk')!;

    const show = (i: number) => {
      const s = steps[i];
      now.setAttribute('d', s.trace);
      const txt = s.n === g.n ? `${s.n} of ${g.n}, all` : `${s.n} of ${g.n}`;
      nout.textContent = txt;
      slider.setAttribute('aria-valuetext', `N equals ${txt}`);
      $('.r-cards').textContent = String(s.cards);
      mk.toggleAttribute('hidden', s.edge_rms_um === null);
      if (s.edge_rms_um !== null) { mk.setAttribute('cx', String(X(s.n))); mk.setAttribute('cy', String(Y(s.edge_rms_um))); }
    };
    slider.addEventListener('input', () => show(+slider.value));

    // layers, the pixel layers load their crop the first time one is switched on
    const svg = $<SVGSVGElement>('.stack-svg'), px = $<SVGGElement>('.st-px'), pxSrc = $<HTMLElement>('.px-src');
    let pixels: Promise<Pixels | null> | null = null;
    const loadPixels = () => (pixels ??= (async () => {
      try {
        const lp: Pixels = await fetch(`${this.dataset.base}pixels/${this.dataset.glyph}.json`).then((r) => r.json());
        const img = (i: Img, cls: string) => `<image class="${cls}" href="${this.dataset.base}pixels/${i.file}" x="-0.5" y="-0.5" width="${i.w}" height="${i.h}" transform="matrix(${i.matrix.join(' ')})" preserveAspectRatio="none" hidden/>`;
        px.innerHTML = img(lp.raw, 'px-raw') + img(lp.cov4, 'px-cov');
        pxSrc.textContent = `Pixel layers: one instance of ${label} from ${cardName(lp.card)}, ${lp.scan} (scanned at ${deg(lp.angle)}), ${lp.id}. The raw crop is ${lp.raw.w} × ${lp.raw.h} px at ${lp.raw.um_per_px.toFixed(1)} µm each, the coverage is on a ${lp.cov4.um_per_px.toFixed(1)} µm grid.`;
        return lp;
      } catch (err) {
        console.warn('stack-view pixels', err);
        return null;
      }
    })());
    const layers: Record<string, Element> = { now, final: $('.st-final'), raw: px, cov: px };
    this.querySelectorAll<HTMLInputElement>('[data-layer]').forEach((c) => c.addEventListener('change', async () => {
      const k = c.dataset.layer!;
      if (k === 'now' || k === 'final') { layers[k].toggleAttribute('hidden', !c.checked); return; }
      if (c.checked && !(await loadPixels())) { c.checked = false; return; }
      const on = (n: string) => this.querySelector<HTMLInputElement>(`[data-layer=${n}]`)!.checked;
      px.querySelector('.px-raw')?.toggleAttribute('hidden', !on('raw'));
      px.querySelector('.px-cov')?.toggleAttribute('hidden', !on('cov'));
      svg.classList.toggle('px-on', on('raw') || on('cov'));
      pxSrc.hidden = !(on('raw') || on('cov'));
    }));
    show(last);
  }
}

customElements.define('stack-view', StackView);
