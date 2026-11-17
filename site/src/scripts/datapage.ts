// The data page's table filters and the stacking panel, which shows one glyph's numbers at a time

import { sweepAxes, type SweepRow } from '../lib/sweepmap';

const num = (v: number | null | undefined, d: number) => (v === null || v === undefined ? 'n/a' : v.toFixed(d));

// filters hide rows with the hidden attribute, so sorting a table keeps them hidden
for (const form of document.querySelectorAll<HTMLFormElement>('form.filters[data-for]')) {
  const table = document.getElementById(form.dataset.for!) as HTMLTableElement;
  const rows = [...table.tBodies[0].rows];
  const count = form.querySelector<HTMLElement>('.count')!;
  const noun = form.dataset.for === 'glyph-table' ? 'glyphs' : 'pairs';
  const keep = (r: HTMLTableRowElement, f: FormData) => {
    if (f.has('min')) return Number(r.dataset.n) >= Number(f.get('min') || 1);
    const q = String(f.get('q') ?? '').trim(), face = f.get('face');
    if (q && r.dataset.char !== q && !r.cells[0].dataset.v!.startsWith(q)) return false;
    if (face && r.dataset.face !== face) return false;
    return !(f.has('noprior') && r.dataset.prior !== undefined);
  };
  const apply = () => {
    const f = new FormData(form);
    let shown = 0;
    for (const r of rows) { r.hidden = !keep(r, f); if (!r.hidden) shown++; }
    count.textContent = `Showing ${shown} of ${rows.length} ${noun}.`;
  };
  form.addEventListener('input', apply);
  form.addEventListener('submit', (e) => e.preventDefault());
  form.hidden = false;
  apply();
}

class StackPanel extends HTMLElement {
  connectedCallback() {
    const select = this.querySelector<HTMLSelectElement>('select')!;
    const ax = sweepAxes(Number(this.dataset.nmax), Number(this.dataset.ymax));
    const sel = this.querySelector<SVGGElement>('.sel')!;
    const curves = this.querySelector<SVGGElement>('.curves')!;
    const label = (name: string) => select.querySelector<HTMLOptionElement>(`option[value="${CSS.escape(name)}"]`)?.textContent?.replace(/, \d+ instances?$/, '') ?? name;
    let token = 0;
    const show = async (name: string, scroll: boolean) => {
      if (!select.querySelector(`option[value="${CSS.escape(name)}"]`)) return;
      select.value = name;
      const mine = ++token;
      const base = this.dataset.base!;
      const [g, sw] = await Promise.all([
        fetch(`${base}glyphs/${name}.json`).then((r) => r.json() as Promise<{ n: number; trace_by_n: { n: number; cards: number; nodes: number; edge_rms_um: number | null }[] }>),
        fetch(`${base}sweep/${name}.json`).then((r) => r.json() as Promise<{ rows: SweepRow[]; recommended_n: number }>),
      ]);
      if (mine !== token) return;
      const text = label(name);
      for (const el of this.querySelectorAll('.sm-name')) el.textContent = text;
      this.querySelector('.sp-n')!.textContent = String(g.n);
      this.querySelector('#sm-d')!.textContent = `${curves.children.length} curves of edge RMS in micrometres against N on a log axis. The picked glyph, ${text}, is drawn in red.`;
      sel.querySelector('.band')!.setAttribute('d', ax.band(sw.rows));
      sel.querySelector('.noise')!.setAttribute('d', ax.line(sw.rows, 'noise_model_um'));
      sel.querySelector('.rms')!.setAttribute('d', ax.line(sw.rows, 'rms_vs_ref_um'));
      const rec = sel.querySelector('.rec')!;
      for (const a of ['x1', 'x2']) rec.setAttribute(a, ax.X(sw.recommended_n).toFixed(1));
      this.querySelector('.sp-trace')!.innerHTML = g.trace_by_n.map((t) => `<tr><td class="r">${t.n}</td><td class="r">${t.cards}</td><td class="r">${t.nodes}</td><td class="r">${num(t.edge_rms_um, 2)}</td></tr>`).join('');
      this.querySelector('.sp-sweep')!.innerHTML = sw.rows.map((t) => `<tr><td class="r">${t.n}</td><td class="r">${num(t.rms_vs_ref_um, 2)}</td><td class="r">${num(t.p10, 2)}</td><td class="r">${num(t.p90, 2)}</td><td class="r">${num(t.noise_model_um, 2)}</td></tr>`).join('');
      this.dataset.shown = name;
      if (scroll) this.scrollIntoView({ block: 'start' });
    };
    const fromHash = (scroll: boolean) => {
      const m = /^#glyph-(.+)$/.exec(decodeURIComponent(location.hash));
      if (m) void show(m[1], scroll);
    };
    select.addEventListener('change', () => {
      history.replaceState(null, '', `#glyph-${select.value}`);
      void show(select.value, false);
    });
    addEventListener('hashchange', () => fromHash(true));
    this.querySelector('form')!.hidden = false;
    this.dataset.shown = select.value;
    fromHash(true);
  }
}
customElements.define('stack-panel', StackPanel);
