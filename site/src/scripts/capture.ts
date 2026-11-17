// Calibration Bench tone readout, a small island over a server-rendered figure

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
