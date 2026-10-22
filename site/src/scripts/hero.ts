interface Inst { path: string }
interface Instances { instances: Inst[]; instances_total: number }

const NS = 'http://www.w3.org/2000/svg';
const SPREAD = 260;
const SETTLED_BY = 0.72;
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const ease = (t: number) => 1 - (1 - t) ** 3;
const smooth = (a: number, b: number, v: number) => { const t = clamp((v - a) / (b - a)); return t * t * (3 - 2 * t); };

/**
 * The letter behind the page. Each printed instance fades in as it nears the stack, once they are all stacked
 * they fade out until only the trace is left, and the trace then dims so the text over it stays readable
 */
class HeroStack extends HTMLElement {
  private bg!: HTMLElement;
  private svg!: SVGSVGElement;
  private run!: HTMLElement;
  private paths: SVGPathElement[] = [];
  private group?: SVGGElement;
  private off: [number, number][] = [];
  private queued = false;

  connectedCallback() {
    if (!document.documentElement.classList.contains('hero-js')) return;
    this.bg = this.querySelector<HTMLElement>('.hero-bg')!;
    this.svg = this.bg.querySelector('svg')!;
    this.run = this.querySelector<HTMLElement>('.hero-run')!;
    addEventListener('scroll', () => this.schedule(), { passive: true });
    addEventListener('resize', () => this.schedule(), { passive: true });
    this.update();
    this.load();
  }

  async load() {
    try {
      const g: Instances = await fetch(`${this.dataset.base}instances/${this.dataset.glyph}.json`).then((r) => r.json());
      this.group = document.createElementNS(NS, 'g');
      this.group.setAttribute('class', 'hero-inst');
      let s = 7;
      const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
      for (const it of g.instances) {
        const p = document.createElementNS(NS, 'path');
        p.setAttribute('d', it.path);
        this.group.append(p);
        this.paths.push(p);
        this.off.push([rnd() * SPREAD, rnd() * SPREAD]);
      }
      this.svg.insertBefore(this.group, this.svg.querySelector('.ink'));
      const note = document.querySelector('[data-hero-note]');
      if (note) note.textContent = `The outlines behind this page are ${g.instances.length} of the ${g.instances_total} printed e’s, drawn at the 0.5 coverage level.`;
      this.update();
    } catch (err) {
      // the static trace stays
      console.warn('hero-stack', err);
    }
  }

  schedule() {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => { this.queued = false; this.update(); });
  }

  update() {
    const span = Math.max(1, this.run.offsetHeight - innerHeight);
    const p = clamp(scrollY / span);
    // after the run the trace dims over a stretch of scroll so the text stays readable
    const dim = clamp((scrollY - span) / (innerHeight * 0.6));
    this.bg.style.opacity = (1 - 0.88 * smooth(0, 1, dim)).toFixed(3);
    this.svg.style.setProperty('--ink-o', smooth(0.7, 0.9, p).toFixed(3));
    if (!this.group) return;
    this.group.style.opacity = (1 - smooth(0.76, 0.96, p)).toFixed(3);
    const n = this.paths.length;
    this.paths.forEach((el, i) => {
      const e = ease(clamp((p / SETTLED_BY - (i / n) * 0.35) / 0.65));
      const [dx, dy] = this.off[i];
      el.setAttribute('transform', `translate(${(dx * (1 - e)).toFixed(1)} ${(dy * (1 - e)).toFixed(1)})`);
      el.style.opacity = (0.12 + 0.88 * smooth(0.05, 0.75, e)).toFixed(3);
    });
  }
}

customElements.define('hero-stack', HeroStack);
