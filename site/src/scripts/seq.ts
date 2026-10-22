// Reveals each step as it scrolls into view and marks the current step in the index
const seq = document.querySelector<HTMLElement>('.seq');
if (seq) {
  const steps = [...seq.querySelectorAll<HTMLElement>('.step')];
  const links = [...document.querySelectorAll<HTMLAnchorElement>('.seq-nav a')];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduce) seq.dataset.js = '';
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) (e.target as HTMLElement).dataset.in = '';
    }
    const visible = steps.filter((s) => { const r = s.getBoundingClientRect(); return r.top < innerHeight * 0.55; });
    const cur = visible.length ? visible[visible.length - 1].id : '';
    links.forEach((a) => a.toggleAttribute('aria-current', a.hash === `#${cur}`));
  }, { rootMargin: '0px 0px -15% 0px', threshold: [0, 0.2] });
  steps.forEach((s) => io.observe(s));
}
