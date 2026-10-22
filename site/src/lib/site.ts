export const base = import.meta.env.BASE_URL.replace(/\/$/, '');
export const href = (p: string) => `${base}${p}`;

// pages marked hidden have no content yet: they are left out of the nav, and their folders start with _ so Astro does not build them
export const pages: { path: string; title: string; milestone: string; blurb: string; hidden?: boolean }[] = [
  { path: '/cards/', hidden: true, title: 'The cards', milestone: 'M6', blurb: 'Why these cards, and which glyphs each contributes.' },
  { path: '/capture/', title: 'Capture', milestone: 'M4', blurb: 'The scanner, the calibration target and the noise floor.' },
  { path: '/cleaning/', title: 'Cleaning', milestone: 'M2', blurb: 'From a scan to one clean glyph, step by step.' },
  { path: '/stacking/', hidden: true, title: 'Stacking', milestone: 'M1', blurb: 'How many printed copies of a letter become one shape.' },
  { path: '/tracing/', hidden: true, title: 'Tracing', milestone: 'M1', blurb: 'From a mask to Bézier outlines.' },
  { path: '/printing-leaves-a-mark/', title: 'Printing leaves a mark', milestone: 'M5', blurb: 'What the stack shows that the design does not.' },
  { path: '/glyphs/', title: 'Glyphs', milestone: 'M1', blurb: 'The character set, one page per glyph.' },
  { path: '/typeset/', title: 'Typeset', milestone: 'M3', blurb: 'Set your own text with real printed letters.' },
  { path: '/log/', hidden: true, title: 'Log', milestone: '', blurb: 'Dated progress entries, newest first.' },
  { path: '/data/', title: 'Data and method', milestone: '', blurb: 'Every data file, how to reproduce it, and the limits.' },
  { path: '/about/', title: 'About', milestone: '', blurb: 'Name, licences, credits and the Fan Content Policy notice.' },
];
