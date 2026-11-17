export const base = import.meta.env.BASE_URL.replace(/\/$/, '');
export const href = (p: string) => `${base}${p}`;

// pages marked hidden have no content yet: they are left out of the nav, and their folders start with _ so Astro does not build them
export const pages: { path: string; title: string; milestone: string; blurb: string; hidden?: boolean }[] = [
  { path: '/cards/', hidden: true, title: 'The cards', milestone: 'M6', blurb: 'Why these cards, and which glyphs each contributes.' },
  { path: '/capture/', title: 'Capture', milestone: 'M4', blurb: 'The scanner, the calibration target and the noise floor.' },
  { path: '/cleaning/', title: 'Cleaning', milestone: 'M2', blurb: 'From a scan to one clean glyph, step by step.' },
  { path: '/stacking/', hidden: true, title: 'Stacking', milestone: 'M1', blurb: 'How many printed copies of a letter become one shape.' },
  { path: '/tracing/', hidden: true, title: 'Tracing', milestone: 'M1', blurb: 'From a mask to Bézier outlines.' },
  { path: '/printing-leaves-a-mark/', title: 'Printing leaves a mark', milestone: 'M5', blurb: 'What the stack shows that the design doesn’t.' },
  { path: '/glyphs/', title: 'Glyphs', milestone: 'M1', blurb: 'The character set, one page per glyph.' },
  { path: '/typeset/', title: 'Typeset', milestone: 'M3', blurb: 'Set your own text with real printed letters.' },
  { path: '/notes/', title: 'Notes', milestone: '', blurb: 'Longer write-ups of problems that came up along the way.' },
  { path: '/log/', hidden: true, title: 'Log', milestone: '', blurb: 'Dated progress entries, newest first.' },
  { path: '/data/', title: 'Data and method', milestone: '', blurb: 'Every data file, how to reproduce it, and the limits.' },
  { path: '/about/', title: 'About', milestone: '', blurb: 'Credits and the Fan Content Policy notice.' },
];

// write-ups listed on the notes page, newest first by date
export const notes: { path: string; title: string; date: string; blurb: string }[] = [
  { path: '/notes/halftone-screen/', title: 'The halftone screen behind the text', date: '2026-10-14', blurb: 'The dots behind the rules text are about the size of a letter’s edge. I tried three ways to get rid of them on one card, and none was worth keeping.' },
  { path: '/notes/text-over-texture/', title: 'Text printed over texture', date: '2026-10-23', blurb: 'Three cards whose backgrounds hid their rules text, and how I got the pipeline from reading 297 of their 675 letters to 573.' },
  { path: '/notes/askew-scans/', title: 'A degree askew', date: '2026-10-26', blurb: 'Cards and the calibration sheet never sit quite square on the scanner. Most of the pipeline doesn’t mind, but two measurements that assumed level lines did.' },
  { path: '/notes/pipeline-performance/', title: 'Fitting the full run on one machine', date: '2026-10-07', blurb: 'Three rounds of work took the projected full run from 3.3 hours and over 200 GB to about 12 minutes and 7 GB, without changing the letters it draws.' },
  { path: '/notes/benchmark/', title: 'Scoring against the printed text', date: '2026-10-27', blurb: 'I know every card’s text ahead of time, so I placed every printed letter on 28 cards and now score each change on the letters it finds and names.' },
  { path: '/notes/reading-every-mark/', title: 'Reading every mark', date: '2026-10-28', blurb: 'An i and its dot, a quote in two halves, a letter and its accent. Joining those pieces took italic i and j from none named to 155, and false labels from 152 to 11.' },
  { path: '/notes/rare-letters/', title: 'Letters printed too few times', date: '2026-10-30', blurb: 'Capitals, digits and rare letters don’t print often enough to stack. Blending their few copies with the measuring stick, drawn the way the cards print, took the traced glyphs from 57 to 166.' },
  { path: '/notes/two-angles/', title: 'Sixteen angles to two', date: '2026-11-16', blurb: 'I cut the scan plan from sixteen passes per card to two. What the extra angles were meant to buy, why I dropped them, and what that gives up.' },
  { path: '/notes/ink-over-screen/', title: 'Ink printed over a screen', date: '2026-10-29', blurb: 'I fitted a model of black ink over a halftone screen on one card. It ruled out a knockout, then trained a small network that cut the edge noise of letters on screen from 10.4 to 7.6 µm.' },
  { path: '/notes/colour-against-grey/', title: 'Colour scans against grey', date: '2026-10-18', blurb: 'Text over a halftone looks sharper in a colour scan. I scanned the calibration target and three cards both ways to see what that is, and what it isn’t.' },
];

export const noteDate = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
export const note = (path: string) => notes.find((n) => n.path === path)!;
