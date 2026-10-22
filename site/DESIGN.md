# Site design conventions

These are the layout and content rules the pages follow. `tests/layout.spec.ts` checks every block on every page is centred and no wider than its class allows. A new page needs an entry there.

## Layout: a breakout grid, text left aligned inside it

Everything in `main` is a centred block with equal margins on both sides. Prose takes the text column, and a figure that needs more room asks for it with a class. Text stays left aligned inside its block. Nothing is centre-aligned text, and nothing hangs off the left edge of the window.

| Class     | Width (centred)                      | Use                                                     |
|-----------|--------------------------------------|---------------------------------------------------------|
| (none)    | `--measure`, 38rem                   | Prose, headings, ledes, notes, small controls           |
| `.mid`    | 46rem                                | Figures with a little more room: cards, charts          |
| `.wide`   | `--container`, 62rem                 | Tables, the glyph sheet, the stack, the typesetter      |
| `.full`   | the whole page less the gutters      | The home page hero                                      |
| `.flow`   | passes the rule on to its children   | A wrapper that holds a mix of text and figures          |

The header and footer sit on the `.wide` edge, so a wide figure starts exactly where the wordmark starts. The layout rule is one block in `global.css`. It uses `!important` on `max-width` and the inline margins on purpose, because many components set their own and none of them may break it. Do not add `.column` style wrappers to pages. `tests/layout.spec.ts` checks every block on every page against its class, so a new page needs only an entry in the path list.

## Content

- The landing page is the hero and the "why". Statistics belong on the data page.
- Interactive figures avoid the nitty-gritty numbers. Tables, sweeps and spacing go on the data page, and the figure links to them.
- Notes are plain text with a red rule in the margin. No proofreader's labels such as "stet" or "placeholder".
- Figure labels are HTML in one standard size, not SVG text that scales with its figure. Axis ticks inside a chart are the exception.
- Layer lists read front to back, and the list order is the stacking order.

## Colour and controls

Paper, ink, proofing red, one grey and one blue. Ink drawn as black with alpha is inverted in dark mode by the global `.ink-img` rule. Radio buttons, checkboxes and sliders share the red accent.

## Pages that are not ready

A page with no content yet is hidden twice: its entry in `src/lib/site.ts` carries `hidden: true`, which leaves it out of the nav, and its folder in `src/pages` starts with an underscore (`_tracing`), which Astro does not build. To publish it, rename the folder, drop the flag, and add it back to the path lists in `tests/pages.spec.ts` and `tests/layout.spec.ts`.

## Testing

`npm test` always rebuilds the site before running, so it never audits a stale build.
