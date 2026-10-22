# Nomicon Fonts site

An Astro static site. Every number it shows is read from `data/`.

```
npm ci
npm run dev      # local server
npm run build    # static output in dist/
npm test         # builds, serves and runs Playwright with axe on every page
```

`data/` is the committed pipeline export. `glyphstack.export_site` writes it in place, `-o site/data` being its default. The Pages workflow does not run the export because raw scans are not in CI.

Needs Node 22.12 or newer. `.github/workflows/pages.yml` builds and deploys on push to `main` once Pages is enabled for the repository.

## Crops for the pixel layers and Scan to Glyph

`python -m glyphstack.export_site SESSION --font FONT --scans SCAN_DIR -o site/data` also writes `steps/` and `pixels/`, which feed the Stack's pixel layers with small raw crops, step images and the transforms that place them in font units. The crops are at most 240 × 144 scan pixels, a few letters of rules text. Without `--scans` they are skipped and the existing files stay. The font is a measuring stick and is never copied.

## Compositor data

`export_site.py` writes `metrics.json` (the measuring stick's vertical metrics and space advance, numbers only). The build serves `spacing.json` (advance widths, metrics and the observed pair offsets) and `compose/{glyph}.json` (the mean trace and the sampled instance outlines) to the Typeset page, which loads a glyph's file only when a letter needs it.
