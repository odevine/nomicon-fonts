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

`python -m glyphstack.corners SESSION --font FONT --italic-font ITALIC -o site/data` writes `corners.json`, the median edge offset of each face's stacks from the measuring stick at its corners by kind and angle, with a typical inner and outer corner. The Bleed page and the Data page read it. It holds numbers and corner positions only, never the font's outlines.

## Compositor data

`export_site.py` writes `metrics.json` (the measuring stick's vertical metrics and space advance, numbers only). The build serves `spacing.json` (advance widths, metrics and the observed pair offsets) and `compose/{glyph}.json` (the mean trace and the sampled instance outlines) to the Typeset page, which loads a glyph's file only when a letter needs it.

## The site's font

The site is set in Nomicon Rules, the font this project builds. `python -m glyphstack.fontbuild ... --site site/public/fonts` writes each face as `nomicon-rules-{face}.woff` with `nomicon-rules.json`, which lists the characters each face draws and its Bleed axis. Text falls back to EB Garamond, the stand-in serif, for any character the font doesn't draw and for weight 600, since the font has no bold yet.

The Glyphs page draws each glyph built from the measuring stick in the font, and the Typeset page draws every letter with no scans in it at its own advance, with a third Letters choice that sets every letter in it. Both pages get a Bleed slider. With Designed kerning chosen, the Typeset page takes the kern of any pair the cards didn't measure from the font.
