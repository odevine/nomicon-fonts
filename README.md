# nomicon-fonts

The pipeline in `glyphstack/` takes a card scan and writes glyph crops, SVG outlines, an index and a contact sheet to `build/<scan name>/`.

```
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m glyphstack.pipeline scans/test_scan.pdf
```

`scan.py` squares the card, finds the rules box, deskews on the text and flattens the box texture into an ink-density image where paper is 0 and solid ink is 1. `pipeline.py` thresholds that density at 0.5 and splits it into glyphs, one connected component each, with dots and accents merged into their base. Glyphs that touch in the scan stay together and are marked `suspect_merged` in `index.json`. Pass `--box x,y,w,h` (scan pixels) to override box detection.

Pass `--card-name "Reef Worm"` to align the boxes to that card's text from the scan list spreadsheet. `align.py` fits the known characters to the boxes with a sequence alignment, using each character's height above the baseline, descent and width (learned from the card itself) in place of OCR. Each unit is written to `regular/<glyph name>/<card id>_iNN.*` with a density crop, mask and SVG preview, and `manifest.json` records its measurements. Capitals get a `_uc` folder suffix so case-insensitive filesystems keep `a` and `A` apart. Boxes where two letters touch in the scan are listed under `rejected` so another instance can be used.

Several shifted passes of one card can be given together. `superres.py` registers them with ECC and fuses them onto a 2x grid by iterative back-projection, and the glyphs are then cut from the fused image.

```
.venv/bin/python -m glyphstack.pipeline pass1.tif pass2.tif pass3.tif pass4.tif
```

`pytest` runs a synthetic check that fusing four shifted half-resolution passes lands closer to the original than upscaling one pass.

## Writing a scan table

A scan table is a CSV file that lists a session's scans, one row per scan, with the columns `file`, `card` and `angle`. Both `prepare.py` and `session.py` read it. This one covers one card at the four jig angles and the same four turned a quarter, half of the sixteen 22.5 degree steps, and a second card scanned upright and turned a quarter. The file names give the angle in tenths of a degree.

```
file,card,angle,notes
reefworm_0000.png,ReefWorm,0,
reefworm_0225.png,ReefWorm,22.5,
reefworm_0450.png,ReefWorm,45,
reefworm_0675.png,ReefWorm,67.5,
reefworm_0900.png,ReefWorm,90,
reefworm_1125.png,ReefWorm,112.5,
reefworm_1350.png,ReefWorm,135,
reefworm_1575.png,ReefWorm,157.5,
altair_000.png,Altair,0,
altair_090.png,Altair,90,rescanned after the card slipped
```

**The table can have any name and live in any folder.** Each `file` is read relative to the table's own folder, so the table usually sits beside its scans in `scans/`, which keeps both out of git. `prepare.py` writes a copy of the table beside the cropped scans with the paths pointing at them, and that copy is the one to give the session. Calibration scans are not listed, they are passed with `--cal-a` and `--cal-b`.

**`card` is an ID that groups the scans of one physical card.** The pipeline only checks whether two scans share it and never looks the card up. The site shows it, splitting CamelCase into words, so write it in CamelCase without spaces or accents, such as `ReefWorm` or `Altair`. It cannot contain a colon, because instance IDs are built as `card:char:scan:unit`. A card whose printed name has an accent gets an entry in the `display` table in `site/src/lib/format.ts`, the way `Altair` is shown as Altaïr.

**`angle` is the card's counterclockwise turn in degrees as it appears in the scan image.** The card lies face down on the glass, so a turn that looks counterclockwise from above shows up clockwise in the scan. Write the jig's angle plus any quarter turns, such as 112.5 for the 22.5 degree jig turned a quarter. Negative angles work too, and -22.5 is the same as 337.5. The card outline refines the angle only within a quarter turn, so the table has to give the quarter turn. If every off-axis scan warns that it is turned about twice its notch the other way, the jig turns against this convention, and each angle should be written as 360 minus the angle, such as 337.5 in place of 22.5.

**Every card should have a scan at 0 degrees and at least one more.** The card's first 0 degree scan is the reference that its other scans are matched to. A card without one is warned about and uses its first scan instead, and a card with a single scan gives each of its glyphs only one view.

**Other columns are ignored by the session and kept by `prepare.py`**, so a column of notes is safe.

## Cropping scans first

`prepare.py` crops every scan in a scan table to its card plus a margin (150 px by default) before a session runs. A flatbed scan of one card is about 90 % empty bed, which compresses well in an 8-bit PNG but not in a 16-bit TIFF, where a whole-bed scan is about 770 MB against about 160 MB cropped. The crop slices whole pixels and keeps the format and bit depth, so the cropped scan holds exactly the card region's pixels, and its offset sits on the 8 px grid the card finder works on, so the session finds the same units in it. Originals are never changed.

```
.venv/bin/python -m glyphstack.prepare scans/table.csv -o scans/cropped
```

The output folder gets the cropped scans under their original names, a copy of the scan table pointing at them, and `prepare.json` with each original's sha256, size and crop box. A rerun skips scans whose original and margin are unchanged. Only PNG and TIFF scans are cropped, because other formats would be re-encoded.

## Running a session

`session.py` runs a whole scanning session. It reads a CSV scan table with the columns `file`, `card` and `angle`, checks the calibration scans, extracts and labels the glyphs of every scan, stacks them and traces the glyphs with enough instances.

`angle` is written as described under Writing a scan table. Quarter turns are undone exactly. Any other turn is measured from the card outline and undone for labelling, while the stacked coverage is read from the native pixels so each view is resampled once. A small error in the angle, such as a card sitting slightly off in one of the jigs in `scanner/`, is taken up by the per-scan fit of the glyphs, and every scan whose outline is turned more than 2 degrees from its `angle` is logged as a warning so a slipped card can be rescanned. Every scan of a card is matched to the card's upright scan from glyph positions, which gives each scan's map to that scan and groups the views of each printed glyph into one instance. Within each card, a printed glyph's views are registered against each other and fused into that instance, and each glyph's instances are then registered and stacked. Cards are extracted and fused in parallel, one process each, and glyphs are registered and stacked in parallel, one process each, with `--workers` to set how many.

```
.venv/bin/python -m glyphstack.session scans/cropped/table.csv -o build/session --font /path/to/font.ttf --cal-a cal_a.png --cal-b cal_b.png
```

Flavor text and reminder text are printed in the italic face. Add `--italic-font /path/to/italic.ttf` to label, stack and trace it in the same run. Each unit is matched against the templates of both faces and keeps the face that fits, and an italic glyph has the key `i:` plus its character, so its stack, trace and exports stay apart from the regular glyph. Italic glyphs are never used to estimate scan geometry. A cache under `<out>/work` built with other faces or another profile is rebuilt. `--italic-font PATH` is the same as `--face-font italic=PATH`.

The font is a measuring stick for labelling and tracing, so keep it out of the repository. On a terminal each stage, card and scan draws a live progress bar, and the console shows only warnings, errors, the run log's path and the final result. Elsewhere progress is logged to stderr with a timestamp every few seconds. Every run also writes a full debug log to `<out>/logs`, set with `--log`, and `-v` shows all of it on the console too. When nothing is reported for `--watchdog` seconds (300 by default, 0 turns it off), the stack of every thread in every process is written to that log, so a stuck run shows where it is waiting without being killed. Extracted scans are cached under `<out>/work` so a rerun resumes. Each glyph's registered units are written to `<out>/work/regs/<glyph>` in compressed chunks, and registration and stacking read and write them one batch at a time, so memory stays flat however many instances a glyph has. The extracted units of every scan stay in memory for the whole run, about 50 KB per view. `--cal-a` and `--cal-b` are two scans of the printed target in `scanner/calibration_target.pdf`, the second turned 90 degrees clockwise. The calibration gate stops the session when scale is outside 47.32 ± 0.15 px/mm or the x and y scales differ by more than 0.2 %, `--skip-gate` overrides it. Outputs are `stacks/<glyph>/` (`stack.npz` with the mean, spread and leave-one-out IoU, and the kept instances in `inst/` as chunks that `stack.load` reads lazily), `manifest.json` (per instance views, edge level, halo, pair id, card scale, deviation from the stack and the view against view edge distance), `traces.json` and `calibration.json`. `python -m glyphstack.calibration cal_a.png cal_b.png` runs the gate alone.

## Target profiles

A target profile says what type a run measures and where it is printed. Profiles are TOML files, and `glyphstack/profiles/rules.toml`, the face of rules, flavor and reminder text traced as Nomicon Rules and Nomicon Rules Italic, is the default. Pass `--profile NAME` for another file in that folder or `--profile path/to/profile.toml` for one elsewhere. The session records the profile's path and hash in `manifest.json`, and the export uses the same profile unless told otherwise.

A profile lists the typeface's faces. The first is the primary face: its glyph keys are the bare character, and the print scale and the shared fit come from it. Each other face gives a key prefix, a cache file prefix and a site name suffix, the way the italic face uses `i:`, `it_` and `_it`. Each face also has a `full_name`, the font name it is traced as, which labels it in the scan list spreadsheet. Fonts are never named in a profile. The primary face's font is given with `--font` and each other face's with `--face-font NAME=PATH`.

`[[regions]]` name the parts of the card to read, as fractions of the upright card crop. Components outside every region are not made into units, though their ink still masks neighbouring glyphs, and the scan's paper and ink levels are measured inside the regions. `[paper] mode = "local"` gives each unit the paper level of the ring around its own ink, for type printed on frames whose colour changes across the card, such as a title bar and a type line.

The other sections hold the settings tuned to the type: the segmentation threshold and size limits, the template character set, em size and print scale range, the matching thresholds, the characters scan geometry is estimated from, the stacking thresholds, the glyph the shared fit starts from with its starting values, and the glyph and card the site's step figure uses. `rules.toml` documents each one.

## Exporting data for the site

`export_site.py` reads a session folder and writes the JSON the documentation site renders.

```
.venv/bin/python -m glyphstack.export_site build/session --font /path/to/font.ttf -o site/data
```

It writes `glyphs.json`, `glyphs/<name>.json` (trace, mean iso-contours at 0.1 to 0.9, a low resolution spread grid, every instance's 0.5 contour with its card, scans, edge level, halo, card scale, deviation and leave-one-out IoU), `sweep/<name>.json` (edge error and noise against the number of stacked instances, with the modelled floor and a recommended N), `kerning.json` (each adjacent pair's designed gap and observed mean and spread), `kerning_samples.json` (every measured sample of each pair with its scan), `noise.json`, `calibration.json` and `cards.json`. Pass `--italic-font` to export the italic glyphs of a session run with it. Italic names end in `_it` (`a_it`, `A_uc_it`), their rows in `glyphs.json` have `style` set to `italic`, and the italic advance metrics and kerning go to `metrics.italic.json`, `kerning.italic.json` and `kerning_samples.italic.json`. The font is only a measuring stick. Its outline is not exported whole: each glyph gets at most `--cutaways` short clipped windows (default 3, none with 0) around the largest misfit to the stack, which together stay under 20 % of the outline length. `--simplify` sets the path tolerance in font units (default 0.5). To keep files small each glyph exports at most `--max-instances` randomly chosen instances (default 64, seeded, 0 for all) from those that passed the outlier check. The stack and statistics always use every instance, and each glyph file records how many were exported. The sweep runs on at most `--sweep-max` randomly chosen instances (default 512, seeded, 0 for all), well past the point where its curve flattens, and each sweep file records how many it used. Instances are read from the session's chunks as each glyph is exported, so memory follows one glyph's sample rather than the session. The glyph fits come from the session's `traces.json`, and glyphs are exported one per process, `--workers` of them, default one per core. `noise.json` gives the noise between views of the same printed glyph scanned at different angles, with scale and shear tied across a card's scans, as the headline figure. Pass `--compare-session` with a second session run using `--no-tie` to list the figure with each scan fitted on its own beside it.

## Repository layout

`glyphstack/` is the Python package, named for the process of stacking scanned glyphs rather than for the typeface it measures, and `tests/` holds its tests. `scanner/` holds the printed calibration target and card jigs, with the script that draws the jigs. `site/` is the documentation site, an Astro project whose `data/` folder is the committed output of `export_site`.

Scans, run output and working notes stay out of git. `scans/` holds the raw and cropped scans, which show card artwork. `build/` holds session and export output. `notes/` holds the scan list spreadsheet, which `pipeline.py --card-name` reads by default. `.venv/` is the Python environment.
