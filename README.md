# nomicon-fonts

The pipeline in `glyphstack/` takes a card scan and writes glyph crops, SVG outlines, an index and a contact sheet to `build/<scan name>/`.

```
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m glyphstack.pipeline scans/test_scan.pdf
```

`scan.py` squares the card, finds the rules box, deskews on the text and flattens the box texture into an ink-density image where paper is 0 and solid ink is 1. `pipeline.py` thresholds that density at 0.5 and splits it into glyphs, one connected component each, with dots and accents merged into their base. Glyphs that touch in the scan stay together and are marked `suspect_merged` in `index.json`. Pass `--box x,y,w,h` (scan pixels) to override box detection.

Pass `--cards cards.json --card ReefWorm` to align the boxes to that card's text from a cards file. `align.py` fits the known characters to the boxes with a sequence alignment, using each character's height above the baseline, descent and width (learned from the card itself) in place of OCR. Each unit is written to `regular/<glyph name>/<card id>_iNN.*` with a density crop, mask and SVG preview, and `manifest.json` records its measurements. Capitals get a `_uc` folder suffix so case-insensitive filesystems keep `a` and `A` apart. Boxes where two letters touch in the scan are listed under `rejected` so another instance can be used.

Several shifted passes of one card can be given together. `superres.py` registers them with ECC and fuses them onto a 2x grid by iterative back-projection, and the glyphs are then cut from the fused image.

```
.venv/bin/python -m glyphstack.pipeline pass1.tif pass2.tif pass3.tif pass4.tif
```

`pytest` runs a synthetic check that fusing four shifted half-resolution passes lands closer to the original than upscaling one pass.

## Writing a cards file

A cards file is the one place a scan folder records what was scanned: every card, its printed text, and each scan of it with the angle it lies at. `prepare.py`, `session.py`, `bench.py` and `pipeline.py` all read it. This one covers two cards, each scanned upright and turned a quarter, the two scans every card gets. The file names give the angle in degrees.

```json
{
 "cards": [
  {
   "id": "ReefWorm",
   "name": "Reef Worm",
   "text": {"rules": ["<first rules paragraph as printed>", "<second>"], "flavor": ["<flavor text as printed>"]},
   "scans": [
    {"file": "reefworm_000.png", "angle": 0},
    {"file": "reefworm_090.png", "angle": 90}
   ]
  },
  {
   "id": "Altair",
   "name": "Altaïr Ibn-La'Ahad",
   "text": {"rules": ["<rules text as printed>"]},
   "scans": [
    {"file": "altair_000.png", "angle": 0},
    {"file": "altair_090.png", "angle": 90, "notes": "rescanned after the card slipped"}
   ]
  }
 ]
}
```

**Each scan folder has its own cards file, which can have any name.** Each `file` is read relative to the cards file's own folder, so it sits beside its scans in `scans/`, which keeps both out of git. A folder holding two scan sets of the same cards, such as a colour and a grey set, keeps one cards file for each. `prepare.py` writes a copy beside the cropped scans with the paths pointing at them, and that copy is the one to give the session. Calibration scans are not listed, they are passed with `--cal-a` and `--cal-b`.

**`id` groups the scans of one physical card.** The session only checks whether two scans share it. The site shows it, splitting CamelCase into words, so write it in CamelCase without spaces or accents, such as `ReefWorm` or `Altair`. It cannot contain a colon, because instance IDs are built as `card:char:scan:unit`. A card whose printed name has an accent gets an entry in the `display` table in `site/src/lib/format.ts`, the way `Altair` is shown as Altaïr.

**`text` holds the card's printed text, its `rules` and `flavor` each a list of the section's paragraphs.** A card with no flavor text leaves out `flavor`. Mana symbols are written as `{T}` or `{G}`, and the text is written as printed, since the benchmark's truth is fitted to it. Flavor text, reminder text in parentheses and an ability word before a dash are read as italic, though a die roll's result before a dash, such as `1 or 2 —`, is not. Text between underscores takes the other face, so italic text in a rules paragraph is marked as Photo Op's `any _Magic_ card`, and roman text inside flavor as Frilled Cave-Wurm's `quartermaster of the _Windshark_`. A printed underscore is written `\_`, which the JSON file spells `\\_`, as in Steel Overseer's `ADMIN\\_USER\\_GROUP`, and any other backslash is printed as it stands. `name` is the printed name, `set` and `number` identify the printing, and `box` gives the rules box as y0 y1 x0 x1 fractions of the card for a frame that puts it elsewhere. A frame that splits the text gives a list of boxes in reading order, such as a Saga's reminder band, chapter column and flavor box, or a Planeswalker's first line beside its loyalty cost and the abilities below it, and the text flows from each box's last line into the next. A white-bordered card's border cannot be told from the scanner bed, so its box fractions are of the frame inside the border, and `size_mm` gives that frame's width and height in millimetres in place of the card's 63 by 88.

**`angle` is the card's counterclockwise turn in degrees as it appears in the scan image.** The card lies face down on the glass, so a turn that looks counterclockwise from above shows up clockwise in the scan. Write 0 for the scan in the jig's first corner and 90 for the scan a quarter turn on. Negative angles work too, and -90 is the same as 270. The card outline refines the angle only within a quarter turn, so the file has to give the quarter turn. If every turned scan warns that it is turned about twice its notch the other way, the jig turns against this convention, and each angle should be written as 360 minus the angle, such as 270 in place of 90.

**Every card is scanned at 0 and 90 degrees.** The 0 degree scan is the reference that the 90 degree scan is matched to, and the pair gives each printed glyph two views on different pixel grids, which is what the noise figures compare. A card without one is warned about and uses its first scan instead, and a card with a single scan gives each of its glyphs only one view.

## Cropping scans first

`prepare.py` crops every scan in a cards file to its card plus a margin (150 px by default) before a session runs. A flatbed scan of one card is about 90 % empty bed, which compresses well in an 8-bit PNG but not in a 16-bit TIFF, where a whole-bed scan is about 770 MB against about 160 MB cropped. The crop slices whole pixels and keeps the format and bit depth, so the cropped scan holds exactly the card region's pixels, and its offset sits on the 8 px grid the card finder works on, so the session finds the same units in it. Originals are never changed.

```
.venv/bin/python -m glyphstack.prepare scans/cards.json -o scans/cropped
```

The output folder gets the cropped scans under their original names, a copy of the cards file pointing at them, and `prepare.json` with each original's sha256, size and crop box. A rerun skips scans whose original and margin are unchanged. Only PNG and TIFF scans are cropped, because other formats would be re-encoded.

## Running a session

`session.py` runs a whole scanning session. It reads the scans of a cards file, checks the calibration scans, extracts and labels the glyphs of every scan, stacks them and traces the glyphs with enough instances.

`angle` is written as described under Writing a cards file. Quarter turns are undone exactly. Any other turn is measured from the card outline and undone for labelling, while the stacked coverage is read from the native pixels so each view is resampled once. A small error in the angle, such as a card sitting slightly off in one of the jigs in `scanner/`, is taken up by the per-scan fit of the glyphs, and every scan whose outline is turned more than 2 degrees from its `angle` is logged as a warning so a slipped card can be rescanned. Every scan of a card is matched to the card's upright scan from glyph positions, which gives each scan's map to that scan and groups the views of each printed glyph into one instance. Within each card, a printed glyph's views are registered against each other and fused into that instance, and each glyph's instances are then registered and stacked. Cards are extracted and fused in parallel, one process each, and glyphs are registered and stacked in parallel, one process each, with `--workers` to set how many.

```
.venv/bin/python -m glyphstack.session scans/cropped/cards.json -o build/session --font /path/to/font.ttf --cal-a cal_a.png --cal-b cal_b.png
```

Flavor text and reminder text are printed in the italic face. Add `--italic-font /path/to/italic.ttf` to label, stack and trace it in the same run. Each unit is matched against the templates of both faces and keeps the face that fits, and an italic glyph has the key `i:` plus its character, so its stack, trace and exports stay apart from the regular glyph. Italic glyphs are never used to estimate scan geometry. A cache under `<out>/work` built with other faces or another profile is rebuilt. `--italic-font PATH` is the same as `--face-font italic=PATH`.

The font is a measuring stick for labelling and tracing, so keep it out of the repository. On a terminal each stage, card and scan draws a live progress bar, and the console shows only warnings, errors, the run log's path and the final result. Elsewhere progress is logged to stderr with a timestamp every few seconds. Every run also writes a full debug log to `<out>/logs`, set with `--log`, and `-v` shows all of it on the console too. When nothing is reported for `--watchdog` seconds (300 by default, 0 turns it off), the stack of every thread in every process is written to that log, so a stuck run shows where it is waiting without being killed. Extracted scans are cached under `<out>/work` so a rerun resumes, and a scan is extracted again when the faces, the profile or the source of the modules that find and label units change. Each glyph's registered units are written to `<out>/work/regs/<glyph>` in compressed chunks, and registration and stacking read and write them one batch at a time, so memory stays flat however many instances a glyph has. The extracted units of every scan stay in memory for the whole run, about 50 KB per view. `--cal-a` and `--cal-b` are two scans of the printed target in `scanner/calibration_target.pdf`, the second turned 90 degrees clockwise. The calibration gate stops the session when scale is outside 47.32 ± 0.15 px/mm or the x and y scales differ by more than 0.2 %, `--skip-gate` overrides it. Outputs are `stacks/<glyph>/` (`stack.npz` with the mean, spread and leave-one-out IoU, and the kept instances in `inst/` as chunks that `stack.load` reads lazily), `manifest.json` (per instance views, edge level, halo, pair id, card scale, deviation from the stack and the view against view edge distance, and for an accent its anchor on its base letter), `traces.json` and `calibration.json`. A glyph with fewer units than the profile's `min_units` is stacked too when `few_units` is set, and one with fewer instances than `--min-instances` is traced from its stack blended with its measuring stick outline, drawn with its face's print look. The outline counts as the profile's `prior_weight` instances for a stack of none, falling to none at `--min-instances` as 1 − (n / threshold) ** `prior_falloff`, so the more copies a glyph has, the less it leans on the outline. With the defaults of 6 and 3 that's 6 copies' worth at n = 1, 4.2 at 8 and 1.4 at 11. On 44 well sampled glyphs with copies held out, this curve came within 0.03 µm of the best constant weight on average and reaches none at the threshold, so a glyph doesn't jump when it gets its twelfth copy. A glyph with no copies at all is built from its outline alone. A face's print look is the ink spread, edge width and rounding that draw the measuring stick closest to its traced stacks, the median over its glyphs traced alone. Rounding blurs the outline's distance field before the edge is drawn, which rounds corners and thins strokes narrower than the blur. A blended glyph's entry in `traces.json` says so and gives how far the blend moved from its own stack, and the site marks it. Every other character of GF Latin Core is built as the face's measuring stick builds it. A composite such as É is its parts' entries, traced or built, placed at the font's component offsets, a combining accent is its spacing accent centred on the origin with no advance, and any other character is its outline drawn with the face's look. A dotless ȷ the font does not draw is its j less the dot, for tracing as well as building. Each such entry records `built` as `composed`, `derived` or `outline`, with its parts or its look. These entries stay in the session's output. The measuring stick draws no ẞ, so it is the one character of the set left unbuilt. `python -m glyphstack.calibration cal_a.png cal_b.png` runs the gate alone.

## Target profiles

A target profile says what type a run measures and where it is printed. Profiles are TOML files, and `glyphstack/profiles/rules.toml`, the face of rules, flavor and reminder text traced as Nomicon Rules and Nomicon Rules Italic, is the default. Pass `--profile NAME` for another file in that folder or `--profile path/to/profile.toml` for one elsewhere. The session records the profile's path and hash in `manifest.json`, and the export uses the same profile unless told otherwise.

A profile lists the typeface's faces. The first is the primary face: its glyph keys are the bare character, and the print scale and the shared fit come from it. Each other face gives a key prefix, a cache file prefix and a site name suffix, the way the italic face uses `i:`, `it_` and `_it`. Each face also has a `full_name`, the font name it is traced as, which labels it in the scan list spreadsheet. Fonts are never named in a profile. The primary face's font is given with `--font` and each other face's with `--face-font NAME=PATH`.

`[[regions]]` name the parts of the card to read, as fractions of the upright card crop. Components outside every region are not made into units, though their ink still masks neighbouring glyphs, and the scan's paper and ink levels are measured inside the regions. `[paper] mode = "local"` gives each unit the paper level of the ring around its own ink, for type printed on frames whose colour changes across the card, such as a title bar and a type line.

`rules_texture.toml` reads the same face from text boxes printed over a gradient, a tint or faint art. Its `[texture]` section divides the card by its local paper level and blurs away a halftone screen where the screen is strong against the paper to ink range, leaving a plain box sharp, and `glyphstack/texture.py` documents each setting. Its `[templates]` section sets `scale_search = "count"`, which takes the print scale at which the most units are labelled, and `self_rounds = 1`, which labels the units again against templates learned from the card's own kept units. A profile without these settings reads as before.

A glyph on strong screen can also read its coverage through a small network, which labelling and stacking then use, while finding still reads the prepared image. `glyphstack/coverage_net.py` trains it on glyphs from plain background printed onto the bare screen of the cards that have it, by a forward model of black ink overprinted on a halftone screen, and runs it in numpy. Training needs torch, from the `train` extra, and an earlier session of the profile whose units give the glyphs. The weights come from card scans, so keep them outside git, under `build/`:

```
.venv/bin/pip install -e '.[train]'
.venv/bin/python -m glyphstack.coverage_net train build/session scans/cards.json -o build/coverage_net/rules_texture.npz
.venv/bin/python -m glyphstack.session scans/cards.json -o build/session --profile rules_texture --font /path/to/font.ttf --italic-font /path/to/italic.ttf --coverage-model build/coverage_net/rules_texture.npz
```

A unit whose centre has screen contrast past the first value of the profile's texture `coverage_contrast` takes the network's coverage, in full past the second, and a unit on plain background keeps its own. The extraction cache is keyed on the weights, so new weights read every scan again. A card turned by other than quarter turns keeps its native coverage as before.

With `keep_marks` set, accents stay apart from their letters as marks of their own. A unit just over a letter is labelled against the measuring stick's spacing accents, a mark over a capital takes its accent's capital form (`acute.case` beside `acute`), and each mark records its anchor, the offset of its centre and the gap to its base in x-heights. Each labelled i and j is also split into its dotless stem and its dot, kept beside the whole letter, which feed the `dotlessi`, `dotlessj` and `dotaccent` stacks most accented letters are composed from.

The other sections hold the settings tuned to the type: the segmentation threshold and size limits with the rules that join an i dot or the parts of a punctuation mark into one unit, the template character set, em size and print scale range, the matching thresholds, the characters scan geometry is estimated from, the stacking thresholds, the glyph the shared fit starts from with its starting values, and the glyph and card the site's step figure uses. `rules.toml` documents each one.

## Scoring a session against the printed text

`bench.py` scores a session against the text printed on its cards, so a change to the pipeline can be measured on found and labelled glyphs rather than on its own output. It reads each card's printed text and rules box from the session's cards file. A card the scan cut off along an edge is placed from its intact edges and its printed size, but keeping a few millimetres of bed around every card is safer.

```
.venv/bin/python -m glyphstack.bench truth scans/cropped/cards.json --font /path/to/font.ttf --italic-font /path/to/italic.ttf
.venv/bin/python -m glyphstack.bench score build/session scans/cropped/cards.json
```

`truth` places every printed character of each scan by fitting the font to the card's text, line by line and kerned as the font kerns it, and writes `<scan stem>.json` with an overlay `<scan stem>.png` to `truth/` beside the cards file. Text takes its face as the cards file gives it, and the text is turned level first, since a card can sit a degree askew. A line whose glyphs fit with a mean correlation below 0.5 is fitted again over a wider range of baselines with each glyph free to sit up to 0.25 em off it and to take an exponent form, 0.83 of its size and raised 0.17 em, as the minus signs and 90° of Niv-Mizzet's formula are printed. Lines that fit well never take it. On a line that fits well, a glyph correlating below 0.5 is searched up to 0.3 em across and moved where it reaches 0.5 without landing on a neighbour, as the print sets some glyphs further apart than the font spaces them, such as Bloodletter's Æ and its full stop, and a neighbour the moved glyph lands on is searched the same way. The truth comes from card scans, so it stays out of git with them. Check the overlays by eye before trusting a new card's truth.

`truth` fits only the scans whose truth is missing or out of date. Each truth file records a key made from the scan file's size and time, its angle, the card's text, box and size, the fonts and the source of the modules that fit it, and a scan whose key still matches is skipped. `--force` fits every scan again.

A scan that is fitted again reuses what has not changed. The fit is made of steps: the print scale, each line's placement at each scale, the correlation of a placed glyph, and the pass that moves poorly placed glyphs. Each step's result is kept in `cache/<scan stem>.pkl` in the truth folder, keyed by the step's arguments and the source of the code it runs, and a correlation map is only computed for a step that is not there. Correcting one line of a card's text fits that line again and reads the rest back, and a change to the code of one step leaves the others' results in place. The cache holds results for one darkness image and one pair of fonts, so a changed box or rescanned card starts it afresh. It takes under 100 KB a scan, `--cache` puts it elsewhere and `--no-cache` fits every step afresh.

`check` flags scans whose truth looks off: characters left unplaced or cut by the scan's edge, a low glyph correlation over the scan or on one line, or a disagreement in line count or print scale with the card's other scan. A disagreement usually means the cards file's text differs from what is printed, and is corrected there. `truth` takes `--workers` to fit scans in parallel.

`score` extracts each scan again under the session's profile to count the glyphs it found, and reads the session's labels. Characters cut by the scan's edge, or on a line whose glyphs the fit placed with a mean correlation below 0.5, such as a line of symbols the measuring stick draws differently, are left out and counted as ignored. For each scan it reports the glyphs found, merged, fragmented or stray, labelled, right, in the wrong face, wrong and falsely labelled, split by face and by the screen contrast under each glyph. Where a card has glyphs on both plain and strongly screened background it reports their ink weight ratio and the light ring outside the screened glyphs' edges. Each card is then stacked on its own from 8 units a glyph, so cards printed at different sizes do not move each other's stacks, and the edge deviation of its instances is reported by background. The results go to `bench/bench.json` in the session.

## Exporting data for the site

`export_site.py` reads a session folder and writes the JSON the documentation site renders.

```
.venv/bin/python -m glyphstack.export_site build/session --font /path/to/font.ttf -o site/data
```

It writes `glyphs.json`, `glyphs/<name>.json` (trace, mean iso-contours at 0.1 to 0.9, a low resolution spread grid, every instance's 0.5 contour with its card, scans, edge level, halo, card scale, deviation and leave-one-out IoU), `sweep/<name>.json` (edge error and noise against the number of stacked instances, with the modelled floor and a recommended N), `kerning.json` (each adjacent pair's designed gap and observed mean and spread), `kerning_samples.json` (every measured sample of each pair with its scan), `noise.json`, `calibration.json`, `cards.json` and `built.json`, which lists each glyph built from the measuring stick with how it was built and from what, but not its trace, since that would put the font in the site's data. The character sheet marks these glyphs without drawing them. Pass `--italic-font` to export the italic glyphs of a session run with it. Italic names end in `_it` (`a_it`, `A_uc_it`), their rows in `glyphs.json` have `style` set to `italic`, and the italic advance metrics and kerning go to `metrics.italic.json`, `kerning.italic.json` and `kerning_samples.italic.json`. The font is only a measuring stick. Its outline is not exported whole: each glyph gets at most `--cutaways` short clipped windows (default 3, none with 0) around the largest misfit to the stack, which together stay under 20 % of the outline length. `--simplify` sets the path tolerance in font units (default 0.5). To keep files small each glyph exports at most `--max-instances` randomly chosen instances (default 64, seeded, 0 for all) from those that passed the outlier check. The stack and statistics always use every instance, and each glyph file records how many were exported. The sweep runs on at most `--sweep-max` randomly chosen instances (default 512, seeded, 0 for all), well past the point where its curve flattens, and each sweep file records how many it used. Instances are read from the session's chunks as each glyph is exported, so memory follows one glyph's sample rather than the session. The glyph fits come from the session's `traces.json`, and glyphs are exported one per process, `--workers` of them, default one per core. `noise.json` gives the noise between views of the same printed glyph scanned at different angles, with scale and shear tied across a card's scans, as the headline figure. Pass `--compare-session` with a second session run using `--no-tie` to list the figure with each scan fitted on its own beside it.

## Building the fonts

`fontbuild.py` turns a session into one variable font per face, `NomiconRules[BLED].ttf` and `NomiconRules-Italic[BLED].ttf`, with every character of GF Latin Core the session traced or built. It builds them from the session's traces and the synthetic font baseline, and doesn't read the measuring stick.

```
.venv/bin/python -m glyphstack.fontbuild build/session --baseline build/synthetic_baseline/NomiconRulesBleed25-Regular.ttf --italic-baseline build/synthetic_baseline/NomiconRulesBleed25-Italic.ttf -o build/fonts
```

Adding `--site site/public/fonts` also writes the fonts the site is set in.

**The synthetic font baseline is each face drawn at a Bleed of 25, and it stands in for the glyphs that don't have enough scans yet.** `baseline.py` makes it once from a session and the measuring stick, as two static fonts, `NomiconRulesBleed25-Regular.ttf` and `NomiconRulesBleed25-Italic.ttf`. Its glyphs with no stack of their own are the measuring stick's outlines with a quarter of the print look, so it stays under `build/` and is never committed or released.

```
.venv/bin/python -m glyphstack.baseline build/session --font /path/to/font.ttf --italic-font /path/to/italic.ttf -o build/synthetic_baseline
```

**The Bleed axis, `BLED`, sets the strength of the face's print look in percent.** The print look is the ink spread and rounding measured on the face's traced stacks. Against the measuring stick's sharp outline it fills in acute inner corners, by 54 to 60 µm where the corner is under 60 degrees, and rounds off outer ones by 14 to 58 µm, more the sharper the corner, as `corners.json` records. At 100, the default, every glyph is as the session traced or built it. At 25 a quarter of the print look is left, and at 125 it is a quarter stronger than measured.

**A glyph with no stack of its own starts from its baseline outline, which already carries a quarter of the print look.** That outline is the master at 25 as it is. Above 25 the strength t rises in steps of 0.25. Each step draws the rest of the look on the baseline outline, blurring its distance field by √(t² − 0.25²)·r, since Gaussian blurs add in quadrature, and spreading the ink by (t − 0.25)·d. It then moves the last step's points along their normals onto the new edge and fits each segment to the stretch of edge between its ends, which keeps the points in order where a filled corner swallows them. A traced glyph keeps its trace at 100 and moves at the other strengths by the change the look makes to its baseline outline, so it has the same points at every strength. Composites stay composites of their parts at the font's offsets.

**The baseline is made the same way from the measuring stick's sharp outline, with the strength rising from 0.** Each segment beside a corner is first split where the look stops doing more than spreading the ink, so the pieces can round the corner or fill it in. The baseline keeps those segments, and `fontbuild.py` reads each of its quadratic splines back as the one cubic it was made from.

**A few glyphs stop bleeding before 125.** Where the look closes a slit or pinches two strokes together, the outline's points cannot follow, and a step that lands more than 8 µm from the look's edge is not taken. The glyph keeps the last strength it followed and the run log names it. On the 80 scans of `full_v1` that leaves 8 glyphs at 100 and none lower.

On `full_v1`, measured against the look drawn on the measuring stick itself, each glyph built from the baseline lands a median 1.5 µm from the look's edge at 100 in the regular face and 1.2 µm in the italic, against a floor of about 1.9 µm for reading that edge from its grid. That's about 0.15 µm further than building straight from the measuring stick. The largest misses are slivers in acute crotches, the worst being eth at 100, whose crotch is left 86 µm short of filled. A traced glyph at 100 is its trace. The fonts have 4000 units per em, four times the measuring stick's, since one of its units is about 3 µm at the printed size. The fonts kern with the measuring stick's designed kerning, which the baseline carries, for the pairs of characters the fonts draw: 3189 in the regular face and 3821 in the italic. It doesn't change with Bleed, and the spacing observed on the cards isn't in the fonts.

The fonts ship with the site, as described in `site/README.md`. The measuring stick and the synthetic font baseline stay out of git.

## Repository layout

`glyphstack/` is the Python package, named for the process of stacking scanned glyphs rather than for the typeface it measures, and `tests/` holds its tests. `scanner/` holds the printed calibration target and card jigs, with the script that draws the jigs. `site/` is the documentation site, an Astro project whose `data/` folder is the committed output of `export_site`.

Scans, run output and working notes stay out of git. `scans/` holds the raw and cropped scans, which show card artwork. `build/` holds session and export output. `notes/` holds the scan list spreadsheet, the plan of which cards to scan. `.venv/` is the Python environment.
