"""Runs a scanning session end to end: calibration gate, glyph extraction, labelling, stacking and tracing

The scan table is a CSV with the columns file, card and angle, the card's counterclockwise turn in degrees as it
appears in the scan image (0, 15, 30, 45, 90, 105 and so on). The README's Writing a scan table section has an example.
File paths are relative to the CSV. Every scan of a card is matched to the card's upright scan, so each card needs
at least two scans for its glyph views to be grouped
"""
import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import as_completed
from pathlib import Path

from . import calibration, font, fontfit, stack, target, template, units
from .log import Progress, add_args, announce, log, pool, start, timed

CACHE_VERSION = 2


def read_table(path):
    """Reads the scan table into a list of (scan path, card, angle in degrees)"""
    path = Path(path)
    rows = []
    for r in csv.DictReader(path.open()):
        if not r.get("angle", "").strip():
            raise SystemExit(f"give an angle in degrees for {r['file']}")
        angle = float(r["angle"])
        rows.append((path.parent / r["file"], r["card"], angle))
    cards = defaultdict(list)
    for _, card, angle in rows:
        cards[card].append(angle)
    for card, angles in cards.items():
        if len(angles) < 2:
            log.warning("card %s has one scan, its glyphs have one view each", card)
        if not any(a % 360 == 0 for a in angles):
            log.warning("card %s has no upright scan, its first scan is the reference", card)
    return rows


_TEMPLATES = None


def _init_worker(font_paths, profile_path):
    global _TEMPLATES
    target.use(profile_path)
    _TEMPLATES = template.Templates({st: font.load_font(p) for st, p in font_paths.items()})


def extract_card(jobs, scale=None):
    """Extracts and labels the given scans of one card in order, searching the print scale fully only when none is known yet

    jobs is a list of (scan path, card, angle). Returns {scan stem: cached scan dict}
    """
    out = {}
    faces = sorted(_TEMPLATES.fonts)
    prog = Progress(f"card {jobs[0][1]}", len(jobs))
    for path, card, angle in jobs:
        log.info("%s (%s, %g deg)", path.name, card, angle)
        r = units.extract_scan(path, angle)
        kept, scale = template.identify(r["units"], _TEMPLATES, around=scale)
        log.info("kept %d of %d units at print scale %.3f", len(kept), len(r["units"]), scale)
        out[path.stem] = dict(units=kept, scale=scale, card=card, angle=angle, to_up=r["to_up"],
                              paper=r["paper"], ink=r["ink"], faces=faces, profile=target.get().sha256, version=CACHE_VERSION)
        prog.step(f"{path.name}, {len(kept)} units")
    return out


def extract_all(rows, work, font_paths, workers=None):
    """Extracts and labels every scan, one card per process, reusing cached results built with the same faces and profile from the work folder"""
    faces = sorted(font_paths)
    prof = target.get()
    scans, todo = {}, defaultdict(list)
    for path, card, angle in rows:
        cache = work / "units" / f"{path.stem}.pkl"
        if cache.exists():
            cached = pickle.load(cache.open("rb"))
            if cached.get("faces") == faces and cached.get("profile") == prof.sha256 and cached.get("version") == CACHE_VERSION:
                scans[path.stem] = cached
                continue
            log.info("%s cache was built with other faces, another profile or an older pipeline, rebuilding", path.name)
        todo[card].append((path, card, angle))
    log.info("%d scans from cache, %d to extract from %d cards", len(scans), sum(map(len, todo.values())), len(todo))
    (work / "units").mkdir(parents=True, exist_ok=True)
    known = {s["card"]: s["scale"] for s in scans.values()}
    workers = workers or min(os.cpu_count() or 1, max(len(todo), 1))
    prog = Progress("extracting cards", len(todo), every=0)
    with pool(workers, _init_worker, (font_paths, prof.path)) as ex:
        futs = {ex.submit(extract_card, sorted(jobs, key=lambda j: j[2] % 360 != 0), known.get(card)): card for card, jobs in todo.items()}
        for f in as_completed(futs):
            card = futs[f]
            for stem, sc in f.result().items():
                pickle.dump(sc, (work / "units" / f"{stem}.pkl").open("wb"))
                scans[stem] = sc
            prog.step(card)
    return {p.stem: scans[p.stem] for p, _, _ in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="scan table CSV")
    ap.add_argument("-o", "--out", default="build/session", help="output folder")
    target.add_args(ap)
    ap.add_argument("--cal-a", help="calibration scan with the sheet upright")
    ap.add_argument("--cal-b", help="calibration scan with the sheet turned 90 degrees clockwise")
    ap.add_argument("--iterations", type=int, default=2, help="geometry refinement passes")
    ap.add_argument("--no-tie", action="store_true", help="fit each scan's affine on its own instead of tying it across a card's two scans")
    ap.add_argument("--min-instances", type=int, default=12, help="fit and trace glyphs with at least this many instances")
    ap.add_argument("--skip-gate", action="store_true", help="continue even when the calibration gate fails")
    ap.add_argument("--workers", type=int, help="processes for extraction, one card each, default one per core")
    add_args(ap)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start(a, out, "session")
    prof = target.use(a.profile or target.DEFAULT)
    log.info("profile %s (%s)", prof.name, prof.path)

    if a.cal_a and a.cal_b:
        with timed("calibration"):
            cal = calibration.analyse(a.cal_a, a.cal_b)
            (out / "calibration.json").write_text(json.dumps(cal, indent=1))
        s = cal["summary"]
        log.info("scale %.3f px/mm, x/y anisotropy %.3f %%, edge width %.0f um", s["px_per_mm"], 100 * s["scan_anisotropy"], s["edge_w1090_um_median"])
        fails = calibration.gate(cal)
        for f in fails:
            log.error("calibration gate: %s", f)
        if fails and not a.skip_gate:
            sys.exit("calibration gate failed, rescan or pass --skip-gate")
    else:
        log.warning("no calibration scans given, scale is unchecked")

    rows = read_table(a.table)
    font_paths = target.font_paths(a)
    fonts = {st: font.load_font(p) for st, p in font_paths.items()}
    with timed("extraction and labelling"):
        scans = extract_all(rows, out / "work", font_paths, a.workers)
    result = stack.build(scans, out / "work", iterations=a.iterations, tie=not a.no_tie)
    result["profile"] = dict(name=prof.name, path=prof.path, sha256=prof.sha256)
    stack.save(result, out)
    with timed("tracing"):
        shared, traces = fontfit.fit_all(fonts, result["stacks"], a.min_instances)
    (out / "traces.json").write_text(json.dumps(traces, indent=1))
    announce("wrote %s with %d glyph traces", out, len(traces))


if __name__ == "__main__":
    main()
