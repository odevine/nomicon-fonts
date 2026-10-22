"""Exports stacked glyphs, per-instance contours, noise sweeps, kerning and calibration as JSON for the documentation site

The designed font is a measuring stick. Its outline is never written whole: the only designed geometry in
the export is a few short cutaways clipped around the spots where the stack differs from it most
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from . import font as fnt
from . import fontfit, kerning, site_crops, styles, sweep, target
from .contours import UM_PER_PX, contours_at
from .log import Progress, add_args, announce, log, start, timed
from .stack import fname

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SIMPLIFY_FU = 0.5
SPREAD_STEP = 8
CUTAWAY_RADIUS_FU = 70
CUTAWAY_MAX_FRACTION = 0.2
CUTAWAY_SEPARATION_FU = 160
MAX_INSTANCES = 64
SAMPLE_SEED = 2026
# a glyph with fewer instances or cards than this is marked low confidence
CONFIDENT_INSTANCES = 20
CONFIDENT_CARDS = 5
SERIES_N = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96]


def meta_doc(min_instances):
    """The thresholds the site quotes, so its text follows the pipeline's settings"""
    return dict(min_instances=min_instances, confident_instances=CONFIDENT_INSTANCES, confident_cards=CONFIDENT_CARDS)


def path_d(polys, p, tol=SIMPLIFY_FU):
    """SVG path data for polylines given in grid coordinates, as simplified closed paths in font units"""
    out = []
    for c in polys:
        fu = fontfit.to_font_units(c, p).astype(np.float32)
        fu = cv2.approxPolyDP(fu.reshape(-1, 1, 2), tol, True).reshape(-1, 2)
        if len(fu) >= 3:
            out.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in fu) + "Z")
    return "".join(out)


def spread_grid(sd, p):
    """Low resolution map of the per-pixel standard deviation across instances, with its grid to font unit transform"""
    small = cv2.resize(sd, None, fx=1 / SPREAD_STEP, fy=1 / SPREAD_STEP, interpolation=cv2.INTER_AREA)
    s, ar, th = p[:3]
    M = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([s, s * ar])
    Mi = np.linalg.inv(M)
    origin = Mi @ (np.array([SPREAD_STEP / 2 - 0.5] * 2) - p[3:5])
    # cell (column j, row i) sits at origin + j * step * Mi[:, 0] + i * step * Mi[:, 1] in font units
    t = [Mi[0, 0] * SPREAD_STEP, Mi[1, 0] * SPREAD_STEP, Mi[0, 1] * SPREAD_STEP, Mi[1, 1] * SPREAD_STEP, origin[0], origin[1]]
    data = np.clip(small / 0.5 * 255, 0, 255).astype(np.uint8)
    return dict(w=int(data.shape[1]), h=int(data.shape[0]), full_scale=0.5, to_fu=[round(float(v), 4) for v in t], data=data.ravel().tolist())


def cutaways(g, p, mean_contours, count, radius=CUTAWAY_RADIUS_FU):
    """Short clipped pieces of the designed outline around the largest distances to the stack contour

    Each piece is a window of `radius` font units around its centre. The pieces together stay under
    CUTAWAY_MAX_FRACTION of the outline length, so no whole outline or whole contour can be rebuilt
    """
    if count <= 0 or not mean_contours:
        return []
    from scipy.spatial import cKDTree

    tree = cKDTree(np.vstack(mean_contours))
    total = sum(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1).sum() for c in g.contours)
    fu_pts = [np.stack([c[:, 0], -c[:, 1]], 1) for c in g.contours]
    # the designed outline carries the fitted ink spread p[5], so the misfit is the distance to the stack contour minus that
    dev = [np.abs(tree.query(pl)[0] - p[5]) for pl in g.to_img(p)]
    allfu, alldev = np.vstack(fu_pts), np.concatenate(dev)
    chosen = []
    for i in np.argsort(-alldev):
        if all(np.hypot(*(allfu[i] - allfu[j])) > CUTAWAY_SEPARATION_FU for j in chosen):
            chosen.append(int(i))
        if len(chosen) == count:
            break
    out, used = [], 0.0
    while True:
        out, used = [], 0.0
        for i in chosen:
            segs = []
            for c in fu_pts:
                near = np.hypot(*(c - allfu[i]).T) <= radius
                run = []
                for k, flag in enumerate(near):
                    if flag:
                        run.append(c[k])
                    elif run:
                        segs.append(np.array(run))
                        run = []
                if run:
                    segs.append(np.array(run))
            used += sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in segs if len(s) > 1)
            out.append(dict(
                center=[round(float(allfu[i][0]), 1), round(float(allfu[i][1]), 1)], radius=radius, deviation_um=round(float(alldev[i] * UM_PER_PX / 4), 2),
                path="".join("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in s) for s in segs if len(s) > 1),
            ))
        if used <= CUTAWAY_MAX_FRACTION * total or radius < 10:
            return out
        radius = radius * 0.8


def trace_series(stack, rows, fit, g, p, sd=None):
    """The glyph traced from the first N instances of one fixed random order, for N in SERIES_N and the full stack

    Each entry is the same trace the full stack gets, so the page can show how the outline settles as N grows.
    sd is the glyph's placed distance field, shared by every N
    """
    if sd is None:
        sd = fontfit.placed_distance(g, p, stack["mean"].shape)
    inst = stack["inst"].astype(np.float32)
    n_all = len(inst)
    order = np.random.default_rng(SAMPLE_SEED).permutation(n_all)
    out = []
    for k in sorted({k for k in SERIES_N if k < n_all} | {n_all}):
        first = order[:k]
        cards = len({rows[i]["card"] for i in first})
        if k == n_all:
            out.append(dict(n=k, cards=cards, trace=fit["trace"], nodes=fit["nodes"], edge_rms_um=round(fit["rms"], 3)))
            continue
        m = inst[first].mean(0)
        d, nodes, _ = fontfit.trace(m, p)
        c = contours_at(m, 0.5)
        rms = None if not c else round(fontfit.edge_stats(g, p, m, c, sd)[0], 3)
        out.append(dict(n=k, cards=cards, trace=d, nodes=nodes, edge_rms_um=rms))
    return out


def export_glyph(key, stack, rows, fit, g, p, ref, ncut, draws, tol=SIMPLIFY_FU, max_instances=MAX_INSTANCES, sd=None):
    """The JSON document of one glyph, from its glyph key"""
    ch, style = styles.char_of(key), styles.style_of(key)
    mean = stack["mean"]
    inst_rows = [r for r in rows if r["char"] == key and r["kept"]]
    inst = stack["inst"].astype(np.float32)
    assert len(inst_rows) == len(inst), (key, len(inst_rows), len(inst))
    # the stack, sweep and statistics use every instance, only the exported per-instance paths are sampled
    pick = np.arange(len(inst))
    if max_instances and len(inst) > max_instances:
        pick = np.sort(np.random.default_rng(SAMPLE_SEED).choice(len(inst), max_instances, replace=False))
    instances = []
    for k in pick:
        r, a = inst_rows[k], inst[k]
        instances.append(dict(
            id=styles.instance_id(key, k), card=r["card"], scans=r["scans"], pair_id=r["pair_id"], edge_level=[round(v, 3) for v in r["edge_level"]], halo_px=r["halo_px"][0],
            card_scale=round(r["card_scale"], 4), print_scale=r["print_scale"], deviation_um=None if r["deviation_um"] is None else round(r["deviation_um"], 2),
            loo_iou=round(r["loo_iou"], 4), path=path_d(contours_at(a, 0.5), p, tol),
        ))
    mean_c = contours_at(mean, 0.5)
    sw = sweep.sweep_glyph(inst, ref, draws)
    return dict(
        name=styles.glyph_name(key), char=ch, unicode=f"U+{ord(ch):04X}", style=style, adv=fit["adv"], n=fit["n"], cards=stack["cards"], trace=fit["trace"], nodes=fit["nodes"],
        instances_total=len(inst), instances_exported=len(pick), contours=fit["nc"], edge_rms_um=round(fit["rms"], 3), iou=round(fit["iou"], 4), recommended_n=sw["recommended_n"],
        mean={f"{lv:.1f}": path_d(contours_at(mean, lv), p, tol) for lv in LEVELS}, spread=spread_grid(stack["sd"], p), instances=instances,
        cutaways=cutaways(g, p, mean_c, ncut), trace_by_n=trace_series(stack, inst_rows, fit, g, p, sd),
    ), sw


def font_metrics(font):
    """Vertical metrics and the space advance of the measuring stick in font units, numbers only"""
    os2 = font["OS/2"]
    return dict(
        upm=font["head"].unitsPerEm, ascent=font["hhea"].ascent, descent=font["hhea"].descent, line_gap=font["hhea"].lineGap,
        x_height=os2.sxHeight, cap_height=os2.sCapHeight, space_adv=font["hmtx"][font.getBestCmap()[32]][0],
    )


def pair_histogram(man, width=1.0, top=40.0):
    """Counts of the edge RMS between views of each printed glyph scanned at different angles, as a pair would show it, in micrometre bins"""
    d = np.array([r["pair_dev_um"] for r in man["instances"] if r.get("pair_dev_um") is not None])
    counts, edges = np.histogram(d, bins=np.arange(0, top + width, width))
    return dict(bin_um=width, counts=counts.tolist(), n=int(len(d)), over=int((d >= top).sum()), median_um=round(float(np.median(d)), 3), rms_um=round(float(np.sqrt((d**2).mean())), 3))


def noise_doc(man, other=None):
    """Noise between views of the same printed glyph, the headline fit with its histogram and the other fit beside it when given"""
    out = dict(headline=dict(man["noise"], hist=pair_histogram(man)))
    out["headline"]["note"] = "views of the same printed glyph scanned at different angles, with scale and shear tied across the card's scans"
    if other is not None:
        out["alternative"] = dict(other["noise"], hist=pair_histogram(other), note="each scan's affine fitted on its own, which absorbs per scan placement distortion")
    return out


def calibration_doc(c):
    """Scanner calibration for the site: the summary, the ruler fits, every edge width and the tone steps of both scans"""
    widths = [e["w1090_px"] * UM_PER_PX for s in ("a", "b") for e in c[s]["edges"]]
    return dict(summary=c["summary"], rulers={s: c[s]["rulers"] for s in ("a", "b")}, edge_w1090_um=sorted(round(w, 2) for w in widths), tone=c["tone"], tone_percent_black=list(range(0, 101, 10)))


def load_stacks(folder):
    """Reads stacks.npz and manifest.json from a session folder"""
    Z = np.load(folder / "stacks.npz")
    man = json.loads((folder / "manifest.json").read_text())
    stacks = {}
    for ch, m in man["glyphs"].items():
        k = fname(ch)
        stacks[ch] = dict(mean=Z[f"{k}__mean"], sd=Z[f"{k}__sd"], inst=Z[f"{k}__inst"], n=m["n"], cards=m["cards"])
    return stacks, man


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", help="folder written by glyphstack.session")
    target.add_args(ap, "target profile, default the one the session was run with")
    ap.add_argument("-o", "--out", default="site/data")
    ap.add_argument("--min-instances", type=int, default=12)
    ap.add_argument("--cutaways", type=int, default=3, help="clipped designed-outline windows per glyph, 0 for none")
    ap.add_argument("--simplify", type=float, default=SIMPLIFY_FU, help="Douglas-Peucker tolerance in font units for exported paths")
    ap.add_argument("--only", default="", help="export only these characters, for quick checks")
    ap.add_argument("--max-instances", type=int, default=MAX_INSTANCES, help="export at most this many randomly chosen instances per glyph, 0 for all")
    ap.add_argument("--compare-session", help="a second session folder run with --no-tie, its noise is listed beside the headline figure")
    ap.add_argument("--scans", help="folder with the session's scan PNGs, writes the raw crops and step images for the Stack's pixel layers and Scan to Glyph, skipped when not given")
    ap.add_argument("--draws", type=int, default=40, help="random subsets per N in the sweep")
    add_args(ap)
    a = ap.parse_args()
    session, out = Path(a.session), Path(a.out)
    # the run log goes with the session so it never lands in the committed site data
    start(a, session, "export_site")
    (out / "glyphs").mkdir(parents=True, exist_ok=True)
    (out / "sweep").mkdir(exist_ok=True)
    stacks, man = load_stacks(session)
    rec = man.get("profile", {})
    # the session's own profile file when it still exists, else the built-in profile of the same name
    prof = target.use(a.profile or (rec["path"] if rec and Path(rec["path"]).exists() else rec.get("name", target.DEFAULT)))
    if prof.sha256 != man.get("profile", {}).get("sha256", prof.sha256):
        log.warning("profile %s has changed since the session was run", prof.path)
    fonts = {face: fnt.load_font(p) for face, p in target.font_paths(a).items()}
    font = fonts[prof.primary.name]
    missing = [k for k in stacks if styles.style_of(k) not in fonts]
    if missing:
        log.warning("%d glyphs of a face without a font are not exported, pass --face-font for their face", len(missing))
    with timed("fitting the measuring stick"):
        shared, fits = fontfit.fit_all(fonts, stacks, a.min_instances)
    index = []
    wanted = [k for k in fits if not a.only or styles.char_of(k) in a.only]
    prog = Progress("exporting glyphs", len(wanted), every=5)
    for key in wanted:
        fit = fits[key]
        g = fontfit.FontGlyph(fonts[styles.style_of(key)], styles.char_of(key))
        p = np.array(fit["p"])
        sd = fontfit.placed_distance(g, p, stacks[key]["mean"].shape)
        ref = fontfit.reference_contours(g, p, sd.shape, sd)
        doc, sw = export_glyph(key, stacks[key], man["instances"], fit, g, p, ref, a.cutaways, a.draws, a.simplify, a.max_instances, sd)
        name = doc["name"]
        (out / "glyphs" / f"{name}.json").write_text(json.dumps(doc, separators=(",", ":")))
        (out / "sweep" / f"{name}.json").write_text(json.dumps(sw, separators=(",", ":")))
        index.append(dict(name=name, char=doc["char"], unicode=doc["unicode"], style=doc["style"], source="scanned", n=doc["n"], cards=len(doc["cards"]), nodes=doc["nodes"],
                          edge_rms_um=doc["edge_rms_um"], iou=doc["iou"], recommended_n=doc["recommended_n"], low_confidence=doc["n"] < CONFIDENT_INSTANCES or len(doc["cards"]) < CONFIDENT_CARDS))
        prog.step(repr(key))
    (out / "glyphs.json").write_text(json.dumps(index, indent=1))

    # kerning from the cached per-scan units
    scans = {}
    for f in (session / "work" / "units").glob("*.pkl"):
        scans[f.stem] = pickle.load(f.open("rb"))
    levels = {n: s["edge_level"] for n, s in man["scans"].items()}
    rows = kerning.measure_pairs(scans, levels, man["card_S"], font, shared[0], shared[5])
    (out / "kerning.json").write_text(json.dumps(kerning.summarise(rows), separators=(",", ":")))
    for face, f in fonts.items():
        if face == prof.primary.name:
            continue
        fu_px = fontfit.face_transform(shared, font, f)[0]
        face_rows = kerning.measure_pairs(scans, levels, man["card_S"], f, fu_px, shared[5], face)
        (out / f"kerning.{face}.json").write_text(json.dumps(kerning.summarise(face_rows), separators=(",", ":")))
        (out / f"metrics.{face}.json").write_text(json.dumps(font_metrics(f), indent=1))

    other = json.loads((Path(a.compare_session) / "manifest.json").read_text()) if a.compare_session else None
    (out / "noise.json").write_text(json.dumps(noise_doc(man, other), indent=1))

    cal = session / "calibration.json"
    if cal.exists():
        (out / "calibration.json").write_text(json.dumps(calibration_doc(json.loads(cal.read_text())), indent=1))
    cards = {}
    for n, s in man["scans"].items():
        cards.setdefault(s["card"], dict(scans=[], print_scale=s["print_scale"], units=0))["scans"].append(dict(scan=n, angle=s["angle"], edge_level=s["edge_level"], rot_deg=s["rot_deg"]))
    for n, s in scans.items():
        cards[s["card"]]["units"] += len(s["units"])
    (out / "cards.json").write_text(json.dumps(cards, indent=1))
    (out / "metrics.json").write_text(json.dumps(font_metrics(font), indent=1))
    (out / "meta.json").write_text(json.dumps(meta_doc(a.min_instances), indent=1))
    if a.scans:
        site_crops.export(session, a.scans, out, wanted)
    announce("wrote %s: %d glyphs, %d kerning pairs", out, len(index), len({r["pair"] for r in rows}))


if __name__ == "__main__":
    main()
