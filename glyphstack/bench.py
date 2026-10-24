"""Scores a session against the printed text of its cards, with the measures the scan cleaning work is judged on

    python -m glyphstack.bench truth TABLE --font REGULAR.ttf --italic-font ITALIC.ttf
    python -m glyphstack.bench check TABLE
    python -m glyphstack.bench score SESSION TABLE

The table is the session's scan table with a `name` column holding each card's printed name as the scan list's Card
text sheet spells it, and optionally a `box` column of y0 y1 x0 x1 fractions of the card for a frame whose rules box
sits elsewhere. A card the scan cut off along an edge is measured from its intact edges. Corrections to the sheet's
text can go in `card_texts.json` beside the table, a map from printed name to text in the sheet's format. `truth` places every printed character of each scan with `truth.fit` and writes
`<scan stem>.json` and an overlay `<scan stem>.png` to the truth folder, default `truth/` beside the table. The truth
comes from card scans and the card text, so it stays outside git with them. `check` flags scans whose truth looks off:
characters left unplaced, a low glyph correlation over the scan or on one line, or a disagreement in line count or
print scale with the other scan of the card, which usually means the card text differs from what is printed.

`score` reads the session's cached units and, under the session's profile, extracts each scan again to count the
glyphs it found. Characters cut by the scan's edge, or on a line the fit placed too poorly to trust, are left out
and counted as ignored. It reports per card and per face the glyphs found, merged, fragmented or stray, labelled,
right, in the wrong face, wrong and falsely labelled, and splits them by the screen contrast under each glyph. For each card
with glyphs on both plain and strongly screened background it reports their ink weight ratio and the light ring
outside the strongly screened glyphs' edges. Each card is then stacked on its own, so cards printed at different
sizes do not move each other's stacks, and each instance's edge deviation is reported by its background
"""
import argparse
import csv
import json
import os
import pickle
from concurrent.futures import as_completed
from pathlib import Path

import cv2
import numpy as np
import openpyxl
from scipy import ndimage as ndi

from . import calibration, font, stack, styles, target, texture, truth, units
from .log import Progress, add_args, announce, log, pool, start
from .scan import load_gray

XLSX = Path(__file__).resolve().parents[1] / "notes" / "mtg_rules_scan_list.xlsx"
# a unit's centre may sit this far outside its character's box and still be a find of it
FIND_PX = 6
# glyphs are stacked per card from this many units, fewer than a session's since one card has few of each
CARD_MIN_UNITS = 8
RING_PX = (3, 5)
# characters on a line whose glyphs correlate less than this on average are too poorly placed to score against, as on
# a line of symbols the measuring stick font draws differently or of superscripts it has no size for
LINE_NCC = 0.5
# a card's width and height, and the margin card_crop leaves around an uncut card
CARD_MM = (63.0, 88.0)
CARD_MARGIN_PX = 20


def read_table(path):
    """The scan table's rows as dicts with the scan path, card, angle, printed name and rules box"""
    path = Path(path)
    rows = []
    for r in csv.DictReader(path.open()):
        if not r.get("name", "").strip():
            raise SystemExit(f"give the printed card name for {r['file']} in a name column")
        box = tuple(float(v) for v in r["box"].split()) if r.get("box", "").strip() else truth.BOX
        rows.append(dict(path=path.parent / r["file"], card=r["card"], angle=float(r["angle"]), name=r["name"].strip(), box=box))
    return rows


def card_texts(xlsx=XLSX, overrides=None):
    """{printed name: text} from the scan list's Card text sheet, with any card in the overrides JSON file replacing it

    The overrides file maps printed names to text in the sheet's format, for corrections the sheet does not have yet
    """
    ws = openpyxl.load_workbook(xlsx, read_only=True)["Card text"]
    texts = {r[1]: r[5] for r in ws.iter_rows(min_row=2, values_only=True) if r[1] and r[5]}
    if overrides and Path(overrides).exists():
        fixes = json.loads(Path(overrides).read_text())
        for name in fixes:
            log.info("text of %s from %s", name, overrides)
        texts.update(fixes)
    return texts


def overrides_path(a):
    return Path(a.texts) if a.texts else Path(a.table).parent / "card_texts.json"


def upright_crop(path, angle):
    """The upright card crop the session cuts from a scan, in whose pixels its units are"""
    return upright_frame(path, angle)[0]


def upright_frame(path, angle):
    """The upright card crop and the card's full frame in its pixels, rebuilt from the intact sides of a card the scan cut off

    A side of the card that runs into the edge of the scan is cut. The frame keeps the opposite, intact side and puts
    the cut one a card's length away, from CARD_MM at the calibrated scale. Turns other than quarter turns are taken as
    uncut, since their crop is cut again after turning
    """
    g = load_gray(path)
    crop, (oy, ox) = units.card_crop(g)
    up, _, _ = units.upright(crop, angle)
    H, W = up.shape
    frame = [CARD_MARGIN_PX, H - CARD_MARGIN_PX, CARD_MARGIN_PX, W - CARD_MARGIN_PX]
    if angle % 90:
        return up, tuple(frame)
    # cut sides of the crop in the scan as top, right, bottom, left, turned with the crop, which rot90 turns counterclockwise
    cut = [oy == 0, ox + crop.shape[1] >= g.shape[1], oy + crop.shape[0] >= g.shape[0], ox == 0]
    for _ in range((4 - int(round(angle / 90)) % 4) % 4):
        cut = cut[1:] + cut[:1]
    top, right, bottom, left = cut
    h, w = CARD_MM[1] * calibration.NOMINAL_PX_PER_MM, CARD_MM[0] * calibration.NOMINAL_PX_PER_MM
    if top and not bottom:
        frame[0] = frame[1] - h
    elif bottom and not top:
        frame[1] = frame[0] + h
    if left and not right:
        frame[2] = frame[3] - w
    elif right and not left:
        frame[3] = frame[2] + w
    return up, tuple(int(round(v)) for v in frame)


def score(all_units, kept, chars, ignore=()):
    """Sorts a scan's units against its characters, returns counts and one row per unit near the text

    A unit whose box holds exactly one character's centre, and whose own centre lies within FIND_PX of that
    character's box, is a find of it. A unit holding two or more centres is merged, one near a box that holds none
    is a fragment and one away from every box is stray. A find is right when its label is the character in its face.
    Units centred on a character in `ignore`, one cut by the scan's edge or too poorly placed to trust, are not counted
    """
    tc = np.array([((c["y0"] + c["y1"]) / 2, (c["x0"] + c["x1"]) / 2) for c in chars])
    ty0, ty1 = min(c["y0"] for c in chars) - 20, max(c["y1"] for c in chars) + 20
    tx0, tx1 = min(c["x0"] for c in chars) - 20, max(c["x1"] for c in chars) + 20
    labels = {tuple(u["bbox"]): u["ch"] for u in kept}
    found, rows = {}, []
    for u in all_units:
        y0, y1, x0, x1 = u["bbox"]
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        if not (ty0 <= cy <= ty1 and tx0 <= cx <= tx1):
            continue
        if any(c["y0"] - FIND_PX <= cy <= c["y1"] + FIND_PX and c["x0"] - FIND_PX <= cx <= c["x1"] + FIND_PX for c in ignore):
            continue
        inside = np.nonzero((tc[:, 0] >= y0) & (tc[:, 0] < y1) & (tc[:, 1] >= x0) & (tc[:, 1] < x1))[0]
        r = dict(bbox=tuple(int(v) for v in u["bbox"]), label=labels.get(tuple(u["bbox"])))
        near = [i for i, c in enumerate(chars) if c["y0"] - FIND_PX <= cy <= c["y1"] + FIND_PX and c["x0"] - FIND_PX <= cx <= c["x1"] + FIND_PX]
        if len(inside) >= 2:
            r["kind"] = "merged"
        elif len(inside) == 1 and int(inside[0]) in near:
            i = int(inside[0])
            r["kind"], r["char"] = "found", i
            area = (y1 - y0) * (x1 - x0)
            if i not in found or area > found[i][0]:
                found[i] = (area, r)
        else:
            r["kind"] = "fragment" if near else "stray"
        rows.append(r)
    finds = {i: r for i, (_, r) in found.items()}
    for r in rows:
        if r["kind"] == "found" and finds.get(r["char"]) is not r:
            r["kind"] = "fragment"
    out = dict(truth=len(chars), ignored=len(ignore), found=len(finds), merged=0, fragments=0, stray=0, labelled=0, right=0, wrong_face=0, wrong=0, false_labels=0)
    for r in rows:
        if r["kind"] in ("merged", "stray"):
            out[r["kind"]] += 1
        elif r["kind"] == "fragment":
            out["fragments"] += 1
        if r["label"] is None:
            continue
        if r["kind"] != "found":
            out["false_labels"] += 1
            continue
        c = chars[r["char"]]
        out["labelled"] += 1
        key = styles.key(c["face"], c["ch"])
        if r["label"] == key:
            out["right"] += 1
        elif styles.char_of(r["label"]) == c["ch"]:
            out["wrong_face"] += 1
        else:
            out["wrong"] += 1
    return out, rows


def background(chars, g):
    """Each character's background as plain or strong from the screen contrast at its centre, or None between them"""
    lo, hi = texture.DEFAULTS["descreen_contrast"]
    rc = texture.screen_contrast(g, dict(texture.DEFAULTS))
    out = []
    for c in chars:
        v = float(rc[(c["y0"] + c["y1"]) // 2, (c["x0"] + c["x1"]) // 2])
        out.append("plain" if v < lo else "strong" if v >= hi else None)
    return out


def edge_profile(g, unit, pad=12):
    """Grey against signed distance from a glyph's own edge, as a share of the way from its ink to the texture around it"""
    y0, y1, x0, x1 = unit["box"]
    Y0, X0 = max(y0 - pad, 0), max(x0 - pad, 0)
    win = g[Y0 : y1 + pad, X0 : x1 + pad].astype(np.float32)
    hint = np.zeros(win.shape, bool)
    hint[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0] = unit["mask"]
    ring = ndi.binary_dilation(hint, iterations=12) & ~ndi.binary_dilation(hint, iterations=6)
    core = ndi.binary_erosion(hint, iterations=2)
    if ring.sum() < 30 or core.sum() < 10:
        return None
    tex, ink = np.median(win[ring]), np.median(win[core])
    mask = ndi.binary_fill_holes(cv2.GaussianBlur(win, (0, 0), 2.0) < (tex + ink) / 2) & ndi.binary_dilation(hint, iterations=4)
    d = np.round(ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask))
    near = ndi.binary_dilation(hint, iterations=12)
    n = (win - ink) / max(tex - ink, 1)
    return {int(b): float(n[near & (d == b)].mean()) for b in range(-6, 11) if (near & (d == b)).sum() >= 5}


def ring_height(profiles):
    """Mean grey 3 to 5 px outside the edge as a share of the texture level, 1 when there is no ring"""
    v = [p[b] for p in profiles if p for b in range(RING_PX[0], RING_PX[1] + 1) if b in p]
    return round(float(np.mean(v)), 3) if v else None


def weight_ratio(rights):
    """Median over characters of the strong background's mean ink area over the plain background's"""
    by = {}
    for key, bg, area in rights:
        if bg:
            by.setdefault(key, {}).setdefault(bg, []).append(area)
    ratios = [np.mean(v["strong"]) / np.mean(v["plain"]) for v in by.values() if "strong" in v and "plain" in v]
    return dict(chars=len(ratios), ratio=round(float(np.median(ratios)), 3) if ratios else None)


_FONTS = None


def _init_truth(profile, font_paths):
    global _FONTS
    target.use(profile)
    _FONTS = {face: font.load_font(p) for face, p in font_paths.items()}


def mark(t):
    """Marks each character of a truth that cannot be scored against: cut by the scan's edge, or on a line placed too poorly

    A character is `cut` when its box runs past the upright crop, and `uncertain` when its line's glyphs average less
    correlation than LINE_NCC. Both are left out of scoring. Returns the truth
    """
    # truth fitted before the crop's shape was recorded cannot tell a cut character
    H, W = t.get("shape") or (np.inf, np.inf)
    means = {}
    for c in t["chars"]:
        means.setdefault(c["line"], []).append(c["ncc"])
    for c in t["chars"]:
        c["cut"] = c["x0"] < 0 or c["y0"] < 0 or c["x1"] > W or c["y1"] > H
        c["uncertain"] = bool(np.mean(means[c["line"]]) < LINE_NCC)
    return t


def load_truth(path):
    return mark(json.loads(Path(path).read_text()))


def truth_scan(r, text, out):
    """Fits one scan's truth and writes its file and overlay, returns a one line summary"""
    g, frame = upright_frame(r["path"], r["angle"])
    t = truth.fit(g, text, _FONTS, r["box"], frame=frame)
    t.update(scan=r["path"].name, card=r["card"], name=r["name"], angle=r["angle"], shape=list(g.shape))
    mark(t)
    (out / f"{r['path'].stem}.json").write_text(json.dumps(t, ensure_ascii=False))
    truth.overlay(g, t, out / f"{r['path'].stem}.png")
    return f"{r['path'].name}: {len(t['chars'])} characters on {t['lines']} lines at print scale {t['scale']:.3f}, text turned {t['skew_deg']:+.2f} deg"


def cmd_truth(a):
    rows = read_table(a.table)
    out = Path(a.out or Path(a.table).parent / "truth")
    out.mkdir(parents=True, exist_ok=True)
    prof = target.use(a.profile or target.DEFAULT)
    texts = card_texts(a.xlsx, overrides_path(a))
    todo = []
    for r in rows:
        if (out / f"{r['path'].stem}.json").exists() and not a.force:
            log.info("%s has truth, pass --force to fit it again", r["path"].name)
            continue
        if r["name"] not in texts:
            raise SystemExit(f"{r['name']} is not in the Card text sheet")
        todo.append(r)
    paths = {"regular": a.font, "italic": a.italic_font}
    prog = Progress("fitting truth", len(todo))
    with pool(a.workers or os.cpu_count() or 1, _init_truth, (prof.path, paths)) as ex:
        futs = {ex.submit(truth_scan, r, texts[r["name"]], out): r for r in todo}
        for f in as_completed(futs):
            try:
                announce("%s", f.result())
            except Exception as e:
                log.error("%s: %s", futs[f]["path"].name, e)
                # an earlier fit of the scan would otherwise be read as its truth
                for ext in (".json", ".png"):
                    (out / f"{futs[f]['path'].stem}{ext}").unlink(missing_ok=True)
            prog.step(futs[f]["path"].name)


def expected_chars(text):
    """The characters of a card's text the truth places, every one but spaces and mana symbols"""
    return [(ch, face) for par in truth.paragraphs(text) for ch, face in par if ch != " " and face != "symbol"]


def check(truths, expected):
    """Flags for each scan whose truth looks off against its own card's other scan or against every scan

    truths is {scan stem: truth dict}, expected {scan stem: expected character count}. A scan is flagged when
    characters went unplaced, when its median correlation or one of its lines' mean correlation sits low against all
    scans, or when it disagrees with the other scan of its card on line count or print scale
    """
    med = {s: float(np.median([c["ncc"] for c in t["chars"]])) for s, t in truths.items()}
    line = {}
    for s, t in truths.items():
        by = {}
        for c in t["chars"]:
            by.setdefault(c["line"], []).append(c["ncc"])
        line[s] = {k: float(np.mean(v)) for k, v in by.items()}
    m = np.median(list(med.values()))
    mad = np.median([abs(v - m) for v in med.values()]) or 0.01
    flags = {s: [] for s in truths}
    for s, t in truths.items():
        if len(t["chars"]) != expected[s]:
            flags[s].append(f"{len(t['chars'])} of {expected[s]} characters placed")
        cut = sum(c.get("cut", False) for c in t["chars"])
        if cut:
            flags[s].append(f"{cut} characters cut by the scan's edge")
        unsure = sum(c.get("uncertain", False) for c in t["chars"])
        if unsure:
            flags[s].append(f"{unsure} characters left out of scoring")
        if med[s] < m - 3 * mad:
            flags[s].append(f"median correlation {med[s]:.2f} against {m:.2f} over all scans")
        low = {k: v for k, v in line[s].items() if v < 0.5}
        if low:
            flags[s].append("low lines " + ", ".join(f"{k + 1} at {v:.2f}" for k, v in sorted(low.items())))
    by_card = {}
    for s, t in truths.items():
        by_card.setdefault(t["card"], []).append(s)
    for card, stems in by_card.items():
        for a_, b_ in zip(stems, stems[1:]):
            ta, tb = truths[a_], truths[b_]
            if ta["lines"] != tb["lines"]:
                for x in (a_, b_):
                    flags[x].append(f"{ta['lines']} lines in {a_} against {tb['lines']} in {b_}")
            if abs(ta["scale"] - tb["scale"]) > 0.011:
                for x in (a_, b_):
                    flags[x].append(f"print scale {ta['scale']:.3f} in {a_} against {tb['scale']:.3f} in {b_}")
    return dict(median=m, scans={s: dict(card=truths[s]["card"], median_ncc=round(med[s], 3), lines=truths[s]["lines"], scale=truths[s]["scale"],
                                         chars=len(truths[s]["chars"]), flags=flags[s]) for s in truths})


def cmd_check(a):
    rows = read_table(a.table)
    tdir = Path(a.truth or Path(a.table).parent / "truth")
    texts = card_texts(a.xlsx, overrides_path(a))
    truths, expected = {}, {}
    for r in rows:
        f = tdir / f"{r['path'].stem}.json"
        if not f.exists():
            log.warning("%s has no truth", r["path"].name)
            continue
        truths[r["path"].stem] = load_truth(f)
        expected[r["path"].stem] = len(expected_chars(texts[r["name"]]))
    res = check(truths, expected)
    (tdir / "check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    w = max(len(s) for s in res["scans"]) + 2
    print(f"{'scan':<{w}}{'card':<24}{'chars':>6}{'lines':>6}{'scale':>7}{'ncc':>6}  flags")
    for s, v in res["scans"].items():
        print(f"{s:<{w}}{v['card']:<24}{v['chars']:>6}{v['lines']:>6}{v['scale']:>7.3f}{v['median_ncc']:>6.2f}  {'; '.join(v['flags'])}")
    announce("%d of %d scans flagged, wrote %s", sum(bool(v["flags"]) for v in res["scans"].values()), len(res["scans"]), tdir / "check.json")


def cmd_score(a):
    session = Path(a.session)
    prof = json.loads((session / "manifest.json").read_text())["profile"]
    target.use(prof["path"])
    rows = read_table(a.table)
    tdir = Path(a.truth or Path(a.table).parent / "truth")
    res = dict(session=str(session), profile=prof["name"], scans={}, cards={})
    caches, tags = {}, {}
    for r in rows:
        stem = r["path"].stem
        allchars = load_truth(tdir / f"{stem}.json")["chars"]
        chars = [c for c in allchars if not (c["cut"] or c["uncertain"])]
        ignore = [c for c in allchars if c["cut"] or c["uncertain"]]
        cache = pickle.load(open(session / "work" / "units" / f"{stem}.pkl", "rb"))
        caches[stem] = cache
        all_units = units.extract_scan(r["path"], r["angle"])["units"]
        counts, urows = score(all_units, cache["units"], chars, ignore)
        g = upright_crop(r["path"], r["angle"])
        bg = background(chars, g)
        by = {tuple(int(v) for v in u["bbox"]): (k, u) for k, u in enumerate(cache["units"])}
        split, rights, profiles = {}, [], {"plain": [], "strong": []}
        for u in urows:
            if u["kind"] != "found":
                continue
            c = chars[u["char"]]
            b = bg[u["char"]]
            key = styles.key(c["face"], c["ch"])
            cell = split.setdefault(f"{c['face']}/{b or 'between'}", dict(found=0, right=0))
            cell["found"] += 1
            if u["label"] == key:
                cell["right"] += 1
                k, unit = by[u["bbox"]]
                tags[(stem, k)] = b
                rights.append((key, b, float(unit["cov"].sum())))
                if b:
                    profiles[b].append(edge_profile(g, unit))
        res["scans"][stem] = dict(card=r["card"], **counts, by_face_and_background=dict(sorted(split.items())), scale=cache["scale"],
                                  weight_strong_over_plain=weight_ratio(rights),
                                  ring={b: ring_height(p) for b, p in profiles.items()})
    for card in sorted({r["card"] for r in rows}):
        stems = [r["path"].stem for r in rows if r["card"] == card]
        sc = {s: caches[s] for s in stems}
        if max(np.unique([u["ch"] for s in sc.values() for u in s["units"]], return_counts=True)[1], default=0) < CARD_MIN_UNITS:
            res["cards"][card] = dict(stacked=0)
            continue
        work = Path(a.out or session / "bench") / "stacks" / card
        built = stack.build(sc, work, stacks_dir=work / "stacks", workers=a.workers, min_units=CARD_MIN_UNITS)
        dev = {}
        for inst in built["instances"]:
            if not inst["kept"] or inst["deviation_um"] is None:
                continue
            b = tags.get((inst["scans"][0], inst["units"][0]), "unscored")
            dev.setdefault(b or "between", []).append(inst["deviation_um"])
        res["cards"][card] = dict(stacked=len(built["stacks"]), deviation_um={b: dict(n=len(v), median=round(float(np.median(v)), 2)) for b, v in sorted(dev.items())})
    out = Path(a.out or session / "bench")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, indent=1))
    report(res)
    announce("wrote %s", out / "bench.json")


def report(res):
    keys = ["truth", "ignored", "found", "merged", "fragments", "stray", "labelled", "right", "wrong_face", "wrong", "false_labels"]
    w = max(len(s) for s in res["scans"]) + 2
    print(f"profile {res['profile']}")
    print(f"{'scan':<{w}}" + "".join(f"{k[:10]:>11}" for k in keys) + f"{'scale':>8}")
    for s, v in res["scans"].items():
        print(f"{s:<{w}}" + "".join(f"{v[k]:>11}" for k in keys) + f"{v['scale']:>8.3f}")
    print(f"{'total':<{w}}" + "".join(f"{sum(v[k] for v in res['scans'].values()):>11}" for k in keys))
    print("\nright of found, by face and background")
    for s, v in res["scans"].items():
        cells = "  ".join(f"{k} {c['right']}/{c['found']}" for k, c in v["by_face_and_background"].items())
        print(f"  {s:<{w}}{cells}")
        wr, ring = v["weight_strong_over_plain"], v["ring"]
        if wr["ratio"] is not None or ring.get("strong") is not None:
            print(f"  {'':<{w}}strong over plain weight {wr['ratio']} over {wr['chars']} characters, ring plain {ring.get('plain')} strong {ring.get('strong')}")
    print("\nedge deviation from each card's own stacks, median um")
    for c, v in res["cards"].items():
        cells = "  ".join(f"{b} {d['median']} ({d['n']})" for b, d in v.get("deviation_um", {}).items())
        print(f"  {c:<24}{v['stacked']:>3} glyphs  {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truth", help="fit every printed character of each scan from its card's text")
    t.add_argument("table")
    t.add_argument("--font", required=True, help="the regular face's TrueType font, a measuring stick never copied anywhere")
    t.add_argument("--italic-font", required=True, help="the italic face's TrueType font")
    t.add_argument("--xlsx", default=XLSX, help="the scan list spreadsheet with the Card text sheet")
    t.add_argument("--texts", help="JSON of corrected card texts by printed name, default card_texts.json beside the table")
    t.add_argument("--profile", help="profile whose template scale settings the fit uses, default rules")
    t.add_argument("-o", "--out", help="truth folder, default truth/ beside the table")
    t.add_argument("--force", action="store_true", help="fit scans that already have truth again")
    t.add_argument("--workers", type=int, help="processes fitting scans, one scan each, default one per core")
    c = sub.add_parser("check", help="flag scans whose truth looks off against their card's other scan or against every scan")
    c.add_argument("table")
    c.add_argument("--truth", help="truth folder, default truth/ beside the table")
    c.add_argument("--xlsx", default=XLSX, help="the scan list spreadsheet with the Card text sheet")
    c.add_argument("--texts", help="JSON of corrected card texts by printed name, default card_texts.json beside the table")
    s = sub.add_parser("score", help="score a session run on the table against its truth")
    s.add_argument("session")
    s.add_argument("table")
    s.add_argument("--truth", help="truth folder, default truth/ beside the table")
    s.add_argument("-o", "--out", help="output folder, default bench/ in the session")
    s.add_argument("--workers", type=int, help="processes for stacking each card")
    for p in (t, c, s):
        add_args(p)
    a = ap.parse_args()
    folder = Path(a.session) / "bench" if a.cmd == "score" else Path(a.table).parent / "truth"
    start(a, Path(getattr(a, "out", None) or getattr(a, "truth", None) or folder), f"bench_{a.cmd}")
    dict(truth=cmd_truth, check=cmd_check, score=cmd_score)[a.cmd](a)


if __name__ == "__main__":
    main()
