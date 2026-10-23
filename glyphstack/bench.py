"""Scores a session against the printed text of its cards, with the measures the scan cleaning work is judged on

    python -m glyphstack.bench truth TABLE --font REGULAR.ttf --italic-font ITALIC.ttf
    python -m glyphstack.bench score SESSION TABLE

The table is the session's scan table with a `name` column holding each card's printed name as the scan list's Card
text sheet spells it, and optionally a `box` column of y0 y1 x0 x1 fractions of the upright card crop for a frame
whose rules box sits elsewhere. `truth` places every printed character of each scan with `truth.fit` and writes
`<scan stem>.json` and an overlay `<scan stem>.png` to the truth folder, default `truth/` beside the table. The truth
comes from card scans and the card text, so it stays outside git with them.

`score` reads the session's cached units and, under the session's profile, extracts each scan again to count the
glyphs it found. It reports per card and per face the glyphs found, merged, fragmented or stray, labelled, right, in
the wrong face, wrong and falsely labelled, and splits them by the screen contrast under each glyph. For each card
with glyphs on both plain and strongly screened background it reports their ink weight ratio and the light ring
outside the strongly screened glyphs' edges. Each card is then stacked on its own, so cards printed at different
sizes do not move each other's stacks, and each instance's edge deviation is reported by its background
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import openpyxl
from scipy import ndimage as ndi

from . import font, stack, styles, target, texture, truth, units
from .log import add_args, announce, log, start
from .scan import load_gray

XLSX = Path(__file__).resolve().parents[1] / "notes" / "mtg_rules_scan_list.xlsx"
# a unit's centre may sit this far outside its character's box and still be a find of it
FIND_PX = 6
# glyphs are stacked per card from this many units, fewer than a session's since one card has few of each
CARD_MIN_UNITS = 8
RING_PX = (3, 5)


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


def card_texts(xlsx=XLSX):
    """{printed name: text} from the scan list's Card text sheet"""
    ws = openpyxl.load_workbook(xlsx, read_only=True)["Card text"]
    return {r[1]: r[5] for r in ws.iter_rows(min_row=2, values_only=True) if r[1] and r[5]}


def upright_crop(path, angle):
    """The upright card crop the session cuts from a scan, in whose pixels its units are"""
    crop, _ = units.card_crop(load_gray(path))
    up, _, _ = units.upright(crop, angle)
    return up


def score(all_units, kept, chars):
    """Sorts a scan's units against its characters, returns counts and one row per unit near the text

    A unit whose box holds exactly one character's centre, and whose own centre lies within FIND_PX of that
    character's box, is a find of it. A unit holding two or more centres is merged, one near a box that holds none
    is a fragment and one away from every box is stray. A find is right when its label is the character in its face
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
    out = dict(truth=len(chars), found=len(finds), merged=0, fragments=0, stray=0, labelled=0, right=0, wrong_face=0, wrong=0, false_labels=0)
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


def cmd_truth(a):
    rows = read_table(a.table)
    out = Path(a.out or Path(a.table).parent / "truth")
    out.mkdir(parents=True, exist_ok=True)
    target.use(a.profile or target.DEFAULT)
    fonts = {"regular": font.load_font(a.font), "italic": font.load_font(a.italic_font)}
    texts = card_texts(a.xlsx)
    for r in rows:
        dst = out / f"{r['path'].stem}.json"
        if dst.exists() and not a.force:
            log.info("%s has truth, pass --force to fit it again", r["path"].name)
            continue
        if r["name"] not in texts:
            raise SystemExit(f"{r['name']} is not in the Card text sheet")
        g = upright_crop(r["path"], r["angle"])
        t = truth.fit(g, texts[r["name"]], fonts, r["box"])
        t.update(scan=r["path"].name, card=r["card"], name=r["name"], angle=r["angle"])
        dst.write_text(json.dumps(t, ensure_ascii=False))
        truth.overlay(g, t, out / f"{r['path'].stem}.png")
        announce("%s: %d characters on %d lines at print scale %.3f, text turned %+.2f deg", r["path"].name, len(t["chars"]), t["lines"], t["scale"], t["skew_deg"])


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
        chars = json.loads((tdir / f"{stem}.json").read_text())["chars"]
        cache = pickle.load(open(session / "work" / "units" / f"{stem}.pkl", "rb"))
        caches[stem] = cache
        all_units = units.extract_scan(r["path"], r["angle"])["units"]
        counts, urows = score(all_units, cache["units"], chars)
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
    keys = ["truth", "found", "merged", "fragments", "stray", "labelled", "right", "wrong_face", "wrong", "false_labels"]
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
    t.add_argument("--profile", help="profile whose template scale settings the fit uses, default rules")
    t.add_argument("-o", "--out", help="truth folder, default truth/ beside the table")
    t.add_argument("--force", action="store_true", help="fit scans that already have truth again")
    s = sub.add_parser("score", help="score a session run on the table against its truth")
    s.add_argument("session")
    s.add_argument("table")
    s.add_argument("--truth", help="truth folder, default truth/ beside the table")
    s.add_argument("-o", "--out", help="output folder, default bench/ in the session")
    s.add_argument("--workers", type=int, help="processes for stacking each card")
    for p in (t, s):
        add_args(p)
    a = ap.parse_args()
    start(a, Path(a.out or (Path(a.table).parent / "truth" if a.cmd == "truth" else Path(a.session) / "bench")), f"bench_{a.cmd}")
    (cmd_truth if a.cmd == "truth" else cmd_score)(a)


if __name__ == "__main__":
    main()
