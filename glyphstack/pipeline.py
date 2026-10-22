"""Scans to glyph crops, vector outlines and a contact sheet"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import target
from .align import align, card_targets, card_text
from .scan import load_scan, rules_box_density
from .styles import glyph_folder
from .superres import fuse

def touches_edge(stat, W, H):
    """True for components cut by the crop border, such as the text box frame"""
    x, y, w, h = stat[:4]
    return x <= 0 or y <= 0 or x + w >= W or y + h >= H


def segment(bw, min_area=100):
    """Returns glyph boxes from connected components, merging dots and accents into the parent"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    H, W = bw.shape
    comps = [
        tuple(stats[i][:4]) + (i,)
        for i in range(1, n)
        if stats[i][4] >= min_area and not touches_edge(stats[i], W, H)
    ]
    comps.sort(key=lambda c: c[0])
    bands = line_bands(bw)
    merged = []
    for x, y, w, h, i in comps:
        for m in merged:
            mx, my, mw, mh, ids = m
            overlap = min(x + w, mx + mw) - max(x, mx)
            vgap = max(y, my) - min(y + h, my + mh)
            tall = max(h, mh)
            small = min(h, mh) < 0.45 * tall
            same_line = band_of(bands, y + h / 2) == band_of(bands, my + mh / 2)
            if same_line and overlap > 0.5 * min(w, mw) and small and vgap < 0.45 * tall:
                nx, ny = min(x, mx), min(y, my)
                m[:] = [nx, ny, max(x + w, mx + mw) - nx, max(y + h, my + mh) - ny, ids + [i]]
                break
        else:
            merged.append([x, y, w, h, [i]])
    return merged, labels


def line_bands(bw):
    """Returns (start, end) row ranges of text lines from gaps in the row ink profile"""
    ink = (bw > 0).sum(axis=1) > 0
    bands, start = [], None
    for y, on in enumerate(ink):
        if on and start is None:
            start = y
        elif not on and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(ink)))
    return bands


def band_of(bands, cy):
    return next((i for i, (y0, y1) in enumerate(bands) if y0 <= cy < y1), -1)


def group_lines(boxes, bw):
    """Groups boxes into text lines using gaps in the row ink profile, returns lists in reading order"""
    bands = line_bands(bw)
    lines = [[] for _ in bands]
    for b in boxes:
        i = band_of(bands, b[1] + b[3] / 2)
        if i >= 0:
            lines[i].append(b)
    return [sorted(l, key=lambda b: b[0]) for l in lines if l]


def trace(mask, eps=0.6):
    """Traces a binary glyph mask to an SVG path string in y-down pixel units, holes included"""
    cs, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    parts = []
    for c in cs:
        if len(c) < 8:
            continue
        pts = cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(float)
        parts.append(_smooth_path(pts))
    return " ".join(parts)


def _smooth_path(pts):
    """Closed Catmull-Rom spline through points, emitted as cubic beziers"""
    n = len(pts)
    d = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6
        c2 = p2 - (p3 - p1) / 6
        d.append(f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}")
    return " ".join(d) + "Z"


def contact_sheet(crops, path, cell=140, labels=None, metrics=None, xh_cell=44):
    """Grid of glyph masks. With metrics (baseline row in the crop, x-height) every glyph is scaled by the
    same factor and sits on a shared baseline, as it would in text. Without, each crop fills its cell"""
    cols = 16
    rows = (len(crops) + cols - 1) // cols
    base_y = int(cell * 0.7)
    sheet = np.full((rows * cell, cols * cell), 255, np.uint8)
    for k, c in enumerate(crops):
        y, x = divmod(k, cols)
        if metrics:
            base, xh = metrics[k]
            s = xh_cell / xh
            r = cv2.resize(255 - c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            top, left = base_y - int(round(base * s)), (cell - r.shape[1]) // 2
        else:
            s = (cell - 16) / max(c.shape)
            r = cv2.resize(255 - c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            top, left = 8, 8
        r0, c0 = max(top, 0), max(left, 0)
        r1, c1 = min(top + r.shape[0], cell), min(left + r.shape[1], cell)
        sheet[y * cell + r0 : y * cell + r1, x * cell + c0 : x * cell + c1] = r[r0 - top : r1 - top, c0 - left : c1 - left]
        if metrics:
            cv2.line(sheet, (x * cell, y * cell + base_y), ((x + 1) * cell - 1, y * cell + base_y), 225, 1)
        if labels and labels[k]:
            cv2.putText(sheet, labels[k], (x * cell + 4, y * cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 90, 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (x * cell, y * cell), ((x + 1) * cell - 1, (y + 1) * cell - 1), 200, 1)
    cv2.imwrite(str(path), sheet)


def valley_cut(mask, frac):
    """Column index of the thinnest ink within 15% of the expected split position of a touching pair"""
    cols = (mask > 0).sum(axis=0)
    w = len(cols)
    lo, hi = max(int((frac - 0.15) * w), 1), min(int((frac + 0.15) * w), w - 1)
    return lo + int(np.argmin(cols[lo:hi]))


def units_from_alignment(glyphs, labels):
    
    """Turns aligned boxes into glyph units, joining multi-part marks and setting aside touching pairs"""
    units, rejected, i = [], [], 0
    while i < len(glyphs):
        g = glyphs[i]
        if g["align"] == "split" and i + 1 < len(glyphs) and glyphs[i + 1]["align"] == "split":
            parts = [g, glyphs[i + 1]]
            i += 2
        else:
            parts = [g]
            i += 1
        if g["align"] == "merged" and len(g["char"]) == 2:
            box = np.isin(labels[g["y"] : g["y"] + g["h"], g["x"] : g["x"] + g["w"]], g["ids"]).astype(np.uint8)
            cut = g["x"] + valley_cut(box, g["split_at"])
            for ch, cols in zip(g["char"], [(g["x"], cut), (cut, g["x"] + g["w"])]):
                units.append({"char": ch, "parts": [g], "cols": cols})
        elif g["char"] is None:
            rejected.append({"id": g["id"], "chars": None, "flags": ["unmatched"]})
        else:
            units.append({"char": g["char"], "parts": parts})
    return units, rejected


def write_unit(out, style, card_id, n, unit, labels, density, margin_frac=0.25):
    """Writes density crop, mask and preview for one unit and returns its manifest entry"""
    parts = unit["parts"]
    ids = [i for p in parts for i in p["ids"]]
    x0, y0 = min(p["x"] for p in parts), min(p["y"] for p in parts)
    x1, y1 = max(p["x"] + p["w"] for p in parts), max(p["y"] + p["h"] for p in parts)
    mg = int(margin_frac * parts[0]["xh"])
    X0, Y0 = max(x0 - mg, 0), max(y0 - mg, 0)
    X1, Y1 = min(x1 + mg, density.shape[1]), min(y1 + mg, density.shape[0])
    mask = np.isin(labels[Y0:Y1, X0:X1], ids).astype(np.uint8) * 255
    base = parts[0]["y"] + parts[0]["h"] - parts[0]["bottom"] * parts[0]["xh"]
    if "cols" in unit:
        keep = np.zeros(mask.shape[1], bool)
        keep[max(unit["cols"][0] - X0, 0) : max(unit["cols"][1] - X0, 0)] = True
        mask[:, ~keep] = 0
        xs = np.nonzero(mask.any(axis=0))[0]
        lo, hi = max(xs.min() - mg, 0), min(xs.max() + mg + 1, mask.shape[1])
        mask, X0, X1 = mask[:, lo:hi], X0 + lo, X0 + hi
    crop = (density[Y0:Y1, X0:X1] * 65535).astype(np.uint16)
    folder = out / style / glyph_folder(unit["char"])
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{card_id}_i{n:02d}"
    cv2.imwrite(str(folder / f"{stem}.density.tif"), crop)
    cv2.imwrite(str(folder / f"{stem}.mask.png"), mask)
    h, w = mask.shape
    (folder / f"{stem}.preview.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path fill-rule="evenodd" d="{trace(mask)}"/></svg>'
    )
    return mask, {
        "glyph": glyph_folder(unit["char"]), "unicode": f"U+{ord(unit['char']):04X}", "style": style,
        "card": card_id, "file": str((folder / stem).relative_to(out)),
        "bbox": [int(X0), int(Y0), int(X1 - X0), int(Y1 - Y0)],
        "x_height_px": round(float(parts[0]["xh"]), 1), "baseline_row": round(float(base - Y0), 1), "threshold": 0.5,
        "align_cost": round(max(p["cost"] or 0 for p in parts), 2),
        "flags": ["joined parts"] if len(parts) > 1 else ["cut at ink valley"] if "cols" in unit else [],
        "review": None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scans", nargs="+", help="one scan, or several shifted passes of the same card to fuse")
    ap.add_argument("-o", "--out", default="build")
    ap.add_argument("--box", help="rules box as x,y,w,h in scan pixels, default auto-detected")
    ap.add_argument("--card-name", help="card name as on the Card text tab, enables alignment and glyph naming")
    ap.add_argument("--card-id", help="id used in file names, default is the scan file stem")
    ap.add_argument("--style", default="regular", choices=["regular", "italic"])
    ap.add_argument("--xlsx", default="notes/mtg_rules_scan_list.xlsx")
    a = ap.parse_args()
    card_id = a.card_id or Path(a.scans[0]).stem
    out = Path(a.out) / card_id
    out.mkdir(parents=True, exist_ok=True)

    box = tuple(map(int, a.box.split(","))) if a.box else None
    stages = [rules_box_density(load_scan(path), box) for path in a.scans]
    density, info = stages[0]
    if len(stages) > 1:
        density = np.clip(fuse([d for d, _ in stages], 2), 0, 1)
    info["passes"] = len(stages)
    info["scale"] = 2 if len(stages) > 1 else 1
    cv2.imwrite(str(out / "density.png"), (255 - density * 255).astype(np.uint8))
    bw = ((density > 0.5) * 255).astype(np.uint8)
    boxes, labels = segment(bw, min_area=100 * info["scale"] ** 2)
    lines = group_lines([b[:5] for b in boxes], bw)

    glyphs = [
        {"id": f"l{li:02d}_g{gi:02d}", "line": li, "x": int(bx), "y": int(by), "w": int(bw_), "h": int(bh), "ids": ids}
        for li, line in enumerate(lines)
        for gi, (bx, by, bw_, bh, ids) in enumerate(line)
    ]
    widths = [g["w"] for g in glyphs]
    median_w = float(np.median(widths))
    for g in glyphs:
        g["suspect_merged"] = bool(g["w"] > 1.6 * median_w)

    manifest, rejected, crops, names, metrics = [], [], [], [], []
    if a.card_name:
        info["alignment_cost"] = round(float(align(glyphs, card_text(a.xlsx, a.card_name))), 2)
        needed = set(card_targets(a.xlsx, a.card_name, target.get().face(a.style).full_name))
        units, rejected = units_from_alignment(glyphs, labels)
        counts = {}
        for u in units:
            k = glyph_folder(u["char"])
            counts[k] = counts.get(k, 0) + 1
            mask, entry = write_unit(out, a.style, card_id, counts[k], u, labels, density)
            entry["needed"] = u["char"] in needed
            manifest.append(entry)
            crops.append(mask)
            names.append(u["char"])
            metrics.append((entry["baseline_row"], entry["x_height_px"]))
    else:
        for g in glyphs:
            crops.append(np.isin(labels[g["y"] : g["y"] + g["h"], g["x"] : g["x"] + g["w"]], g["ids"]).astype(np.uint8) * 255)
            names.append("")
    (out / "manifest.json").write_text(json.dumps({"scan": info, "instances": manifest, "rejected": rejected}, indent=1))
    contact_sheet(crops, out / "contact_sheet.png", labels=names, metrics=metrics or None)
    if a.card_name:
        got = {e["unicode"] for e in manifest if e["needed"]}
        want = {f"U+{ord(c):04X}" for c in needed}
        print(f"needed {len(want)}: {''.join(sorted(needed))}, extracted {len(got & want)}, missing {sorted(want - got)}")
    print(f"deskew {info['deskew_deg']:.2f} deg, {len(lines)} lines, {len(glyphs)} boxes, {len(manifest)} units, {len(rejected)} set aside -> {out}")


if __name__ == "__main__":
    main()
