"""Cuts one greyscale card scan into glyph units, each a clean ink coverage map with its neighbours masked"""
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import target, texture
from .log import log, timed
from .scan import load_gray

PAD = 16
HALO_PX = 3
DILATE_PX = 7
TRIM_PX = 2
# a scan whose outline is turned further than this from its scan table angle is reported as a warning
TILT_WARN_DEG = 2


def card_crop(g):
    """Crops the card from the scan, returns the crop and its (row, column) origin in the scan"""
    small = cv2.resize(g, None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_AREA)
    # paper under the card is brighter than 235 only on bright cards, which a stricter cut would break
    mask = cv2.morphologyEx((small < 235).astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    _, _, st, _ = cv2.connectedComponentsWithStats(mask)
    i = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h = st[i, :4]
    y0, x0 = max(y * 8 - 20, 0), max(x * 8 - 20, 0)
    return g[y0 : (y + h) * 8 + 20, x0 : (x + w) * 8 + 20], (y0, x0)


def segment(g):
    """Labels ink components and merges i and j dots into their stems, returns the label image and component dicts

    The ink threshold, size limits and dot rules come from the profile's segment section. A dot joins the stem below it
    whose top end it overlaps, the columns of the stem's top fifth, so the dot of a slanted italic stem, which sits to
    the right of the stem's foot, still finds it. A dot smaller than the components the profile keeps joins only a
    stem narrow at its top, as an i or j is, so a speck of texture above an n or m stays out of it
    """
    cfg = target.get()["segment"]
    ink = ndi.gaussian_filter(g.astype(np.float32), 1.2) < cfg["ink_level"]
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    areas = ndi.sum(ink, lab, range(1, n + 1))
    comps, small = [], []
    big, box = cfg["max_size"], cfg["dot_max_box"]
    dot_min = cfg.get("dot_min_area", cfg["min_area"])
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        c = dict(id=i, y0=sl[0].start, y1=sl[0].stop, x0=sl[1].start, x1=sl[1].stop, area=int(areas[i - 1]), ids=[i])
        if areas[i - 1] >= cfg["min_area"] and h <= big and w <= big and h >= cfg["min_height"]:
            comps.append(c)
        elif dot_min <= areas[i - 1] < cfg["min_area"] and h <= box and w <= box:
            small.append(c)
    used = set()
    tall, gap = cfg["dot_stem_min_height"], cfg["dot_max_gap"]
    narrow = cfg.get("dot_narrow_stem", 0.6)
    tops = {}

    def top_end(o):
        """The columns of a component's top fifth, and that width over its height"""
        if o["id"] not in tops:
            h = o["y1"] - o["y0"]
            cols = np.nonzero((lab[o["y0"] : o["y0"] + max(h // 5, 3), o["x0"] : o["x1"]] == o["id"]).any(0))[0] + o["x0"]
            tops[o["id"]] = (cols.min(), cols.max() + 1, (cols.max() + 1 - cols.min()) / h)
        return tops[o["id"]]

    dots = [c for c in comps if c["area"] < cfg["dot_max_area"] and c["x1"] - c["x0"] <= box and c["y1"] - c["y0"] <= box] + small
    for c in dots:
        best = None
        for o in comps:
            if o is c or o["id"] in used or o["y1"] - o["y0"] <= tall or not (o["y0"] > c["y0"] - 2 and 0 <= o["y0"] - c["y1"] < gap):
                continue
            t0, t1, ratio = top_end(o)
            if c in small and ratio > narrow:
                continue
            overlap = min(c["x1"], t1 + 4) - max(c["x0"], t0 - 4)
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, o)
        if best is not None:
            s = best[1]
            s["ids"] = s["ids"] + c["ids"]
            s["y0"] = min(s["y0"], c["y0"])
            s["area"] += c["area"]
            used.add(c["id"])
    if "mark_max_height" in cfg:
        join_marks(lab, comps, used, cfg)
    return lab, [c for c in comps if c["id"] not in used]


def join_marks(lab, comps, used, cfg):
    """Joins the parts of a punctuation mark into one component, in place, adding each joined part's id to used

    Two quote halves of one size side by side become one double quote, the two bars of = one over the other become
    one sign, the dot of a colon or semicolon joins the dot or comma below it, and the dot of ! or ? joins a stem above
    it whose foot is at most dot_narrow_stem of its height across. A mark part is no taller than mark_max_height and smaller than dot_max_area, and a dot also fits in
    dot_max_box. Every part is a component the profile keeps on its own, so specks of a halftone screen never join
    """
    box, tall, narrow = cfg["dot_max_box"], cfg["dot_stem_min_height"], cfg.get("dot_narrow_stem", 0.6)
    h = lambda c: c["y1"] - c["y0"]
    w = lambda c: c["x1"] - c["x0"]
    live = lambda: [c for c in comps if c["id"] not in used]
    mark = lambda c: h(c) <= cfg["mark_max_height"] and c["area"] < cfg["dot_max_area"]
    dot = lambda c: mark(c) and h(c) <= box and w(c) <= box

    def join(s, c):
        s["ids"] = s["ids"] + c["ids"]
        s["y0"], s["y1"], s["x0"], s["x1"] = min(s["y0"], c["y0"]), max(s["y1"], c["y1"]), min(s["x0"], c["x0"]), max(s["x1"], c["x1"])
        s["area"] += c["area"]
        used.add(c["id"])

    def nearest(a, fits, gap):
        best = [o for o in live() if o is not a and fits(o)]
        return min(best, key=gap) if best else None

    # a quote half pairs with the nearest half to its right of the same size and height
    for a in sorted([c for c in live() if mark(c) and h(c) > box], key=lambda c: c["x0"]):
        if a["id"] in used:
            continue
        fits = lambda o: (mark(o) and h(o) > box and o["x0"] >= a["x1"] and o["x0"] - a["x1"] <= cfg["quote_max_gap"]
                          and abs(h(o) - h(a)) <= 0.25 * max(h(o), h(a)) and max(o["area"], a["area"]) <= 1.5 * min(o["area"], a["area"])
                          and min(o["y1"], a["y1"]) - max(o["y0"], a["y0"]) >= 0.7 * min(h(o), h(a)))
        b = nearest(a, fits, lambda o: o["x0"] - a["x1"])
        if b is not None:
            join(a, b)
    if "equals_gap" in cfg:
        lo, hi = cfg["equals_gap"]
        bar = lambda c: w(c) >= 3 * h(c) and h(c) <= box
        # the two bars of = are flat, of one width and in line, one just over the other
        for a in sorted([c for c in live() if bar(c)], key=lambda c: c["y0"]):
            if a["id"] in used:
                continue
            fits = lambda o: (bar(o) and lo <= o["y0"] - a["y1"] <= hi and abs(w(o) - w(a)) <= 0.1 * max(w(o), w(a))
                              and min(o["x1"], a["x1"]) - max(o["x0"], a["x0"]) >= 0.9 * min(w(o), w(a)))
            b = nearest(a, fits, lambda o: o["y0"] - a["y1"])
            if b is not None:
                join(a, b)
    lo, hi = cfg["colon_gap"]
    # a colon's or semicolon's dot over the dot or comma below it
    for a in sorted([c for c in live() if dot(c)], key=lambda c: c["y0"]):
        if a["id"] in used:
            continue
        fits = lambda o: mark(o) and lo <= o["y0"] - a["y1"] <= hi and min(o["x1"], a["x1"] + 4) - max(o["x0"], a["x0"] - 4) > 0
        b = nearest(a, fits, lambda o: o["y0"] - a["y1"])
        if b is not None:
            join(a, b)
    lo, hi = cfg["foot_gap"]
    # the dot of ! or ? under the foot of its stem
    for a in [c for c in live() if dot(c)]:
        if a["id"] in used:
            continue

        def fits(o):
            if h(o) <= tall or not lo <= a["y0"] - o["y1"] <= hi:
                return False
            cols = np.nonzero(np.isin(lab[o["y1"] - max(h(o) // 5, 3) : o["y1"], o["x0"] : o["x1"]], o["ids"]).any(0))[0] + o["x0"]
            return len(cols) and (cols.max() + 1 - cols.min()) / h(o) <= narrow and min(a["x1"], cols.max() + 5) - max(a["x0"], cols.min() - 4) > 0

        s = nearest(a, fits, lambda o: a["y0"] - o["y1"])
        if s is not None:
            join(s, a)


def in_regions(comps, shape):
    """The components whose centre lies in one of the profile's regions, each tagged with the first region it lies in"""
    out = []
    for c in comps:
        cx, cy = (c["x0"] + c["x1"]) / 2, (c["y0"] + c["y1"]) / 2
        r = next((r for r in target.get().regions if r.contains(cx, cy, shape)), None)
        if r is not None:
            out.append(dict(c, region=r.name))
    return out


def region_mask(shape):
    """Pixels inside any of the profile's regions"""
    m = np.zeros(shape, bool)
    for r in target.get().regions:
        x, y, w, h = r.box
        m[int(np.floor(y * shape[0])) : int(np.ceil((y + h) * shape[0])), int(np.floor(x * shape[1])) : int(np.ceil((x + w) * shape[1]))] = True
    return m


def scan_levels(g, lab, comps, kept=None):
    """Paper and solid ink grey levels from rings around and cores inside the glyph ink

    allink and the ring cover every component, so neighbours outside the regions still mask a unit and stay out of
    its paper. The levels themselves are measured inside the regions, on the kept components when given.
    Returns paper, ink, the ink mask and the ring mask
    """
    paper_cfg = target.get()["paper"]
    inner, outer = paper_cfg["ring_px"]
    allink = np.isin(lab, [i for c in comps for i in c["ids"]])
    ring = ndi.binary_dilation(allink, iterations=outer) & ~ndi.binary_dilation(allink, iterations=inner)
    own = allink if kept is None else np.isin(lab, [i for c in kept for i in c["ids"]])
    inside = region_mask(g.shape)
    paper = float(np.median(g[ring & inside]))
    ink = float(np.median(g[ndi.binary_erosion(own, iterations=paper_cfg["core_px"])]))
    return paper, ink, allink, ring


def local_paper(g, ring, box, paper):
    """Paper level of the ring pixels around one unit, or the scan's level when there are too few of them"""
    cfg = target.get()["paper"]
    m = cfg["local_window_px"]
    y0, y1, x0, x1 = box
    sl = (slice(max(y0 - m, 0), y1 + m), slice(max(x0 - m, 0), x1 + m))
    vals = g[sl][ring[sl]]
    return float(np.median(vals)) if len(vals) >= cfg["local_min_px"] else paper


def coverage_units(g, lab, comps, paper, ink, allink, halo=HALO_PX, ring=None):
    """Builds one unit per component: coverage with halftone suppressed and neighbouring ink masked by a halo

    Coordinates in `box` index the scan crop, `mask` is the unit's own component inside that box and
    `bbox` is the component extent in the same crop frame. The box is trimmed to the window around the ink,
    so the coverage carries no empty padding. With the ring mask given each unit takes its own local paper level,
    which it records as `paper`
    """
    units = []
    for c in comps:
        y0, y1 = max(c["y0"] - PAD, 0), min(c["y1"] + PAD, g.shape[0])
        x0, x1 = max(c["x0"] - PAD, 0), min(c["x1"] + PAD, g.shape[1])
        m = np.isin(lab[y0:y1, x0:x1], c["ids"])
        others = ndi.binary_dilation(allink[y0:y1, x0:x1] & ~m, iterations=halo)
        win = (ndi.binary_dilation(m, iterations=DILATE_PX) & ~others) | m
        ys, xs = np.nonzero(win)
        ty0, ty1, tx0, tx1 = max(ys.min() - TRIM_PX, 0), min(ys.max() + 1 + TRIM_PX, win.shape[0]), max(xs.min() - TRIM_PX, 0), min(xs.max() + 1 + TRIM_PX, win.shape[1])
        win, m = win[ty0:ty1, tx0:tx1], m[ty0:ty1, tx0:tx1]
        y0, y1, x0, x1 = y0 + ty0, y0 + ty1, x0 + tx0, x0 + tx1
        p = paper if ring is None else local_paper(g, ring, (y0, y1, x0, x1), paper)
        cov = coverage(g[y0:y1, x0:x1], p, ink) * win
        u = dict(
            cov=cov.astype(np.float32), box=(y0, y1, x0, x1), mask=m, halo_px=halo,
            cy=(c["y0"] + c["y1"]) / 2, cx=(c["x0"] + c["x1"]) / 2, bbox=(c["y0"], c["y1"], c["x0"], c["x1"]),
        )
        if "region" in c:
            u["region"] = c["region"]
        if ring is not None:
            u["paper"] = p
        units.append(u)
    return units


def coverage(g, paper, ink):
    """Ink coverage of grey levels with the halftone floor and the solid ceiling from the profile clipped off"""
    lo, hi = target.get()["paper"]["coverage_clip"]
    cov = np.clip((paper - g.astype(np.float32)) / (paper - ink), 0, 1)
    return np.clip((cov - lo) / (hi - lo), 0, 1)


def card_angle(g):
    """Counterclockwise turn of the card in degrees within +-45, from the long edge of its outline"""
    small = cv2.resize(g, None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_AREA)
    mask = cv2.morphologyEx((small < 235).astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    box = cv2.boxPoints(cv2.minAreaRect(max(cs, key=cv2.contourArea)))
    edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
    dx, dy = max(edges, key=lambda e: np.hypot(*e))
    if dy > 0:
        dx, dy = -dx, -dy
    # the long edge points up on an upright card, and turns left as the card turns counterclockwise
    return float(((np.degrees(np.arctan2(-dx, -dy)) + 45) % 90) - 45)


def upright(g, angle, name=None):
    """Turns a scan's card crop upright, returns the upright crop and the 2x3 map from the quarter-turned crop to it

    angle is the card's counterclockwise turn in degrees as it appears in the scan image. Quarter turns are undone exactly with rot90.
    Any rest is measured from the card outline near its nominal value and undone by one cubic warp, after which
    the card is cropped again. With a scan name, the outline is measured on every scan and a turn more than
    TILT_WARN_DEG from the scan table is logged as a warning
    """
    q = int(round(angle / 90)) % 4
    rest = angle - 90 * round(angle / 90)
    g = np.ascontiguousarray(np.rot90(g, (4 - q) % 4))
    if rest == 0 and name is None:
        return g, g, np.eye(2, 3)
    # the outline cannot tell +45 from -45, so the turn nearest the nominal one is taken
    measured = min(card_angle(g) + np.array([-90.0, 0.0, 90.0]), key=lambda m: abs(m - rest))
    if name is not None:
        log.debug("%s outline is turned %+.2f deg from the scan table's %g", name, measured - rest, angle)
        if abs(measured - rest) > TILT_WARN_DEG:
            log.warning("%s is turned %+.1f deg from the %g deg in the scan table, check how it sits in the jig", name, measured - rest, angle)
    if rest == 0:
        return g, g, np.eye(2, 3)
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -measured, 1.0)
    rot = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    crop, (oy, ox) = card_crop(rot)
    M[:, 2] -= [ox, oy]
    return np.ascontiguousarray(crop), g, M


def native_units(units, native, to_up, paper, ink):
    """Adds each unit's coverage cut from the quarter-turned native pixels, so stacking resamples a turned scan only once

    `ncov` is the native coverage with the unit's window carried over, `nbox` its (y0, y1, x0, x1) in the native crop
    """
    inv = cv2.invertAffineTransform(to_up)
    for u in units:
        y0, y1, x0, x1 = u["box"]
        corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float) @ inv[:, :2].T + inv[:, 2]
        nx0, ny0 = np.maximum(np.floor(corners.min(0)).astype(int) - 2, 0)
        nx1, ny1 = np.minimum(np.ceil(corners.max(0)).astype(int) + 3, native.shape[::-1])
        # window of the native box in unit box coordinates, pulled back with nearest neighbour
        A = to_up.copy()
        A[:, 2] += A[:, :2] @ [nx0, ny0] - [x0, y0]
        win = cv2.warpAffine((u["cov"] > 0).astype(np.uint8) | u["mask"].astype(np.uint8), A, (int(nx1 - nx0), int(ny1 - ny0)), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        win = cv2.dilate(win, np.ones((3, 3), np.uint8))
        u["ncov"] = (coverage(native[ny0:ny1, nx0:nx1], u.get("paper", paper), ink) * win).astype(np.float32)
        u["nbox"] = (int(ny0), int(ny1), int(nx0), int(nx1))


def extract_scan(path, angle=0):
    """Loads a scan, crops the card, turns it upright from its counterclockwise angle in the scan image and returns its units and levels

    For a card turned by other than quarter turns, `to_up` maps the quarter-turned native crop to the upright crop
    and every unit also carries its native coverage. The profile's texture settings are applied once the card is
    found and turned, to the upright and the native pixels alike, since flattening lifts the bed around the card
    to the paper level the outline is told apart from
    """
    with timed(f"extracting {Path(path).name}"):
        crop, origin = card_crop(load_gray(path))
        crop, native, to_up = upright(crop, angle, Path(path).name)
        if texture.active():
            same = native is crop
            crop = texture.prepare(crop)
            native = crop if same else texture.prepare(native)
        lab, comps = segment(crop)
        kept = in_regions(comps, crop.shape)
        paper, ink, allink, ring = scan_levels(crop, lab, comps, kept)
        local = target.get()["paper"]["mode"] == "local"
        units = coverage_units(crop, lab, kept, paper, ink, allink, ring=ring if local else None)
        if native is not crop:
            native_units(units, native, to_up, paper, ink)
        log.info("card %dx%d px, %d components, %d in regions, paper %.0f ink %.0f", crop.shape[1], crop.shape[0], len(comps), len(kept), paper, ink)
    return dict(units=units, paper=paper, ink=ink, shape=crop.shape, angle=angle, to_up=to_up, origin=origin, ncomp=len(comps), nkept=len(kept))
