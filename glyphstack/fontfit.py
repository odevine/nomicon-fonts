"""Fits a designed outline to a stacked glyph as a measuring stick, and traces the stack to Bezier paths in font units"""
import cv2
import numpy as np
import potrace
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from . import font as fnt
from . import styles, target
from .contours import UM_PER_PX, contours_at, sym_dev

NAMES = ["s", "ar", "th", "tx", "ty", "d"]
FIXED_BOUNDS = dict(ar=(0.9, 1.1), th=(-0.05, 0.05), tx=(-1e4, 1e4), ty=(-1e4, 1e4), d=(-1, 8))


def bounds():
    """Fit bounds, with the scale range from the profile since it follows the size the type is printed at"""
    return dict(FIXED_BOUNDS, s=tuple(target.get()["fit"]["scale_bounds"]))


class FontGlyph:
    """A designed outline as dense polylines in font units with its advance width"""

    def __init__(self, font, ch, spacing=1.0):
        contours, _ = fnt.glyph_contours(font, ch)
        self.contours = fnt.densify(contours, spacing)
        self.adv = font["hmtx"][font.getBestCmap()[ord(ch)]][0]

    def to_img(self, prm):
        s, ar, th, tx, ty = prm[:5]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([s, s * ar])
        return [np.stack([c[:, 0], -c[:, 1]], 1) @ R.T + [tx, ty] for c in self.contours]

    def signed_dist(self, prm, pts):
        """Signed distance in grid px from pts to the placed outline, positive outside"""
        poly = self.to_img(prm)
        return np.where(fnt.inside_evenodd(poly, pts), -1, 1) * outline_dist(poly, pts)


def outline_dist(poly, pts):
    """Unsigned distance in grid px from pts to closed polylines, measured to their segments

    Measuring to segments rather than vertices keeps the distance smooth in the placement, so the fit does not
    settle in the small dips the vertex spacing leaves
    """
    V = np.vstack(poly)
    starts = np.cumsum([0] + [len(c) for c in poly])
    j = cKDTree(V).query(pts)[1]
    ci = np.searchsorted(starts, j, side="right") - 1
    s0, n = starts[ci], np.diff(starts)[ci]
    dist = np.full(len(pts), np.inf)
    # with evenly spaced vertices the nearest point lies on one of the two segments meeting at the nearest vertex
    for off in (-1, 0):
        a = s0 + (j + off - s0) % n
        b = s0 + (j + off + 1 - s0) % n
        A, d = V[a], V[b] - V[a]
        t = np.clip(((pts - A) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
        dist = np.minimum(dist, np.linalg.norm(pts - A - t[:, None] * d, axis=1))
    return dist


def init_translation(g, pts, prm):
    """Copies prm with the translation that puts the outline centre on the contour centre"""
    fx = np.vstack(g.contours)
    fc, pc = (fx.min(0) + fx.max(0)) / 2, (pts.min(0) + pts.max(0)) / 2
    p = np.array(prm, float)
    p[3] = pc[0] - p[0] * fc[0]
    p[4] = pc[1] + p[0] * p[1] * fc[1]
    return p


def fit(g, pts, x0, free):
    """Robust least squares of the outline to the contour points over the named free parameters"""
    idx = [NAMES.index(n) for n in free]
    x0 = np.array(x0, float)

    def res(v):
        p = x0.copy()
        p[idx] = v
        return g.signed_dist(p, pts) - p[5]

    b = bounds()
    sol = least_squares(res, x0[idx], loss="soft_l1", f_scale=1.5, bounds=([b[n][0] for n in free], [b[n][1] for n in free]), x_scale=[0.01 if n == "s" else 1 for n in free])
    p = x0.copy()
    p[idx] = sol.x
    return p


def shared_transform(font, mean, ch=None):
    """Free fit of size, aspect, rotation, ink spread and offset of the profile's seed glyph to its stack"""
    fit_cfg = target.get()["fit"]
    g = FontGlyph(font, ch or fit_cfg["seed_glyph"])
    pts = np.vstack(contours_at(mean, 0.5))
    return fit(g, pts, init_translation(g, pts, fit_cfg["seed"]), NAMES)


def placed_distance(g, p, shape, band=12):
    """Signed distance field of the placed outline on a stack's grid, positive outside

    Distances are exact within `band` grid px of the outline (widened to clear the ink spread) and clamped to
    plus or minus the band further out, which leaves the contour at the ink spread and the filled region unchanged
    """
    H, W = shape
    band = int(max(band, abs(p[5]) + 6))
    poly = g.to_img(p)
    inside, edge = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    for c in poly:
        pts = [np.round(c * 16).astype(np.int32)]
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, pts, 1, cv2.LINE_8, shift=4)
        inside ^= m
        cv2.polylines(edge, pts, True, 1, 1, cv2.LINE_8, shift=4)
    yy, xx = np.nonzero(cv2.dilate(edge, np.ones((2 * band + 1, 2 * band + 1), np.uint8)))
    d = np.full((H, W), float(band))
    d[yy, xx] = outline_dist(poly, np.stack([xx, yy], 1).astype(float))
    return np.where(inside > 0, -d, d)


def reference_contours(g, p, shape, sd=None):
    """Polylines of the placed outline with the ink spread applied, in the stack's grid coordinates"""
    return contours_at(placed_distance(g, p, shape) if sd is None else sd, p[5])


def edge_stats(g, p, mean, contour, sd=None):
    """Edge RMS in micrometres and IoU of a stack against its placed outline, with the ink spread applied

    Pass the glyph's `placed_distance` as sd to reuse it across calls on the same grid
    """
    if sd is None:
        sd = placed_distance(g, p, mean.shape)
    rms = float(np.sqrt((sym_dev(contour, contours_at(sd, p[5])) ** 2).mean()) * UM_PER_PX)
    si, fi = mean > 0.5, sd <= p[5]
    return rms, float((si & fi).sum() / (si | fi).sum())


def to_font_units(q, p):
    """Maps points in the stack's grid coordinates to font units, y down, origin at the glyph origin"""
    s, ar, th, tx, ty = p[:5]
    Mi = np.linalg.inv(np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([s, s * ar]))
    return (np.asarray(q, float) - [tx, ty]) @ Mi.T


def trace(mean, p):
    """Potrace Bezier fit of the 0.5 contour of a stack as SVG path strings in font units, y down, and a node count"""
    big = cv2.resize(mean, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # potrace traces the True region, so the ink (coverage above 0.5) is the complement
    paths = potrace.Bitmap(big <= 0.5).trace(turdsize=40, turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY, alphamax=1.0, opticurve=True, opttolerance=0.6)
    s, ar, th, tx, ty = p[:5]
    Mi = np.linalg.inv(np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([s, s * ar]))

    def fu(pt):
        # potrace works in corner coordinates of the 2x image, contours_at in pixel centre coordinates of the stack
        v = Mi @ (np.array([pt.x / 2 - 0.5, pt.y / 2 - 0.5]) - [tx, ty])
        return v[0], v[1]

    out = []
    for c in paths:
        a = fu(c.start_point)
        d = f"M{a[0]:.1f},{a[1]:.1f}"
        for sg in c.segments:
            if sg.is_corner:
                c1, e = fu(sg.c), fu(sg.end_point)
                d += f"L{c1[0]:.1f},{c1[1]:.1f}L{e[0]:.1f},{e[1]:.1f}"
            else:
                c1, c2, e = fu(sg.c1), fu(sg.c2), fu(sg.end_point)
                d += f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {e[0]:.1f},{e[1]:.1f}"
        out.append(d + "Z")
    return "".join(out), sum(len(c.segments) for c in paths), len(paths)


def face_transform(shared, primary, other):
    """The shared transform of the primary face carried to another face, with the scale corrected for the other font's units per em"""
    p = np.array(shared, float)
    p[0] *= primary["head"].unitsPerEm / other["head"].unitsPerEm
    return p


def fit_all(fonts, stacks, min_n=12):
    """Shared transform from the profile's seed glyph, then a translation-only placement, trace and statistics for each glyph with min_n instances

    fonts is {face: font}, a bare font counts as the primary face. Glyph keys come from `styles`, and a glyph of
    another face is placed with the shared transform scaled to that font's units per em
    """
    first = styles.primary()
    if not isinstance(fonts, dict):
        fonts = {first: fonts}
    reg = fonts[first]
    seed = target.get()["fit"]["seed_glyph"]
    if seed not in stacks:
        raise SystemExit(f"the seed glyph {seed!r} has no stack, so the shared transform cannot be fitted")
    shared = shared_transform(reg, stacks[seed]["mean"])
    per_style = {st: shared if st == first else face_transform(shared, reg, f) for st, f in fonts.items()}
    out = {}
    for key, s in stacks.items():
        st, ch = styles.style_of(key), styles.char_of(key)
        font = fonts.get(st)
        if font is None or s["n"] < min_n or ord(ch) not in font.getBestCmap():
            continue
        g = FontGlyph(font, ch)
        contour = contours_at(s["mean"], 0.5)
        if not contour:
            continue
        p = fit(g, np.vstack(contour), init_translation(g, np.vstack(contour), per_style[st]), ["tx", "ty"])
        rms, iou = edge_stats(g, p, s["mean"], contour)
        path, nodes, nc = trace(s["mean"], p)
        out[key] = dict(trace=path, adv=g.adv, n=s["n"], ncards=len(s["cards"]), rms=rms, iou=iou, nodes=nodes, nc=nc, p=p.tolist())
    return shared, out
