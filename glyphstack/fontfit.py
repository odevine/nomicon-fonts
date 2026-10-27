"""Fits a designed outline to a stacked glyph as a measuring stick, and traces the stack to Bezier paths in font units"""
import os

import cv2
import numpy as np
import potrace
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import ndtr

from . import font as fnt
from . import styles, target
from .contours import UM_PER_PX, contours_at, edge_rms_um, sym_dev
from .log import pool

NAMES = ["s", "ar", "th", "tx", "ty", "d"]
FIXED_BOUNDS = dict(ar=(0.9, 1.1), th=(-0.05, 0.05), tx=(-1e4, 1e4), ty=(-1e4, 1e4), d=(-1, 8))
# blur widths in grid px tried for the print's rounding, and the distance field band that keeps the widest exact
ROUNDING = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
LOOK_BAND = 48


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


def blur(sd, r):
    return sd if r == 0 else gaussian_filter(sd, r)


def print_look(g, p, mean, near=10):
    """The ink spread d, edge width w and rounding r in grid px that render the placed outline closest to a stack's coverage near its edge

    Rounding blurs the outline's distance field before the edge is drawn, which rounds corners and thins strokes
    narrower than the blur as the print does
    """
    sd = placed_distance(g, p, mean.shape, LOOK_BAND)
    band = np.abs(sd - p[5]) < near
    best = None
    for r in ROUNDING:
        f, m = blur(sd, r)[band], mean[band]
        sol = least_squares(lambda v: ndtr((v[0] - f) / v[1]) - m, [p[5], 2.0], bounds=([-4, 0.4], [12, 10]))
        if best is None or sol.cost < best[0]:
            best = (sol.cost, dict(d=float(sol.x[0]), w=float(sol.x[1]), r=r))
    return best[1]


def face_look(looks):
    """One look for a face, the median of its glyphs' spread, edge width and rounding"""
    return {k: float(np.median([lk[k] for lk in looks])) for k in ("d", "w", "r")}


def render(g, p, shape, look):
    """The placed outline as coverage on a stack's grid, drawn with a print look"""
    return ndtr((look["d"] - blur(placed_distance(g, p, shape, LOOK_BAND), look["r"])) / look["w"]).astype(np.float32)


def with_prior(g, p, mean, n, look, weight):
    """A stack of n instances blended with the measuring stick's outline rendered as the print would show it, which counts as weight instances"""
    return ((n * mean + weight * render(g, p, mean.shape, look)) / (n + weight)).astype(np.float32)


def from_outline(g, p, look):
    """The outline rendered with a print look on a grid of its own, and the placement that maps that grid to font units"""
    p = np.array(p, float)
    p[3] = p[4] = 0.0
    pts = np.vstack(g.to_img(p))
    margin = LOOK_BAND
    p[3], p[4] = margin - pts[:, 0].min(), margin - pts[:, 1].min()
    shape = tuple(int(np.ceil(v)) + 2 * margin for v in (np.ptp(pts[:, 1]), np.ptp(pts[:, 0])))
    return render(g, p, shape, look), p


def traced_outline(g, p, look):
    """The trace of an outline drawn with a print look, its node and contour counts and the placement it was traced at"""
    mean, p = from_outline(g, p, look)
    return trace(mean, p) + (p.tolist(),)


def to_font_units(q, p):
    """Maps points in the stack's grid coordinates to font units, y down, origin at the glyph origin"""
    s, ar, th, tx, ty = p[:5]
    Mi = np.linalg.inv(np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([s, s * ar]))
    return (np.asarray(q, float) - [tx, ty]) @ Mi.T


def trace(mean, p):
    """Potrace Bezier fit of the 0.5 contour of a stack as SVG path strings in font units, y down, and a node count"""
    big = cv2.resize(mean, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # potrace traces the True region, so the ink (coverage above 0.5) is the complement
    try:
        paths = potrace.Bitmap(big <= 0.5).trace(turdsize=40, turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY, alphamax=1.0, opticurve=True, opttolerance=0.6)
    except ValueError:
        # the curve optimisation can take the square root of a negative on rare shapes, which leaves its curves unjoined
        paths = potrace.Bitmap(big <= 0.5).trace(turdsize=40, turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY, alphamax=1.0, opticurve=False)
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


def fit_all(fonts, stacks, min_n=12, chars=(), workers=None):
    """Shared transform from the profile's seed glyph, then a translation-only placement, trace and statistics for each glyph with min_n instances

    fonts is {face: font}, a bare font counts as the primary face. Glyph keys come from `styles`, and a glyph of
    another face is placed with the shared transform scaled to that font's units per em. Each face's print look is
    the median of its traced glyphs'. With the profile's fit prior_weight set, a glyph with fewer than min_n
    instances is traced from its stack blended with its outline drawn with that look, and records that it leans on
    the prior and how far the blend moved from its own stack. Each of `chars` a face's font draws and no stack
    traces is built from its outline drawn with the look, and records that it was built so
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
    weight = target.get()["fit"].get("prior_weight")
    out, placed = {}, {}
    for key, s in stacks.items():
        st, ch = styles.style_of(key), styles.char_of(key)
        font = fonts.get(st)
        if font is None or (s["n"] < min_n and not weight) or not styles.drawn_by(font, ch):
            continue
        g = FontGlyph(font, ch)
        contour = contours_at(s["mean"], 0.5)
        if not contour:
            continue
        placed[key] = (g, fit(g, np.vstack(contour), init_translation(g, np.vstack(contour), per_style[st]), ["tx", "ty"]))
    todo = [(st, styles.key(st, ch)) for st, font in fonts.items() for ch in chars if styles.key(st, ch) not in placed and styles.drawn_by(font, ch)]
    todo = [(k, FontGlyph(fonts[st], styles.char_of(k))) for st, k in todo]
    todo = [(k, g) for k, g in todo if g.contours]
    with pool(workers or os.cpu_count() or 1) as ex:
        well = [k for k in placed if stacks[k]["n"] >= min_n]
        futs = {k: ex.submit(print_look, *placed[k], stacks[k]["mean"]) for k in well}
        looks = {}
        for k in well:
            looks.setdefault(styles.style_of(k), []).append(futs[k].result())
        face = {st: face_look(lk) for st, lk in looks.items()}
        for st in fonts:
            face.setdefault(st, face.get(first))
        built = {k: ex.submit(traced_outline, g, per_style[styles.style_of(k)], face[styles.style_of(k)]) for k, g in todo if face[styles.style_of(k)]}
        for key, (g, p) in placed.items():
            s = stacks[key]
            mean, extra = s["mean"], {}
            if s["n"] < min_n:
                mean = with_prior(g, p, s["mean"], s["n"], face[styles.style_of(key)], weight)
                extra = dict(prior=True, prior_weight=weight, moved_um=edge_rms_um(mean, s["mean"]))
            contour = contours_at(mean, 0.5)
            rms, iou = edge_stats(g, p, mean, contour)
            path, nodes, nc = trace(mean, p)
            out[key] = dict(trace=path, adv=g.adv, n=s["n"], ncards=len(s["cards"]), rms=rms, iou=iou, nodes=nodes, nc=nc, p=p.tolist(), **extra)
        for k, g in todo:
            if k in built:
                path, nodes, nc, p = built[k].result()
                out[k] = dict(trace=path, adv=g.adv, n=stacks[k]["n"] if k in stacks else 0, nodes=nodes, nc=nc, p=p, built="outline", look=face[styles.style_of(k)])
    return shared, out
