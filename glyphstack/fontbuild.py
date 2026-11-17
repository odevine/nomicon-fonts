"""Variable fonts of each face of a session with a Bleed axis, built from its traces and the synthetic font baseline

Bleed is the strength of the face's print look in percent, the ink spread and rounding measured on its traced
stacks, which fill in sharp inner corners and round off outer ones. At 100 every glyph is as the session traced or
built it, at 25 a quarter of the print look is left and at 125 it is a quarter stronger. The synthetic font baseline,
which glyphstack.baseline makes, is each face drawn at a Bleed of 25 and is the lowest master as it is. A glyph with
no stack of its own is its baseline outline with its points moved onto the edge the rest of the print look draws at
each strength. A traced glyph keeps its trace at 100 and moves by the change the print look makes to its baseline
outline at the other strengths, so every master of a glyph has the same points. Composites stay composites of their parts.
The fonts kern with the baseline's kerning, the measuring stick's designed kerning, which doesn't change with Bleed. With --weight
the fonts also have a Weight axis from 400 to a semibold at 600, or to the weight given, each glyph's regular outline grown with the same points

    python -m glyphstack.fontbuild build/session --baseline build/synthetic_baseline/NomiconRulesBleed25-Regular.ttf \
        --italic-baseline build/synthetic_baseline/NomiconRulesBleed25-Italic.ttf -o build/fonts --site site/public/fonts
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from fontTools import varLib
from fontTools.agl import UV2AGL
from fontTools.cu2qu import curves_to_quadratic
from fontTools.designspaceLib import AxisDescriptor, DesignSpaceDocument, InstanceDescriptor, SourceDescriptor
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.otlLib.builder import buildStatTable
from fontTools.pens.basePen import BasePen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree

from . import font as fnt
from . import fontfit, log, styles, target
from .contours import UM_PER_PX, contours_at, sym_dev

# past a quarter more than measured the look closes more slits than a fixed set of points can follow
AXIS = dict(tag="BLED", name="Bleed", minimum=25, default=100, maximum=125)
# strengths between these interpolate to within about 1 µm of the look drawn at them
MASTERS = (25, 50, 100, 125)
# the output's units per em as a multiple of the measuring stick's, since one unit of a 1000 unit em is about 3 µm
# at the printed size, coarser than the 1 µm the masters are fitted to
SCALE = 4
# a joint where the outline turns by more than this many degrees is a corner, which the print look rounds or fills
CORNER_DEG = 25.0
# how far in grid px each point is searched along its normal for the print look's edge
REACH = 48.0
# points each segment is fitted through
SAMPLES = 9
# grid px around an outline on its own grid, enough for the widest blur and REACH
MARGIN = 96
# how far in output units a curve may move when made quadratic
MAX_ERR = 0.5
# segments shorter than this in the measuring stick's units are folded into their neighbour
MIN_SEGMENT = 2.0
# a glyph whose next strength step lands further than this in µm from the print look's edge keeps the last step
# it fitted, where the look closes a slit or cuts a stroke that its points cannot follow
MAX_MISFIT = 8.0
# a segment between two corners shorter than this in grid px is a slit's tip, folded into one corner
TIP_PX = 3.0
# the optional Weight axis, whose bold master is the regular grown by BOLD
WEIGHT = dict(tag="wght", name="Weight", minimum=400, default=400, maximum=700)
# how far the bold grows each side in measuring stick units, across and up and down, the measuring stick's bold
# stems being 30 wider each side than its regular and its horizontal strokes 5
BOLD = dict(x=30.0, y=5.0)
WEIGHT_NAMES = {500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold"}


def set_top_weight(top):
    """Puts the Weight axis's top master at top, its outline grown by the measuring stick's bold scaled linearly from 400 to 700"""
    s = (top - 400) / 300
    WEIGHT["maximum"] = top
    BOLD.update(x=30.0 * s, y=5.0 * s)
# how far in grid px a traced outline's move from the measuring stick is smoothed along it, which takes out the jitter
# of reading the move from a different outline while keeping the change at a filled corner
SMOOTH_PX = 2.0
# points each curve of a traced outline is moved through
MOVE_SAMPLES = 32
# the shortest a moved handle may get as a fraction of its length on the trace
MIN_HANDLE = 0.1
# the longest an outer corner's point may move as a multiple of the offset
MITRE = 2.0
# how far apart in measuring stick units the points of a filled notch sit
UNLOOP_GAP = 0.25


class SegmentPen(BasePen):
    """Records an outline as closed contours of cubic segments, kind L for a line and C for a curve, in font units y up"""

    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.contours, self.cur = [], None

    def _moveTo(self, p):
        self.start = self.last = np.array(p, float)
        self.cur = []

    def _lineTo(self, p):
        p = np.array(p, float)
        if np.linalg.norm(p - self.last) > 1e-6:
            self.cur.append(("L", line(self.last, p)))
        self.last = p

    def _curveToOne(self, a, b, c):
        c = np.array(c, float)
        self.cur.append(("C", np.array([self.last, a, b, c], float)))
        self.last = c

    def _closePath(self):
        self._lineTo(self.start)
        if self.cur:
            self.contours.append(closed(self.cur))
        self.cur = None

    _endPath = _closePath


class SplinePen(SegmentPen):
    """A SegmentPen that reads each quadratic spline as the one cubic it was made from, for fonts this module wrote"""

    def qCurveTo(self, *points):
        if points[-1] is None:
            return super().qCurveTo(*points)
        pts = np.array([self.last, *points], float)
        if len(pts) == 2:
            return self._lineTo(pts[1])
        mids = [pts[0]] + [(pts[i] + pts[i + 1]) / 2 for i in range(1, len(pts) - 2)] + [pts[-1]]
        u = np.linspace(0, 1, 9)[:, None]
        Q = np.vstack([(1 - u) ** 2 * a + 2 * (1 - u) * u * c + u ** 2 * b for a, c, b in zip(mids[:-1], pts[1:-1], mids[1:])])
        self.cur.append(("C", cubic_through(Q)))
        self.last = pts[-1]
        self._BasePen__currentPoint = tuple(pts[-1])


def cubic_through(Q):
    """The cubic with Q's ends that fits its points best, each taken at its fraction of the length along Q"""
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    u = (L / L[-1])[:, None]
    b1, b2 = 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u ** 2
    rest = Q - (1 - u) ** 3 * Q[0] - u ** 3 * Q[-1]
    inner = np.linalg.lstsq(np.hstack([b1, b2]), rest, rcond=None)[0]
    return np.array([Q[0], inner[0], inner[1], Q[-1]])


def line(a, b):
    return np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b], float)


def closed(c):
    """A contour whose last segment ends exactly where its first starts, so every master drops the same closing point"""
    c[-1][1][3] = c[0][1][0]
    return c


def design(font, ch, name=None, k=1, pen=SegmentPen):
    """The outline of a character in font units divided by k, or of the glyph so named, a dotless letter the font does not draw being its dotted letter less the dot"""
    stem = styles.STEMS.get(ch) if name is None and ord(ch) not in font.getBestCmap() else None
    name = name or font.getBestCmap()[ord(stem or ch)]
    gs = font.getGlyphSet()
    name = name if name in gs or name + ".part" not in gs else name + ".part"
    pen = pen(gs)
    gs[name].draw(pen)
    cs = [[(kind, P / k) for kind, P in c] for c in pen.contours]
    if stem:
        cs = [c for c in cs if c is not max(cs, key=lambda c: min(P[:, 1].min() for _, P in c))]
    return [merge_tiny(c) for c in cs]


def merge_tiny(c, min_fu=MIN_SEGMENT, tip_fu=0.0):
    """The contour with segments shorter than min_fu, or than tip_fu between two corners, folded into the next one, whose start moves back by the same amount

    Some outlines step by under a unit where two strokes meet, or end a slit in a flat tip a pixel wide, and a corner
    that small has no direction to fill along
    """
    c = [(k, P.copy()) for k, P in c]
    i = 0
    while len(c) > 3 and i < len(c):
        k, P = c[i]
        size = max(np.linalg.norm(P[3] - P[0]), np.abs(P - P[0]).max())
        cn = corners(c)
        if size < min_fu or (size < tip_fu and cn[i] and cn[(i + 1) % len(c)]):
            nk, N = c[(i + 1) % len(c)]
            N[:2] += P[0] - N[0]
            if nk == "L":
                N[:] = line(N[0], N[3])
            del c[i]
            i = 0
            continue
        i += 1
    return closed(c)


def parse_trace(d):
    """The contours of a trace's SVG path, in font units y down as fontfit writes it, as segments y up"""
    out = []
    for sub in re.findall(r"M[^M]*", d):
        toks = re.findall(r"[CL]|-?[\d.]+,-?[\d.]+", sub[1:])
        pts = lambda ts: [np.array([float(v) for v in t.split(",")]) * [1, -1] for t in ts]
        last = pts(toks[:1])[0]
        c, i = [], 1
        while i < len(toks):
            if toks[i] == "L":
                p = pts(toks[i + 1:i + 2])[0]
                if np.linalg.norm(p - last) > 1e-6:
                    c.append(("L", line(last, p)))
                last, i = p, i + 2
            else:
                a, b, e = pts(toks[i + 1:i + 4])
                c.append(("C", np.array([last, a, b, e])))
                last, i = e, i + 4
        if np.linalg.norm(last - c[0][1][0]) > 1e-6:
            c.append(("L", line(last, c[0][1][0])))
        out.append(closed(c))
    return out


def bez(P, u):
    u = np.asarray(u, float)[:, None]
    return (1 - u) ** 3 * P[0] + 3 * (1 - u) ** 2 * u * P[1] + 3 * (1 - u) * u ** 2 * P[2] + u ** 3 * P[3]


def dbez(P, u):
    u = np.asarray(u, float)[:, None]
    return 3 * (1 - u) ** 2 * (P[1] - P[0]) + 6 * (1 - u) * u * (P[2] - P[1]) + 3 * u ** 2 * (P[3] - P[2])


def split(P, u):
    a, b, c = P[:3] + u * (P[1:] - P[:3])
    d, e = a + u * (b - a), b + u * (c - b)
    f = d + u * (e - d)
    return np.array([P[0], a, d, f]), np.array([f, e, c, P[3]])


def heading(P, end):
    """Unit tangent at a segment's start or end, from the first control point that does not sit on it"""
    for v in ((P[3] - P[2], P[3] - P[1], P[3] - P[0]) if end else (P[1] - P[0], P[2] - P[0], P[3] - P[0])):
        n = np.linalg.norm(v)
        if n > 1e-6:
            return v / n
    return np.zeros(2)


def corners(c):
    """Whether each segment's start is a corner, where the outline turns by more than CORNER_DEG"""
    return [np.degrees(np.arccos(np.clip(heading(c[i - 1][1], True) @ heading(c[i][1], False), -1, 1))) > CORNER_DEG for i in range(len(c))]


def arc_param(P, s, from_end):
    """The parameter at arc length s from a segment's start or end, and the segment's length"""
    u = np.linspace(0, 1, 65)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(bez(P, u), axis=0), axis=1))])
    if from_end:
        return float(np.interp(L[-1] - s, L, u)), L[-1]
    return float(np.interp(s, L, u)), L[-1]


def refine(c, split_fu):
    """The contour with each curve halved and the segments beside each corner split split_fu from it, and which joints are corners

    split_fu is a distance in font units, or a function of a segment and whether the corner is at its end that gives
    one. The pieces next to a corner become curves, so the print look can round the corner or fill it in
    """
    reach = split_fu if callable(split_fu) else lambda P, end: split_fu
    cn = corners(c)
    out, joint = [], []
    for i, (kind, P) in enumerate(c):
        c0, c1 = cn[i], cn[(i + 1) % len(c)]
        cuts = [0.5] if kind == "C" else []
        if (c0 or c1) and arc_param(P, 0, False)[1] > 3 * MIN_SEGMENT:
            length = arc_param(P, 0, False)[1]
            cap = (0.3 if c0 and c1 else 0.45) * length
            s0, s1 = min(reach(P, False), cap), min(reach(P, True), cap)
            cuts = [u for u in cuts if (not c0 or u > arc_param(P, s0, False)[0]) and (not c1 or u < arc_param(P, s1, True)[0])]
            cuts += ([arc_param(P, s0, False)[0]] if c0 else []) + ([arc_param(P, s1, True)[0]] if c1 else [])
        pieces, rest, done = [], P, 0.0
        for u in sorted(cuts):
            a, rest = split(rest, (u - done) / (1 - done))
            pieces.append(a)
            done = u
        pieces.append(rest)
        for j, q in enumerate(pieces):
            near = (j == 0 and c0) or (j == len(pieces) - 1 and c1)
            out.append(("C" if near else kind, q))
            joint.append(c0 and j == 0)
    return closed(out), joint


def corner_reach(pl, look, t, t0=0.0):
    """How far along a segment from its corner the print look at strength t does more than spread the ink, in font units, times 1.3

    Splitting there leaves the whole fill or rounding of the corner to the pieces beside it, however acute the corner
    """
    F = pl.field(t, look, t0)
    plain = -(t - t0) * look["d"]
    least = look["r"] / pl.M[0, 0]

    def reach(P, end):
        u = np.linspace(0, 1, 201)
        pts = bez(P, u[::-1] if end else u)
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        x = pl.grid(pts)
        v = map_coordinates(F, [x[:, 1], x[:, 0]], order=1)
        calm = np.where(np.abs(v - plain) < 1.0)[0]
        return max(least, 1.3 * (L[calm[0]] if len(calm) else L[-1]))

    return reach


class Placed:
    """Outline contours on a grid of their own, placed the way the face's glyphs are placed on their stacks"""

    def __init__(self, contours, p):
        p = np.array(p, float)
        s, ar, th = p[:3]
        self.M = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([s, s * ar])
        self.p, self.T = p, np.zeros(2)
        pts = self.grid(np.vstack([P for c in contours for _, P in c]))
        self.T = MARGIN - pts.min(0)
        self.p[3:5] = self.T
        self.shape = tuple(int(np.ceil(v)) + 2 * MARGIN for v in np.ptp(pts, 0)[::-1])
        self.poly = [np.vstack([bez(self.grid(P), np.linspace(0, 1, 24, endpoint=False)) for _, P in c]) for c in contours]
        self.sd = fontfit.placed_distance(self, self.p, self.shape, fontfit.LOOK_BAND)

    def to_img(self, p):
        return self.poly

    def grid(self, pts):
        return (np.asarray(pts, float) * [1, -1]) @ self.M.T + self.T

    def font(self, q):
        return ((np.asarray(q, float) - self.T) @ np.linalg.inv(self.M).T) * [1, -1]

    def field(self, t, look, t0=0.0):
        """The print look's edge at strength t is where this is 0, negative in the ink, for an outline that already carries strength t0

        Blurs compose in quadrature, so the outline drawn at t0 takes the rest of the blur and of the ink spread
        """
        return fontfit.blur(self.sd, np.sqrt(t ** 2 - t0 ** 2) * look["r"]) - (t - t0) * look["d"]

    def outward(self, x, n, base=None):
        """Normals flipped to point out of the ink of base, the outline's own distance field unless given"""
        base = self.sd if base is None else base
        a = map_coordinates(base, [(x + n)[:, 1], (x + n)[:, 0]], order=1)
        b = map_coordinates(base, [(x - n)[:, 1], (x - n)[:, 0]], order=1)
        return np.where((a >= b)[:, None], n, -n)

    def project(self, F, x, n, flags=False, base=None):
        """Each point moved along its normal to the first place F is 0, outward from a point inside the ink and inward from one outside it

        The direction matters inside a filled corner, where the nearest edge can be the far side of a thin stroke. A
        point with no edge within REACH that way takes the nearest edge, and one with none at all stays where it is.
        With flags, also whether each move stayed clear of the rest of the edge of base, the outline's own distance
        field unless given, which a point in a slit that fills in does not, since its ray crosses the slit's other side
        """
        base = self.sd if base is None else base
        s = np.linspace(-REACH, REACH, int(8 * REACH) + 1)
        mid = len(s) // 2
        rays = x[:, None, :] + s[None, :, None] * n[:, None, :]
        at = [rays[..., 1].ravel(), rays[..., 0].ravel()]
        v = map_coordinates(F, at, order=1, mode="nearest").reshape(rays.shape[:2])
        sd = map_coordinates(base, at, order=1, mode="nearest").reshape(rays.shape[:2])
        out, ok = np.zeros(len(x)), np.ones(len(x), bool)
        clear = int(8 * 1.5)
        for i, vi in enumerate(v):
            j = np.where(np.sign(vi[:-1]) != np.sign(vi[1:]))[0]
            ahead = j[(s[j] >= 0) == (vi[mid] < 0)]
            j = ahead if len(ahead) else j
            if len(j):
                j = j[np.argmin(np.abs(s[j]))]
                out[i] = s[j] + (s[1] - s[0]) * vi[j] / (vi[j] - vi[j + 1])
                lo, hi = sorted((mid, j))
                way = sd[i, lo:hi + 1]
                far = np.abs(np.arange(lo, hi + 1) - mid) > clear
                ok[i] = not (far.any() and np.sign(way[far]).min() != np.sign(way[far]).max())
        moved = x + out[:, None] * n
        return (moved, ok) if flags else moved


def normals(P, u):
    d = dbez(P, u)
    d /= np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
    return np.stack([d[:, 1], -d[:, 0]], 1)


def fit_segment(kind, P, A, B, free0, free3, us, Q, rounds=6):
    """Control points of a cubic from A to B through the points Q, its handles kept in P's directions except at corners

    The fit starts from the parameters us and moves each point's parameter to the nearest point of the last fit
    """
    if kind == "L":
        return line(A, B)
    if len(Q) == 0:
        return np.array([A, A + (P[1] - P[0]), B - (P[3] - P[2]), B])
    dense = np.linspace(0, 1, 201)
    for _ in range(rounds):
        G = fit_once(P, A, B, free0, free3, us, Q)
        us = dense[cKDTree(bez(G, dense)).query(Q)[1]]
        us = np.clip(np.maximum.accumulate(us), 1e-3, 1 - 1e-3)
    return fit_once(P, A, B, free0, free3, us, Q)


def fit_once(P, A, B, free0, free3, us, Q):
    b = [(1 - us) ** 3, 3 * (1 - us) ** 2 * us, 3 * (1 - us) * us ** 2, us ** 3]
    rhs = Q - b[0][:, None] * A - b[3][:, None] * B
    cols, take = [], []
    T0, T3 = P[1] - P[0], P[3] - P[2]
    z = np.zeros_like(us)
    if free0 or np.linalg.norm(T0) < 1e-6:
        cols += [np.concatenate([b[1], z]), np.concatenate([z, b[1]])]
        take.append(lambda v: np.array([v[0], v[1]]))
    else:
        T0 = T0 / np.linalg.norm(T0)
        rhs = rhs - b[1][:, None] * A
        cols.append(np.concatenate([b[1] * T0[0], b[1] * T0[1]]))
        take.append(lambda v: A + v[0] * T0)
    k0 = len(cols)
    if free3 or np.linalg.norm(T3) < 1e-6:
        cols += [np.concatenate([b[2], z]), np.concatenate([z, b[2]])]
        take.append(lambda v: np.array([v[0], v[1]]))
    else:
        T3 = T3 / np.linalg.norm(T3)
        rhs = rhs - b[2][:, None] * B
        cols.append(np.concatenate([-b[2] * T3[0], -b[2] * T3[1]]))
        take.append(lambda v: B - v[0] * T3)
    v = np.linalg.lstsq(np.stack(cols, 1), np.concatenate([rhs[:, 0], rhs[:, 1]]), rcond=None)[0]
    return np.array([A, take[0](v[:k0]), take[1](v[k0:]), B])


class Edge:
    """The print look's edge at one strength as dense polylines, for walking along it between two points"""

    def __init__(self, F):
        self.polys = contours_at(-F, 0.0, min_len=0)
        self.owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(self.polys)]) if self.polys else np.zeros(0, int)
        self.index = np.concatenate([np.arange(len(p)) for p in self.polys]) if self.polys else np.zeros(0, int)
        self.tree = cKDTree(np.vstack(self.polys)) if self.polys else None

    def between(self, A, B, m):
        """m points evenly spaced along the edge from near A to near B by the shorter way round, with their chord parameters, None when A and B are on different polylines"""
        if self.tree is None:
            return None
        (ka, kb) = self.tree.query([A, B])[1]
        if self.owner[ka] != self.owner[kb]:
            return None
        poly = self.polys[self.owner[ka]]
        n, ia, ib = len(poly), self.index[ka], self.index[kb]
        fw = (ib - ia) % n
        idx = (ia + np.arange(fw + 1)) % n if fw <= n - fw else (ia - np.arange(n - fw + 1)) % n
        path = np.vstack([A, poly[idx][1:-1], B]) if len(idx) > 2 else np.vstack([A, B])
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        if L[-1] < 1e-6:
            return np.zeros((0, 2)), np.zeros(0)
        us = np.linspace(0, 1, m + 2)[1:-1]
        Q = np.stack([np.interp(us * L[-1], L, path[:, 0]), np.interp(us * L[-1], L, path[:, 1])], 1)
        return Q, us


def arm(g, i, forward, dist=3.0):
    """The point dist grid px along the outline from joint i, forward or back, or the farthest the contour reaches"""
    J, n = g[i][1][0], len(g)
    u = np.linspace(0, 1, 33)
    for step in range(n):
        P = g[(i + step) % n][1] if forward else g[(i - 1 - step) % n][1]
        pts = bez(P, u if forward else u[::-1])
        far = np.where(np.linalg.norm(pts - J, axis=1) >= dist)[0]
        if len(far):
            return pts[far[0]]
    return pts[-1]


def bisector(g, i, pl, base=None):
    """The direction out of the ink that halves the angle at corner i, from the outline 3 px either side of it"""
    J = g[i][1][0]
    a, b = arm(g, i, True) - J, arm(g, i, False) - J
    v = a / np.linalg.norm(a) + b / np.linalg.norm(b)
    if np.linalg.norm(v) < 1e-3:
        v = np.array([-a[1], a[0]])
    v = v / np.linalg.norm(v)
    return pl.outward(J[None], v[None] * 2, base)[0] / 2


def contained(G, A, B, Q, P):
    """Whether a fitted segment stays near the stretch of edge it was fitted to, which a walk the wrong way round does not"""
    pts = np.vstack([A, B, Q])
    pad = 0.5 * np.ptp(P, 0).max() + 4 + max(np.linalg.norm(A - P[0]), np.linalg.norm(B - P[3]))
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    span = np.linalg.norm(np.diff(np.vstack([A, Q, B]), axis=0), axis=1).sum()
    return bool((G >= lo).all() and (G <= hi).all() and span < 3 * np.linalg.norm(np.diff(P, axis=0), axis=1).sum() + 2 * REACH)


def settle(J2, ok, g, edge):
    """Joints whose move crossed the outline placed along the edge between their nearest good neighbours, spaced as they are on the outline"""
    n = len(J2)
    if ok.all() or not ok.any():
        return J2
    J2 = J2.copy()
    seg = np.array([np.linalg.norm(np.diff(P, axis=0), axis=1).sum() for _, P in g])
    good = np.where(ok)[0]
    for a, b in zip(good, np.roll(good, -1)):
        run = [(a + m) % n for m in range(1, (b - a) % n or n)]
        if not run:
            continue
        walk = edge.between(J2[a], J2[b], 400)
        if walk is None:
            continue
        path = np.vstack([J2[a], walk[0], J2[b]])
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        arc = np.cumsum([seg[(a + m) % n] for m in range(len(run) + 1)])
        if L[-1] > 3 * arc[-1] + 2 * REACH:
            continue
        for m, idx in enumerate(run):
            f = arc[m] / arc[-1]
            J2[idx] = [np.interp(f * L[-1], L, path[:, 0]), np.interp(f * L[-1], L, path[:, 1])]
    return J2


def standins(contours, joints, pl, look, ts, step=0.25, t0=0.0):
    """Refined measuring stick outlines with their points moved onto the print look's edge at each strength of ts, in font units, and the strength they stop at


    The strength rises in steps from t0, the strength the outline already carries, each moving the last step's points a little, so a point is never sent across
    another part of the outline. Each joint moves along its normal, or a corner along its bisector, to the edge, and
    each segment is then fitted to the stretch of edge between its two joints, which keeps the points in order where a
    filled corner swallows them. A step that lands more than MAX_MISFIT from the edge is not taken, and the outline
    keeps the last strength it followed, which is returned, None when it followed every step
    """
    out, cur, base, held = {round(t0, 6): contours}, contours, pl.sd, None
    for t in np.arange(t0 + step, max(ts) + step / 2, step):
        F = pl.field(t, look, t0)
        if held is None:
            nxt = advance(cur, joints, pl, F, base)
            if (misfit_um(nxt, pl, F) or 0) > MAX_MISFIT:
                held = t - step
            else:
                cur = nxt
        out[round(float(t), 6)] = cur
        base = F
    return [out[round(float(t), 6)] for t in ts], held


def advance(contours, joints, pl, F, base):
    """The contours moved from the edge of the field base onto the edge of F"""
    edge = Edge(F)
    us0 = np.linspace(0, 1, SAMPLES + 2)[1:-1]
    out, k = [], 0
    for c in contours:
        g = [(kind, pl.grid(P)) for kind, P in c]
        n = len(g)
        J = np.array([P[0] for _, P in g])
        nin = pl.outward(J, np.vstack([normals(P, [0.0]) for _, P in g]), base)
        nout = pl.outward(J, np.vstack([normals(g[i - 1][1], [1.0]) for i in range(n)]), base)
        nj = nin + nout
        nj /= np.maximum(np.linalg.norm(nj, axis=1), 1e-9)[:, None]
        for i in range(n):
            if joints[k + i]:
                nj[i] = bisector(g, i, pl, base)
        J2, ok = pl.project(F, J, nj, flags=True, base=base)
        J2 = settle(J2, ok, g, edge)
        cc = []
        for i, (kind, P) in enumerate(g):
            A, B = J2[i], J2[(i + 1) % n]
            free = joints[k + i], joints[k + (i + 1) % n]
            x = bez(P, us0)
            ray = pl.project(F, x, pl.outward(x, normals(P, us0), base), base=base), us0
            G = None
            for Q, us in filter(None, (edge.between(A, B, SAMPLES), ray)):
                fitted = fit_segment(kind, P, A, B, *free, us, Q)
                if contained(fitted, A, B, Q, P):
                    G = fitted
                    break
            if G is None:
                G = np.array([A, A + P[1] - P[0], B + P[2] - P[3], B])
            cc.append((kind, pl.font(G)))
        k += n
        out.append(closed(cc))
    return out


def outline_move(at1, att, pl, per=64):
    """The move of the measuring stick outline from master at1 to att at the closest point of at1 to each point, on the grid

    The two masters share their points, so the same curve parameters on both give each place on the outline its move,
    and a point between two samples takes the move interpolated between them, which changes smoothly along the outline
    """
    us = np.linspace(0, 1, per, endpoint=False)
    a = [np.vstack([bez(pl.grid(P), us) for _, P in c]) for c in at1]
    move = np.vstack([np.vstack([bez(pl.grid(P), us) for _, P in c]) - p for c, p in zip(att, a)])
    nxt = np.concatenate([o + (np.arange(len(p)) + 1) % len(p) for o, p in zip(np.cumsum([0] + [len(p) for p in a]), a)])
    prv = np.empty_like(nxt)
    prv[nxt] = np.arange(len(nxt))
    a = np.vstack(a)
    tree = cKDTree(a)

    def at(x):
        i = tree.query(x)[1]
        best, out = np.full(len(x), np.inf), np.zeros((len(x), 2))
        for j0, j1 in ((prv[i], i), (i, nxt[i])):
            e = a[j1] - a[j0]
            f = np.clip(((x - a[j0]) * e).sum(1) / np.maximum((e * e).sum(1), 1e-12), 0, 1)
            d = np.linalg.norm(a[j0] + f[:, None] * e - x, axis=1)
            take = d < best
            best[take], out[take] = d[take], (move[j0] + f[:, None] * (move[j1] - move[j0]))[take]
        return out

    return at


def smooth_along(X, v, sigma):
    """Values at the points of a closed polyline smoothed by a Gaussian of sigma in its own units of length along it"""
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.vstack([X, X[:1]]), axis=0), axis=1))])
    if L[-1] < 1e-9:
        return v
    s = np.arange(0, L[-1], sigma / 8)
    even = np.stack([np.interp(s, L, np.append(v[:, k], v[0, k])) for k in range(v.shape[1])], 1)
    even = gaussian_filter1d(even, 8.0, axis=0, mode="wrap")
    return np.stack([np.interp(L[:-1], np.append(s, L[-1]), np.append(even[:, k], even[0, k])) for k in range(v.shape[1])], 1)


def turn(v, a):
    """A vector turned by a radians"""
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])


def displaced(trace, at1, att, pl):
    """A traced outline moved by the change the print look makes to the measuring stick outline, from its master at1 to att

    Each place on the trace takes the move of the closest place on at1, smoothed along the trace by SMOOTH_PX. Each
    joint moves with it and turns its two handles by the same angle, the turn the moved outline makes there, so a
    smooth joint stays smooth, and each curve's handle lengths are fitted to its moved points
    """
    if att is at1:
        return trace
    at = outline_move(at1, att, pl)
    u = np.linspace(0, 1, MOVE_SAMPLES + 1)
    b = [(1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u ** 2, u ** 3]
    out = []
    for c in trace:
        G = [pl.grid(P) for _, P in c]
        n = len(G)
        X = np.array([bez(P, u) for P in G])
        flat = X[:, :-1].reshape(-1, 2)
        m = smooth_along(flat, at(flat), SMOOTH_PX).reshape(n, MOVE_SAMPLES, 2)
        O = X + np.concatenate([m, np.roll(m[:, :1], -1, axis=0)], 1)
        ang = lambda v: np.arctan2(v[..., 1], v[..., 0])
        # the turn at each joint, the mean of the moved outline's turn just before it and just after it
        before = np.roll(ang(O[:, -1] - O[:, -2]) - ang(X[:, -1] - X[:, -2]), 1)
        after = ang(O[:, 1] - O[:, 0]) - ang(X[:, 1] - X[:, 0])
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
        th = 0.5 * (wrap(before) + wrap(after))
        cc = []
        for i, ((kind, _), P) in enumerate(zip(c, G)):
            A, B = O[i, 0], O[i, -1]
            if kind == "L":
                cc.append((kind, pl.font(line(A, B))))
                continue
            T0, T3 = turn(heading(P, False), th[i]), turn(heading(P, True), th[(i + 1) % n])
            l0, l3 = np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[3] - P[2])
            rhs = O[i, 1:-1] - (b[0] + b[1])[1:-1, None] * A - (b[2] + b[3])[1:-1, None] * B
            M = np.stack([np.concatenate([b[1][1:-1] * T0[0], b[1][1:-1] * T0[1]]), np.concatenate([-b[2][1:-1] * T3[0], -b[2][1:-1] * T3[1]])], 1)
            h0, h3 = np.linalg.lstsq(M, np.concatenate([rhs[:, 0], rhs[:, 1]]), rcond=None)[0]
            # a handle never flips back over its point
            h0, h3 = max(h0, MIN_HANDLE * l0), max(h3, MIN_HANDLE * l3)
            cc.append((kind, pl.font(np.array([A, A + h0 * T0, B - h3 * T3, B]))))
        out.append(closed(cc))
    return out


def dense(P, spacing=0.5):
    """Points along a segment about spacing apart"""
    n = max(8, int(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / spacing))
    return bez(P, np.linspace(0, 1, n, endpoint=False))


def misfit_um(contours, pl, F):
    """RMS distance in µm from a built master to the edge of the field it was fitted to"""
    poly = [np.vstack([dense(pl.grid(P)) for _, P in c]) for c in contours]
    edge = contours_at(-F, 0.0)
    if not edge:
        return None
    return float(np.sqrt((sym_dev(poly, edge) ** 2).mean()) * UM_PER_PX)


def crossing(A, B, C, D):
    """Where segment AB crosses segment CD, None when they don't"""
    r, q = B - A, D - C
    den = r[0] * q[1] - r[1] * q[0]
    if abs(den) < 1e-12:
        return None
    t = ((C - A)[0] * q[1] - (C - A)[1] * q[0]) / den
    u = ((C - A)[0] * r[1] - (C - A)[1] * r[0]) / den
    return A + t * r if 0 <= t <= 1 and 0 <= u <= 1 else None


def unloop(pts, inner, reach=12):
    """Control points with each loop a grown inner corner makes, where the sides before it cross the sides after it, closed up onto the crossing"""
    n = len(pts)
    reach = min(reach, n // 2 - 1)
    for i in np.where(inner)[0]:
        best = None
        for jb in range(i - 2, i - reach - 1, -1):
            for ja in range(i + 1, i + reach):
                X = crossing(pts[jb % n], pts[(jb + 1) % n], pts[ja % n], pts[(ja + 1) % n])
                if X is not None and (best is None or ja - jb < best[1] - best[0]):
                    best = (jb, ja, X)
        if best:
            # the points close up along the two sides either side of the crossing, a fraction of a unit apart, so no segment has no length
            jb, ja, X = best
            ub = pts[(jb + 1) % n] - pts[jb % n]
            ua = pts[(ja + 1) % n] - pts[ja % n]
            ub, ua = ub / max(np.linalg.norm(ub), 1e-9), ua / max(np.linalg.norm(ua), 1e-9)
            k = ja - jb
            for m, j in enumerate(range(jb + 1, ja + 1)):
                s = (m + 1 - (k + 1) / 2) * UNLOOP_GAP
                pts[j % n] = X + s * (ua if s > 0 else ub)
    return pts


def grow(contours, a, b):
    """Contours with every point, handles included, moved out of the ink by a across and b up and down, mitred at corners

    Each point moves along the mitre of the control polygon's edges either side of it, worked out in a space scaled
    by 1/a across and 1/b up and down so the offset is the same each way, then scaled back. An outer corner's mitre
    longer than MITRE times the offset is cut to that, which keeps sharp points from shooting out. Where the grown
    sides of an inner corner cross, the points between close up onto the crossing, as the ink fills the notch
    """
    area = lambda c: sum(P[0, 0] * P[3, 1] - P[3, 0] * P[0, 1] for _, P in c)
    # the contour with the largest area is an outer one, and out of the ink is to its left when it runs clockwise
    side = np.sign(area(max(contours, key=lambda c: abs(area(c))))) or -1.0
    out = []
    for c in contours:
        pts = np.vstack([P[:3] for _, P in c]) / [a, b]
        n = len(pts)
        moved, inner = np.empty_like(pts), np.zeros(n, bool)
        for i in range(n):
            back = next((pts[i] - pts[(i - j) % n] for j in range(1, n) if np.linalg.norm(pts[i] - pts[(i - j) % n]) > 1e-6), None)
            ahead = next((pts[(i + j) % n] - pts[i] for j in range(1, n) if np.linalg.norm(pts[(i + j) % n] - pts[i]) > 1e-6), None)
            if back is None:
                moved[i] = pts[i]
                continue
            n1, n2 = (np.array([v[1], -v[0]]) * side / np.linalg.norm(v) for v in (back, ahead))
            m = (n1 + n2) / max(1 + n1 @ n2, 1e-9)
            u, v = back * [a, b], ahead * [a, b]
            inner[i] = ahead @ n1 > 1e-6 and np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))) > CORNER_DEG
            # an inner corner moves to where its grown sides meet, up to 4 times the offset across
            far = np.linalg.norm(m * [a, b]) / (4 * a) if inner[i] else np.linalg.norm(m) / MITRE
            moved[i] = pts[i] + m / max(1.0, far)
        moved = unloop(moved * [a, b], inner)
        segs = [np.vstack([moved[3 * k:3 * k + 3], moved[(3 * k + 3) % n]]) for k in range(len(c))]
        out.append(closed([(kind, S if kind == "C" else line(S[0], S[3])) for (kind, _), S in zip(c, segs)]))
    return out


def zones(font, k):
    """The heights the bold keeps, descender, baseline, x-height and cap height, in measuring stick units"""
    os2 = font["OS/2"]
    return [os2.sTypoDescender / k, 0.0, os2.sxHeight / k, os2.sCapHeight / k]


def hold_heights(contours, heights):
    """Contours grown BOLD["y"] up and down moved back so their tops and bottoms sit on the heights they grew from,
    and moved BOLD["x"] right so the left side bearing stays"""
    b = BOLD["y"]
    lo, hi = heights[:2], heights[2:]
    grown = [h - b for h in lo] + [h + b for h in hi]

    def move(P):
        y = np.interp(P[:, 1], grown, heights, left=np.nan, right=np.nan)
        y = np.where(P[:, 1] < grown[0], P[:, 1] + b, np.where(P[:, 1] > grown[-1], P[:, 1] - b, y))
        return np.stack([P[:, 0] + BOLD["x"], y], 1)

    return [closed([(kind, move(P)) for kind, P in c]) for c in contours]


def embolden(contours, heights):
    """The bold of a regular outline, grown by BOLD with its heights held, with the same points"""
    return hold_heights(grow(contours, BOLD["x"], BOLD["y"]), heights)


def sources(weight=False):
    """The location of each master, Bleed at every weight"""
    weights = (WEIGHT["minimum"], WEIGHT["maximum"]) if weight else (None,)
    return [{AXIS["name"]: m} | ({WEIGHT["name"]: w} if w else {}) for w in weights for m in MASTERS]


def quadratic(masters):
    """TrueType glyphs of one glyph's masters, each curve converted to quadratics with the same points in every master"""
    pens = [TTGlyphPen(None) for _ in masters]
    for ci in range(len(masters[0])):
        for pen, m in zip(pens, masters):
            pen.moveTo(tuple(m[ci][0][1][0] * SCALE))
        for si, (kind, _) in enumerate(masters[0][ci]):
            if kind == "L":
                for pen, m in zip(pens, masters):
                    pen.lineTo(tuple(m[ci][si][1][3] * SCALE))
                continue
            splines = curves_to_quadratic([[tuple(q) for q in m[ci][si][1] * SCALE] for m in masters], [MAX_ERR] * len(masters))
            for pen, sp in zip(pens, splines):
                pen.qCurveTo(*sp[1:])
        for pen in pens:
            pen.closePath()
    return [pen.glyph() for pen in pens]


class AnyGlyph:
    """Stands in for the glyph set a composite's pen checks its components against, since they are built alongside it"""

    def __contains__(self, name):
        return True


def composite(parts):
    pen = TTGlyphPen(AnyGlyph())
    for name, (dx, dy) in parts:
        pen.addComponent(name, (1, 0, 0, 1, round(dx * SCALE), round(dy * SCALE)))
    return pen.glyph()


def char_name(ch):
    return styles.NAMES.get(ch) or UV2AGL.get(ord(ch), f"uni{ord(ch):04X}")


def base_bleed(font):
    """The Bleed a source font was saved at and its units per measuring stick unit, 0 and 1 for the measuring stick itself"""
    family = font["name"].getDebugName(1) if "name" in font else None
    m = re.search(r"Bleed (\d+(?:\.\d+)?)$", family or "")
    return (float(m.group(1)) / 100, SCALE) if m else (0.0, 1)


def face_glyphs(font, traces, face, chars, weight=False):
    """{glyph name: [glyph of each master]}, the cmap and the advances of each master of one face, with a summary of how each glyph was made

    The source is the measuring stick or a static font this module saved at a Bleed of its own, whose outlines already
    carry that much of the print look, so each glyph's masters start from it and take only the rest. With weight the
    masters of `sources` come at the regular weight and again at the bold, whose source outline is the regular's grown
    by `embolden` and drawn with the print look the same way, and whose traced glyphs move by the change the bold
    makes to the source outline at each Bleed, which keeps the printed shape's own departures from the measuring stick
    """
    t0, k = base_bleed(font)
    looks = [e["look"] for k, e in traces.items() if styles.style_of(k) == face and e.get("built") == "outline"]
    if not looks:
        raise SystemExit(f"the {face} face has no glyph built with its print look in traces.json")
    look = looks[0]
    shared = next(e["p"] for k, e in traces.items() if styles.style_of(k) == face and "p" in e)
    ts = [m / 100 for m in MASTERS]
    if t0 > min(ts):
        raise SystemExit(f"the {face} source is saved at Bleed {t0 * 100:g}, past the lowest master {MASTERS[0]}")
    want, cmap = {}, {}
    taken = {char_name(ch) for ch in chars}
    weights = 2 if weight else 1
    heights = zones(font, k)
    wider = lambda w, extra=2 * BOLD["x"]: [w] * len(ts) + [w + extra] * len(ts) * (weights - 1)

    def need(ch, name=None):
        """The glyph name of a character or a named glyph of the measuring stick, queued for building"""
        gname = char_name(ch) if ch else (name if name not in taken else name + ".part")
        if gname not in want:
            want[gname] = (ch, name)
        return gname

    for ch in chars:
        if styles.key(face, ch) in traces or (not ch.strip() and ord(ch) in font.getBestCmap()):
            cmap[ord(ch)] = need(ch)
    glyphs, adv, made = {}, {}, {}
    todo = list(want)
    while todo:
        gname = todo.pop(0)
        if gname in glyphs:
            continue
        ch, name = want[gname]
        e = traces.get(styles.key(face, ch)) if ch else None
        if e is None and ch and ord(ch) in font.getBestCmap() and not font["glyf"][font.getBestCmap()[ord(ch)]].numberOfContours:
            glyphs[gname] = [TTGlyphPen(None).glyph()] * len(ts) * weights
            adv[gname], made[gname] = wider(font["hmtx"][font.getBestCmap()[ord(ch)]][0] / k, 0), "empty"
            continue
        if e and e.get("parts"):
            parts = [(need(p["char"], p["glyph"]), p["offset"]) for p in e["parts"]]
            todo += [n for n, _ in parts]
            glyphs[gname] = [composite(parts)] * len(ts) * weights
            adv[gname], made[gname] = wider(e["adv"]), e["built"]
            continue
        contours = design(font, ch or "", None if ch else name, k, SplinePen if t0 else SegmentPen)
        if not contours:
            continue
        contours = [merge_tiny(c, tip_fu=TIP_PX / shared[0]) for c in contours]
        pl = Placed(contours, shared)
        if t0:
            # a saved font's outlines were refined when it was built
            rc, joints = contours, [j for c in contours for j in corners(c)]
        else:
            ref = [refine(c, corner_reach(pl, look, ts[-1])) for c in contours]
            rc, joints = [r[0] for r in ref], [j for r in ref for j in r[1]]
        masters, held = standins(rc, joints, pl, look, ts, t0=t0)
        if weight:
            bold = embolden(rc, heights)
            more, held_bold = standins(bold, joints, Placed(bold, shared), look, ts, t0=t0)
            if held_bold is not None:
                log.log.warning("%s %s bold stops bleeding at %d, past it the print look closes a gap its points cannot follow", face, gname, round(held_bold * 100))
        made[gname] = "outline"
        if e and not e.get("built"):
            trace = parse_trace(e["trace"])
            at1 = masters[ts.index(1.0)]
            masters = [displaced(trace, at1, m, pl) for m in masters + (more if weight else [])]
            made[gname] = "prior" if e.get("prior") else "traced"
        elif weight:
            masters = masters + more
        if held is not None:
            log.log.warning("%s %s stops bleeding at %d, past it the print look closes a gap its points cannot follow", face, gname, round(held * 100))
        glyphs[gname] = quadratic(masters)
        adv[gname] = wider(e["adv"] if e else font["hmtx"][name if name in font["hmtx"].metrics else name + ".part" if name else font.getBestCmap()[ord(ch)]][0] / k)
    return glyphs, cmap, adv, made


def master_font(font, face, glyphs, cmap, adv, i, family, style):
    order = [".notdef"] + sorted(glyphs)
    t0, k = base_bleed(font)
    fb = FontBuilder(round(font["head"].unitsPerEm * SCALE / k), isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph()} | {n: g[i] for n, g in glyphs.items()})
    glyf = fb.font["glyf"]
    metrics = {}
    for n in order:
        g = glyf[n]
        g.recalcBounds(glyf)
        metrics[n] = (round(adv[n][i] * SCALE if n in adv else font["hmtx"][".notdef"][0] / k * SCALE), getattr(g, "xMin", 0))
    fb.setupHorizontalMetrics(metrics)
    os2, hhea = font["OS/2"], font["hhea"]
    m = lambda v: round(v * SCALE / k)
    fb.setupHorizontalHeader(ascent=m(hhea.ascent), descent=m(hhea.descent), lineGap=m(hhea.lineGap))
    italic = face != styles.primary()
    fb.setupNameTable(dict(familyName=family, styleName=style, version="Version 0.100", psName=f"{family.replace(' ', '')}-{style}"))
    fb.setupOS2(version=4, sTypoAscender=m(os2.sTypoAscender), sTypoDescender=m(os2.sTypoDescender), sTypoLineGap=m(os2.sTypoLineGap),
                usWinAscent=m(os2.usWinAscent), usWinDescent=m(os2.usWinDescent), sxHeight=m(os2.sxHeight),
                sCapHeight=m(os2.sCapHeight), fsSelection=(1 << 0 if italic else 1 << 6) | (1 << 7), achVendID="NONE")
    fb.setupPost(italicAngle=font["post"].italicAngle)
    fb.font["head"].macStyle = 2 if italic else 0
    return fb.font


def kern_pairs(font):
    """{(first glyph, second glyph): x advance adjustment} of a font's kern feature, the first subtable to kern a pair winning as it does in shaping"""
    out = {}
    if "GPOS" not in font:
        return out
    gpos = font["GPOS"].table
    lookups = sorted({i for fr in gpos.FeatureList.FeatureRecord if fr.FeatureTag == "kern" for i in fr.Feature.LookupListIndex})
    x = lambda v: getattr(v, "XAdvance", 0) or 0 if v else 0
    for li in lookups:
        for st in gpos.LookupList.Lookup[li].SubTable:
            st = getattr(st, "ExtSubTable", None) or st
            if st.Format == 1:
                for a, ps in zip(st.Coverage.glyphs, st.PairSet):
                    for r in ps.PairValueRecord:
                        out.setdefault((a, r.SecondGlyph), x(r.Value1))
            else:
                second = {}
                for g, c in st.ClassDef2.classDefs.items():
                    second.setdefault(c, []).append(g)
                for a in st.Coverage.glyphs:
                    for c, r in enumerate(st.Class1Record[st.ClassDef1.classDefs.get(a, 0)].Class2Record):
                        for b in second.get(c, []) if x(r.Value1) else []:
                            out.setdefault((a, b), x(r.Value1))
    return {k: v for k, v in out.items() if v}


def add_kerning(font, source, factor):
    """Adds the source font's kerning between the characters both fonts draw to font, each value times factor, and returns how many pairs it kerns"""
    chars = {g: c for c, g in source.getBestCmap().items()}
    names = font.getBestCmap()
    pairs = {}
    for (a, b), v in kern_pairs(source).items():
        if a in chars and b in chars and chars[a] in names and chars[b] in names:
            pairs[(names[chars[a]], names[chars[b]])] = round(v * factor)
    pairs = {k: v for k, v in pairs.items() if v}
    if pairs:
        lines = "".join(f"  pos \\{a} \\{b} {v};\n" for (a, b), v in sorted(pairs.items()))
        addOpenTypeFeaturesFromString(font, f"languagesystem DFLT dflt;\nlanguagesystem latn dflt;\nfeature kern {{\n{lines}}} kern;\n")
    return len(pairs)


def build_face(font, traces, face, family, style, chars, out, weight=False):
    """Writes one face's variable font, with a Weight axis when weight is set, and returns its path and how each glyph was made"""
    glyphs, cmap, adv, made = face_glyphs(font, traces, face, chars, weight)
    axes = [AXIS] + ([WEIGHT] if weight else [])
    ds = DesignSpaceDocument()
    for ax in axes:
        ds.addAxis(AxisDescriptor(**ax))
    for i, loc in enumerate(sources(weight)):
        src = SourceDescriptor()
        src.font = master_font(font, face, glyphs, cmap, adv, i, family, style)
        src.location, src.name = loc, f"{style} " + " ".join(f"{k.lower()} {v}" for k, v in loc.items())
        ds.addSource(src)
    inst = InstanceDescriptor()
    inst.familyName, inst.styleName, inst.location = family, style, {ax["name"]: ax["default"] for ax in axes}
    ds.addInstance(inst)
    vf, _, _ = varLib.build(ds)
    italic = face != styles.primary()
    stat = [dict(tag=AXIS["tag"], name=AXIS["name"], values=[dict(value=AXIS["default"], name="Regular", flags=0x2)])]
    if weight:
        stat.append(dict(tag=WEIGHT["tag"], name=WEIGHT["name"], values=[dict(value=400, name="Regular", flags=0x2) | (dict(linkedValue=700) if WEIGHT["maximum"] == 700 else {}), dict(value=WEIGHT["maximum"], name=WEIGHT_NAMES.get(WEIGHT["maximum"], str(WEIGHT["maximum"])))]))
    buildStatTable(vf, stat + [dict(tag="ital", name="Italic", values=[dict(value=1, name="Italic")] if italic else [dict(value=0, name="Roman", flags=0x2, linkedValue=1)])])
    log.log.info("%s %s kerns %d pairs", family, style, add_kerning(vf, font, SCALE / base_bleed(font)[1]))
    path = out / (family.replace(" ", "") + ("-Italic" if italic else "") + f"[{','.join(ax['tag'] for ax in axes)}].ttf")
    vf.save(path)
    return path, made


def static(path, bleed, out=None):
    """Saves a variable font fixed at one Bleed as a static font in out, beside it by default, its family named for the strength"""
    font = instantiateVariableFont(TTFont(path), {AXIS["tag"]: bleed})
    name = font["name"]
    family, style = name.getDebugName(1) + f" Bleed {bleed:g}", name.getDebugName(2)
    if "STAT" in font:
        del font["STAT"]
    ps = family.replace(" ", "") + "-" + style.replace(" ", "")
    for nid, value in {1: family, 2: style, 3: f"{name.getDebugName(5)};{ps}", 4: f"{family} {style}", 6: ps}.items():
        name.setName(value, nid, 3, 1, 0x409)
    name.removeNames(nameID=16)
    name.removeNames(nameID=17)
    out = Path(out or Path(path).parent) / (ps + ".ttf")
    font.save(out)
    return out


def load_session(session, profile=None):
    """The profile, traces and GF Latin Core characters of a session"""
    man = json.loads((session / "manifest.json").read_text())
    rec = man.get("profile", {})
    prof = target.use(profile or (rec["path"] if rec and Path(rec["path"]).exists() else rec.get("name", target.DEFAULT)))
    traces = json.loads((session / "traces.json").read_text())
    chars = [chr(c) for c in json.loads(styles.LATIN_CORE.read_text())["codepoints"]]
    return prof, traces, chars


def style_of(prof, face):
    return "Regular" if face == prof.primary.name else face.title()


def report(path, made, chars):
    counts = {k: list(made.values()).count(k) for k in sorted(set(made.values()))}
    missing = [c for c in chars if unicodedata.category(c) != "Zs" and char_name(c) not in made]
    log.announce("%s: %s, %s, not built: %s", path, len(made), counts, "".join(missing) or "none")


def baseline_paths(a, prof):
    """{face name: synthetic baseline path} from --baseline, --italic-baseline and --face-baseline"""
    out = {prof.primary.name: a.baseline}
    for pair in list(a.face_baseline) + ([f"italic={a.italic_baseline}"] if a.italic_baseline else []):
        name, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"--face-baseline takes NAME=PATH, got {pair!r}")
        prof.face(name)
        out[name] = path
    return out


def publish(paths, site):
    """Writes each face's font as WOFF into site with nomicon-rules.json, which lists the files, the characters each draws and the axes"""
    site.mkdir(parents=True, exist_ok=True)
    index = dict(faces={}, chars={}, axes={})
    for face, path in paths.items():
        font = TTFont(path)
        name = f"nomicon-rules-{face}.woff"
        font.flavor = "woff"
        font.save(site / name)
        index["faces"][face] = name
        index["chars"][face] = sorted(font.getBestCmap())
        index["axes"] = {a.axisTag: [a.minValue, a.defaultValue, a.maxValue] for a in font["fvar"].axes}
    (site / "nomicon-rules.json").write_text(json.dumps(index, separators=(",", ":")) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="folder written by glyphstack.session")
    ap.add_argument("-o", "--out", default="build/fonts")
    ap.add_argument("--site", help="also write the fonts as WOFF into this folder of the site, such as site/public/fonts")
    ap.add_argument("--profile", help="target profile, a name in glyphstack/profiles or a path to a TOML file")
    ap.add_argument("--baseline", required=True, help="the primary face's synthetic font baseline, from glyphstack.baseline")
    ap.add_argument("--italic-baseline", help="the italic face's synthetic font baseline")
    ap.add_argument("--face-baseline", action="append", default=[], metavar="NAME=PATH", help="another face's synthetic font baseline, repeat for each face")
    ap.add_argument("--weight", type=int, nargs="?", const=600, help="add a Weight axis from 400 to a top master grown from the regular, 600 when no weight is given")
    log.add_args(ap)
    a = ap.parse_args()
    session, out = Path(a.session), Path(a.out)
    log.start(a, session, "fontbuild")
    prof, traces, chars = load_session(session, a.profile)
    out.mkdir(parents=True, exist_ok=True)
    if a.weight:
        set_top_weight(a.weight)
    built = {}
    for face, path in baseline_paths(a, prof).items():
        font = fnt.load_font(path)
        if not base_bleed(font)[0]:
            raise SystemExit(f"{path} is not a synthetic font baseline, make one with glyphstack.baseline")
        built[face], made = build_face(font, traces, face, prof.primary.full_name, style_of(prof, face), chars, out, bool(a.weight))
        report(built[face], made, chars)
    if a.site:
        publish(built, Path(a.site))


if __name__ == "__main__":
    main()
