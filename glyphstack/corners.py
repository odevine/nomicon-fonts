"""How far each face's stacks sit from the measuring stick at its corners, the fill of inner corners and the rounding of outer ones, by corner angle

A corner is a joint of the measuring stick's outline where it turns by more than fontbuild.CORNER_DEG. From each
corner a ray runs along the bisector, out of the ink, and the edge offset is where the stack's trace crosses it less
where the sharp outline spread by the face's ink spread crosses it, in µm, positive with the ink further out. An inner
corner that fills has a positive offset and an outer corner that rounds a negative one. The same offset to the print
look's edge shows what the look leaves over. Only glyphs traced from their own copies count, since a glyph blended with
the measuring stick is pulled toward it. Each close up in FIGURES also keeps the stretch of the measuring stick's outline
inside its square, a few millimetres of one letter at the printed size, so the page can show the sharp corner it measures from

    python -m glyphstack.corners build/session --font REGULAR.ttf --italic-font ITALIC.ttf -o site/data
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from . import fontbuild as fb
from . import fontfit, log, styles, target
from . import font as fnt
from .contours import UM_PER_PX, UP

# a glyph needs this many instances to count, the same as a glyph traced without the prior
MIN_INSTANCES = 12
# corner angles in degrees, the opening of the outer corner's ink or of the inner corner's white wedge
BINS = (0, 60, 90, 120, 160)
# an offset further than this in µm is a ray that crossed another stroke, not the corner's edge
MAX_UM = 150.0
# places drawn close up on the Bleed page in the measuring stick's units, each with the quadrant of the square its corner sits in
FIGURES = {"regular": [("e", (118, 277), "top left"), ("e", (427, 108), "top right")]}
# how far the square's centre moves from the corner, as (x, y) in half widths with y up, so the corner lands mid quadrant
QUADRANTS = {"top left": (0.5, -0.5), "top right": (-0.5, -0.5), "bottom left": (0.5, 0.5), "bottom right": (-0.5, 0.5)}
# half the width of a close up in the measuring stick's units, under half a millimetre at the printed size
FIGURE_HALF = 30


class Outline:
    """Contours on the grid of a Placed measuring stick outline, with their own distance field"""

    def __init__(self, contours, pl):
        self.M, self.T, self.p, self.shape = pl.M, pl.T, pl.p, pl.shape
        self.poly = [np.vstack([fb.bez(pl.grid(P), np.linspace(0, 1, 24, endpoint=False)) for _, P in c]) for c in contours]
        self.sd = fontfit.placed_distance(self, self.p, self.shape, fontfit.LOOK_BAND)

    def to_img(self, p):
        return self.poly


def crossing(F, x, u):
    """Where F crosses 0 along the ray x + s u nearest s = 0, in grid px, None if it doesn't within fb.REACH"""
    s = np.linspace(-fb.REACH, fb.REACH, int(8 * fb.REACH) + 1)
    pts = x + s[:, None] * u
    v = map_coordinates(F, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
    j = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    if not len(j):
        return None
    j = j[np.argmin(np.abs(s[j]))]
    return float(s[j] + (s[1] - s[0]) * v[j] / (v[j] - v[j + 1]))


def glyph_corners(contours, trace, pl, look):
    """Each corner of the outline as (kind, angle, offset to the stack, offset of the stack from the print look, x, y) in degrees, µm and font units"""
    stack = Outline(trace, pl).sd
    spread, printed = pl.sd - look["d"], pl.field(1.0, look)
    out = []
    for c in contours:
        flags = fb.corners(c)
        for i, (_, P) in enumerate(c):
            if not flags[i]:
                continue
            t_in, t_out = fb.heading(c[i - 1][1], True), fb.heading(P, False)
            angle = float(np.degrees(np.arccos(np.clip(-t_in @ t_out, -1, 1))))
            if angle >= BINS[-1]:
                continue
            g = lambda v: (v * [1, -1]) @ pl.M.T
            b = g(t_out) - g(t_in)
            if np.linalg.norm(b) < 1e-9:
                continue
            b /= np.linalg.norm(b)
            x = pl.grid(P[:1])[0]
            # the bisector points into the wedge between the two edges, ink at an outer corner and white at an inner one
            inner = map_coordinates(pl.sd, [[x[1] + 2 * b[1]], [x[0] + 2 * b[0]]], order=1)[0] > 0
            u = b if inner else -b
            s = [crossing(F, x, u) for F in (stack, spread, printed)]
            if None in s:
                continue
            off, left = (s[0] - s[1]) * UM_PER_PX / UP, (s[0] - s[2]) * UM_PER_PX / UP
            if abs(off) <= MAX_UM:
                out.append(("inner" if inner else "outer", angle, off, left, float(P[0, 0]), float(P[0, 1])))
    return out


def summary(rows):
    """Median offsets and counts by kind and angle bin"""
    out = []
    for kind in ("inner", "outer"):
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            sel = [r for r in rows if r[0] == kind and lo <= r[1] < hi]
            if sel:
                out.append(dict(kind=kind, lo=lo, hi=hi, n=len(sel), stack_um=round(float(np.median([r[2] for r in sel])), 1),
                                left_um=round(float(np.median([r[3] for r in sel])), 1)))
    return out


def snippet(contours, cx, cy, half):
    """The stretch of an outline inside the square of half width half around (cx, cy), as an open SVG path in font units y down"""
    out = []
    for c in contours:
        pts = np.vstack([fb.dense(P, 0.5) for _, P in c])
        inside = (np.abs(pts[:, 0] - cx) <= half) & (np.abs(pts[:, 1] - cy) <= half)
        # start the walk outside the square so a run never wraps around the contour's start
        if inside.all() or not inside.any():
            continue
        k = int(np.argmin(inside))
        pts, inside = np.roll(pts, -k, 0), np.roll(inside, -k)
        run = []
        for q, keep in zip(pts, inside):
            if keep:
                run.append(q)
            elif run:
                out.append(run)
                run = []
        if run:
            out.append(run)
    return "".join("M" + "L".join(f"{x:.1f},{-y:.1f}" for x, y in r[::2] + [r[-1]]) for r in out if len(r) > 1)


def figure(font, traces, face, look, shared, ch, at, quadrant, half=FIGURE_HALF):
    """A close up of one place on a traced glyph, with the corner at at in the given quadrant: the corners measured there and the measuring stick's outline inside the square"""
    k = styles.key(face, ch)
    cs = [fb.merge_tiny(c, tip_fu=fb.TIP_PX / shared[0]) for c in fb.design(font, ch)]
    found = glyph_corners(cs, fb.parse_trace(traces[k]["trace"]), fb.Placed(cs, shared), look)
    dx, dy = QUADRANTS[quadrant]
    cx, cy = round(at[0] + dx * half), round(at[1] + dy * half)
    near = [r for r in found if abs(r[4] - cx) <= half and abs(r[5] - cy) <= half]
    return dict(char=ch, glyph=styles.glyph_name(k), x=cx, y=cy, half=half, stick=snippet(fb.design(font, ch), cx, cy, half),
                corners=[dict(kind=r[0], angle=round(r[1]), stack_um=round(r[2], 1)) for r in near])


def face_corners(font, traces, face):
    """The corner summary of one face, the glyphs it was measured on and its close ups"""
    entries = {k: e for k, e in traces.items() if styles.style_of(k) == face}
    look = next(e["look"] for e in entries.values() if e.get("built") == "outline")
    shared = next(e["p"] for e in entries.values() if "p" in e)
    rows, used = [], 0
    for k, e in entries.items():
        if e.get("built") or e.get("prior") or not e.get("trace") or e.get("n", 0) < MIN_INSTANCES:
            continue
        cs = fb.design(font, styles.char_of(k))
        if not cs:
            continue
        cs = [fb.merge_tiny(c, tip_fu=fb.TIP_PX / shared[0]) for c in cs]
        rows += glyph_corners(cs, fb.parse_trace(e["trace"]), fb.Placed(cs, shared), look)
        used += 1
    figures = [figure(font, traces, face, look, shared, ch, at, q) for ch, at, q in FIGURES.get(face, [])]
    return dict(glyphs=used, corners=len(rows), bins=summary(rows), figures=figures)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="folder written by glyphstack.session")
    ap.add_argument("-o", "--out", default="site/data")
    target.add_args(ap)
    log.add_args(ap)
    a = ap.parse_args()
    session, out = Path(a.session), Path(a.out)
    log.start(a, session, "corners")
    prof, traces, _ = fb.load_session(session, a.profile)
    doc = {face: face_corners(fnt.load_font(path), traces, face) for face, path in target.font_paths(a).items()}
    (out / "corners.json").write_text(json.dumps(doc, indent=1) + "\n")
    for face, d in doc.items():
        log.announce("%s: %d corners on %d glyphs", face, d["corners"], d["glyphs"])


if __name__ == "__main__":
    main()
