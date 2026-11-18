"""How much more an outline wiggles at one Bleed than at another, glyph by glyph, from a variable font

An outline turns through much the same angles at every Bleed when the print look only rounds its corners and moves
its edges, so a dent shows up as turning the outline does at 0 and not at 75. Two numbers measure it, both read
along the outline every SPACING units. The first is the outline's absolute turning in degrees, each lobe of one sign
counted as its net turn so that jitter under LOBE_DEG cancels. The second is the count of inflections, where the
curvature changes sign between lobes that turn more than LOBE_DEG

    python -m glyphstack.dents build/fonts/NomiconRules[BLED].ttf --at 0 --against 75
"""
import argparse

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

# points each curve is drawn with
STEPS = 24
# how far apart in font units of a 4000 unit em an outline is read, about 6 µm at the printed size, coarser than the
# unit a TrueType point is rounded to, so a handle shorter than a unit cannot kink it
SPACING = 8.0
# the least turning in degrees either side of an inflection for it to count, more than a quadratic's rounding to whole units makes
LOBE_DEG = 1.0


class PolyPen(BasePen):
    """Records an outline as closed polylines, each curve drawn with STEPS points"""

    def __init__(self, glyphset=None):
        super().__init__(glyphset)
        self.polys, self.cur = [], None

    def _moveTo(self, p):
        self.cur = [np.array(p, float)]

    def _lineTo(self, p):
        self.cur.append(np.array(p, float))

    def _qCurveToOne(self, a, b):
        p0, a, b = self.cur[-1], np.array(a, float), np.array(b, float)
        u = np.linspace(0, 1, STEPS + 1)[1:, None]
        self.cur += list((1 - u) ** 2 * p0 + 2 * (1 - u) * u * a + u ** 2 * b)

    def _curveToOne(self, a, b, c):
        p0, a, b, c = self.cur[-1], *(np.array(q, float) for q in (a, b, c))
        u = np.linspace(0, 1, STEPS + 1)[1:, None]
        self.cur += list((1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * a + 3 * (1 - u) * u ** 2 * b + u ** 3 * c)

    def _closePath(self):
        P = np.array(self.cur)
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
        P = P[keep]
        if len(P) > 1 and np.linalg.norm(P[-1] - P[0]) < 1e-9:
            P = P[:-1]
        if len(P) > 2:
            self.polys.append(P)
        self.cur = None

    _endPath = _closePath


def resample(P, spacing):
    """A closed polyline's points spacing apart along it"""
    Q = np.vstack([P, P[:1]])
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    s = np.linspace(0, L[-1], max(8, int(round(L[-1] / spacing))), endpoint=False)
    return np.stack([np.interp(s, L, Q[:, 0]), np.interp(s, L, Q[:, 1])], 1)


def turns(P):
    """The signed turning in degrees at each point of a closed polyline"""
    d = np.diff(np.vstack([P, P[:1]]), axis=0)
    a = np.arctan2(d[:, 1], d[:, 0])
    return np.degrees((a - np.roll(a, 1) + np.pi) % (2 * np.pi) - np.pi)


def lobes(t):
    """The turning of each run of one sign around a closed polyline, the zero turns along straight stretches joining the run they sit in"""
    s = np.sign(t)
    nz = np.where(s != 0)[0]
    if not len(nz):
        return np.zeros(0)
    t = np.roll(t, -nz[0])
    s = np.sign(t)
    out, cur, sign = [], 0.0, s[0]
    for v, sv in zip(t, s):
        if sv and sv != sign:
            out.append(cur)
            cur, sign = 0.0, sv
        cur += v
    out.append(cur)
    if len(out) > 1 and np.sign(out[0]) == np.sign(out[-1]):
        out[0] += out.pop()
    return np.array(out)


def merged(t, least=None):
    """The turning of each lobe of a closed polyline, a lobe that turns less than least folded with its two neighbours into one"""
    least = LOBE_DEG if least is None else least
    lb = list(lobes(t))
    while len(lb) > 2:
        i = int(np.argmin(np.abs(lb)))
        if abs(lb[i]) >= least:
            break
        n = len(lb)
        lb[(i - 1) % n] += lb[i] + lb[(i + 1) % n]
        for j in sorted({i, (i + 1) % n}, reverse=True):
            del lb[j]
    if len(lb) == 2 and min(abs(v) for v in lb) < least:
        lb = [sum(lb)]
    return np.array(lb)


def inflections(t, least=None):
    """How many times a closed polyline's curvature changes sign between lobes that each turn at least least"""
    n = len(merged(t, least))
    return n if n > 1 else 0


def measure(polys, spacing=SPACING):
    """Absolute turning in degrees of a glyph's polylines read spacing apart, counting each lobe as its net turning, and their inflections"""
    lbs = [merged(turns(resample(P, spacing))) for P in polys]
    return float(sum(np.abs(lb).sum() for lb in lbs)), int(sum(len(lb) if len(lb) > 1 else 0 for lb in lbs))


def outlines(font, bleed, names=None):
    """{glyph name: polylines} of the simple glyphs of a variable font at one Bleed"""
    inst = instantiateVariableFont(TTFont(font) if isinstance(font, str) else font, {"BLED": bleed})
    gs, glyf = inst.getGlyphSet(), inst["glyf"]
    out = {}
    for n in names or inst.getGlyphOrder():
        if n in glyf and glyf[n].numberOfContours > 0:
            pen = PolyPen(gs)
            gs[n].draw(pen)
            out[n] = pen.polys
    return out


def compare(font, at=0, against=75, names=None):
    """{glyph name: (extra turning, extra inflections)} of each simple glyph at Bleed at over against"""
    a, b = outlines(TTFont(font), at, names), outlines(TTFont(font), against, names)
    out = {}
    for n in a:
        (ta, ia), (tb, ib) = measure(a[n]), measure(b[n])
        out[n] = (ta - tb, ia - ib)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("font", nargs="+")
    ap.add_argument("--at", type=float, default=0)
    ap.add_argument("--against", type=float, default=75)
    ap.add_argument("--glyphs", help="comma separated glyph names, all simple glyphs by default")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    names = a.glyphs.split(",") if a.glyphs else None
    for f in a.font:
        res = compare(f, a.at, a.against, names)
        extra = np.array([v[0] for v in res.values()])
        print(f"{f}: {len(res)} glyphs at {a.at:g} over {a.against:g}, extra turning median {np.median(extra):.1f}° total {extra.sum():.0f}°, "
              f"{sum(v[1] > 0 for v in res.values())} glyphs with more inflections, {sum(max(v[1], 0) for v in res.values())} more in all")
        for n, (t, i) in sorted(res.items(), key=lambda kv: -kv[1][0])[:a.top]:
            print(f"  {n:14} {t:7.1f}° {i:+3d}")


if __name__ == "__main__":
    main()
