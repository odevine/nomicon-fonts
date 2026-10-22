"""Reads glyph outlines, advances and kerning from a TrueType font used as a measuring stick"""
import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont
from matplotlib.path import Path


class FlattenPen(BasePen):
    """Flattens a glyph outline into closed polylines in font units"""

    def __init__(self, glyph_set, steps=48):
        super().__init__(glyph_set)
        self.steps = steps
        self.contours = []
        self._cur = None

    def _moveTo(self, p):
        self._cur = [p]

    def _lineTo(self, p):
        self._cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = np.array(self._cur[-1])
        t = np.linspace(0, 1, self.steps + 1)[1:, None]
        pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * np.array(p1) + 3 * (1 - t) * t**2 * np.array(p2) + t**3 * np.array(p3)
        self._cur.extend(map(tuple, pts))

    def _qCurveToOne(self, p1, p2):
        p0 = np.array(self._cur[-1])
        t = np.linspace(0, 1, self.steps + 1)[1:, None]
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * np.array(p1) + t**2 * np.array(p2)
        self._cur.extend(map(tuple, pts))

    def _closePath(self):
        if self._cur and len(self._cur) > 2:
            self.contours.append(np.array(self._cur, float))
        self._cur = None

    _endPath = _closePath


def load_font(path):
    return TTFont(path)


def glyph_contours(font, ch):
    """Returns the flattened contours of a character and its glyph name"""
    name = font.getBestCmap()[ord(ch)]
    pen = FlattenPen(font.getGlyphSet())
    font.getGlyphSet()[name].draw(pen)
    return pen.contours, name


def densify(contours, spacing):
    """Resamples closed polylines to roughly even spacing"""
    out = []
    for c in contours:
        cc = np.vstack([c, c[:1]])
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cc, axis=0), axis=1))])
        u = np.linspace(0, s[-1], max(int(s[-1] / spacing), 8), endpoint=False)
        out.append(np.stack([np.interp(u, s, cc[:, 0]), np.interp(u, s, cc[:, 1])], 1))
    return out


def inside_evenodd(polys, pts):
    """True where pts lie in the filled region, even-odd over all contours so counters are excluded"""
    pts = np.asarray(pts, float)
    cnt = np.zeros(len(pts), int)
    for c in polys:
        cnt += Path(np.vstack([c, c[:1]])).contains_points(pts).astype(int)
    return cnt % 2 == 1


class Kerning:
    """Horizontal advance and side bearings of glyphs plus the pair adjustments of the font's kern feature"""

    def __init__(self, font):
        self.font = font
        self.cmap = font.getBestCmap()
        self.hmtx = font["hmtx"]
        self.glyf = font["glyf"] if "glyf" in font else None
        self.lookups = pair_lookups(font)

    def advance(self, ch):
        return self.hmtx[self.cmap[ord(ch)]][0]

    def bounds(self, ch):
        name = self.cmap[ord(ch)]
        if self.glyf is not None:
            g = self.glyf[name]
            return g.xMin, g.xMax
        pen = BoundsPen(self.font.getGlyphSet())
        self.font.getGlyphSet()[name].draw(pen)
        return (pen.bounds[0], pen.bounds[2]) if pen.bounds else (0, 0)

    def pair(self, a, b):
        """Kerning adjustment in font units between two characters, summed over the kern lookups, 0 when the pair has none

        Within one lookup the first subtable that covers the first glyph decides, as in OpenType layout
        """
        ga, gb = self.cmap[ord(a)], self.cmap[ord(b)]
        total = 0
        for subtables in self.lookups:
            for st in subtables:
                cov = st.Coverage.glyphs
                if ga not in cov:
                    continue
                if st.Format == 1:
                    for pv in st.PairSet[cov.index(ga)].PairValueRecord:
                        if pv.SecondGlyph == gb and pv.Value1:
                            total += getattr(pv.Value1, "XAdvance", 0) or 0
                            break
                    else:
                        continue
                    break
                c1 = st.ClassDef1.classDefs.get(ga, 0)
                c2 = st.ClassDef2.classDefs.get(gb, 0)
                v = st.Class1Record[c1].Class2Record[c2].Value1
                total += (getattr(v, "XAdvance", 0) or 0) if v else 0
                break
        return total


def pair_lookups(font):
    """The pair positioning subtables of each lookup the kern feature uses, or of every pair lookup when there is no kern feature"""
    if "GPOS" not in font or not font["GPOS"].table.LookupList:
        return []
    table = font["GPOS"].table
    idx = sorted({i for fr in (table.FeatureList.FeatureRecord if table.FeatureList else []) if fr.FeatureTag == "kern" for i in fr.Feature.LookupListIndex})
    lookups = [table.LookupList.Lookup[i] for i in idx] or list(table.LookupList.Lookup)
    out = []
    for lk in lookups:
        # a font may wrap its pair subtables in extension subtables
        subs = [getattr(st, "ExtSubTable", st) for st in lk.SubTable]
        subs = [st for st in subs if getattr(st, "LookupType", lk.LookupType) == 2 or type(st).__name__ == "PairPos"]
        if subs:
            out.append(subs)
    return out
