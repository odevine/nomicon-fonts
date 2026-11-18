"""OpenType features of the Nomicon Rules fonts, with anchors placed on the built outlines

The composites of the measuring stick place each accent on its letter at an offset, and the combining accents are
those spacing accents moved onto the origin, so an offset says where on its letter an accent's middle goes. `anchors`
reads each letter's top and bottom anchors from those offsets and, for a letter no composite uses, from its outline
at the height the composites put accents at. Everything is read from the default master, Bleed 100 at the regular
weight, in the font's own units. The bold master grows each glyph about its own middle, so the same anchors hold at
every weight, as the composites' fixed offsets do

`extras` lists the glyphs the features substitute in, each a composite of one glyph the font draws, and `text` writes
the feature file: mark and mkmk from the anchors, ccmp dropping the dot of i and j under an accent above, case
raising punctuation to the capitals, locl for Romanian, Moldavian and Catalan, and kern
"""
import math

import numpy as np
from fontTools.pens.basePen import BasePen

# combining accents by where they attach, with the spacing accent each is built from
TOP = {0x0300: "grave", 0x0301: "acute", 0x0302: "circumflex", 0x0303: "tilde", 0x0304: "macron", 0x0306: "breve",
       0x0307: "dotaccent", 0x0308: "dieresis", 0x030A: "ring", 0x030B: "hungarumlaut", 0x030C: "caron"}
BOTTOM = {0x0326: "commaaccent", 0x0327: "cedilla"}
OGONEK = {0x0328: "ogonek"}
# each letter drawn with an ogonek and the letter it is drawn on
OGONEK_LETTERS = {"Aogonek": "A", "aogonek": "a", "Eogonek": "E", "eogonek": "e", "Iogonek": "I", "iogonek": "i",
                  "Uogonek": "U", "uogonek": "u"}
# letters that lose their dot under an accent above, and the dotless glyph each becomes
SOFT_DOTTED = {"i": "dotlessi", "j": "dotlessj"}
# punctuation centred on the lowercase, raised for capitals by case
CASE = ["hyphen", "endash", "emdash", "parenleft", "parenright", "bracketleft", "bracketright", "braceleft",
        "braceright", "guilsinglleft", "guilsinglright", "guillemotleft", "guillemotright", "bullet"]
# inverted marks hang below the baseline, raised for capitals to sit on it
CASE_INVERTED = ["exclamdown", "questiondown"]
# a case form moved less than this fraction of the em is not made
CASE_LEAST = 0.025
# the Catalan middle dot between two l keeps this fraction of the middle dot's side bearings
CAT_BEARING = 0.5
# the gap between two stacked accents as a fraction of the gap between the x-height and an accent above it
STACK_GAP = 0.5
# the part of a glyph's height read for the middle of its top or bottom
BAND = 0.25


class FlatPen(BasePen):
    """The points of an outline with each curve sampled, for reading where its ink is"""

    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.pts = []

    def _moveTo(self, p):
        self.pts.append(p)

    def _lineTo(self, p):
        self.pts.append(p)

    def _curveToOne(self, a, b, c):
        p0 = np.array(self._getCurrentPoint(), float)
        u = np.linspace(0, 1, 9)[1:, None]
        self.pts += list(map(tuple, (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * np.array(a) + 3 * (1 - u) * u ** 2 * np.array(b) + u ** 3 * np.array(c)))

    def _qCurveToOne(self, a, b):
        p0 = np.array(self._getCurrentPoint(), float)
        u = np.linspace(0, 1, 9)[1:, None]
        self.pts += list(map(tuple, (1 - u) ** 2 * p0 + 2 * (1 - u) * u * np.array(a) + u ** 2 * np.array(b)))


def ink(font, name):
    """The sampled points of a glyph's outline, components included, or an empty array for a glyph with none"""
    pen = FlatPen(font.getGlyphSet())
    font.getGlyphSet()[name].draw(pen)
    return np.array(pen.pts, float).reshape(-1, 2)


def slope(font):
    """How far right the italic's strokes lean per unit up"""
    return math.tan(math.radians(-font["post"].italicAngle))


def names(font, table):
    """{glyph name of each combining accent the font draws: its spacing accent}"""
    cmap = font.getBestCmap()
    return {cmap[cp]: spacing for cp, spacing in table.items() if cp in cmap and spacing in font["hmtx"].metrics}


def components(font, name):
    g = font["glyf"][name]
    return [(c.glyphName, c.x, c.y) for c in g.components] if g.isComposite() else []


def middle(pts, lo, hi, at, lean):
    """The middle of the ink between heights lo and hi, moved along the slant to height at"""
    band = pts[(pts[:, 1] >= lo) & (pts[:, 1] <= hi)]
    band = band if len(band) else pts
    y = (band[:, 1].min() + band[:, 1].max()) / 2
    return (band[:, 0].min() + band[:, 0].max()) / 2 + (at - y) * lean


class Anchors:
    """Each glyph's anchors for mark and mkmk, in font units on the default master

    A letter an accent is composed on takes its anchor from where the composites put the accent's middle, the median
    over its composites. Any other letter takes the height the composites put accents at for a letter as tall as it,
    and the middle of its outline's top or bottom there, corrected by the median difference the two make on the
    letters that have both. An accented letter's top anchor is its accent's own, so another accent stacks on it
    """

    def __init__(self, font):
        self.font, self.lean = font, slope(font)
        self.x_height = font["OS/2"].sxHeight
        self.top_marks, self.bottom_marks, self.ogonek_marks = names(font, TOP), names(font, BOTTOM), names(font, OGONEK)
        self.marks = self.top_marks | self.bottom_marks | self.ogonek_marks
        spacing_top, spacing_bottom = set(self.top_marks.values()), set(self.bottom_marks.values())
        self.half = lambda n: font["hmtx"][n][0] / 2
        self.pts = {n: ink(font, n) for n in font.getGlyphOrder()}
        glyphs = [n for n in font.getGlyphOrder() if n != ".notdef" and n not in self.marks and len(self.pts[n])]
        seen = {"top": {}, "bottom": {}}
        for n in glyphs:
            cs = components(font, n)
            if len(cs) < 2 or cs[0][1:] != (0, 0):
                continue
            for accent, dx, dy in cs[1:]:
                kind = "top" if accent in spacing_top else "bottom" if accent in spacing_bottom else None
                if kind:
                    seen[kind].setdefault(cs[0][0], []).append((dx + self.half(accent), dy + (self.x_height if kind == "top" else 0)))
        self.seen = {k: {n: tuple(np.median(np.array(v), axis=0)) for n, v in d.items()} for k, d in seen.items()}
        self.top, self.bottom = {}, {}
        plain = [n for n in glyphs if not components(font, n)]
        rise = [self.seen["top"][n][1] - self.pts[n][:, 1].max() for n in self.seen["top"] if n in plain]
        self.rise = float(np.median(rise)) if rise else 0.0
        self.top_fix = self.fix("top", plain)
        self.bottom_fix = self.fix("bottom", plain)
        for n in glyphs:
            self.top[n], self.bottom[n] = self.place(n)
        self.ogonek = self.ogonek_anchors()
        self.mark_top = self.stack()

    def guess(self, n, kind):
        """A glyph's anchor read from its outline alone"""
        P = self.pts[n]
        lo, hi = P[:, 1].min(), P[:, 1].max()
        if kind == "top":
            y = hi + self.rise
            return middle(P, hi - BAND * (hi - lo), hi, y, self.lean), y
        y = 0.0 if lo > -0.05 * self.font["head"].unitsPerEm else lo
        return middle(P, lo, lo + BAND * (hi - lo), y, self.lean), y

    def fix(self, kind, plain):
        """The median of where the composites put an anchor less where the outline alone would"""
        d = [self.seen[kind][n][0] - self.guess(n, kind)[0] for n in self.seen[kind] if n in plain]
        return float(np.median(d)) if d else 0.0

    def place(self, n):
        cs = components(self.font, n)
        if cs and cs[0][0] in self.top:
            base, bx, by = cs[0]
            top, bottom = self.top[base], self.bottom[base]
            top, bottom = (top[0] + bx, top[1] + by), (bottom[0] + bx, bottom[1] + by)
            for accent, dx, dy in cs[1:]:
                if accent in self.top_marks.values():
                    t = self.accent_top(accent)
                    top = (t[0] + dx, t[1] + dy)
            return top, bottom
        out = []
        for kind, fix in (("top", self.top_fix), ("bottom", self.bottom_fix)):
            if n in self.seen[kind]:
                out.append(self.seen[kind][n])
            else:
                x, y = self.guess(n, kind)
                out.append((x + fix, y))
        return out

    def gap(self):
        """The median gap between the x-height and the bottom of an accent above it"""
        return float(np.median([self.pts[m][:, 1].min() - self.x_height for m in self.top_marks]))

    def accent_top(self, accent):
        """Where an accent stacked on this spacing accent attaches, in the spacing accent's own units"""
        P = self.pts[accent]
        y = P[:, 1].max() - (1 - STACK_GAP) * self.gap()
        return self.half(accent) + (y - self.x_height) * self.lean, y

    def stack(self):
        """{combining accent above: the anchor another accent above stacks on}"""
        out = {}
        for m, spacing in self.top_marks.items():
            x, y = self.accent_top(spacing)
            out[m] = (x - self.half(spacing), y)
        return out

    def ogonek_anchors(self):
        """{letter: where an ogonek attaches}, read from the letters the font draws with one by matching the ogonek's ink below the baseline"""
        if not self.ogonek_marks:
            return {}
        spacing = next(iter(self.ogonek_marks.values()))
        O = self.pts[spacing]
        below = lambda P: P[P[:, 1] < -0.1 * self.font["head"].unitsPerEm]
        out = {}
        for drawn, base in OGONEK_LETTERS.items():
            if drawn in self.pts and base in self.top and len(below(self.pts[drawn])) and len(below(O)):
                a, b = below(self.pts[drawn]), below(O)
                dx = (a[:, 0].min() + a[:, 0].max() - b[:, 0].min() - b[:, 0].max()) / 2
                out[base] = (dx + self.half(spacing), 0.0)
                if base in SOFT_DOTTED and SOFT_DOTTED[base] in self.top:
                    out[SOFT_DOTTED[base]] = out[base]
        return out


def bounds(font, name):
    P = ink(font, name)
    return P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()


def extras(font):
    """{new glyph name: (glyph it is drawn from, x offset, y offset, change to its advance)}, the forms the features substitute in

    A case form is moved so its middle sits halfway up the capitals, or an inverted mark so it sits on the baseline,
    and along the italic's slant as it rises. The Catalan middle dot keeps CAT_BEARING of its side bearings
    """
    have = set(font.getGlyphOrder())
    cap, upm, lean = font["OS/2"].sCapHeight, font["head"].unitsPerEm, slope(font)
    out = {}

    def raised(name, src, dy, dx=0, dadv=0):
        if abs(dy) >= CASE_LEAST * upm:
            out[name] = (src, round(dx + dy * lean), round(dy), dadv)

    for n in CASE + CASE_INVERTED:
        if n in have:
            x0, y0, x1, y1 = bounds(font, n)
            raised(n + ".case", n, -y0 if n in CASE_INVERTED else cap / 2 - (y0 + y1) / 2)
    if "periodcentered" in have and "l" in have:
        x0, y0, x1, y1 = bounds(font, "periodcentered")
        adv = font["hmtx"]["periodcentered"][0]
        lsb, rsb = x0, adv - x1
        dx = round(-(1 - CAT_BEARING) * lsb)
        dadv = round(-(1 - CAT_BEARING) * (lsb + rsb))
        out["periodcentered.loclCAT"] = ("periodcentered", dx, 0, dadv)
        if "L" in have:
            raised("periodcentered.loclCAT.case", "periodcentered", cap / 2 - (y0 + y1) / 2, dx, dadv)
    return out


def anchor(p):
    return f"<anchor {round(p[0])} {round(p[1])}>"


def text(font, kern=""):
    """The feature file of a font that already has the glyphs `extras` lists, kern lines included"""
    a = Anchors(font)
    have = set(font.getGlyphOrder())
    bases = [n for n in font.getGlyphOrder() if n != ".notdef" and n not in a.marks]
    out = ["languagesystem DFLT dflt;", "languagesystem latn dflt;", "languagesystem latn ROM;", "languagesystem latn MOL;", "languagesystem latn CAT;", ""]
    out.append("table GDEF {\n  GlyphClassDef [" + " ".join(bases) + "], , [" + " ".join(a.marks) + "], ;\n} GDEF;\n")
    out += [f"markClass {m} <anchor 0 {a.x_height}> @top;" for m in a.top_marks]
    out += [f"markClass {m} <anchor 0 0> @bottom;" for m in a.bottom_marks]
    out += [f"markClass {m} <anchor 0 0> @ogonek;" for m in a.ogonek_marks]
    out.append("")
    if kern:
        out.append(f"feature kern {{\n  lookupflag IgnoreMarks;\n{kern}}} kern;\n")
    mark = []
    if a.top_marks:
        mark += [f"  pos base {n} {anchor(a.top[n])} mark @top;" for n in bases if n in a.top]
    if a.bottom_marks:
        mark += [f"  pos base {n} {anchor(a.bottom[n])} mark @bottom;" for n in bases if n in a.bottom]
    if a.ogonek_marks:
        mark += [f"  pos base {n} {anchor(p)} mark @ogonek;" for n, p in a.ogonek.items()]
    if mark:
        out.append("feature mark {\n" + "\n".join(mark) + "\n} mark;\n")
    if a.top_marks:
        out.append("feature mkmk {\n" + "\n".join(f"  pos mark {m} {anchor(p)} mark @top;" for m, p in a.mark_top.items()) + "\n} mkmk;\n")
    out.append(ccmp(a, have))
    case = [(n.removesuffix(".case"), n) for n in sorted(have) if n.endswith(".case") and n.removesuffix(".case") in have]
    if case:
        out.append("feature case {\n" + "".join(f"  sub {s} by {n};\n" for s, n in case) + "} case;\n")
    out.append(locl(font, have))
    return "\n".join(o for o in out if o is not None)


def ccmp(a, have):
    """ccmp dropping the dot of i and j under an accent above, also past accents below, and splitting į so its ogonek can stay"""
    soft = {s: d for s, d in SOFT_DOTTED.items() if s in have and d in have}
    if not soft or not a.top_marks:
        return None
    top = " ".join(a.top_marks)
    below = " ".join(a.bottom_marks | a.ogonek_marks)
    src, dst = " ".join(soft), " ".join(soft.values())
    rules = [f"    sub [{src}]' lookup ccmp_dotless [{top}];"]
    if below:
        rules.append(f"    sub [{src}]' lookup ccmp_dotless [{below}] [{top}];")
    split = ""
    ogonek = next(iter(a.ogonek_marks), None)
    if "iogonek" in have and "i" in soft and ogonek:
        split = f"lookup ccmp_split {{\n  sub iogonek by {soft['i']} {ogonek};\n}} ccmp_split;\n\n"
        rules.append(f"    sub iogonek' lookup ccmp_split [{top}];")
        if below:
            rules.append(f"    sub iogonek' lookup ccmp_split [{below}] [{top}];")
    return (f"lookup ccmp_dotless {{\n  sub [{src}] by [{dst}];\n}} ccmp_dotless;\n\n{split}"
            f"feature ccmp {{\n  lookup ccmp_soft {{\n" + "\n".join(rules) + "\n  } ccmp_soft;\n} ccmp;\n")


def locl(font, have):
    """locl with the comma below for the cedilla under S and T in Romanian and Moldavian, and the Catalan middle dot between two l"""
    cmap = font.getBestCmap()
    pairs = [(cmap.get(c), cmap.get(m)) for c, m in ((0x015E, 0x0218), (0x015F, 0x0219), (0x0162, 0x021A), (0x0163, 0x021B))]
    rom = [(c, m) for c, m in pairs if c and m]
    cat = []
    if "periodcentered.loclCAT" in have:
        cat.append("  sub l periodcentered' l by periodcentered.loclCAT;")
        capital = "periodcentered.loclCAT.case" if "periodcentered.loclCAT.case" in have else "periodcentered.loclCAT"
        if "L" in have:
            cat.append(f"  sub L periodcentered' L by {capital};")
    if not rom and not cat:
        return None
    # lookups written inside the feature would also apply to the default language
    out = []
    if rom:
        out.append("lookup locl_rom {\n" + "".join(f"  sub {c} by {m};\n" for c, m in rom) + "} locl_rom;\n")
    if cat:
        out.append("lookup locl_cat {\n" + "\n".join(cat) + "\n} locl_cat;\n")
    out += ["feature locl {", "  script latn;"]
    if rom:
        out += ["  language ROM exclude_dflt;", "  lookup locl_rom;", "  language MOL exclude_dflt;", "  lookup locl_rom;"]
    if cat:
        out += ["  language CAT exclude_dflt;", "  lookup locl_cat;"]
    out.append("} locl;\n")
    return "\n".join(out)
