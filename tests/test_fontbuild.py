import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

from glyphstack import fontbuild, target

V = [(0, 700), (150, 700), (300, 150), (450, 700), (600, 700), (350, 0), (250, 0)]
LOOK = dict(d=2.5, w=3.3, r=8.0)
P = [0.6, 1.0, 0.0, 0.0, 0.0, 2.0]


def poly(pts):
    pen = TTGlyphPen(None)
    pen.moveTo(pts[0])
    for q in pts[1:]:
        pen.lineTo(q)
    pen.closePath()
    return pen.glyph()


def box(x0, y0, x1, y1):
    return poly([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])


def measuring_stick():
    """A font with a V whose notch is acute, a box A, an acute accent and Á composed from them"""
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A", "V", "acute", "Aacute"])
    fb.setupCharacterMap({ord("A"): "A", ord("V"): "V", ord("´"): "acute", ord("Á"): "Aacute"})
    pen = TTGlyphPen({"A": box(0, 0, 600, 700), "acute": box(0, 500, 100, 600)})
    pen.addComponent("A", (1, 0, 0, 1, 0, 0))
    pen.addComponent("acute", (1, 0, 0, 1, 250, 220))
    fb.setupGlyf({".notdef": box(0, 0, 10, 10), "A": box(0, 0, 600, 700), "V": poly(V), "acute": box(0, 500, 100, 600), "Aacute": pen.glyph()})
    fb.setupHorizontalMetrics({".notdef": (100, 0), "A": (650, 0), "V": (620, 0), "acute": (300, 0), "Aacute": (650, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupOS2()
    fb.setupPost()
    return fb.font


def traces():
    """Session entries for the font: A traced as a slightly wider box, V and the accent built from outlines, Á composed"""
    return {"A": dict(trace="M-5.0,0.0L-5.0,-700.0L605.0,-700.0L605.0,0.0Z", adv=650, n=20, p=P),
            "V": dict(trace="", adv=620, n=0, p=P, built="outline", look=LOOK),
            "´": dict(trace="", adv=300, n=0, p=P, built="outline", look=LOOK),
            "Á": dict(adv=650, n=0, built="composed", parts=[dict(char="A", glyph="A", offset=[0, 0]), dict(char="´", glyph="acute", offset=[250, 220])])}


def points(font, name):
    pen = DecomposingRecordingPen(font.getGlyphSet())
    font.getGlyphSet()[name].draw(pen)
    return np.array([q for _, args in pen.value for q in args if q is not None], float) / fontbuild.SCALE


def built(tmp_path):
    target.use("rules")
    path, made = fontbuild.build_face(measuring_stick(), traces(), "regular", "Test Rules", "Regular", list("AV´Á"), tmp_path)
    return path, made


def test_the_axis_runs_from_no_print_look_to_a_quarter_more_than_measured(tmp_path):
    path, made = built(tmp_path)
    vf = TTFont(path)
    (ax,) = vf["fvar"].axes
    assert (ax.axisTag, ax.minValue, ax.defaultValue, ax.maxValue) == ("BLED", 0, 100, 125)
    assert made == {"A": "traced", "V": "outline", "acute": "outline", "Aacute": "composed"}
    assert vf["glyf"]["Aacute"].isComposite()


def test_bleed_0_is_the_measuring_stick_and_100_is_the_trace(tmp_path):
    path, _ = built(tmp_path)
    at0 = instantiateVariableFont(TTFont(path), {"BLED": 0})
    at100 = instantiateVariableFont(TTFont(path), {"BLED": 100})
    v = points(at0, "V")
    assert all(np.abs(v - q).sum(1).min() <= 0.5 for q in V)
    a = points(at100, "A")
    assert np.allclose([a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()], [-5, 605, 0, 700], atol=0.5)


def test_the_print_look_fills_the_acute_notch_and_spreads_the_ink(tmp_path):
    path, _ = built(tmp_path)
    notch = []
    for b in (0, 100, 125):
        v = points(instantiateVariableFont(TTFont(path), {"BLED": b}), "V")
        middle = v[np.abs(v[:, 0] - 300) < 40]
        notch.append(middle[:, 1].min() if b == 0 else middle[middle[:, 1] > 100][:, 1].min())
    # the white wedge above the notch fills from its point, further the stronger the look
    assert notch[0] < notch[1] < notch[2]
    assert notch[1] - notch[0] > 10
