from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from glyphstack import build


def box(x0, y0, x1, y1):
    pen = TTGlyphPen(None)
    pen.moveTo((x0, y0))
    pen.lineTo((x0, y1))
    pen.lineTo((x1, y1))
    pen.lineTo((x1, y0))
    pen.closePath()
    return pen.glyph()


def tiny_font():
    """A font with A, an acute accent, Á composed from them and a Private Use comma accent"""
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A", "acute", "Aacute", "commaaccent"])
    fb.setupCharacterMap({ord("A"): "A", ord("´"): "acute", ord("Á"): "Aacute", 0xF6C3: "commaaccent"})
    acute = box(0, 500, 100, 600)
    pen = TTGlyphPen({"A": box(0, 0, 600, 700), "acute": acute})
    pen.addComponent("A", (1, 0, 0, 1, 0, 0))
    pen.addComponent("acute", (1, 0, 0, 1, 250, 220))
    fb.setupGlyf({".notdef": box(0, 0, 10, 10), "A": box(0, 0, 600, 700), "acute": acute, "Aacute": pen.glyph(), "commaaccent": box(0, -200, 80, -50)})
    fb.setupHorizontalMetrics({".notdef": (100, 0), "A": (650, 0), "acute": (300, 0), "Aacute": (650, 0), "commaaccent": (200, 0)})
    fb.setupHorizontalHeader()
    return fb.font


def test_a_composite_is_built_from_its_parts_at_the_fonts_offsets():
    f = tiny_font()
    ps = build.parts(f, "Á")
    assert ps == [("A", "A", (0, 0)), ("´", "acute", (250, 220))]
    assert build.parts(f, "A") is None
    a = dict(trace="M0.0,-700.0L600.0,-700.0Z", nodes=2, nc=1, prior=True)
    acute = dict(trace="M0.0,-600.0L100.0,-600.0Z", nodes=2, nc=1, built="outline")
    e = build.compose(f, "Á", ps, [a, acute])
    # paths are y down, so the accent's offset of 220 up moves it to y -820
    assert e["trace"] == "M0.0,-700.0L600.0,-700.0ZM250.0,-820.0L350.0,-820.0Z"
    assert e["adv"] == 650 and e["built"] == "composed" and e["nodes"] == 4
    assert [p["source"] for p in e["parts"]] == ["prior", "outline"]


def test_a_combining_accent_is_its_spacing_accent_centred_on_the_origin_with_no_advance():
    f = tiny_font()
    ps = build.parts(f, "\u0301")
    assert ps == [("´", "acute", (-150, 0))]
    e = build.compose(f, "\u0301", ps, [dict(trace="M0.0,-600.0L300.0,-600.0Z", nodes=2, nc=1)])
    assert e["trace"] == "M-150.0,-600.0L150.0,-600.0Z" and e["adv"] == 0 and e["built"] == "derived"
    # a part with only a Private Use code point is named by its glyph
    assert build.parts(f, "\u0326") == [(None, "commaaccent", (-100, 0))]


def test_a_dotless_letter_the_font_does_not_draw_is_its_dotted_letter_less_the_dot():
    from glyphstack import fontfit, styles

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "j"])
    fb.setupCharacterMap({ord("j"): "j"})
    pen = TTGlyphPen(None)
    for x0, y0, x1, y1 in ((0, -200, 100, 450), (0, 550, 100, 650)):
        box(x0, y0, x1, y1).draw(pen, None)
    fb.setupGlyf({".notdef": box(0, 0, 10, 10), "j": pen.glyph()})
    fb.setupHorizontalMetrics({".notdef": (100, 0), "j": (250, 0)})
    fb.setupHorizontalHeader()
    f = fb.font
    assert styles.drawn_by(f, "ȷ")
    g = fontfit.FontGlyph(f, "ȷ")
    assert len(g.contours) == 1 and max(c[:, 1].max() for c in g.contours) <= 450 and g.adv == 250
