import json

import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

from glyphstack import baseline, fontbuild, target

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


def test_bleed_runs_from_a_quarter_of_the_print_look_to_a_quarter_more_and_optical_size_from_6_to_72_pt(tmp_path):
    path, made = built(tmp_path)
    vf = TTFont(path)
    axes = [(ax.axisTag, ax.minValue, ax.defaultValue, ax.maxValue) for ax in vf["fvar"].axes]
    assert axes == [("BLED", 25, 100, 125), ("opsz", 6, 8, 72)]
    assert path.name == "TestRules[BLED,opsz].ttf"
    assert made == {"A": "traced", "V": "outline", "acute": "outline", "Aacute": "composed"}
    assert vf["glyf"]["Aacute"].isComposite()


def test_bleed_25_stays_near_the_measuring_stick_and_100_is_the_trace(tmp_path):
    path, _ = built(tmp_path)
    at25 = instantiateVariableFont(TTFont(path), {"BLED": 25})
    at100 = instantiateVariableFont(TTFont(path), {"BLED": 100})
    v = points(at25, "V")
    # a quarter of the look moves the outer corners by a few units at most
    assert all(np.abs(v - q).sum(1).min() <= 8 for q in V if q[1] == 700)
    a = points(at100, "A")
    assert np.allclose([a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()], [-5, 605, 0, 700], atol=0.5)


def test_the_print_look_fills_the_acute_notch_and_spreads_the_ink(tmp_path):
    path, _ = built(tmp_path)
    notch = []
    for b in (25, 100, 125):
        v = points(instantiateVariableFont(TTFont(path), {"BLED": b}), "V")
        middle = v[np.abs(v[:, 0] - 300) < 40]
        notch.append(middle[middle[:, 1] > 100][:, 1].min())
    # the white wedge above the notch fills from its point, further the stronger the look
    assert notch[0] < notch[1] < notch[2]
    assert notch[1] - notch[0] > 10


def edge(font, glyph):
    return np.vstack([fontbuild.dense(P) for c in fontbuild.design(font, "", glyph, fontbuild.SCALE) for _, P in c])


def near(a, b):
    """The furthest any point of either dense outline is from the other"""
    return max(np.linalg.norm(a[:, None] - b[None], axis=2).min(1).max(), np.linalg.norm(b[:, None] - a[None], axis=2).min(1).max())


def notch(font):
    """The height of the V's notch, which the print look fills from its point"""
    v = points(font, "V")
    middle = v[np.abs(v[:, 0] - 300) < 40]
    return middle[middle[:, 1] > 100][:, 1].min()


def at(path, **loc):
    return instantiateVariableFont(TTFont(path), {k.replace("bled", "BLED"): v for k, v in loc.items()})


def test_the_print_look_grows_as_the_size_shrinks_and_the_default_size_is_the_bleed_look(tmp_path):
    path, _ = built(tmp_path)
    assert np.allclose(points(at(path, bled=100, opsz=8), "V"), points(at(path, bled=100), "V"))
    assert np.allclose(points(at(path, bled=100, opsz=8), "A")[:, 0].min(), -5, atol=0.5)
    heights = [notch(at(path, bled=100, opsz=o)) for o in (6, 8, 72)]
    # the stronger the look the further the notch fills, so smaller sizes fill more
    assert heights[0] > heights[1] > heights[2]
    # the traced A moves by the change the look makes to its outline at the other sizes
    assert all(np.abs(points(at(path, bled=100, opsz=o), "A") - points(at(path, bled=100, opsz=8), "A")).max() >= 0.5 for o in (6, 72))


def test_optical_size_scales_the_measured_shares_of_spread_and_rounding():
    assert fontbuild.size_gain(8) == {"d": 1, "r": 1}
    g = fontbuild.size_gain(4)
    assert np.isclose(g["d"], 1 + fontbuild.PHYSICAL["d"]) and np.isclose(g["r"], 1 + fontbuild.PHYSICAL["r"])
    # a lighter look than an outline already carries cannot be drawn from it
    assert fontbuild.strength(25, 72, t0=0.25) == (0.25, 0.25)
    assert fontbuild.strength(100, 6) == (1 + fontbuild.PHYSICAL["d"] / 3, 1 + fontbuild.PHYSICAL["r"] / 3)


def test_a_glyph_that_stops_bleeding_sooner_at_a_small_size_keeps_the_default_size_look_past_it():
    ts = [m / 100 for m in fontbuild.MASTERS["BLED"]]
    cols = {6: ["6@25", "6@50", "6@100", "6@125"], 8: ["8@25", "8@50", "8@100", "8@125"], 72: ["72@25", "72@50", "72@100", "72@125"]}
    # at 6 pt the look stops at 50, at 8 pt it follows every step and at 72 pt it stops at 25, which is lighter anyway
    out = fontbuild.no_lighter(cols, {6: 0.5, 8: 1.25, 72: 0.25}, ts, t0=0.25)
    assert out[6] == ["6@25", "6@50", "8@100", "8@125"] and out[8] == cols[8] and out[72] == cols[72]


def test_optical_size_interpolates_in_one_over_size(tmp_path):
    path, _ = built(tmp_path)
    seg = TTFont(path)["avar"].segments["opsz"]
    keys = sorted(seg)
    norm = lambda o: (o - 8) / (72 - 8) if o >= 8 else (o - 8) / (8 - 6)
    mapped = lambda o: np.interp(norm(o), keys, [seg[k] for k in keys])
    for o in (7, 12, 16, 24, 48):
        want = (1 / 8 - 1 / o) / (1 / 8 - 1 / 72) if o >= 8 else -(1 / o - 1 / 8) / (1 / 6 - 1 / 8)
        assert abs(mapped(o) - want) < 0.01


def synthetic_baseline(tmp_path):
    target.use("rules")
    tmp_path.mkdir()
    path, _ = baseline.make(measuring_stick(), traces(), "regular", "Test Rules", "Regular", list("AV´Á"), tmp_path)
    return path


def test_the_synthetic_baseline_is_the_measuring_stick_built_at_bleed_25(tmp_path):
    path, _ = built(tmp_path)
    static = TTFont(synthetic_baseline(tmp_path / "baseline"))
    assert "fvar" not in static and "gvar" not in static
    assert (static["name"].getDebugName(1), static["name"].getDebugName(6)) == ("Test Rules Bleed 25", "TestRulesBleed25-Regular")
    # the fonts' quadratics are fitted to all their masters at once, so the two can differ by a point but not in shape
    assert near(edge(static, "V"), edge(instantiateVariableFont(TTFont(path), {"BLED": 25}), "V")) <= 0.5


def test_a_build_from_the_synthetic_baseline_starts_from_it_and_reaches_the_same_print_look(tmp_path):
    path, _ = built(tmp_path)
    source = TTFont(synthetic_baseline(tmp_path / "baseline"))
    assert fontbuild.base_bleed(source) == (0.25, fontbuild.SCALE)
    (tmp_path / "again").mkdir()
    again, made = fontbuild.build_face(source, traces(), "regular", "Test Rules", "Regular", list("AV´Á"), tmp_path / "again")
    assert made == {"A": "traced", "V": "outline", "acute": "outline", "Aacute": "composed"}
    vf = TTFont(again)
    assert vf["head"].unitsPerEm == 4000 and vf["hmtx"]["V"][0] == 620 * fontbuild.SCALE
    at = lambda p, b, g: edge(instantiateVariableFont(TTFont(p), {"BLED": b}), g)
    # the lowest master is the source's outline, less the short pieces folded into their neighbours, and the others land where the build from the measuring stick put them
    assert near(at(again, 25, "V"), edge(source, "V")) <= 1
    assert near(at(again, 100, "V"), at(path, 100, "V")) <= 3 and near(at(again, 125, "V"), at(path, 125, "V")) <= 3
    assert np.allclose(at(again, 100, "A")[:, 0].min(), -5, atol=0.5)
    # at Bleed 25 above the default size the look would be lighter than the baseline carries, so it stays the baseline
    big = edge(instantiateVariableFont(TTFont(again), {"BLED": 25, "opsz": 72}), "V")
    assert near(big, edge(source, "V")) <= 1
    assert [p.name for p in (tmp_path / "again").iterdir()] == ["TestRules[BLED,opsz].ttf"]


def test_publish_writes_woff_fonts_and_an_index_for_the_site(tmp_path):
    path, _ = built(tmp_path)
    fontbuild.publish({"regular": path}, tmp_path / "site")
    index = json.loads((tmp_path / "site" / "nomicon-rules.json").read_text())
    assert index["faces"] == {"regular": "nomicon-rules-regular.woff"} and index["axes"] == {"BLED": [25, 100, 125], "opsz": [6, 8, 72]}
    assert ord("V") in index["chars"]["regular"] and TTFont(tmp_path / "site" / "nomicon-rules-regular.woff").flavor == "woff"


def test_the_fonts_carry_the_source_kerning_through_the_baseline(tmp_path):
    from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
    stick = measuring_stick()
    addOpenTypeFeaturesFromString(stick, "feature kern { pos A V -80; pos V acute 30; } kern;")
    target.use("rules")
    (tmp_path / "baseline").mkdir()
    base, _ = baseline.make(stick, traces(), "regular", "Test Rules", "Regular", list("AV´Á"), tmp_path / "baseline")
    assert fontbuild.kern_pairs(TTFont(base)) == {("A", "V"): -80 * fontbuild.SCALE, ("V", "acute"): 30 * fontbuild.SCALE}
    path, _ = fontbuild.build_face(TTFont(base), traces(), "regular", "Test Rules", "Regular", list("AV´Á"), tmp_path)
    assert fontbuild.kern_pairs(TTFont(path)) == {("A", "V"): -320, ("V", "acute"): 120}
