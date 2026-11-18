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
    fb.setupOS2(sxHeight=500, sCapHeight=700, sTypoDescender=-200)
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


def test_the_axis_runs_from_a_quarter_of_the_print_look_to_a_quarter_more_than_measured(tmp_path):
    path, made = built(tmp_path)
    vf = TTFont(path)
    (ax,) = vf["fvar"].axes
    assert (ax.axisTag, ax.minValue, ax.defaultValue, ax.maxValue) == ("BLED", 25, 100, 125)
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
    assert np.allclose(points(static, "V"), points(instantiateVariableFont(TTFont(path), {"BLED": 25}), "V"))


def test_a_build_from_the_synthetic_baseline_starts_from_it_and_reaches_the_same_print_look(tmp_path):
    path, _ = built(tmp_path)
    source = TTFont(synthetic_baseline(tmp_path / "baseline"))
    assert fontbuild.base_bleed(source) == (0.25, fontbuild.SCALE)
    (tmp_path / "again").mkdir()
    again, made = fontbuild.build_face(source, traces(), "regular", "Test Rules", "Regular", list("AV´Á"), tmp_path / "again")
    assert made == {"A": "traced", "V": "outline", "acute": "outline", "Aacute": "composed"}
    vf = TTFont(again)
    assert vf["head"].unitsPerEm == 4000 and vf["hmtx"]["V"][0] == 620 * fontbuild.SCALE
    edge = lambda f, g: np.vstack([fontbuild.dense(P) for c in fontbuild.design(f, "", g, fontbuild.SCALE) for _, P in c])
    at = lambda p, b, g: edge(instantiateVariableFont(TTFont(p), {"BLED": b}), g)
    near = lambda a, b: max(np.linalg.norm(a[:, None] - b[None], axis=2).min(1).max(), np.linalg.norm(b[:, None] - a[None], axis=2).min(1).max())
    # the lowest master is the source's outline, less the short pieces folded into their neighbours, and the others land where the build from the measuring stick put them
    assert near(at(again, 25, "V"), edge(source, "V")) <= 1
    assert near(at(again, 100, "V"), at(path, 100, "V")) <= 3 and near(at(again, 125, "V"), at(path, 125, "V")) <= 3
    assert np.allclose(at(again, 100, "A")[:, 0].min(), -5, atol=0.5)
    assert [p.name for p in (tmp_path / "again").iterdir()] == ["TestRules[BLED].ttf"]


def test_publish_writes_woff_fonts_and_an_index_for_the_site(tmp_path):
    path, _ = built(tmp_path)
    fontbuild.publish({"regular": path}, tmp_path / "site")
    index = json.loads((tmp_path / "site" / "nomicon-rules.json").read_text())
    assert index["faces"] == {"regular": "nomicon-rules-regular.woff"} and index["axes"] == {"BLED": [25, 100, 125]}
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


def bold_built(tmp_path):
    target.use("rules")
    return fontbuild.build_face(measuring_stick(), traces(), "regular", "Test Rules", "Regular", list("AV´Á"), tmp_path, weight=True)


def test_the_weight_axis_leaves_the_regular_as_it_was(tmp_path):
    path, made = bold_built(tmp_path)
    (tmp_path / "plain").mkdir()
    plain, _ = built(tmp_path / "plain")
    vf = TTFont(path)
    assert [(a.axisTag, a.minValue, a.defaultValue, a.maxValue) for a in vf["fvar"].axes] == [("BLED", 25, 100, 125), ("wght", 400, 400, 700)]
    assert path.name == "TestRules[BLED,wght].ttf" and made == {"A": "traced", "V": "outline", "acute": "outline", "Aacute": "composed"}
    edge = lambda f, g: np.vstack([fontbuild.dense(P) for c in fontbuild.design(f, "", g, fontbuild.SCALE, fontbuild.SplinePen) for _, P in c])
    for b in (25, 60, 100, 125):
        regular = instantiateVariableFont(TTFont(path), {"BLED": b, "wght": 400})
        before = instantiateVariableFont(TTFont(plain), {"BLED": b})
        for g in ("A", "V", "acute"):
            # the quadratic splines are shared with the bold, so their points differ but the outline stays within the conversion's tolerance
            d = np.linalg.norm(edge(regular, g)[:, None] - edge(before, g)[None], axis=2)
            assert max(d.min(0).max(), d.min(1).max()) <= 2 * fontbuild.MAX_ERR
        assert all(regular["hmtx"][g] == before["hmtx"][g] for g in ("A", "V", "acute", "Aacute"))


def test_the_bold_grows_stems_across_keeps_its_heights_and_widens_the_advance(tmp_path):
    path, _ = bold_built(tmp_path)
    bold = instantiateVariableFont(TTFont(path), {"BLED": 100, "wght": 700})
    a = points(bold, "A")
    # the traced box grows BOLD x each side and moves right as far, so its left side bearing stays, its top and bottom hold
    x, y = fontbuild.BOLD["x"], fontbuild.BOLD["y"]
    assert np.allclose([a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()], [-5, 605 + 2 * x, 0, 700], atol=1.5)
    assert y < x
    for g, adv in (("A", 650), ("V", 620), ("Aacute", 650)):
        assert bold["hmtx"][g][0] == (adv + 2 * x) * fontbuild.SCALE
    assert bold["glyf"]["Aacute"].isComposite()
    # the V's strokes thicken, so the notch between them rises
    middle = lambda f: (lambda v: v[(np.abs(v[:, 0] - 300 - x) < 40) & (v[:, 1] > 100)][:, 1].min())(points(f, "V"))
    assert middle(bold) > middle(instantiateVariableFont(TTFont(path), {"BLED": 100, "wght": 400})) + 20


def curved_v():
    """A trace of the V as cubics, each stroke split at a smooth joint and the acute notch rounded by one curve"""
    q = lambda *pts: " ".join(f"{x:.1f},{-y:.1f}" for x, y in pts)
    a, b = np.array([270.0, 260.0]), np.array([330.0, 260.0])
    straight = lambda p, r: q(p + (r - p) / 3, p + 2 * (r - p) / 3, r)
    d = "M" + q((0, 700)) + " L" + q((150, 700))
    for p, r in (((150, 700), (210, 480)), ((210, 480), a)):
        d += " C" + straight(np.array(p, float), np.array(r, float))
    d += " C" + q(a + (15, -55), b + (-15, -55), b)
    for p, r in ((b, (390, 480)), ((390, 480), (450, 700))):
        d += " C" + straight(np.array(p, float), np.array(r, float))
    d += " L" + q((600, 700))
    for p, r in (((600, 700), (475, 350)), ((475, 350), (350, 0))):
        d += " C" + straight(np.array(p, float), np.array(r, float))
    d += " L" + q((250, 0))
    for p, r in (((250, 0), (125, 350)), ((125, 350), (0, 700))):
        d += " C" + straight(np.array(p, float), np.array(r, float))
    return d + "Z"


def joint_kinks(font, name, at):
    """The turn in degrees at the on-curve point of a glyph nearest each of at, from the point before it to the point after it"""
    glyf = font["glyf"]
    coords, ends, flags = glyf[name].getCoordinates(glyf)
    c = np.array(coords, float) / fontbuild.SCALE
    out = []
    for q in at:
        i = int(np.argmin(np.linalg.norm(c - q, axis=1)))
        a, b = c[i] - c[i - 1], c[(i + 1) % len(c)] - c[i]
        out.append(abs(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))))
    return out


def test_a_traced_glyph_beside_a_filled_notch_keeps_its_smooth_joints_smooth_at_every_bleed(tmp_path):
    target.use("rules")
    t = traces() | {"V": dict(trace=curved_v(), adv=620, n=20, p=P)}
    path, made = fontbuild.build_face(measuring_stick(), t, "regular", "Test Rules", "Regular", list("AV´Á"), tmp_path)
    assert made["V"] == "traced"
    # the joints either side of the notch's curve and halfway down each inner side of the strokes
    joints = [(270, 260), (330, 260), (210, 480), (390, 480)]
    for b in (25, 50, 125):
        at = instantiateVariableFont(TTFont(path), {"BLED": b})
        assert max(joint_kinks(at, "V", joints)) < 1
    # the notch still opens up at 25, where the measuring stick's notch is sharper
    v = lambda b: (lambda pts: pts[(np.abs(pts[:, 0] - 300) < 40) & (pts[:, 1] > 100)][:, 1].min())(points(instantiateVariableFont(TTFont(path), {"BLED": b}), "V"))
    assert v(25) < v(100) - 5


def test_a_traced_stem_beside_a_filled_notch_keeps_an_even_width_where_its_baseline_moves_unevenly():
    trace = fontbuild.parse_trace(curved_v())
    pl = fontbuild.Placed(fontbuild.design(measuring_stick(), "V"), P)

    def change(q):
        """The baseline's change, filling the notch and with a short lump along the left stroke's inner side"""
        notch = 8 * np.exp(-np.sum((q - [300, 240]) ** 2, 1) / (2 * 20 ** 2))
        lump = 3 * np.exp(-((q[:, 1] - 480) / 25) ** 2 - ((q[:, 0] - 210) / 15) ** 2)
        return np.stack([lump, notch], 1)

    moved = [[(kind, Q + change(Q)) for kind, Q in c] for c in trace]
    out = fontbuild.displaced(trace, trace, [fontbuild.closed(c) for c in moved], pl)
    outline = np.vstack([fontbuild.bez(Q, np.linspace(0, 1, 200)) for _, Q in out[0]])

    def inner(y):
        """Where the left stroke's inner side crosses height y"""
        a, b = outline, np.roll(outline, -1, 0)
        k = (a[:, 1] - y) * (b[:, 1] - y) < 0
        return float(np.sort(a[k, 0] + (y - a[k, 1]) * (b[k, 0] - a[k, 0]) / (b[k, 1] - a[k, 1]))[1])

    ys = np.linspace(360, 620, 27)
    x = np.array([inner(y) for y in ys])
    # the trace draws the side straight, so the baseline's lump stays out of it
    assert np.ptp(x - np.polyval(np.polyfit(ys, x, 1), ys)) < 0.5
    # the notch still fills
    notch = lambda c: min(fontbuild.bez(Q, np.linspace(0, 1, 50))[:, 1].min() for _, Q in c[0] if abs(Q[:, 0].mean() - 300) < 20 and Q[:, 1].min() > 100)
    assert notch(out) > notch(trace) + 1
