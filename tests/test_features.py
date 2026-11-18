import io

import numpy as np
import uharfbuzz as hb
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from glyphstack import features, fontbuild

from test_fontbuild import bold_built, box, built

X, CAP = 500, 700


def composite(*parts):
    pen = TTGlyphPen(fontbuild.AnyGlyph())
    for name, dx, dy in parts:
        pen.addComponent(name, (1, 0, 0, 1, dx, dy))
    return pen.glyph()


def font():
    """A font laid out the way the measuring stick's are: accents drawn above the x-height on a 300 unit advance, combining accents built from them on the origin, accented letters composed

    Á and á are left out of the cmap, so the shaper keeps a letter and a combining accent apart rather than composing them
    """
    outlines = {
        "A": box(0, 0, 600, CAP), "V": box(0, 0, 500, CAP), "S": box(50, 0, 550, CAP), "a": box(50, 0, 450, X),
        "i": [box(100, 0, 200, X), box(100, 600, 200, 700)], "dotlessi": box(100, 0, 200, X),
        "j": [box(100, -200, 200, X), box(100, 600, 200, 700)], "dotlessj": box(100, -200, 200, X),
        "l": box(100, 0, 200, 750), "L": box(100, 0, 500, CAP),
        "acute": box(100, 550, 200, 650), "cedilla": box(100, -200, 200, -50), "commaaccent": box(120, -220, 180, -60),
        "ogonek": box(150, -200, 250, 0), "hyphen": box(50, 220, 350, 280), "parenleft": box(100, -150, 200, 650),
        "exclamdown": box(100, -200, 200, X), "periodcentered": box(150, 220, 250, 320),
    }
    glyphs = {n: g for n, g in outlines.items() if not isinstance(g, list)}
    for n, parts in outlines.items():
        if isinstance(parts, list):
            pen = TTGlyphPen(None)
            for g in parts:
                g.draw(pen, None)
            glyphs[n] = pen.glyph()
    adv = {n: 300 for n in ("acute", "cedilla", "ogonek")} | {"commaaccent": 300, "i": 300, "dotlessi": 300, "j": 300, "dotlessj": 300, "l": 300,
                                                                 "A": 650, "V": 550, "S": 600, "a": 500, "L": 550, "hyphen": 400, "parenleft": 300,
                                                                 "exclamdown": 300, "periodcentered": 400}
    glyphs |= {"acutecomb": composite(("acute", -150, 0)), "uni0327": composite(("cedilla", -150, 0)),
               "uni0326": composite(("commaaccent", -150, 0)), "uni0328": composite(("ogonek", -150, 0)),
               "Aacute": composite(("A", 0, 0), ("acute", 180, 200)), "aacute": composite(("a", 0, 0), ("acute", 100, 0)),
               "Scedilla": composite(("S", 0, 0), ("cedilla", 150, 0)), "Scommaaccent": composite(("S", 0, 0), ("commaaccent", 150, 0)),
               "iogonek": composite(("i", 0, 0), ("ogonek", 50, 0))}
    adv |= {"acutecomb": 0, "uni0327": 0, "uni0326": 0, "uni0328": 0, "Aacute": 650, "aacute": 500, "Scedilla": 600, "Scommaaccent": 600, "iogonek": 300}
    cmap = {ord("A"): "A", ord("V"): "V", ord("S"): "S", ord("a"): "a", ord("i"): "i", ord("j"): "j", ord("ı"): "dotlessi", ord("ȷ"): "dotlessj",
            ord("l"): "l", ord("L"): "L", ord("´"): "acute", ord("¸"): "cedilla", ord("˛"): "ogonek", ord("-"): "hyphen", ord("("): "parenleft",
            ord("¡"): "exclamdown", ord("·"): "periodcentered", 0x301: "acutecomb", 0x327: "uni0327", 0x326: "uni0326", 0x328: "uni0328",
            ord("Ş"): "Scedilla", ord("Ș"): "Scommaaccent", ord("į"): "iogonek"}
    order = [".notdef"] + sorted(glyphs)
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf({".notdef": box(0, 0, 10, 10)} | glyphs)
    glyf = fb.font["glyf"]
    for n in order:
        glyf[n].recalcBounds(glyf)
    fb.setupHorizontalMetrics({n: (adv.get(n, 100), getattr(glyf[n], "xMin", 0)) for n in order})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupOS2(sxHeight=X, sCapHeight=CAP, sTypoDescender=-200)
    fb.setupPost()
    return fb.font


def featured():
    """The font with the glyphs features.extras lists and its features compiled"""
    f = font()
    order = f.getGlyphOrder()
    ex = features.extras(f)
    glyf, hmtx = f["glyf"], f["hmtx"]
    for name, (src, dx, dy, dadv) in ex.items():
        order.append(name)
        glyf[name] = composite((src, dx, dy))
        hmtx[name] = (hmtx[src][0] + dadv, 0)
    f.setGlyphOrder(order)
    addOpenTypeFeaturesFromString(f, features.text(f, "  pos A V -50;\n"))
    return f, ex


def shape(f, text, language=None, feats=None):
    buf = io.BytesIO()
    f.save(buf)
    hbf = hb.Font(hb.Face(buf.getvalue()))
    b = hb.Buffer()
    b.add_str(text)
    b.guess_segment_properties()
    if language:
        b.language = language
    hb.shape(hbf, b, feats or {})
    order = f.getGlyphOrder()
    return [(order[i.codepoint], p.x_offset, p.y_offset, p.x_advance) for i, p in zip(b.glyph_infos, b.glyph_positions)]


def test_mark_puts_a_combining_accent_where_the_composite_puts_the_spacing_accent():
    f, _ = featured()
    for text, base, dx, dy in (("A\u0301", "A", 180, 200), ("a\u0301", "a", 100, 0)):
        (b, *_), (m, mx, my, madv) = shape(f, text)
        assert (b, m, madv) == (base, "acutecomb", 0)
        # the accent's ink starts at the base's advance plus the mark's offset, less the half advance the combining accent is moved by
        assert (f["hmtx"][base][0] + mx - 150, my) == (dx, dy)


def test_mark_places_an_accent_on_a_letter_no_composite_uses_above_its_top():
    f, _ = featured()
    (_, *_), (_, mx, my, _) = shape(f, "V\u0301")
    acute_bottom = 550 + my
    assert acute_bottom > CAP
    # near the middle of the V's top, 250, corrected by how far the composites put accents from the middle of their letters' tops
    assert abs(f["hmtx"]["V"][0] + mx - 250) <= 20


def test_mkmk_stacks_a_second_accent_above_the_first():
    f, _ = featured()
    _, (_, x1, y1, _), (_, x2, y2, _) = shape(f, "a\u0301\u0301")
    # the second accent's bottom clears the first's top by half the gap between the x-height and an accent above it
    assert (550 + y2) - (650 + y1) == 25
    # an accented letter's top anchor is its accent's, so an accent added to á lands where the second one does
    assert features.Anchors(f).top["aacute"][1] == X + y2


def test_ccmp_drops_the_dot_of_i_and_j_under_an_accent_above_and_keeps_it_otherwise():
    f, _ = featured()
    assert [g for g, *_ in shape(f, "i\u0301")] == ["dotlessi", "acutecomb"]
    assert [g for g, *_ in shape(f, "j\u0301")] == ["dotlessj", "acutecomb"]
    assert [g for g, *_ in shape(f, "i\u0327\u0301")] == ["dotlessi", "uni0327", "acutecomb"]
    assert [g for g, *_ in shape(f, "į\u0301")] == ["dotlessi", "uni0328", "acutecomb"]
    assert [g for g, *_ in shape(f, "\u012f\u0327\u0301")] == ["dotlessi", "uni0328", "uni0327", "acutecomb"]
    assert [g for g, *_ in shape(f, "i\u0327")] == ["i", "uni0327"]
    assert [g for g, *_ in shape(f, "ij")] == ["i", "j"]


def test_ccmp_keeps_the_ogonek_where_the_letter_draws_it():
    f, _ = featured()
    _, (_, ox, oy, _), _ = shape(f, "į\u0301")
    assert (f["hmtx"]["dotlessi"][0] + ox - 150, oy) == (50, 0)


def test_case_raises_punctuation_to_the_capitals():
    f, ex = featured()
    assert [g for g, *_ in shape(f, "(-", feats={"case": True})] == ["parenleft.case", "hyphen.case"]
    assert [g for g, *_ in shape(f, "(-")] == ["parenleft", "hyphen"]
    # the hyphen's middle moves from 250 to halfway up the capitals, the inverted mark onto the baseline
    assert ex["hyphen.case"] == ("hyphen", 0, CAP // 2 - 250, 0)
    assert ex["exclamdown.case"][2] == 200


def test_locl_takes_the_comma_below_in_romanian_and_moldavian():
    f, _ = featured()
    assert [g for g, *_ in shape(f, "Ş")] == ["Scedilla"]
    for lang in ("ro", "mo"):
        assert [g for g, *_ in shape(f, "Ş", language=lang)] == ["Scommaaccent"]


def test_locl_tightens_the_catalan_middle_dot_between_two_l():
    f, ex = featured()
    assert [g for g, *_ in shape(f, "l·l")] == ["l", "periodcentered", "l"]
    glyphs = shape(f, "l·l", language="ca")
    assert [g for g, *_ in glyphs] == ["l", "periodcentered.loclCAT", "l"]
    assert glyphs[1][3] < f["hmtx"]["periodcentered"][0]
    assert [g for g, *_ in shape(f, "L·L", language="ca")] == ["L", "periodcentered.loclCAT.case", "L"]
    assert [g for g, *_ in shape(f, "a·a", language="ca")] == ["a", "periodcentered", "a"]


def test_kern_still_kerns_with_the_features_beside_it():
    f, _ = featured()
    assert shape(f, "AV")[0][3] == f["hmtx"]["A"][0] - 50


def test_a_composite_takes_the_accent_below_its_character_names():
    stick = font()
    cedilla = dict(char="¸", glyph="cedilla", offset=[150, 0])
    comma = dict(char=None, glyph="commaaccent", offset=[150, 0])
    assert fontbuild.below_accent(stick, "Ș", [dict(char="S", glyph="S", offset=[0, 0]), cedilla], 1)[1]["glyph"] == "commaaccent"
    assert fontbuild.below_accent(stick, "Ş", [dict(char="S", glyph="S", offset=[0, 0]), comma], 1)[1] == cedilla | dict(offset=[150.0, 0])
    assert fontbuild.below_accent(stick, "Ç", [cedilla], 1) == [cedilla]


def test_the_fonts_carry_release_names_installable_embedding_and_smart_dropout(tmp_path):
    path, _ = built(tmp_path)
    vf = TTFont(path)
    name = vf["name"]
    assert {r.platformID for r in name.names} == {3}
    for nid in (0, 1, 2, 3, 4, 5, 6, 13, 14, 25):
        assert name.getName(nid, 3, 1, 0x409), nid
    assert name.getDebugName(4) == "Test Rules Regular" and name.getDebugName(5) == f"Version {fontbuild.VERSION}"
    # head stores the version as 16.16 fixed point
    assert abs(vf["head"].fontRevision - float(fontbuild.VERSION)) < 1e-4
    assert vf["OS/2"].fsType == 0 and vf["gasp"].gaspRange == {0xFFFF: 0x000F}
    assert "SCANCTRL" in " ".join(vf["prep"].program.getAssembly())
    assert vf["glyf"][".notdef"].numberOfContours == 2
    assert vf["OS/2"].sTypoLineGap == 0 == vf["hhea"].lineGap
    assert vf["glyf"]["V"].flags[0] & 0x40 and vf["glyf"]["Aacute"].components[0].flags & 0x400


def test_the_windows_metrics_clear_every_master_and_the_line_metrics_are_shared(tmp_path):
    path, _ = built(tmp_path)
    vf = TTFont(path)
    # Á's accent tops out at 820 and grows past it at the strongest Bleed
    assert vf["OS/2"].usWinAscent >= 820 * fontbuild.SCALE and vf["OS/2"].usWinDescent >= 0
    other = tmp_path / "other.ttf"
    o = TTFont(path)
    o["OS/2"].usWinAscent, o["OS/2"].sTypoAscender = 9999, 1
    o.save(other)
    fontbuild.share_metrics([path, other])
    a, b = TTFont(path), TTFont(other)
    assert a["OS/2"].usWinAscent == b["OS/2"].usWinAscent == 9999
    assert b["OS/2"].sTypoAscender == a["OS/2"].sTypoAscender


def test_the_named_instances_are_the_face_at_bleed_100_and_every_standard_weight(tmp_path):
    path, _ = built(tmp_path)
    vf = TTFont(path)
    name = vf["name"]
    assert [(name.getDebugName(i.subfamilyNameID), i.coordinates) for i in vf["fvar"].instances] == [("Regular", {"BLED": 100})]
    try:
        fontbuild.set_top_weight(600)
        (tmp_path / "bold").mkdir()
        bold, _ = bold_built(tmp_path / "bold")
    finally:
        fontbuild.set_top_weight(700)
    vf = TTFont(bold)
    got = [(vf["name"].getDebugName(i.subfamilyNameID), i.coordinates["wght"]) for i in vf["fvar"].instances]
    assert got == [("Regular", 400), ("Medium", 500), ("SemiBold", 600)]
    assert all(i.coordinates["BLED"] == 100 for i in vf["fvar"].instances)
