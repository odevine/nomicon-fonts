import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from glyphstack import fontfit, kerning, styles, template


def tiny_font(upm, shear=0):
    """A font whose only glyph `a` is a 400 by 500 box per 1000 units, sheared by `shear` units at the top"""
    k = upm / 1000
    pen = TTGlyphPen(None)
    pts = [(100, 0), (500, 0), (500 + shear, 500), (100 + shear, 500)]
    pen.moveTo((round(pts[0][0] * k), round(pts[0][1] * k)))
    for x, y in pts[1:]:
        pen.lineTo((round(x * k), round(y * k)))
    pen.closePath()
    fb = FontBuilder(upm, isTTF=True)
    fb.setupGlyphOrder([".notdef", "a"])
    fb.setupCharacterMap({ord("a"): "a"})
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph(), "a": pen.glyph()})
    fb.setupHorizontalMetrics({".notdef": (500, 0), "a": (600, 100)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({})
    fb.setupOS2()
    fb.setupPost()
    return fb.font


def test_keys_name_the_face_and_the_character():
    assert styles.key("regular", "a") == "a" and styles.key("italic", "a") == "i:a"
    assert styles.key("italic", ":") == "i::"
    assert [styles.style_of(k) for k in ("a", ":", "i:a", "i::")] == ["regular", "regular", "italic", "italic"]
    assert [styles.char_of(k) for k in ("a", "i:a", "i::")] == ["a", "a", ":"]


def test_stems_names_and_ids_never_collide_across_faces():
    keys = [styles.key(s, c) for s in styles.names() for c in "aA.:"]
    for fn in (styles.stem, styles.glyph_name):
        assert len({fn(k) for k in keys}) == len(keys)
    assert styles.stem("a") == "a" and styles.stem(".") == "ord46" and styles.stem("i:a") == "it_a"
    assert styles.stem("A") == "A_uc" and styles.stem("i:A") == "it_A_uc"


def test_stems_stay_distinct_on_a_case_insensitive_filesystem():
    keys = [styles.key(s, c) for s in styles.names() for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;"]
    assert len({styles.stem(k).lower() for k in keys}) == len(keys)
    assert styles.glyph_name("a") == "a" and styles.glyph_name("i:A") == "A_uc_it" and styles.glyph_name("i:.") == "period_it"
    assert styles.instance_id("a", 3) == "a-003" and styles.instance_id("i:a", 3) == "a-it-003"


def test_pair_keys_of_the_two_faces_are_distinct_lengths():
    assert styles.pair_key("r", "n") == "rn" and styles.pair_key("i:r", "i:n") == "i:rn"


def test_template_size_follows_the_em():
    t1 = template.render_templates(tiny_font(1000), 1.0, "a")["a"]
    t2 = template.render_templates(tiny_font(2000), 1.0, "a", style="italic")["i:a"]
    assert abs(t1["h"] - t2["h"]) < 0.5 and abs(t1["w"] - t2["w"]) < 0.5


def test_templates_merge_faces_and_scale_search_can_use_the_regular_one():
    tp = template.Templates({"regular": tiny_font(1000), "italic": tiny_font(1000, shear=150)})
    assert set(tp.at(1.0)) == {"a", "i:a"}
    assert set(tp.at(1.0, only=("regular",))) == {"a"}
    assert set(template.Templates(tiny_font(1000)).at(1.0)) == {"a"}
    # the slanted italic box is wider than the upright one
    both = tp.at(1.0)
    assert both["i:a"]["w"] > both["a"]["w"]


def test_face_transform_corrects_the_scale_for_units_per_em():
    shared = np.array([0.6, 1.01, 0.002, 10.0, 20.0, 1.5])
    p = fontfit.face_transform(shared, tiny_font(1000), tiny_font(2000))
    assert np.isclose(p[0], 0.3) and np.allclose(p[1:], shared[1:])
    assert shared[0] == 0.6


def test_kerning_pairs_are_measured_within_one_face(monkeypatch):
    class NoKern(kerning.Kerning):
        def __init__(self, font):
            self.cmap, self.hmtx, self.glyf = font.getBestCmap(), font["hmtx"], font["glyf"]

        def pair(self, a, b):
            return 0

    monkeypatch.setattr(kerning, "Kerning", NoKern)
    cov = np.zeros((20, 30), np.float32)
    cov[:, 5:12] = 1
    unit = lambda ch, x: dict(cov=cov, box=(0, 20, x, x + 30), bbox=(0, 20, x + 5, x + 12), cx=x + 8, ch=ch)
    scans = {"s": dict(card="c", angle=0, units=[unit("a", 100), unit("i:a", 120), unit("i:a", 140)])}
    args = (scans, {"s": 0.5}, {"c": [[1.0, 0], [0, 1.0]]}, tiny_font(1000), 0.6, 1.0)
    assert kerning.measure_pairs(*args) == []
    italic = kerning.measure_pairs(*args, "italic")
    assert [r["pair"] for r in italic] == ["aa"] and italic[0]["style"] == "italic"


def test_a_character_that_fits_both_faces_takes_the_face_of_its_neighbours():
    unit = lambda ch, cx, cy=100: dict(ch=ch, cx=cx, cy=cy, bbox=(cy - 10, cy + 10, cx - 5, cx + 5))
    kept = [unit("a", 10), unit("i:a", 300), unit("i:a", 320, cy=900)]
    out = template.resolve_faces([unit(".", 290), unit(".", 20), unit(".", 310, cy=900), unit(".", 50, cy=5000)], kept)
    assert [u["ch"] for u in out] == ["i:.", ".", "i:.", "."]
