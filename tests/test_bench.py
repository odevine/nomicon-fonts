import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from glyphstack import bench, font, target, truth

FONT = Path(os.environ.get("NOMICON_FONT", ""))
ITALIC = Path(os.environ.get("NOMICON_ITALIC_FONT", ""))
needs_fonts = pytest.mark.skipif(not (FONT.is_file() and ITALIC.is_file()), reason="set NOMICON_FONT and NOMICON_ITALIC_FONT to TrueType fonts")


def faces(par):
    return "".join({"regular": "r", "italic": "i", "symbol": "s"}[f] for _, f in par)


def test_flavor_reminder_text_and_ability_words_read_as_italic():
    pars = truth.paragraphs("Rules:\nDraw a card. (It goes to your hand.)\nLandfall — Gain 1 life.\n{T}: Add {G}.\nFlavor:\n“Grow.”")
    assert len(pars) == 4
    first = "".join(c for c, _ in pars[0])
    assert faces(pars[0]) == "r" * first.index("(") + "i" * (len(first) - first.index("("))
    assert faces(pars[1]).startswith("i" * len("Landfall — ")) and faces(pars[1])[len("Landfall — ")] == "r"
    assert pars[2][0] == ("{T}", "symbol") and pars[2][-2] == ("{G}", "symbol")
    assert set(faces(pars[3])) == {"i"}
    assert set(faces(truth.paragraphs("Rules:\nChoose one — Draw a card.")[0])) == {"r"}


def test_words_are_split_at_spaces_and_wrapped_to_the_ink_width():
    words = truth.tokens([(c, "regular") for c in "aa bbb c"])
    assert ["".join(c for c, _ in w) for w in words] == ["aa", "bbb", "c"]
    widths = [20.0, 30.0, 10.0, 40.0]
    # 20 + 5 + 30 + 5 + 10 = 70 is nearest an ink width of 72
    assert truth.wrap_end(widths, 5.0, 72, 0, 4) == 2
    assert truth.wrap_end(widths, 5.0, 72, 1, 2) == 1


def test_units_are_sorted_into_finds_merges_fragments_and_strays():
    chars = [dict(ch="a", face="regular", y0=100, y1=140, x0=100, x1=130), dict(ch="b", face="regular", y0=100, y1=150, x0=140, x1=170),
             dict(ch="c", face="italic", y0=100, y1=140, x0=200, x1=230), dict(ch="d", face="regular", y0=100, y1=150, x0=300, x1=330)]
    u = lambda y0, y1, x0, x1: dict(bbox=(y0, y1, x0, x1))
    units = [u(100, 140, 100, 130), u(100, 150, 140, 170), u(100, 140, 200, 230), u(100, 150, 295, 345), u(100, 150, 245, 290), u(130, 140, 300, 310)]
    # a and its right label, b labelled as another letter, c in the wrong face, two merged into one by d, a texture blob
    units[3] = u(100, 150, 200, 330)
    kept = [dict(units[0], ch="a"), dict(units[1], ch="h"), dict(units[2], ch="c"), dict(units[4], ch="o")]
    out, rows = bench.score(units, kept, chars)
    assert (out["found"], out["right"], out["wrong"], out["wrong_face"]) == (3, 1, 1, 1)
    assert (out["merged"], out["stray"], out["fragments"], out["false_labels"]) == (1, 1, 1, 1)


def test_units_on_ignored_characters_are_not_counted():
    chars = [dict(ch="a", face="regular", y0=100, y1=140, x0=100, x1=130), dict(ch="|", face="italic", y0=100, y1=150, x0=200, x1=215)]
    units = [dict(bbox=(100, 140, 100, 130)), dict(bbox=(100, 150, 198, 217))]
    kept = [dict(units[0], ch="a"), dict(units[1], ch="l")]
    out, rows = bench.score(units, kept, chars[:1], ignore=chars[1:])
    # the unit on the ignored bar is neither a stray nor a false label
    assert (out["truth"], out["ignored"], out["right"], out["stray"], out["false_labels"]) == (1, 1, 1, 0, 0)
    assert len(rows) == 1


def test_characters_on_a_poorly_placed_line_or_past_the_crop_are_marked():
    t = dict(shape=[500, 400], chars=[dict(line=0, ncc=0.9, x0=10, x1=40, y0=10, y1=50), dict(line=0, ncc=0.8, x0=380, x1=410, y0=10, y1=50),
                                      dict(line=1, ncc=0.3, x0=10, x1=40, y0=100, y1=140), dict(line=1, ncc=0.4, x0=50, x1=80, y0=100, y1=140)])
    bench.mark(t)
    assert [c["cut"] for c in t["chars"]] == [False, True, False, False]
    assert [c["uncertain"] for c in t["chars"]] == [False, False, True, True]


def test_weight_ratio_compares_each_character_on_both_backgrounds():
    rights = [("a", "plain", 100.0), ("a", "plain", 110.0), ("a", "strong", 95.0), ("e", "plain", 80.0), ("e", "strong", 88.0), ("o", "plain", 70.0)]
    w = bench.weight_ratio(rights)
    assert w["chars"] == 2 and w["ratio"] == pytest.approx(np.median([95 / 105, 88 / 80]), abs=1e-3)


def test_lines_are_found_with_their_baselines_and_ink_widths():
    target.use(target.DEFAULT)
    D = np.zeros((600, 1400), np.float32)
    # three lines of x-height blocks, the last a short paragraph end, under a frame bar along the top
    for base, right in ((150, 1300), (280, 1200), (410, 500)):
        for x in range(100, right, 45):
            D[base - 50 : base, x : x + 30] = 1
        D[base - 80 : base - 50, 100:130] = 1
    D[:12] = 1
    lines = truth.text_lines(D, truth.Glyphs({}, 0.9))
    assert [l["base"] for l in lines] == pytest.approx([150, 280, 410], abs=2)
    # each line ends at its last block's right side
    assert [l["x1"] for l in lines] == [max(range(100, r, 45)) + 30 for r in (1300, 1200, 500)]


@needs_fonts
def test_fit_places_rendered_text_where_it_was_drawn():
    target.use(target.DEFAULT)
    fonts = {"regular": font.load_font(str(FONT)), "italic": font.load_font(str(ITALIC))}
    text = "Rules:\nThe quick brown fox jumps.\nFlavor:\nPack my box with jugs."
    gl = truth.Glyphs(fonts, 0.88)
    H, W = 1400, 3000
    card = np.full((int(H / 0.315), W), 210, np.uint8)
    y0 = int(truth.BOX[0] * card.shape[0])
    x0 = int(truth.BOX[2] * W) + 120
    drawn = []
    for base, par in ((y0 + 200, truth.paragraphs(text)[0]), (y0 + 330, truth.paragraphs(text)[1])):
        pen = float(x0)
        for ch, face in par:
            t = gl(ch, face)
            if t["img"] is not None:
                ox, oy = t["off"]
                h, w = t["img"].shape
                sl = (slice(base + oy, base + oy + h), slice(int(pen) + ox, int(pen) + ox + w))
                card[sl] = np.minimum(card[sl], (210 - 185 * t["img"]).astype(np.uint8))
                ys, xs = np.nonzero(t["img"] > 0.5)
                drawn.append((ch, int(pen) + ox + (xs.min() + xs.max()) / 2, base + oy + (ys.min() + ys.max()) / 2))
            pen += t["adv"]
    card = cv2.GaussianBlur(card, (0, 0), 1.0)
    got = truth.fit(card, text, fonts, scales=[0.87, 0.88, 0.89])
    assert got["scale"] == pytest.approx(0.88) and len(got["chars"]) == len(drawn)
    err = [np.hypot((c["x0"] + c["x1"]) / 2 - x, (c["y0"] + c["y1"]) / 2 - y) for c, (_, x, y) in zip(got["chars"], drawn)]
    assert [c["ch"] for c in got["chars"]] == [d[0] for d in drawn]
    assert max(err) < 3


def test_a_kerned_neighbour_inside_a_units_box_does_not_make_it_merged():
    # a T whose bar reaches over the o after it: the o's centre lies in the T's box but its ink does not
    chars = [dict(ch="T", face="regular", y0=100, y1=160, x0=100, x1=150), dict(ch="o", face="regular", y0=125, y1=160, x0=135, x1=165)]
    mask = np.zeros((60, 60), bool)
    mask[0:8, 0:60] = True
    mask[0:60, 22:30] = True
    t = dict(bbox=(100, 160, 100, 160), box=(100, 160, 100, 160), mask=mask)
    o = dict(bbox=(125, 160, 135, 165))
    out, rows = bench.score([t, o], [dict(t, ch="T"), dict(o, ch="o")], chars)
    assert (out["found"], out["right"], out["merged"], out["false_labels"]) == (2, 2, 0, 0)
    # ink that reaches the o's middle does make the unit a merge
    mask[30:52, 40:56] = True
    out, _ = bench.score([t], [dict(t, ch="T")], chars)
    assert out["merged"] == 1


@needs_fonts
def test_fit_takes_the_exponent_form_where_a_glyph_is_printed_small_and_raised(monkeypatch):
    target.use(target.DEFAULT)
    # every line is fitted again, so the test does not hang on how poorly the plain fit does
    monkeypatch.setattr(truth, "WIDE_NCC", 1.01)
    fonts = {"regular": font.load_font(str(FONT)), "italic": font.load_font(str(ITALIC))}
    text = "Rules:\nso E 9090 N 9090 on."
    gl = truth.Glyphs(fonts, 0.88)
    W = 3000
    card = np.full((int(1400 / 0.315), W), 210, np.uint8)
    base = int(truth.BOX[0] * card.shape[0]) + 200
    pen = float(int(truth.BOX[2] * W) + 120)
    small = []
    for ch, face in truth.paragraphs(text)[0]:
        sm = ch in "90"
        t = gl(ch, face, sm)
        if t["img"] is not None:
            ox, oy = t["off"]
            h, w = t["img"].shape
            sl = (slice(base + oy, base + oy + h), slice(int(pen) + ox, int(pen) + ox + w))
            card[sl] = np.minimum(card[sl], (210 - 185 * t["img"]).astype(np.uint8))
            small.append(sm)
        pen += t["adv"]
    card = cv2.GaussianBlur(card, (0, 0), 1.0)
    got = truth.fit(card, text, fonts, scales=[0.88])
    assert [bool(c.get("small")) for c in got["chars"]] == small
    assert min(c["ncc"] for c in got["chars"]) > 0.8
