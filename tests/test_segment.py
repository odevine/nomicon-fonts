import cv2
import numpy as np

from glyphstack import target, units


def page():
    return np.full((200, 400), 220, np.uint8)


def slanted_i(g, x, slant=0.25, dot=(14, 14)):
    """An italic i: a 12 px stem 60 px tall leaning right by `slant`, with its dot over the stem's top, not its foot"""
    pts = np.array([[x, 160], [x + 12, 160], [x + 12 + 60 * slant, 100], [x + 60 * slant, 100]], np.int32)
    cv2.fillPoly(g, [pts], 20)
    dx = int(x + 60 * slant + 18 * slant)
    g[82 : 82 + dot[0], dx : dx + dot[1]] = 20


def ids(comps):
    return sorted(len(c["ids"]) for c in comps)


def test_an_italic_dot_joins_its_slanted_stem():
    target.use(target.DEFAULT)
    g = page()
    slanted_i(g, 100)
    _, comps = units.segment(g)
    assert len(comps) == 1 and len(comps[0]["ids"]) == 2


def test_a_dot_below_the_component_size_limit_still_joins_an_i():
    target.use(target.DEFAULT)
    g = page()
    # an 11 px dot falls under min_area after the blur but above dot_min_area
    slanted_i(g, 100, slant=0.0, dot=(11, 11))
    _, comps = units.segment(g)
    assert len(comps) == 1 and len(comps[0]["ids"]) == 2


def test_a_small_speck_above_a_wide_letter_stays_out():
    target.use(target.DEFAULT)
    g = page()
    # an n-like arch, as wide at its top as it is tall, with a speck of texture just above it
    g[100:160, 100:112] = 20
    g[100:160, 148:160] = 20
    g[100:112, 100:160] = 20
    g[84:94, 125:135] = 20
    _, comps = units.segment(g)
    assert ids(comps) == [1]


def boxes(comps):
    return sorted((c["y0"], c["x0"], len(c["ids"])) for c in comps)


def test_quote_halves_side_by_side_join_into_one_mark():
    target.use(target.DEFAULT)
    g = page()
    # two 15 px wide, 36 px tall halves 7 px apart, as a double quote prints
    g[60:96, 100:115] = 20
    g[60:96, 122:137] = 20
    _, comps = units.segment(g)
    assert ids(comps) == [2]
    assert (comps[0]["x0"], comps[0]["x1"]) == (100, 137)


def test_a_colon_and_a_semicolon_join_their_dots():
    target.use(target.DEFAULT)
    g = page()
    # a colon's two 18 px dots 26 px apart, and a semicolon's dot over a comma
    g[100:118, 100:118] = 20
    g[144:162, 100:118] = 20
    g[100:118, 200:218] = 20
    g[144:180, 202:216] = 20
    _, comps = units.segment(g)
    assert ids(comps) == [2, 2]


def test_the_dot_of_an_exclamation_mark_joins_its_stem():
    target.use(target.DEFAULT)
    g = page()
    g[60:135, 100:116] = 20
    g[145:163, 99:117] = 20
    _, comps = units.segment(g)
    assert ids(comps) == [2] and comps[0]["y1"] == 163


def test_marks_beside_letters_and_an_ellipsis_stay_apart():
    target.use(target.DEFAULT)
    g = page()
    # a period on the baseline beside a stem, an ellipsis, and a comma beside a quote half of another size
    g[100:160, 40:52] = 20
    g[142:160, 56:74] = 20
    for x in (120, 150, 180):
        g[142:160, x : x + 18] = 20
    g[60:96, 260:275] = 20
    g[140:160, 280:294] = 20
    _, comps = units.segment(g)
    assert ids(comps) == [1] * len(comps)


def test_two_specks_of_texture_one_over_another_make_no_mark():
    target.use(target.DEFAULT)
    g = page()
    # dots under the profile's min_area, as specks of a halftone screen are
    g[100:111, 100:111] = 20
    g[135:146, 100:111] = 20
    _, comps = units.segment(g)
    assert comps == []


def test_a_dot_under_a_wide_foot_stays_apart():
    target.use(target.DEFAULT)
    g = page()
    # a blob of screen the size of a dot just under a letter as wide at its foot as it is tall
    g[80:140, 100:160] = 20
    g[90:130, 110:150] = 220
    g[148:166, 120:138] = 20
    _, comps = units.segment(g)
    assert ids(comps) == [1, 1]
