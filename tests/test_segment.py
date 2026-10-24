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
