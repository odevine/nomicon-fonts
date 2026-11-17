import numpy as np

from glyphstack import corners, fontbuild, target
from glyphstack.contours import contours_at
from test_fontbuild import LOOK, P, V, measuring_stick


def test_a_stack_drawn_with_the_print_look_fills_the_inner_corner_and_rounds_the_outer_ones():
    target.use("rules")
    cs = fontbuild.design(measuring_stick(), "V")
    pl = fontbuild.Placed(cs, P)
    # a stack that is exactly the print look's edge, as straight segments in font units
    edge = contours_at(-pl.field(1.0, LOOK), 0.0)
    trace = [fontbuild.closed([("L", fontbuild.line(a, b)) for a, b in zip(q[:-1], q[1:])]) for q in (pl.font(e[::4]) for e in edge)]
    rows = corners.glyph_corners(cs, trace, pl, LOOK)
    inner = [r for r in rows if r[0] == "inner"]
    outer = [r for r in rows if r[0] == "outer"]
    assert len(inner) >= 1 and len(outer) >= 4
    assert all(r[2] > 10 for r in inner) and all(r[2] < 0 for r in outer)
    # measured against the look itself, nothing is left over
    assert all(abs(r[3]) < 3 for r in rows)
    s = corners.summary(rows)
    assert {b["kind"] for b in s} == {"inner", "outer"} and sum(b["n"] for b in s) == len(rows)
