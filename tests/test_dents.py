import numpy as np

from glyphstack import dents


def rect(w=400, h=300):
    pts = []
    for a, b in [((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))]:
        pts += list(np.linspace(a, b, 40, endpoint=False))
    return np.array(pts, float)


def bitten(depth=6, width=40):
    """The rectangle with a V bite taken out of the middle of its top edge"""
    P = rect()
    top = (P[:, 1] == 300) & (np.abs(P[:, 0] - 200) < width / 2)
    P[top, 1] -= depth * (1 - np.abs(P[top, 0] - 200) / (width / 2))
    return P


def test_a_bite_in_a_straight_edge_adds_turning_and_two_inflections():
    (t0, i0), (t1, i1) = dents.measure([rect()]), dents.measure([bitten()])
    assert abs(t0 - 360) < 1e-6 and i0 == 0
    # into the bite, out the other side and back to the edge, each about the bite's slope
    slope = np.degrees(np.arctan(6 / 20))
    assert abs((t1 - t0) - 4 * slope) < 0.1 * slope and i1 == 2


def test_rounding_a_curves_points_to_whole_units_does_not_count_as_a_dent():
    # an ellipse of 16 quadratics, its on-curve points and handles rounded as a TrueType glyph rounds them
    pen = dents.PolyPen()
    u = np.linspace(0, 2 * np.pi, 33)
    at = lambda a, r=1.0: tuple(np.round([1000.3 * r * np.cos(a), 700.7 * r * np.sin(a)]))
    pen.moveTo(at(0))
    for a in u[2::2]:
        pen.qCurveTo(at(a - u[1], 1 / np.cos(u[1])), at(a))
    pen.closePath()
    turning, inflections = dents.measure(pen.polys)
    assert abs(turning - 360) < 1 and inflections == 0


def test_lobes_add_up_to_the_net_turn():
    t = dents.turns(bitten())
    assert abs(dents.lobes(t).sum() - 360) < 1e-6
    assert abs(dents.merged(t).sum() - 360) < 1e-6
