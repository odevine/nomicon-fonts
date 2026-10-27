import re

import numpy as np

from glyphstack import styles, target, template


def test_a_twin_takes_its_low_form_on_the_baseline_and_its_high_form_above_it():
    target.use(target.DEFAULT)
    em = target.get()["templates"]["px_per_em"] / template.UP
    # letters 0.5 em tall on a baseline at y 1000, and a mark beside them
    letters = [dict(bbox=(1000 - int(0.5 * em), 1000, x, x + 60)) for x in range(100, 700, 80)]
    low = dict(bbox=(990, 1030, 720, 760))
    high = dict(bbox=(1000 - int(0.75 * em), 1000 - int(0.45 * em), 720, 760))
    assert template.place_twin(low, letters + [low], 1.0, "„”") == "„"
    assert template.place_twin(high, letters + [high], 1.0, "„”") == "”"
    # with no letters beside it a mark is left unlabelled
    assert template.place_twin(low, [low], 1.0, "„”") is None


def test_a_low_mark_takes_the_face_of_the_letters_on_its_line():
    target.use(target.DEFAULT)
    em = target.get()["templates"]["px_per_em"] / template.UP
    letters = [dict(bbox=(1000 - int(0.7 * em), 1000, x, x + 60), cx=x + 30, ch="i:A") for x in range(100, 400, 80)]
    above = [dict(bbox=(1000 - int(1.9 * em), 1000 - int(1.2 * em), 420, 480), cx=450, ch="A")]
    under = dict(bbox=(1002, 1012, 410, 490), cx=450, ch="_", scale=1.0)
    assert template.resolve_faces([under], letters + above)[0]["ch"] == "i:_"


class Font(dict):
    """The two tables anchor reads, an x-height of 450 units in 1000"""

    def __init__(self):
        super().__init__({"OS/2": type("T", (), {"sxHeight": 450})(), "head": type("T", (), {"unitsPerEm": 1000})()})


FONTS = type("T", (), {"fonts": {"regular": Font(), "italic": Font()}})()


def dotted_i():
    """A labelled i on a 40 by 30 window: a stem with its dot 6 px over it"""
    stem = np.zeros((40, 30), bool)
    stem[14:38, 10:18] = True
    dot = np.zeros((40, 30), bool)
    dot[2:9, 10:17] = True
    cov = (stem | dot).astype(np.float32)
    return dict(ch="i", cov=cov, mask=stem | dot, parts=[stem, dot], box=(100, 140, 200, 230), bbox=(102, 138, 210, 218), cx=214, cy=120, scale=1.0)


def test_a_labelled_i_splits_into_a_dotless_stem_and_a_dot_with_its_anchor():
    target.use(target.DEFAULT)
    u = dotted_i()
    stem, dot = template.split_dotted([u], FONTS, 1.0)
    assert stem["ch"] == "ı" and dot["ch"] == "˙" and stem["part_of"] == dot["part_of"] == "i"
    assert stem["bbox"] == (114, 138, 210, 218) and dot["bbox"] == (102, 109, 210, 217)
    assert np.allclose(stem["cov"].sum(), 24 * 8) and np.allclose(dot["cov"].sum(), 49)
    xh = 0.45 * target.get()["templates"]["px_per_em"] / template.UP
    assert dot["anchor"]["base"] == "ı" and np.isclose(dot["anchor"]["gap"], round(5 / xh, 3)) and np.isclose(dot["anchor"]["dx"], round(-0.5 / xh, 3))
    # the stems and dots are kept beside the letter, which bench alone scores
    assert "parts" not in stem and "part_of" not in u


def test_a_mark_over_a_capital_takes_the_capital_form_of_its_accent():
    target.use(target.DEFAULT)
    em = target.get()["templates"]["px_per_em"] / template.UP
    E = dict(ch="E", bbox=(1000 - int(0.7 * em), 1000, 100, 160), cx=130)
    e = dict(ch="e", bbox=(1000 - int(0.45 * em), 1000, 300, 350), cx=325)
    acute = lambda x, y1: dict(ch="´", bbox=(y1 - 18, y1, x, x + 16), cx=x + 8)
    out = template.anchor_marks([E, e, acute(120, E["bbox"][0] - 10), acute(318, e["bbox"][0] - 10)], FONTS, 1.0)
    marks = [u for u in out if "anchor" in u]
    assert [m["ch"] for m in marks] == [styles.CASE["´"], "´"] and [m["anchor"]["base"] for m in marks] == ["E", "e"]
    assert styles.glyph_folder(styles.CASE["´"]) == "acute.case"


def square(side=200.0, step=1.0):
    """A square outline of the given side in font units as a dense closed polyline"""
    t = np.arange(0, side, step)
    return [np.concatenate([np.stack([t, 0 * t], 1), np.stack([0 * t + side, t], 1), np.stack([side - t, 0 * t + side], 1), np.stack([0 * t, side - t], 1)])]


def test_the_prior_renders_the_outline_on_its_spread_contour_and_blends_as_instances():
    from glyphstack import fontfit
    from glyphstack.contours import contours_at

    g = object.__new__(fontfit.FontGlyph)
    g.contours = square()
    # scale 0.2 puts the 200 unit square at 40 grid px, 30 px in, with a spread of 2 px
    p = np.array([0.2, 1.0, 0.0, 30.0, 70.0, 2.0])
    look = dict(d=2.0, w=1.5, r=0.0)
    prior = fontfit.render(g, p, (100, 100), look)
    xs = np.vstack(contours_at(prior, 0.5))[:, 0]
    assert abs(xs.min() - 28) < 0.3 and abs(xs.max() - 72) < 0.3
    stack = np.zeros((100, 100), np.float32)
    blend = fontfit.with_prior(g, p, stack, 2, look, 2)
    assert np.allclose(blend, prior / 2)


def test_the_print_look_finds_the_spread_edge_and_rounding_a_stack_was_drawn_with():
    from glyphstack import fontfit

    g = object.__new__(fontfit.FontGlyph)
    g.contours = square()
    p = np.array([0.4, 1.0, 0.0, 60.0, 140.0, 2.0])
    stack = fontfit.render(g, p, (200, 200), dict(d=3.0, w=2.0, r=6.0))
    look = fontfit.print_look(g, p, stack)
    assert look["r"] == 6.0 and abs(look["d"] - 3.0) < 0.1 and abs(look["w"] - 2.0) < 0.1
    # drawn on a grid of its own, the outline traces back to the same square in font units
    mean, q = fontfit.from_outline(g, p, look)
    path, nodes, _ = fontfit.trace(mean, q)
    xs = [float(v) for v in re.findall(r"(-?[\d.]+),", path)]
    # the spread of 3 grid px is 7.5 font units at scale 0.4
    assert abs(min(xs) + 7.5) < 1 and abs(max(xs) - 207.5) < 1
