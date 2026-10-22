import cv2
import numpy as np

from glyphstack import export_site, fontfit, kerning, sweep
from glyphstack.contours import contours_at


def blob(size=(120, 100)):
    img = np.zeros(size, np.float32)
    cv2.ellipse(img, (50, 60), (22, 34), 0, 0, 360, 1, -1)
    return cv2.GaussianBlur(img, (0, 0), 2.0)


def test_grid_ends_at_all_instances():
    assert sweep.grid(12)[-1] == 12 and sweep.grid(107)[-1] == 107
    assert sweep.grid(107) == sorted(set(sweep.grid(107)))


def test_noise_model_recovers_floor_and_noise():
    rows = [dict(n=n, rms_vs_ref_um=float(np.sqrt(4.0**2 + 8.0**2 / n))) for n in sweep.grid(60)]
    b, s = sweep.noise_model(rows)
    assert abs(b - 4.0) < 0.05 and abs(s - 8.0) < 0.05
    # 1 - sqrt(N / N') falls below 7 % between 14 and 16 on the pilot grid
    assert sweep.recommended_n([dict(n=n) for n in sweep.BASE_NS], s) == 16


def test_sweep_error_falls_with_more_instances():
    rng = np.random.default_rng(0)
    base = blob()
    inst = np.clip(np.array([np.roll(base, tuple(rng.integers(-2, 3, 2)), (0, 1)) + rng.normal(0, 0.08, base.shape) for _ in range(20)]), 0, 1)
    rows = sweep.sweep(inst, contours_at(base, 0.5), draws=6)
    assert rows[0]["rms_vs_ref_um"] > rows[-1]["rms_vs_ref_um"]
    assert rows[-1]["n"] == 20 and "noise_um" in rows[0] and "noise_um" not in rows[-1]


def test_ink_edges_and_neighbour_choice():
    cov = np.zeros((20, 30), np.float32)
    cov[:, 5:12] = 1
    a = dict(cov=cov, box=(0, 20, 100, 130), bbox=(0, 20, 105, 112), cx=108, ch="r")
    b = dict(cov=cov, box=(0, 20, 120, 150), bbox=(0, 20, 125, 132), cx=128, ch="n")
    far = dict(cov=cov, box=(0, 20, 400, 430), bbox=(0, 20, 405, 412), cx=408, ch="x")
    assert kerning.ink_edges(a, 0.5) == (105, 112)
    assert kerning.right_neighbour(a, [a, b, far]) is b
    assert kerning.right_neighbour(b, [a, b, far]) is None


def test_summarise_groups_pairs():
    rows = [dict(scan="s", card="c", pair="rn", base=10.0, kern=30, dev_nokern=31.0 + i, dev_kern=1.0 + i) for i in range(3)]
    out = kerning.summarise(rows)["rn"]
    assert out["n"] == 3 and out["kern"] == 30 and abs(out["mean_dev_kern"] - 2.0) < 1e-9


def test_trace_nodes_sit_on_the_half_contour():
    from scipy.spatial import cKDTree

    mean = blob()
    p = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    path, nodes, nc = fontfit.trace(mean, p)
    import re

    # node positions are the last coordinate pair of each command, control points of curves sit off the contour
    pts = np.array([[float(v) for v in re.findall(r"[-\d.]+,[-\d.]+", args)[-1].split(",")] for _, args in re.findall(r"([MLC])([^MLCZ]*)", path)])
    d = cKDTree(np.vstack(contours_at(mean, 0.5))).query(pts)[0]
    assert d.mean() < 0.4


class Circle:
    """A stand in for a designed outline, a 1000 unit circle"""

    def __init__(self):
        t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        self.contours = [np.stack([500 * np.cos(t), 500 * np.sin(t)], 1)]

    def to_img(self, p):
        return [c * p[0] * [1, -1] + p[3:5] for c in self.contours]


def test_cutaways_never_cover_a_whole_outline():
    g, p = Circle(), np.array([0.6, 1.0, 0.0, 0.0, 0.0, 1.5])
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    stack_contour = [np.stack([310 * np.cos(t) * 1.02, 300 * np.sin(t)], 1)]
    out = export_site.cutaways(g, p, stack_contour, 3)
    assert 1 <= len(out) <= 3
    total = 2 * np.pi * 500
    length = 0.0
    for c in out:
        for seg in c["path"].split("M")[1:]:
            pts = np.array([[float(v) for v in q.split(",")] for q in seg.split("L")])
            length += np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    assert length <= export_site.CUTAWAY_MAX_FRACTION * total + 1
    assert export_site.cutaways(g, p, stack_contour, 0) == []


def test_path_d_is_in_font_units():
    p = np.array([0.5, 1.0, 0.0, 10.0, 20.0, 0.0])
    sq = np.array([[10, 20], [60, 20], [60, 70], [10, 70]], float)
    d = export_site.path_d([sq], p)
    assert d.startswith("M0.0,0.0L100.0,0.0")


def test_meta_carries_the_session_threshold_and_the_confidence_rule():
    m = export_site.meta_doc(12)
    assert m["min_instances"] == 12 and m["confident_instances"] == export_site.CONFIDENT_INSTANCES and m["confident_cards"] == export_site.CONFIDENT_CARDS


def test_banded_distance_matches_the_full_field_near_the_outline():
    class Square:
        def to_img(self, prm):
            t = np.linspace(0, 1, 40, endpoint=False)[:, None]
            a, b, c, d = np.array([20, 20]), np.array([60, 20]), np.array([60, 70]), np.array([20, 70])
            return [np.vstack([a + (b - a) * t, b + (c - b) * t, c + (d - c) * t, d + (a - d) * t]).astype(float)]

    g, p = Square(), np.array([1, 1, 0, 0, 0, 1.5])
    yy, xx = np.mgrid[0:90, 0:80]
    full = fontfit.FontGlyph.signed_dist(g, p, np.stack([xx.ravel(), yy.ravel()], 1).astype(float)).reshape(90, 80)
    band = fontfit.placed_distance(g, p, (90, 80))
    near = np.abs(full) < 12
    assert np.allclose(band[near], full[near])
    assert ((band <= p[5]) == (full <= p[5])).mean() > 0.999
