import numpy as np

from glyphstack import fontfit, site_crops


def test_to_fu_matches_font_units_with_aspect():
    p = np.array([0.59, 0.995, -0.01, 200.0, 480.0, 2.4])
    asp = 0.98
    F = site_crops.to_fu(p, asp)
    pts = np.array([[10.0, 20.0], [100.5, 300.25], [0.0, 0.0]])
    # grid pixel centres of the aspect-scaled stack, as the stack applies it by resizing rows
    scaled = np.stack([pts[:, 0], (pts[:, 1] + 0.5) * asp - 0.5], 1)
    expect = fontfit.to_font_units(scaled, p)
    got = pts @ F[:, :2].T + F[:, 2]
    assert np.allclose(got, expect, atol=1e-6)


def test_compose_applies_inner_first():
    inner = np.array([[1.0, 0, 5], [0, 1, -3]])
    outer = np.array([[2.0, 0, 0], [0, 2, 1]])
    M = site_crops.compose(outer, inner)
    assert np.allclose(M @ [1, 1, 1], outer @ np.append(inner @ [1, 1, 1], 1))


def test_rel_path_round_trips_a_square():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    d = site_crops.rel_path([sq])
    assert d.startswith("M0,0l10,0 0,10 -10,0") and d.endswith("z")


def test_native_to_reg_recovers_a_known_warp():
    import cv2

    rng = np.random.default_rng(1)
    cov = cv2.GaussianBlur((rng.random((60, 50)) > 0.7).astype(np.float32), (0, 0), 3)
    cov = (cov / cov.max()).astype(np.float32)
    reg = np.clip(cv2.resize(cov, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), 0, 1)
    reg = cv2.warpAffine(reg, np.float32([[1, 0, 6], [0, 1, -4]]), (reg.shape[1] + 20, reg.shape[0] + 20))
    A, cc = site_crops.native_to_reg(cov, reg, 1.0)
    # the pixel centre (10, 20) of the unit lands at 4x + 1.5 in the upsampled grid, then shifts by (6, -4)
    assert cc > 0.95
    assert np.allclose(A @ [10, 20, 1], [10 * 4 + 1.5 + 6, 20 * 4 + 1.5 - 4], atol=0.6)
