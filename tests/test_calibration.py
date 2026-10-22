import numpy as np

from glyphstack import calibration as cal


def synthetic_ruler(pitch=47.3, ticks=61, tilt=0.0, size=(420, 3200)):
    """Dark baseline with 1 mm ticks hanging below it, on white paper"""
    img = np.full(size, 255, np.uint8)
    base = 120
    for x in range(100, 100 + int(pitch * (ticks - 1)) + 10):
        img[base + int(round((x - 100) * np.tan(np.radians(tilt)))) - 3 : base + int(round((x - 100) * np.tan(np.radians(tilt)))) + 3, x] = 0
    for i in range(ticks):
        x = int(round(100 + i * pitch))
        y0 = base + int(round((x - 100) * np.tan(np.radians(tilt))))
        img[y0 : y0 + 120, x - 3 : x + 4] = 0
    return img


def test_fit_scale_recovers_pitch():
    pos = 100 + 47.3 * np.arange(61) + np.random.default_rng(0).normal(0, 0.2, 61)
    scale, resid = cal.fit_scale(pos)
    assert abs(scale - 47.3) < 0.01
    assert resid < 0.5


def test_measure_ruler_on_synthetic_target():
    img = synthetic_ruler()
    tilt, pos = cal.measure_ruler(img, (100, 117, 2840, 130), vertical=False, pad=100)
    scale, _ = cal.fit_scale(pos)
    assert len(pos) == 61
    assert abs(scale - 47.3) < 0.05
    assert abs(tilt) < 0.05


def test_find_rulers_and_squares_by_size():
    comps = [(0, 0, 2840, 160, 50000), (0, 0, 150, 2840, 50000), (0, 0, 500, 500, 212500), (0, 0, 465, 465, 183800), (0, 0, 500, 500, 100000)]
    rulers = cal.find_rulers(comps)
    assert [v for _, v in rulers] == [False, True]
    assert cal.find_squares(comps) == [(0, 0, 500, 500), (0, 0, 465, 465)]


def result(sy_a, sy_b, sx_a=47.0, sx_b=47.0):
    return {"summary": {"px_per_mm": (sy_a + sy_b) / 2, "scan_anisotropy": max(abs(sy_a - sy_b) / sy_a, abs(sx_a - sx_b) / sx_a)}}


def test_gate_passes_pilot_values_and_fails_bad_scale():
    assert cal.gate(result(47.3246, 47.3197)) == []
    assert any("scale" in f for f in cal.gate(result(47.8, 47.8)))
    assert any("differ" in f for f in cal.gate(result(47.3, 47.5)))
