import numpy as np

from glyphstack import calibration

GREYS = [255, 219, 150, 108, 81, 57, 43, 34, 30, 28, 24]


def test_tone_steps_are_read_back_from_a_synthetic_wedge():
    s = 47.32
    sw, sh = int(4 * s), int(6 * s)
    g = np.full((900, 2700), 250, np.uint8)
    x0, y0 = 300, 250
    for i, v in enumerate(GREYS):
        g[y0 : y0 + sh, x0 + i * sw : x0 + (i + 1) * sw] = v
    # a distractor that is dark but the wrong shape
    g[600:800, 100:260] = 10
    got = calibration.tone_steps(g, s, 1.0)
    assert got is not None and len(got) == 11
    assert np.allclose(got, GREYS, atol=1)


def test_tone_steps_return_none_without_a_wedge():
    assert calibration.tone_steps(np.full((600, 900), 250, np.uint8), 47.32) is None
