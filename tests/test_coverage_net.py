import numpy as np
import pytest
from scipy import ndimage as ndi

from glyphstack import coverage_net, target, units


def random_weights(seed=0):
    rng = np.random.default_rng(seed)
    W, c = {}, 2
    for k in range(len(coverage_net.DILATIONS)):
        W[f"conv{k}_w"] = rng.normal(0, 0.2, (coverage_net.CHANNELS, c, 3, 3)).astype(np.float32)
        W[f"conv{k}_b"] = rng.normal(0, 0.1, coverage_net.CHANNELS).astype(np.float32)
        c = coverage_net.CHANNELS
    W["out_w"] = rng.normal(0, 0.2, (1, c, 1, 1)).astype(np.float32)
    W["out_b"] = np.zeros(1, np.float32)
    return W


@pytest.fixture
def textured():
    target.use("rules_texture")
    yield
    target.use(target.DEFAULT)
    coverage_net.use(None)


def test_forward_matches_dilated_convolution_with_zero_padding():
    W = random_weights()
    x = np.random.default_rng(1).uniform(0, 1.2, (2, 30, 24)).astype(np.float32)
    ref = x
    for k, d in enumerate(coverage_net.DILATIONS):
        w, b = W[f"conv{k}_w"], W[f"conv{k}_b"]
        out = np.zeros((w.shape[0],) + ref.shape[1:], np.float32)
        for o in range(w.shape[0]):
            for i in range(w.shape[1]):
                kern = np.zeros((2 * d + 1, 2 * d + 1), np.float32)
                kern[::d, ::d] = w[o, i]
                out[o] += ndi.correlate(ref[i], kern, mode="constant")
            out[o] += b[o]
        ref = np.maximum(out, 0)
    ref = 1 / (1 + np.exp(-(np.tensordot(W["out_w"][:, :, 0, 0], ref, 1)[0] + W["out_b"][0])))
    assert np.allclose(coverage_net.forward(x, W), ref, atol=1e-5)


def test_units_on_plain_background_keep_their_coverage(textured):
    g = np.full((300, 300), 200, np.uint8)
    g[100:160, 100:114] = 25
    coverage_net._weights = random_weights()
    read = coverage_net.reader(g)
    assert read((92, 168, 92, 122), np.ones((76, 30), bool)) is None


def test_a_unit_on_strong_screen_takes_the_network_coverage_in_its_window(textured):
    y, x = np.mgrid[:400, :400].astype(np.float32)
    g = 120 + 60 * np.cos(2 * np.pi * (x + y) / 6 / np.sqrt(2)) * np.cos(2 * np.pi * (x - y) / 6 / np.sqrt(2))
    g[170:230, 190:204] = 20
    g = np.clip(g, 0, 255).astype(np.uint8)
    lab, n = ndi.label(g < 40)
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    comps = [dict(ids=[big], y0=170, y1=230, x0=190, x1=204)]
    allink = lab == big
    plain = units.coverage_units(g, lab, comps, 200.0, 20.0, allink)[0]
    coverage_net._weights = random_weights()
    calls = []

    def net(box, mask):
        calls.append(box)
        return np.full(mask.shape, 0.5, np.float32), 1.0

    read = units.coverage_units(g, lab, comps, 200.0, 20.0, allink, net=net)[0]
    assert calls == [read["box"]]
    assert read["box"] == plain["box"]
    assert np.allclose(read["cov"][plain["cov"] > 0], 0.5)


def test_opaque_ink_over_flat_paper_reads_ink_where_covered_and_paper_elsewhere():
    c = np.zeros((40, 40), np.float32)
    c[10:30, 10:30] = 1
    out = coverage_net.print_over(c, np.full((40, 40), 200.0, np.float32), 25.0, 200.0, alpha=1.0, sigma=0.0, gamma=0.0)
    assert out[20, 20] == pytest.approx(25.0, abs=0.5)
    assert out[2, 2] == pytest.approx(200.0, abs=0.5)
