import cv2
import numpy as np
import pytest

from glyphstack import target, template, texture, units


@pytest.fixture
def textured():
    target.use("rules_texture")
    yield target.get()
    target.use(target.DEFAULT)


def screen(shape, period=6.0, depth=30.0, seed=0):
    """A 45 degree dot screen of the given period and peak to peak depth in grey levels"""
    y, x = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    return depth / 2 * np.cos(2 * np.pi * (x + y) / period / np.sqrt(2)) * np.cos(2 * np.pi * (x - y) / period / np.sqrt(2))


def letters(shape, ink=25.0):
    """A row of solid 14 px strokes 60 px tall, the size of rules text at 1200 dpi, as a mask"""
    m = np.zeros(shape, bool)
    for x0 in range(60, shape[1] - 60, 70):
        m[100:160, x0 : x0 + 14] = True
    return m


def test_the_default_profile_leaves_the_scan_untouched():
    target.use(target.DEFAULT)
    g = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
    assert not texture.active()
    assert texture.prepare(g) is g


def test_flattening_lets_the_ink_threshold_find_strokes_over_a_dark_gradient(textured):
    shape = (260, 1200)
    paper = np.tile(np.linspace(200, 80, shape[1], dtype=np.float32), (shape[0], 1))
    ink = letters(shape)
    g = np.clip(np.where(ink, 25.0, paper) + screen(shape, depth=10), 0, 255).astype(np.uint8)
    target.use(target.DEFAULT)
    _, plain = units.segment(g)
    target.use("rules_texture")
    _, flat = units.segment(texture.prepare(g))
    strokes = len(range(60, shape[1] - 60, 70))
    # the dark end falls under the fixed threshold as one component too large to be a glyph
    assert len(plain) < strokes
    assert len(flat) == strokes


def test_descreening_follows_the_screen_contrast(textured):
    shape = (260, 1200)
    ink = letters(shape)
    weak = np.clip(np.where(ink, 25.0, 200.0) + screen(shape, depth=20), 0, 255).astype(np.uint8)
    strong = np.clip(np.where(ink, 25.0, 140.0) + screen(shape, depth=90), 0, 255).astype(np.uint8)
    s = texture.settings()
    lo, hi = s["descreen_contrast"]
    assert np.median(texture.screen_contrast(weak, s)) < lo
    assert np.median(texture.screen_contrast(strong, s)) > hi
    # a weak screen is left sharp, a strong one loses most of its ripple away from the ink
    assert np.abs(texture.descreen(weak, s) - weak).max() < 1
    away = ~cv2.dilate(ink.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    away[:20], away[-20:], away[:, :20], away[:, -20:] = False, False, False, False
    assert texture.descreen(strong, s)[away].std() < 0.3 * strong[away].std()


def test_profiles_reject_unknown_texture_settings_and_scale_searches(tmp_path):
    text = (target.BUILTIN / "rules_texture.toml").read_text()
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("flatten = true", "flatten = true\nsharpen = 1"))
    with pytest.raises(ValueError, match="unknown texture settings sharpen"):
        target.load(str(bad))
    bad.write_text(text.replace('scale_search = "count"', 'scale_search = "median"'))
    with pytest.raises(ValueError, match="scale_search"):
        target.load(str(bad))


def ring(r_out, r_in, jitter=0.0, seed=0):
    """Coverage of an o-like ring with a ragged edge, as a unit's cov"""
    rng = np.random.default_rng(seed)
    n = 2 * r_out + 20
    y, x = np.mgrid[:n, :n] - n / 2
    d = np.hypot(y, x) + rng.normal(0, jitter, (n, n))
    return ((d < r_out) & (d > r_in)).astype(np.float32)


def test_card_templates_take_the_mean_of_the_kept_units():
    kept = [dict(ch="o", cov=ring(30, 18, seed=s)) for s in range(4)]
    base = {"o": dict(mask=np.ones((10, 10), np.float32), w=2.5, h=2.5)}
    t = template.card_templates(kept, base, min_units=3)
    own = template.unit_mask4(kept[0]["cov"])
    assert template.best_iou(own, t["o"]["mask"]) > 0.97
    assert abs(t["o"]["w"] - own.shape[1] / template.UP) < 1
    # too few units keep the font's template
    assert template.card_templates(kept[:2], base, min_units=3)["o"] is base["o"]


def test_count_scale_lands_on_the_scale_that_labels_the_most_units(monkeypatch):
    target.use(target.DEFAULT)
    best = 0.93
    monkeypatch.setattr(template, "label", lambda us, t, s: us[: max(0, 40 - int(round(abs(s - best) * 1000)))])
    tp = type("T", (), {"at": lambda self, s, only=None: {}})()
    assert template.count_scale(list(range(40)), tp) == best
    assert template.count_scale(list(range(40)), tp, around=0.92) == best
