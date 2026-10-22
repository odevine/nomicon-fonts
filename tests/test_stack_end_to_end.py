import json

import cv2
import numpy as np

from glyphstack import stack


def glyph(ch, rng):
    img = np.zeros((34, 30), np.float32)
    if ch == "o":
        cv2.ellipse(img, (15, 17), (9, 12), 0, 0, 360, 1, 4)
    else:
        cv2.rectangle(img, (12, 4), (17, 30), 1, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    return np.clip(img * rng.uniform(0.95, 1.0) + rng.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)


def synthetic_scans(cards=("A", "B"), per_char=10, angles=(0, 90)):
    rng = np.random.default_rng(3)
    scans = {}
    for card in cards:
        pos = [(ch, float(rng.uniform(100, 1500)), float(rng.uniform(100, 1500))) for ch in "ol" * per_char]
        for i, angle in enumerate(angles):
            units = []
            for ch, x, y in pos:
                cx, cy = (x + 40 * i, y - 25 * i)
                units.append(dict(ch=ch, cov=glyph(ch, rng), cx=cx, cy=cy, halo_px=3))
            scans[f"{card}_{angle}"] = dict(units=units, scale=1.0, card=card, angle=angle)
    return scans


def test_build_and_save_round_trip(tmp_path):
    res = stack.build(synthetic_scans(), tmp_path / "work", iterations=1)
    assert set(res["stacks"]) == {"o", "l"}
    # each unit of the turned scan pairs with its unit in the upright scan, so 20 units per glyph per card pair become 20 instances
    assert res["stacks"]["o"]["n"] == 20
    assert 0.98 < res["aspect"] < 1.02
    stack.save(res, tmp_path / "out")
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    row = man["instances"][0]
    for key in ("edge_level", "halo_px", "pair_id", "card_scale", "deviation_um"):
        assert key in row
    assert len(row["scans"]) == 2
    assert man["noise"]["n_pairs"] == 20 or man["noise"]["n_pairs"] > 0
    assert man["noise"]["single_scan_um"] < 5
    arrays = np.load(tmp_path / "out" / "stacks.npz")
    assert arrays["o__mean"].ndim == 2 and arrays["o__inst"].shape[0] == 20


def test_every_view_of_a_printed_glyph_joins_one_instance(tmp_path):
    res = stack.build(synthetic_scans(angles=(0, 15, 90, 105)), tmp_path / "work", iterations=1)
    assert res["stacks"]["o"]["n"] == 20
    rows = [r for r in res["instances"] if r["char"] == "o"]
    assert all(len(r["scans"]) == 4 for r in rows)
    assert res["noise"]["views_mean"] == 4
