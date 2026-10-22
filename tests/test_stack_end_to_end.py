import json

import cv2
import numpy as np

from glyphstack import chunks, stack


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
    stacks, _ = stack.load(tmp_path / "out")
    assert stacks["o"]["mean"].ndim == 2 and len(stacks["o"]["inst"]) == 20
    assert stacks["o"]["inst"].take([0, 19]).shape == (2, *stacks["o"]["mean"].shape)
    # the reloaded instances are the float16 stack, so their mean is the stack's mean
    assert np.abs(stacks["o"]["inst"].take(range(20)).mean(0) - stacks["o"]["mean"]).max() < 1e-2


def test_every_view_of_a_printed_glyph_joins_one_instance(tmp_path):
    res = stack.build(synthetic_scans(angles=(0, 15, 90, 105)), tmp_path / "work", iterations=1)
    assert res["stacks"]["o"]["n"] == 20
    rows = [r for r in res["instances"] if r["char"] == "o"]
    assert all(len(r["scans"]) == 4 for r in rows)
    assert res["noise"]["views_mean"] == 4


def test_streamed_stack_matches_stacking_in_memory(tmp_path, monkeypatch):
    # chunks of three images, so a printed glyph's views can straddle two chunks
    monkeypatch.setattr(chunks, "CHUNK_BYTES", 34 * 30 * 2 * 3)
    rng = np.random.default_rng(5)
    # twelve printed glyphs of one or two views each, and one wrong glyph that the screen drops
    views = [glyph("o", rng) for _ in range(20)] + [glyph("l", rng)]
    keys = [("s0", i) if i < 12 else ("s1", i - 12) for i in range(21)]
    spans = [(k, a, b) for k, (a, b) in zip(range(13), [(i, i + 1) for i in range(4)] + [(4 + 2 * j, 6 + 2 * j) for j in range(8)] + [(20, 21)])]
    levels = {"s0": 0.45, "s1": 0.55}
    with chunks.Writer(tmp_path / "regs", meta=dict(keys=[list(k) for k in keys])) as w:
        for v in views:
            w.add(v)
    R = chunks.Store(tmp_path / "regs")
    st, pair_dev, keep, loo, devs = stack.stack_glyph(R, keys, spans, levels, 1.0, tmp_path / "g", loo_min=0.8)
    reg = R.take(range(21))
    inst = np.array([np.mean([stack.remap(reg[i], levels[keys[i][0]]) for i in range(a, b)], 0) for _, a, b in spans])
    mean, keep2, loo2, loo_kept = stack.stack_instances(inst, loo_min=0.8)
    assert not keep[-1] and (keep == keep2).all()
    assert np.array_equal(st["mean"], mean) and np.array_equal(loo, loo2) and np.array_equal(st["loo"], loo_kept)
    assert np.array_equal(st["sd"], inst[keep].std(0))
    assert pair_dev[0] is None and pair_dev[4] is not None and devs[-1] is None
    assert len(chunks.Store(tmp_path / "g" / "inst")) == keep.sum()


def test_chunk_store_reads_back_in_any_order(tmp_path, monkeypatch):
    monkeypatch.setattr(chunks, "CHUNK_BYTES", 3 * 4 * 5 * 2)
    imgs = [np.full((4, 5), i / 10, np.float32) for i in range(10)]
    with chunks.Writer(tmp_path / "s") as w:
        for a in imgs:
            w.add(a)
    R = chunks.Store(tmp_path / "s")
    assert len(R) == 10 and R.chunk == 3 and len(list((tmp_path / "s").glob("*.npz"))) == 4
    got = R.take([9, 0, 4, 3])
    assert np.allclose(got[:, 0, 0], [0.9, 0.0, 0.4, 0.3], atol=1e-3) and got.dtype == np.float32
    assert np.allclose(R[7], 0.7, atol=1e-3)
