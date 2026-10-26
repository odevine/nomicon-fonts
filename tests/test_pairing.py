import json

import numpy as np
import pytest

from glyphstack.pairing import match_pair


def test_pairs_found_after_rotation_and_shift():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 2000, (60, 2))
    chars = rng.choice(list("etaoin"), 60)
    th = np.radians(0.5)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    moved = pts @ R.T + [140, -60]
    A = [(c, x, y) for c, (x, y) in zip(chars, pts)]
    order = rng.permutation(60)
    B = [(chars[i], *moved[i]) for i in order]
    pairs, rot = match_pair(A, B)
    assert len(pairs) >= 55
    assert all(order[kb] == ka for ka, kb in pairs)
    assert abs(rot - 0.5) < 0.1


def test_each_glyph_of_the_second_scan_used_once():
    A = [("e", 0, 0), ("e", 5, 0), ("t", 100, 100), ("e", 300, 20)]
    B = [("e", 1, 0), ("t", 101, 100), ("e", 301, 20)]
    pairs, _ = match_pair(A, B)
    assert len({kb for _, kb in pairs}) == len(pairs)


def test_cards_file_lists_every_scan_with_its_angle(tmp_path):
    from glyphstack.session import read_scans

    t = tmp_path / "cards.json"
    t.write_text(json.dumps(dict(cards=[dict(id="X", name="X", scans=[dict(file="a.png", angle=15), dict(file="b.png", angle=105, notes="slipped once"), dict(file="c.png", angle=0)])])))
    assert [(r[0].name, r[1], r[2]) for r in read_scans(t)] == [("a.png", "X", 15.0), ("b.png", "X", 105.0), ("c.png", "X", 0.0)]


def test_cards_file_needs_an_angle_for_every_scan_and_a_plain_id(tmp_path):
    from glyphstack.session import read_scans

    t = tmp_path / "cards.json"
    t.write_text(json.dumps(dict(cards=[dict(id="X", scans=[dict(file="a.png", angle=0), dict(file="b.png")])])))
    with pytest.raises(SystemExit, match="b.png"):
        read_scans(t)
    t.write_text(json.dumps(dict(cards=[dict(id="Reef Worm", scans=[dict(file="a.png", angle=0)])])))
    with pytest.raises(SystemExit, match="Reef Worm"):
        read_scans(t)
