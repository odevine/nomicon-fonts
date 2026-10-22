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


def test_scan_table_reads_angles(tmp_path):
    from glyphstack.session import read_table

    t = tmp_path / "t.csv"
    t.write_text("file,card,angle,notes\na.png,X,15,\nb.png,X,105,slipped once\nc.png,X,0,\n")
    assert [(r[0].name, r[1], r[2]) for r in read_table(t)] == [("a.png", "X", 15.0), ("b.png", "X", 105.0), ("c.png", "X", 0.0)]


def test_scan_table_needs_an_angle_for_every_scan(tmp_path):
    from glyphstack.session import read_table

    t = tmp_path / "t.csv"
    t.write_text("file,card,angle\na.png,X,0\nb.png,X,\n")
    with pytest.raises(SystemExit, match="b.png"):
        read_table(t)
