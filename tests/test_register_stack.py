import logging

import cv2
import numpy as np

from glyphstack import register as rg
from glyphstack import stack
from scipy import ndimage as ndi

from glyphstack.units import card_angle, coverage_units, scan_levels, segment, upright


def blob(shift=(0, 0), size=(120, 100)):
    img = np.zeros(size, np.float32)
    cv2.ellipse(img, (50 + shift[0], 60 + shift[1]), (22, 34), 0, 0, 360, 1, -1)
    return cv2.GaussianBlur(img, (0, 0), 2.0)


def test_register_aligns_shifted_copies():
    imgs = [blob(s) for s in [(0, 0), (4, -3), (-5, 2), (2, 5)]]
    reg = rg.register(imgs)
    assert all(rg.iou(reg[0], r) > 0.97 for r in reg)


def test_remap_sends_level_to_half():
    c = np.array([0.0, 0.4, 1.0], np.float32)
    out = stack.remap(c, 0.4)
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_level_for_area_recovers_threshold():
    ramp = np.tile(np.linspace(0, 1, 1000, dtype=np.float32), (10, 1))
    area = (ramp > 0.37).sum()
    assert abs(stack.level_for_area(ramp, area) - 0.37) < 0.01


def test_loo_drops_wrong_instance():
    good = [blob((i % 3, 0)) for i in range(8)]
    bad = np.roll(blob(), 30, axis=1)
    mean, keep, loo, loo_kept = stack.stack_instances(np.array(good + [bad]))
    assert not keep[-1] and keep[:-1].all()


def test_rotation_scale_roundtrip():
    th = np.radians(0.4)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = np.diag([1.02, 0.99])
    t, _, rest = rg.rotation_scale(R @ S)
    assert abs(t - th) < 1e-3
    assert np.allclose(rest, S, atol=2e-3)


def test_halo_masks_neighbour_ink():
    g = np.full((200, 300), 200, np.uint8)
    g[60:140, 60:80] = 30
    g[60:140, 85:105] = 30
    lab, comps = segment(g)
    paper, ink, allink, _ = scan_levels(g, lab, comps)
    units = coverage_units(g, lab, comps, paper, ink, allink)
    left = min(units, key=lambda u: u["cx"])
    y0, y1, x0, x1 = left["box"]
    neighbour_cols = slice(85 - x0 - 3, 105 - x0)
    assert left["cov"][:, neighbour_cols].max() == 0
    assert left["cov"].max() > 0.9


def test_segment_merges_dot_into_stem():
    g = np.full((200, 100), 200, np.uint8)
    g[70:150, 40:52] = 30
    g[48:62, 40:52] = 30
    _, comps = segment(g)
    assert len(comps) == 1 and len(comps[0]["ids"]) == 2


def test_rectify_matches_upsample_then_affine():
    u = dict(cov=blob((3, -2)))
    M2 = np.array([[1.004, 0.006], [-0.005, 0.998]])
    a = rg.affine_apply(rg.upsample(u["cov"], 0.95), M2)
    b = rg.rectify(u, 0.95, M2)
    ca = np.array(ndi.center_of_mass(a)) - (np.array(a.shape) - 1) / 2
    cb = np.array(ndi.center_of_mass(b)) - (np.array(b.shape) - 1) / 2
    assert np.abs(ca - cb).max() < 0.05 and abs(a.sum() - b.sum()) / a.sum() < 1e-3


def test_rectify_reads_native_coverage_through_the_scan_map():
    cov = blob()
    # the native crop is the upright one shifted by (5, 3), so the native box sits 5 right and 3 down
    to_up = np.array([[1.0, 0, -5], [0, 1.0, -3]])
    u = dict(cov=cov, box=(10, 130, 20, 120), ncov=cov, nbox=(13, 133, 25, 125))
    assert np.allclose(rg.rectify(u, 1.0, np.eye(2), to_up), rg.rectify(dict(cov=cov), 1.0, np.eye(2)), atol=1e-4)


def test_card_angle_and_upright_follow_the_turn():
    g = np.full((900, 900), 250, np.uint8)
    g[200:700, 300:640] = 40
    for angle in (15, 30, 45):
        turned = cv2.warpAffine(g, cv2.getRotationMatrix2D((450, 450), angle, 1.0), (900, 900), borderValue=250)
        assert abs(abs(card_angle(turned)) - angle) < 1
        crop, _, M = upright(turned, angle)
        # OpenCV's matrix for a clockwise turn by the angle has +sin below the diagonal
        assert abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])) - angle) < 1
        assert crop.shape[0] > crop.shape[1]


def test_upright_warns_when_a_scan_sits_off_its_angle(caplog):
    g = np.full((900, 900), 250, np.uint8)
    g[200:700, 300:640] = 40
    turn = lambda a: cv2.warpAffine(g, cv2.getRotationMatrix2D((450, 450), a, 1.0), (900, 900), borderValue=250)
    with caplog.at_level(logging.DEBUG, logger="glyphstack"):
        for table, actual in ((0, 1), (30, 31), (45, 44), (0, 4), (30, 34)):
            upright(turn(actual), table, f"{table}-{actual}.png")
        upright(turn(4), 0)
    warned = sorted(r.getMessage().split()[0] for r in caplog.records if r.levelno == logging.WARNING)
    assert warned == ["0-4.png", "30-34.png"]
