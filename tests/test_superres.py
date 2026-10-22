import cv2
import numpy as np

from glyphstack.superres import fuse


def make_truth():
    img = np.zeros((240, 480), np.uint8)
    cv2.putText(img, "Reef Worm rn 3/3", (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 2.2, 255, 3, cv2.LINE_AA)
    cv2.putText(img, "creature token", (10, 190), cv2.FONT_HERSHEY_TRIPLEX, 2.2, 255, 3, cv2.LINE_AA)
    return img.astype(np.float32) / 255


def capture(truth, shift, ang, rng):
    """Simulates one scan pass at half the truth resolution, with optics blur and sensor noise"""
    h, w = truth.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[:, 2] += shift
    warped = cv2.warpAffine(truth, M, (w, h), flags=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(warped, (0, 0), 1.0)
    low = cv2.resize(blurred, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return low + rng.normal(0, 0.01, low.shape).astype(np.float32)


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))


def test_fusion_beats_single_pass_upscale():
    rng = np.random.default_rng(0)
    truth = make_truth()
    shifts = [(0, 0), (0.9, 0.3), (0.4, 1.1), (1.3, 0.8)]
    passes = [capture(truth, np.array(s), a, rng) for s, a in zip(shifts, [0, 0.05, -0.05, 0.08])]
    fused = fuse(passes, 2)
    single = cv2.resize(passes[0], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    inner = (slice(20, -20), slice(20, -20))
    assert psnr(fused[inner], truth[inner]) > psnr(single[inner], truth[inner]) + 0.5
