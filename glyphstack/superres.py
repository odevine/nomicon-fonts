"""Multi-frame super-resolution: registers shifted passes to sub-pixel accuracy and fuses them onto a 2x grid"""
import cv2
import numpy as np

WARP = cv2.WARP_INVERSE_MAP


def register(ref, img, mode=cv2.MOTION_EUCLIDEAN, iters=200):
    """Estimates the warp that maps ref coordinates to img coordinates with ECC, returns a 2x3 matrix and the correlation"""
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    r = cv2.GaussianBlur(ref, (0, 0), 1.0)
    m = cv2.GaussianBlur(img, (0, 0), 1.0)
    cc, warp = cv2.findTransformECC(r, m, warp, mode, crit, None, 5)
    return warp, cc


def _fine_to_pass(warp, scale):
    """3x3 matrix taking a fine grid pixel to pass coordinates, with pixel centres aligned"""
    off = 0.5 / scale - 0.5
    G = np.array([[1 / scale, 0, off], [0, 1 / scale, off], [0, 0, 1]], np.float64)
    return np.vstack([warp, [0, 0, 1]]).astype(np.float64) @ G


def fuse(passes, scale=2, psf_sigma=1.0, iterations=15, step=1.0):
    """Fuses single-channel float passes onto a grid scale times finer.

    Starts from shift-and-add, then runs iterative back-projection: each pass is re-simulated from the
    estimate with a Gaussian PSF of psf_sigma fine pixels, and its residual is pushed back onto the grid
    """
    ref = passes[0].astype(np.float32)
    h, w = ref.shape
    size = (w * scale, h * scale)
    maps = []
    for p in passes:
        warp = np.eye(2, 3, dtype=np.float32) if p is passes[0] else register(ref, p.astype(np.float32))[0]
        maps.append(_fine_to_pass(warp, scale))

    def to_fine(img, M, interp):
        return cv2.warpAffine(img.astype(np.float32), M[:2], size, flags=interp | WARP)

    def to_pass(est, M):
        blurred = cv2.GaussianBlur(est, (0, 0), psf_sigma)
        return cv2.warpAffine(blurred, np.linalg.inv(M)[:2], (w, h), flags=cv2.INTER_LINEAR | WARP)

    acc = np.zeros(size[::-1], np.float32)
    cover = np.zeros_like(acc)
    for p, M in zip(passes, maps):
        acc += to_fine(p, M, cv2.INTER_CUBIC) * to_fine(np.ones_like(ref), M, cv2.INTER_LINEAR)
        cover += to_fine(np.ones_like(ref), M, cv2.INTER_LINEAR)
    est = acc / np.maximum(cover, 1e-6)

    for _ in range(iterations):
        corr = np.zeros_like(est)
        for p, M in zip(passes, maps):
            resid = p.astype(np.float32) - to_pass(est, M)
            corr += cv2.GaussianBlur(to_fine(resid, M, cv2.INTER_CUBIC), (0, 0), psf_sigma)
        est += step * corr / len(passes) * scale * scale / 2
    return est
