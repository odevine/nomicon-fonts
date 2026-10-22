"""Sub-pixel registration helpers for stacking glyph coverage maps on a 4x grid"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy import ndimage as ndi

UP = 4
MARGIN = 10 * UP
_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
_POOL = ThreadPoolExecutor(os.cpu_count() or 1)


def pmap(fn, items):
    """Maps fn over items on a shared thread pool, OpenCV releases the GIL so ECC calls run in parallel"""
    return list(_POOL.map(fn, items))


def upsample(cov, scale):
    """Resamples a coverage map to the 4x grid at the card's print scale, so one grid px is 5.3 um at scale 1"""
    f = UP / scale
    return np.clip(cv2.resize(cov, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC), 0, 1).astype(np.float32)


def place(img, shape):
    """Centres an image on a zero canvas of the given shape"""
    out = np.zeros(shape, np.float32)
    h, w = img.shape
    H, W = shape
    oy, ox = (H - h) // 2, (W - w) // 2
    ys0, xs0, ys1, xs1 = max(oy, 0), max(ox, 0), min(oy + h, H), min(ox + w, W)
    out[ys0:ys1, xs0:xs1] = img[ys0 - oy : ys1 - oy, xs0 - ox : xs1 - ox]
    return out


def com_canvas(imgs, margin=MARGIN):
    """Places every image on a common canvas with its centre of mass at the canvas centre"""
    # a high percentile rather than the largest, so one oversized unit does not enlarge every canvas of its glyph
    Hc = int(np.percentile([i.shape[0] for i in imgs], 98)) + 2 * margin
    Wc = int(np.percentile([i.shape[1] for i in imgs], 98)) + 2 * margin
    out = []
    for i in imgs:
        cy, cx = ndi.center_of_mass(i)
        oy, ox = int(round(Hc / 2 - cy)), int(round(Wc / 2 - cx))
        c = np.zeros((Hc, Wc), np.float32)
        h, w = i.shape
        ys0, xs0, ys1, xs1 = max(oy, 0), max(ox, 0), min(oy + h, Hc), min(ox + w, Wc)
        c[ys0:ys1, xs0:xs1] = i[ys0 - oy : ys1 - oy, xs0 - ox : xs1 - ox]
        out.append(c)
    return out


def ecc_shift(ref, img):
    """Translation that maps ref to img by ECC, the identity when ECC cannot converge"""
    w = np.eye(2, 3, dtype=np.float32)
    try:
        _, w = cv2.findTransformECC(ref, img, w, cv2.MOTION_TRANSLATION, _CRIT, None, 5)
    except cv2.error:
        pass
    return w


def ecc_affine(ref, img):
    """Affine linear part mapping ref to img by ECC and its correlation, or None on failure"""
    w = np.eye(2, 3, dtype=np.float32)
    try:
        cc, w = cv2.findTransformECC(ref, img, w, cv2.MOTION_AFFINE, _CRIT, None, 5)
    except cv2.error:
        return None
    return w[:, :2].copy(), cc


def warp_to(ref, img, w):
    return cv2.warpAffine(img, w, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderValue=0).clip(0, 1)


def register(imgs, iters=3):
    """Aligns images by centre of mass and then by translation ECC against the running mean"""
    cs = com_canvas(imgs)
    ref = np.mean(cs, 0)
    for _ in range(iters):
        reg = pmap(lambda c: warp_to(ref, c, ecc_shift(ref, c)), cs)
        ref = np.mean(reg, 0)
    return reg


def affine_apply(u, M2):
    """Rectifies an upsampled image into the reference frame, M2 maps reference to image coordinates about the centre"""
    h, w = u.shape
    pad = int(0.2 * max(h, w))
    u2 = np.pad(u, pad)
    H, W = u2.shape
    c = np.array([(W - 1) / 2, (H - 1) / 2])
    Wm = np.hstack([M2, (c - M2 @ c)[:, None]]).astype(np.float32)
    return cv2.warpAffine(u2, Wm, (W, H), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderValue=0).clip(0, 1)


def rectify(u, scale, M2, to_up=None):
    """A unit resampled once onto the 4x grid at the card's print scale and rectified by M2, the same image as
    `affine_apply(upsample(u["cov"], scale), M2)` from a single cubic warp

    When the unit carries native coverage (`ncov`, `nbox`) from a turned scan, the warp reads the native pixels
    through `to_up`, the scan's map from its native crop to the upright crop, so the turn costs no extra resample
    """
    f = UP / scale
    h, w = u["cov"].shape
    hu, wu = int(round(h * f)), int(round(w * f))
    M2 = np.asarray(M2, float)
    pad = 4 + int(np.ceil(max(hu, wu) * np.abs(M2 - np.eye(2)).max()))
    H, W = hu + 2 * pad, wu + 2 * pad
    c = np.array([(W - 1) / 2, (H - 1) / 2])
    # output pixel -> upsampled unit pixel -> unit box pixel
    T = np.hstack([M2 / f, ((c - M2 @ c - pad + 0.5) / f - 0.5)[:, None]])
    src = u["cov"]
    if "ncov" in u and to_up is not None:
        y0, _, x0, _ = u["box"]
        ny0, _, nx0, _ = u["nbox"]
        A = np.vstack([np.asarray(to_up, float), [0, 0, 1]])
        A[:2, 2] += A[:2, :2] @ [nx0, ny0] - [x0, y0]
        T = (np.linalg.inv(A) @ np.vstack([T, [0, 0, 1]]))[:2]
        src = u["ncov"]
    return cv2.warpAffine(src, T.astype(np.float32), (W, H), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderValue=0).clip(0, 1)


def iou(a, b):
    a, b = a > 0.5, b > 0.5
    return (a & b).sum() / max((a | b).sum(), 1)


def rotation_scale(M):
    """Splits an affine 2x2 into its rotation angle in radians and the rotation-free remainder"""
    th = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return th, R, R.T @ M
