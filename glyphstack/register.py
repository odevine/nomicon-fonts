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


def canvas_size(shapes, margin=MARGIN):
    """The common canvas of a glyph's images from their shapes"""
    # a high percentile rather than the largest, so one oversized unit does not enlarge every canvas of its glyph
    return int(np.percentile([s[0] for s in shapes], 98)) + 2 * margin, int(np.percentile([s[1] for s in shapes], 98)) + 2 * margin


def com_offset(img, size):
    """Where on_canvas places an image's top left corner, so its centre of mass lands at the canvas centre"""
    cy, cx = ndi.center_of_mass(img)
    return int(round(size[0] / 2 - cy)), int(round(size[1] / 2 - cx))


def on_canvas(img, size, offset=None):
    """An image placed on a canvas of the given size with its centre of mass at the canvas centre, or at offset when given"""
    Hc, Wc = size
    oy, ox = offset or com_offset(img, size)
    c = np.zeros((Hc, Wc), np.float32)
    h, w = img.shape
    ys0, xs0, ys1, xs1 = max(oy, 0), max(ox, 0), min(oy + h, Hc), min(ox + w, Wc)
    c[ys0:ys1, xs0:xs1] = img[ys0 - oy : ys1 - oy, xs0 - ox : xs1 - ox]
    return c


def com_canvas(imgs, margin=MARGIN):
    """Places every image on a common canvas with its centre of mass at the canvas centre"""
    size = canvas_size([i.shape for i in imgs], margin)
    return [on_canvas(i, size) for i in imgs]


def ecc_shift(ref, img, start=None):
    """Translation that maps ref to img by ECC from start, default the identity, and start itself when ECC cannot converge"""
    w = np.eye(2, 3, dtype=np.float32) if start is None else start.copy()
    try:
        _, w = cv2.findTransformECC(ref, img, w, cv2.MOTION_TRANSLATION, _CRIT, None, 5)
    except cv2.error:
        pass
    return w


def ink_box(img, margin=24, thr=0.02):
    """Rows and columns (y0, y1, x0, x1) of the image's ink plus a margin, the whole image when it has none"""
    ys, xs = np.nonzero(img > thr)
    if len(ys) == 0:
        return 0, img.shape[0], 0, img.shape[1]
    return max(ys.min() - margin, 0), ys.max() + margin + 1, max(xs.min() - margin, 0), xs.max() + margin + 1


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


def register_shifts(load, n, shapes, iters=3, batch=64, size=None):
    """The canvas size, each image's translation and the mean that register() arrives at, holding one batch of images at a time

    load(i) gives image i, shapes are the images' shapes, and size fixes the canvas instead of sizing it from them.
    Each image is loaded once per pass, iters + 1 times in all. Each pass runs ECC on the mean's ink plus a margin,
    starting from the image's shift in the pass before
    """
    size = size or canvas_size(shapes)
    offsets = [None] * n

    def canvas(i):
        img = load(i)
        if offsets[i] is None:
            offsets[i] = com_offset(img, size)
        return on_canvas(img, size, offsets[i])

    batches = [range(b, min(b + batch, n)) for b in range(0, n, batch)]
    # summed in order in float32, which gives the same mean as np.mean over the stacked images
    acc = np.zeros(size, np.float32)
    for b in batches:
        for c in pmap(canvas, b):
            acc += c
    ref = acc / n
    shifts = [np.eye(2, 3, dtype=np.float32)] * n
    for _ in range(iters):
        y0, y1, x0, x1 = ink_box(ref)
        crop = np.ascontiguousarray(ref[y0:y1, x0:x1])

        def one(i):
            c = canvas(i)
            w = ecc_shift(crop, np.ascontiguousarray(c[y0:y1, x0:x1]), shifts[i])
            return w, warp_to(ref, c, w)

        acc = np.zeros(size, np.float32)
        for b in batches:
            for i, (w, r) in zip(b, pmap(one, b)):
                shifts[i] = w
                acc += r
        ref = acc / n
    return size, shifts, ref


def registered(img, size, w, offset=None):
    """One image as register_shifts placed it, from its shift, or placed at offset and then shifted when offset is given"""
    return warp_to(np.empty(size, np.float32), on_canvas(img, size, offset), w)


def register(imgs, iters=3):
    """Aligns images by centre of mass and then by translation ECC against the running mean"""
    size, shifts, _ = register_shifts(imgs.__getitem__, len(imgs), [i.shape for i in imgs], iters)
    return pmap(lambda k: registered(imgs[k], size, shifts[k]), range(len(imgs)))


def affine_apply(u, M2):
    """Rectifies an upsampled image into the reference frame, M2 maps reference to image coordinates about the centre"""
    h, w = u.shape
    pad = int(0.2 * max(h, w))
    u2 = np.pad(u, pad)
    H, W = u2.shape
    c = np.array([(W - 1) / 2, (H - 1) / 2])
    Wm = np.hstack([M2, (c - M2 @ c)[:, None]]).astype(np.float32)
    return cv2.warpAffine(u2, Wm, (W, H), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderValue=0).clip(0, 1)


def _rect_geometry(u, scale, M2):
    f = UP / scale
    h, w = u["cov"].shape
    hu, wu = int(round(h * f)), int(round(w * f))
    pad = 4 + int(np.ceil(max(hu, wu) * np.abs(np.asarray(M2, float) - np.eye(2)).max()))
    return f, pad, (hu + 2 * pad, wu + 2 * pad)


def rectified_shape(u, scale, M2):
    """The shape rectify() gives, without resampling"""
    return _rect_geometry(u, scale, M2)[2]


def rectify(u, scale, M2, to_up=None):
    """A unit resampled once onto the 4x grid at the card's print scale and rectified by M2, the same image as
    `affine_apply(upsample(u["cov"], scale), M2)` from a single cubic warp

    When the unit carries native coverage (`ncov`, `nbox`) from a turned scan, the warp reads the native pixels
    through `to_up`, the scan's map from its native crop to the upright crop, so the turn costs no extra resample
    """
    f, pad, (H, W) = _rect_geometry(u, scale, M2)
    M2 = np.asarray(M2, float)
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
