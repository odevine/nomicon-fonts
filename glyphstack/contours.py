"""Sub-pixel iso-contours of stacked glyphs and edge-to-edge distances between them"""
import contourpy
import numpy as np
from scipy.spatial import cKDTree

UP = 4
UM_PER_PX = 25400 / 1200


def contours_at(img, level=0.5, min_len=60, spacing=0.5):
    """Dense polylines of an iso-level in grid coordinates (x is column), dropping loops shorter than min_len"""
    out = []
    for ln in contourpy.contour_generator(z=img.astype(float)).lines(level):
        seg = np.linalg.norm(np.diff(ln, axis=0), axis=1)
        if seg.sum() < min_len:
            continue
        s = np.concatenate([[0], np.cumsum(seg)])
        u = np.arange(0, s[-1], spacing)
        out.append(np.stack([np.interp(u, s, ln[:, 0]), np.interp(u, s, ln[:, 1])], 1))
    return out


def sym_dev(P, Q):
    """Symmetric nearest-point distances between two sets of polylines, in scan px"""
    P, Q = np.vstack(P), np.vstack(Q)
    return np.concatenate([cKDTree(Q).query(P)[0], cKDTree(P).query(Q)[0]]) / UP


def edge_rms_um(a, b, level=0.5, um_per_px=UM_PER_PX):
    """RMS edge distance in micrometres between the iso-contours of two stacked maps, None if either has none"""
    ca, cb = contours_at(a, level), contours_at(b, level)
    if not ca or not cb:
        return None
    return float(np.sqrt((sym_dev(ca, cb) ** 2).mean()) * um_per_px)
