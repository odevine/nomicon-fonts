"""Measures scanner scale, tilt and edge width from the printed calibration target and gates a session on them"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .scan import load_gray

NOMINAL_PX_PER_MM = 47.32
SCALE_TOL = 0.15
ANISOTROPY_TOL = 0.002
RULER_MM = (60, 180, 240)
TONE_STEPS = 11
TONE_STEP_MM = (4, 6)
SQUARE_MM = (8, 25)
UM_PER_PX = 25400 / 1200


def ink_components(g, thresh=128):
    """Connected components of dark ink as (x, y, w, h, area) rows"""
    n, _, st, _ = cv2.connectedComponentsWithStats((g < thresh).astype(np.uint8), connectivity=8)
    return [tuple(int(v) for v in s) for s in st[1:n]]


def find_rulers(comps, px_per_mm=NOMINAL_PX_PER_MM, tol=0.05):
    """Picks the long thin components whose length matches a printed ruler, returns (bbox, vertical) pairs"""
    out = []
    for x, y, w, h, _ in comps:
        long_, short = max(w, h), min(w, h)
        if short * 10 > long_:
            continue
        if any(abs(long_ / px_per_mm - mm) < tol * mm for mm in RULER_MM):
            out.append(((x, y, w, h), h > w))
    return sorted(out, key=lambda r: r[1])


def find_squares(comps, px_per_mm=NOMINAL_PX_PER_MM):
    """Picks the slanted solid squares by size and by how much of their bounding box they fill"""
    lo, hi = (v * px_per_mm for v in SQUARE_MM)
    return [(x, y, w, h) for x, y, w, h, a in comps if lo < w < hi and lo < h < hi and abs(w - h) < 0.05 * w and 0.80 < a / (w * h) < 0.90]


def measure_ruler(g, bbox, vertical, pad=60):
    """Returns the baseline tilt in degrees and the sub-pixel tick positions along one ruler"""
    x, y, w, h = bbox
    d = 255 - g[y - pad : y + h + pad, x - pad : x + w + pad].astype(np.float32)
    if vertical:
        d = d.T
    W = d.shape[1]
    r0 = int(np.argmax(d.sum(1)))
    # the ticks hang off one side of the baseline, flip so they are always below it
    if d[max(r0 - 60, 0) : r0 - 15].sum() > d[r0 + 15 : r0 + 60].sum():
        d = d[::-1]
        r0 = int(np.argmax(d.sum(1)))
    cov = d / (255 - 9.0)
    xs = np.arange(80, W - 80)
    ey = []
    for xx in xs:
        col = cov[r0 - 25 : r0 + 6, xx]
        k = np.nonzero(col >= 0.5)[0]
        if not len(k) or k[0] == 0:
            ey.append(np.nan)
            continue
        k = k[0]
        ey.append(r0 - 25 + k - 1 + (0.5 - col[k - 1]) / (col[k] - col[k - 1]) + 0.5)
    ey = np.array(ey)
    ok = ~np.isnan(ey)
    line = np.polyfit(xs[ok], ey[ok], 1)
    for _ in range(3):
        ok2 = ok & (np.abs(ey - np.polyval(line, xs)) < 1.5)
        line = np.polyfit(xs[ok2], ey[ok2], 1)
    th = np.arctan(line[0])
    # sample a band 15 to 50 px below the baseline, parallel to it, and average across the band
    u = np.arange(0, W, 1.0)
    vs = np.arange(15, 50, 1.0)
    px = (u[None, :] + vs[:, None] * -np.sin(th)).astype(np.float32)
    py = (line[1] + line[0] * u[None, :] + vs[:, None] * np.cos(th)).astype(np.float32)
    prof = cv2.remap(d, px, py, cv2.INTER_LINEAR, borderValue=0).mean(0)
    smooth = gaussian_filter1d(prof, 1.5)
    peaks, _ = find_peaks(smooth, height=0.5 * np.percentile(smooth, 99), distance=30)
    pos = []
    for p in peaks:
        wgt = prof[p - 8 : p + 9] - prof[p - 8 : p + 9].min()
        pos.append((np.arange(p - 8, p + 9) * wgt).sum() / wgt.sum())
    return float(np.degrees(th)), np.array(pos)


def fit_scale(pos):
    """Fits tick index against position, returns px per mm and the residual RMS in px"""
    pitch = np.median(np.diff(pos))
    idx = np.round((pos - pos[0]) / pitch)
    b, a = np.polyfit(idx, pos, 1)
    return float(b), float((pos - (a + b * idx)).std())


def edge_profile(g, bbox, side, pad=40):
    """Edge spread function of one side of a slanted square, as (distance px, coverage, tilt deg)"""
    x, y, w, h = bbox
    c = g[y - pad : y + h + pad, x - pad : x + w + pad].astype(np.float32)
    H, W = c.shape
    blk = np.median(c[c < 60]) if (c < 60).any() else np.percentile(c, 1)
    wht = np.median(c[c > 240])
    cov = np.clip((wht - c) / (wht - blk), 0, 1)
    pts = []
    if side in ("left", "right"):
        lo, hi = int(H * 0.3), int(H * 0.7)
        for r in range(lo, hi):
            k = np.nonzero(cov[r] >= 0.5)[0]
            if not len(k):
                continue
            k = k[0] if side == "left" else k[-1]
            if k < 2 or k > W - 3:
                continue
            if side == "left":
                pts.append((k - 1 + (0.5 - cov[r, k - 1]) / (cov[r, k] - cov[r, k - 1]), r))
            else:
                pts.append((k + (cov[r, k] - 0.5) / (cov[r, k] - cov[r, k + 1]), r))
        P = np.array(pts)
        A = np.polyfit(P[:, 1], P[:, 0], 1)
        yy, xx = np.mgrid[lo:hi, 0:W]
        dist = (xx - (A[0] * yy + A[1])) / np.sqrt(1 + A[0] ** 2)
        vals = cov[lo:hi, :]
    else:
        lo, hi = int(W * 0.3), int(W * 0.7)
        for cidx in range(lo, hi):
            k = np.nonzero(cov[:, cidx] >= 0.5)[0]
            if not len(k):
                continue
            k = k[0] if side == "top" else k[-1]
            if k < 2 or k > H - 3:
                continue
            if side == "top":
                pts.append((cidx, k - 1 + (0.5 - cov[k - 1, cidx]) / (cov[k, cidx] - cov[k - 1, cidx])))
            else:
                pts.append((cidx, k + (cov[k, cidx] - 0.5) / (cov[k, cidx] - cov[k + 1, cidx])))
        P = np.array(pts)
        A = np.polyfit(P[:, 0], P[:, 1], 1)
        yy, xx = np.mgrid[0:H, lo:hi]
        dist = (yy - (A[0] * xx + A[1])) / np.sqrt(1 + A[0] ** 2)
        vals = cov[:, lo:hi]
    d, v = dist.ravel(), vals.ravel()
    near = np.abs(d) < 14
    # orient the distance axis so that ink is on the positive side
    if v[near & (d > 8)].mean() < 0.5:
        d = -d
    bins = np.arange(-14, 14.001, 0.1)
    k = np.digitize(d[near], bins) - 1
    sums = np.bincount(k, v[near], len(bins))
    cnt = np.bincount(k, minlength=len(bins))
    esf = (sums / np.maximum(cnt, 1))[: len(bins) - 1]
    good = cnt[: len(bins) - 1] > 5
    return ((bins[:-1] + bins[1:]) / 2)[good], esf[good], float(np.degrees(np.arctan(A[0])))


def edge_width(xs, esf):
    """10 to 90 percent rise width in px and the line spread function sigma in px"""

    def cross(level):
        i = np.nonzero(esf >= level)[0][0]
        return xs[i - 1] + (level - esf[i - 1]) / (esf[i] - esf[i - 1]) * (xs[i] - xs[i - 1])

    lsf = np.clip(np.gradient(gaussian_filter1d(esf, 2)), 0, None)
    lsf /= lsf.sum()
    mu = (xs * lsf).sum()
    return float(cross(0.9) - cross(0.1)), float(np.sqrt(((xs - mu) ** 2 * lsf).sum()))


def tone_steps(g, px_per_mm, x_over_y=1.0):
    """Grey levels of the printed tone steps, from 0 to 100 percent black in equal steps, or None when the wedge is not found

    The dark end of the wedge is found on a quarter size copy as a solid block one step tall, its right edge is
    refined at full size, and the other steps follow from the printed step width
    """
    q = 4
    small = cv2.blur(cv2.resize(g, None, fx=1 / q, fy=1 / q, interpolation=cv2.INTER_AREA), (9, 9))
    n, _, st, _ = cv2.connectedComponentsWithStats((small < 100).astype(np.uint8), connectivity=8)
    sw, sh = TONE_STEP_MM[0] * px_per_mm * x_over_y, TONE_STEP_MM[1] * px_per_mm
    found = [tuple(int(v) * q for v in s_[:4]) for s_ in st[1:n] if abs(s_[3] * q - sh) < 0.12 * sh and 5 * sw < s_[2] * q < 9 * sw and s_[4] > 0.8 * s_[2] * s_[3]]
    if len(found) != 1:
        return None
    x, y, w, h = found[0]
    band = g[y + int(0.25 * h) : y + int(0.75 * h)].astype(np.float32).mean(0)
    lo, hi = x + w - 4 * q * 4, x + w + 4 * q * 4
    x1 = lo + int(np.nonzero(band[lo:hi] < 150)[0][-1]) + 1
    x0 = x1 - TONE_STEPS * sw
    return [float(np.median(g[y + int(0.25 * h) : y + int(0.75 * h), int(x0 + i * sw + 0.25 * sw) : int(x0 + i * sw + 0.75 * sw)])) for i in range(TONE_STEPS)]


def analyse_scan(g, vertical_is_sheet_x):
    """Ruler scales, tilts and edge widths of one calibration scan, keyed by sheet axis"""
    comps = ink_components(g)
    rulers = {}
    for bbox, vertical in find_rulers(comps):
        tilt, pos = measure_ruler(g, bbox, vertical)
        scale, resid = fit_scale(pos)
        axis = "x" if vertical == vertical_is_sheet_x else "y"
        rulers[axis] = {"scan_axis": "y" if vertical else "x", "bbox": list(bbox), "ticks": len(pos), "tilt_deg": tilt, "px_per_mm": scale, "resid_px": resid}
    edges = []
    for bbox in find_squares(comps):
        for side in ("left", "right", "top", "bottom"):
            xs, esf, tilt = edge_profile(g, bbox, side)
            w, sigma = edge_width(xs, esf)
            edges.append({"bbox": list(bbox), "side": side, "tilt_deg": tilt, "w1090_px": w, "sigma_px": sigma})
    return {"rulers": rulers, "edges": edges}


def analyse(path_a, path_b):
    """Analyses scan A (sheet upright) and scan B (sheet turned 90 degrees clockwise) and summarises scale"""
    ga, gb = load_gray(path_a), load_gray(path_b)
    a = analyse_scan(ga, vertical_is_sheet_x=False)
    b = analyse_scan(gb, vertical_is_sheet_x=True)
    # sheet y is the reference axis, measured along scan y in A and along scan x in B
    sy_a, sy_b = a["rulers"]["y"]["px_per_mm"], b["rulers"]["y"]["px_per_mm"]
    sx_a, sx_b = a["rulers"]["x"]["px_per_mm"], b["rulers"]["x"]["px_per_mm"]
    widths = [e["w1090_px"] * UM_PER_PX for s in (a, b) for e in s["edges"]]
    summary = {
        "px_per_mm": (sy_a + sy_b) / 2,
        "scan_anisotropy": max(abs(sy_a - sy_b) / sy_a, abs(sx_a - sx_b) / sx_a),
        "print_x_over_y": (sx_a + sx_b) / (sy_a + sy_b),
        "edge_w1090_um_median": float(np.median(widths)) if widths else None,
    }
    # the wedge is read upright, so scan B is turned back a quarter turn, the same turn a card scanned at 90 degrees gets
    ratio = summary["print_x_over_y"]
    tone = {"a": tone_steps(ga, summary["px_per_mm"], ratio), "b": tone_steps(np.ascontiguousarray(np.rot90(gb, 3)), summary["px_per_mm"], ratio)}
    return {"a": a, "b": b, "summary": summary, "tone": tone}


def gate(result, nominal=NOMINAL_PX_PER_MM, scale_tol=SCALE_TOL, aniso_tol=ANISOTROPY_TOL):
    """Returns a list of failure messages, empty when the session passes"""
    s = result["summary"]
    fails = []
    if abs(s["px_per_mm"] - nominal) > scale_tol:
        fails.append(f"scale {s['px_per_mm']:.3f} px/mm is outside {nominal} +- {scale_tol}")
    if s["scan_anisotropy"] > aniso_tol:
        fails.append(f"x/y scales differ by {100 * s['scan_anisotropy']:.3f} %, limit {100 * aniso_tol:.1f} %")
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan_a", help="calibration scan with the sheet upright")
    ap.add_argument("scan_b", help="calibration scan with the sheet turned 90 degrees clockwise")
    ap.add_argument("-o", "--out", help="write the full measurements as JSON")
    a = ap.parse_args()
    res = analyse(a.scan_a, a.scan_b)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))
    fails = gate(res)
    for f in fails:
        print("FAIL", f)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
