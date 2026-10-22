"""How the edge error of a stacked glyph falls as more printed instances are averaged"""
import numpy as np

from .contours import UM_PER_PX, contours_at, sym_dev
from .log import log
from .register import pmap

BASE_NS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28]
GAIN_LIMIT = 0.07


def grid(n):
    """Instance counts to sample, the pilot's grid up to 28 then roughly geometric up to all n"""
    ns = [k for k in BASE_NS if k < n]
    k = 28
    while k * 1.3 < n:
        k = int(round(k * 1.3 / 4) * 4)
        ns.append(k)
    return sorted(set(ns + [n]))


def _rms_um(a, b):
    ca, cb = contours_at(a, 0.5), contours_at(b, 0.5)
    if not ca or not cb:
        return np.nan, np.nan
    return float(np.sqrt((sym_dev(ca, cb) ** 2).mean()) * UM_PER_PX)


def sweep(inst, reference, draws=40, seed=2026):
    """Edge RMS against a reference contour set, and split-half noise, as the number of averaged instances N grows

    reference is a list of polylines in the stack's grid coordinates, for example a placed designed outline
    used as a measuring stick. inst is (n, H, W). Each N is drawn `draws` times without replacement. The
    noise column compares the means of two disjoint subsets of size N, so it needs 2N <= n
    """
    inst = np.asarray(inst, np.float32)
    n = len(inst)
    rng = np.random.default_rng(seed)
    rows = []
    for N in grid(n):
        subsets = [rng.choice(n, N, replace=False) for _ in range(1 if N == n else draws)]
        rms = np.array(pmap(lambda ix: float(np.sqrt((sym_dev(contours_at(inst[ix].mean(0), 0.5) or [np.zeros((1, 2))], reference) ** 2).mean()) * UM_PER_PX), subsets))
        row = dict(n=N, rms_vs_ref_um=float(rms.mean()), p10=float(np.percentile(rms, 10)), p90=float(np.percentile(rms, 90)))
        if 2 * N <= n:
            halves = [rng.permutation(n) for _ in range(draws)]
            d = pmap(lambda p: _rms_um(inst[p[:N]].mean(0), inst[p[N : 2 * N]].mean(0)), halves)
            # each half carries noise sigma/sqrt(N), so their difference carries sqrt(2) times that
            row["noise_um"] = float(np.nanmean(d) / np.sqrt(2))
        rows.append(row)
    return rows


def noise_model(rows):
    """Fits RMS^2(N) = b^2 + s^2/N, returns (b, s) where b is the floor shared with the reference and s the single instance noise, in um"""
    x = np.array([1 / r["n"] for r in rows])
    y = np.array([r["rms_vs_ref_um"] ** 2 for r in rows])
    A = np.stack([np.ones_like(x), x], 1)
    w = 1 / y
    b2, s2 = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    return float(np.sqrt(max(b2, 0))), float(np.sqrt(max(s2, 0)))


def recommended_n(rows, s):
    """First sampled N where the modelled noise falls less than GAIN_LIMIT from the previous sampled N"""
    for prev, cur in zip(rows, rows[1:]):
        if 1 - np.sqrt(prev["n"] / cur["n"]) < GAIN_LIMIT:
            return cur["n"]
    return rows[-1]["n"]


def sweep_glyph(inst, reference, draws=40):
    """Sweep rows, model parameters and the recommended N for one glyph"""
    rows = sweep(inst, reference, draws)
    b, s = noise_model(rows)
    for r in rows:
        r["noise_model_um"] = s / np.sqrt(r["n"])
    log.info("sweep: %d instances, floor %.2f um, single-instance noise %.2f um", len(inst), b, s)
    return dict(rows=rows, floor_um=b, single_noise_um=s, recommended_n=recommended_n(rows, s))
