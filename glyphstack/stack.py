"""Stacks the instances of each glyph from many scans into one coverage map on a 4x grid

Each card's scans are matched to one reference scan from glyph centres, which gives every scan's linear map to
that reference and groups the views of each printed glyph. The card's tilt and scale against a running per-glyph
mean are estimated by ECC on a sample of units and carried to its other scans through those maps. Units are
registered, normalised to a common edge level, averaged over the views of each printed glyph, screened for
outliers and averaged
"""
import json
import os
import pickle
import shutil
from collections import defaultdict
from concurrent.futures import as_completed
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import chunks
from .contours import UM_PER_PX, edge_rms_um
from .log import Progress, log, pool, timed
from . import target
from .pairing import match_map
from .styles import stem as fname
from .register import canvas_size, com_offset, ecc_affine, ink_box, iou, place, pmap, rectified_shape, rectify, register, register_shifts, registered, rotation_scale

AFFINE_MIN_CC = 0.85
# units per scan fitted by affine ECC, and per glyph registered for the reference means between passes
AFFINE_SAMPLE = 60
REF_SAMPLE = 400
GROUP_TOL_PX = 8
SEED = 2026


class ScanUnits:
    """The extracted units of a session's scans, read from work/units one scan at a time with the last few kept in memory"""

    def __init__(self, folder, keep=4):
        self.files = {f.stem: f for f in sorted((Path(folder) / "work" / "units").glob("*.pkl"))}
        self._load = lru_cache(keep)(lambda n: pickle.loads(self.files[n].read_bytes()))

    def __getitem__(self, name):
        return self._load(name)

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return iter(self.files)

    def items(self):
        for n in self.files:
            yield n, self[n]


def units_by_char(scans):
    out = defaultdict(list)
    for n, s in scans.items():
        for k, u in enumerate(s["units"]):
            out[u["ch"]].append((n, k))
    return out


def affine_chars(by_char, n=None):
    """The most frequent primary-face characters of the profile's geometry pool, whose shapes are stable enough to estimate scan geometry from"""
    geo = target.get()["geometry"]
    n = n or geo["count"]
    letters = [c for c in by_char if len(c) == 1 and c in geo["pool"]]
    return "".join(sorted(sorted(letters, key=lambda c: -len(by_char[c]))[:n]))


def sample(items, n, seed=SEED):
    """At most n items in their original order, chosen at random with a fixed seed"""
    if len(items) <= n:
        return list(items)
    return [items[i] for i in sorted(np.random.default_rng(seed).choice(len(items), n, replace=False))]


def grid_unit(scans, nk, M2=np.eye(2)):
    """One unit on the 4x grid, rectified by M2 in a single resample"""
    s = scans[nk[0]]
    return rectify(s["units"][nk[1]], s["scale"], M2, s.get("to_up"))


def bootstrap_means(scans, by_char, chars):
    """Translation-only means of a sample of each glyph in the frame of the scans, the reference for the first affine pass"""
    return {c: np.mean(register(pmap(lambda nk: grid_unit(scans, nk), sample(by_char[c], REF_SAMPLE))), 0) for c in chars}


def trim(img, margin=24, thr=0.02):
    """The image cropped to its ink plus a margin"""
    y0, y1, x0, x1 = ink_box(img, margin, thr)
    return img[y0:y1, x0:x1]


def scan_affines(scans, refs, n=AFFINE_SAMPLE):
    """Affine of a sample of each scan's units against the reference means, as {scan: [2x2 matrices]}

    Each matrix maps the reference glyph frame to the scan's 4x grid. Both images are cropped to their ink first
    """
    trimmed = {c: trim(r.astype(np.float32)) for c, r in refs.items()}

    def one(nk):
        ref = trimmed[scans[nk[0]]["units"][nk[1]]["ch"]]
        x = trim(grid_unit(scans, nk))
        shape = (max(ref.shape[0], x.shape[0]) + 24, max(ref.shape[1], x.shape[1]) + 24)
        a = ecc_affine(place(ref, shape), place(x, shape))
        return a[0] if a and a[1] > AFFINE_MIN_CC else None

    out = {}
    prog = Progress("scan geometry", len(scans))
    for name, s in scans.items():
        keys = sample([(name, k) for k, u in enumerate(s["units"]) if u["ch"] in refs], n)
        out[name] = [m for m in pmap(one, keys) if m is not None]
        prog.step(f"{name}, {len(out[name])} units fitted")
    return out


def centre(u):
    """Centre of the unit's coverage in its scan's upright crop, x then y"""
    if "box" not in u:
        return np.array([u["cx"], u["cy"]])
    cy, cx = ndi.center_of_mass(u["cov"])
    return np.array([u["box"][2] + cx, u["box"][0] + cy])


def fit_affine(A, B, rounds=3):
    """Affine (2x3) mapping points A to B by least squares, dropping points far off the fit each round"""
    keep = np.ones(len(A), bool)
    X = np.hstack([A, np.ones((len(A), 1))])
    for _ in range(rounds):
        M, *_ = np.linalg.lstsq(X[keep], B[keep], rcond=None)
        r = np.linalg.norm(X @ M - B, axis=1)
        keep = r < max(4 * np.median(r[keep]), 1.0)
    return M.T


def card_maps(scans):
    """Matches every scan of a card to the card's reference scan, the first with the card upright, from glyph positions

    Returns {scan: (reference scan, 2x3 affine from reference crop pixels to this scan's crop pixels)} and the views
    of each printed glyph as {(scan, k): group key}, where the group key is the first member's (scan, k)
    """
    cards = defaultdict(list)
    for n, s in scans.items():
        cards[s["card"]].append(n)
    maps, group_of = {}, {}
    for card, ns in cards.items():
        ns = sorted(ns, key=lambda n: (scans[n]["angle"] % 360 != 0, n))
        ref = ns[0]
        cen = {n: np.array([centre(u) for u in scans[n]["units"]]) for n in ns}
        tab = lambda n: [(u["ch"], u["cx"], u["cy"]) for u in scans[n]["units"]]
        maps[ref] = (ref, np.eye(2, 3))
        for n in ns[1:]:
            pairs = match_map(tab(ref), tab(n))[0] if scans[n]["units"] and scans[ref]["units"] else []
            if len(pairs) < 6:
                log.warning("scan %s matched %d glyphs to %s, it is treated as unmatched", n, len(pairs), ref)
                maps[n] = (n, np.eye(2, 3))
                continue
            maps[n] = (ref, fit_affine(cen[ref][[a for a, _ in pairs]], cen[n][[b for _, b in pairs]]))
        # each view joins the nearest group of its character within GROUP_TOL_PX that has no view from its scan yet
        groups = defaultdict(list)
        for n in ns:
            if maps[n][0] != ref:
                continue
            A = np.vstack([maps[n][1], [0, 0, 1]])
            pos = (np.hstack([cen[n], np.ones((len(cen[n]), 1))]) @ np.linalg.inv(A).T)[:, :2] if len(cen[n]) else cen[n]
            for k, u in enumerate(scans[n]["units"]):
                best, bd = None, GROUP_TOL_PX
                for g in groups[u["ch"]]:
                    d = np.hypot(*(g["pos"] - pos[k]))
                    if d < bd and n not in g["scans"]:
                        best, bd = g, d
                if best is None:
                    groups[u["ch"]].append(dict(key=(n, k), pos=pos[k], scans={n}))
                    group_of[(n, k)] = (n, k)
                else:
                    best["scans"].add(n)
                    group_of[(n, k)] = best["key"]
    return maps, group_of


def tie_cards(scans, affines, maps, tie=True):
    """Matrices that rectify each scan into the reference glyph frame, the per-card scale and shear, and the pooled aspect

    With the tie, every unit's affine is carried to its card's reference scan through the centre map and averaged
    there, then carried back, so all scans of a card share one estimate. With tie=False each scan keeps the mean of
    its own units. Returns per-scan matrices, per-card matrices, the pooled y over x aspect and per-scan info
    """
    def lin(n):
        # the centre map in 4x grid units of each scan, which differ when the print scales differ
        ref, A = maps[n]
        return A[:, :2] * scans[ref]["scale"] / scans[n]["scale"]

    M_tie, card_S, info = {}, {}, {}
    cards = defaultdict(list)
    for n, s in scans.items():
        cards[s["card"]].append(n)
    for card, ns in cards.items():
        by_ref = defaultdict(list)
        for n in ns:
            by_ref[maps[n][0]] += [np.linalg.inv(lin(n)) @ M for M in affines[n]]
        G = {r: np.mean(v, 0) for r, v in by_ref.items() if v}
        acc, wsum = 0, 0
        for n in ns:
            own = np.mean(affines[n], 0) if affines[n] else np.eye(2)
            M_tie[n] = lin(n) @ G[maps[n][0]] if tie and maps[n][0] in G else own
            th, _, Sm = rotation_scale(M_tie[n])
            k = max(len(affines[n]), 1)
            acc, wsum = acc + Sm * scans[n]["scale"] * k, wsum + k
            info[n] = dict(rot_deg=float(np.degrees(th)), refined_scale=float(Sm[0, 0] * scans[n]["scale"]), aspect=float(Sm[1, 1] / Sm[0, 0]), units_used=len(affines[n]),
                           reference_scan=maps[n][0], M=M_tie[n].tolist())
        card_S[card] = acc / wsum
    w = [sum(len(affines[n]) for n in cards[c]) for c in card_S]
    asp = float(np.average([card_S[c][1, 1] / card_S[c][0, 0] for c in card_S], weights=w if sum(w) else None))
    return M_tie, card_S, asp, info


def rectify_glyph(scans, M_tie, keys):
    """Rectified 4x coverage of the given units, one resample each"""
    return pmap(lambda nk: grid_unit(scans, nk, M_tie[nk[0]]), keys)


def level_for_area(reg, area, lo=0.2, hi=0.8):
    """Coverage level at which a registered unit has the given ink area, by bisection"""
    # every midpoint lies strictly between lo and hi, so pixels at or below lo never count and pixels above hi always do
    above, band = int((reg > hi).sum()), reg[(reg > lo) & (reg <= hi)]
    for _ in range(18):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if above + (band > mid).sum() > area else (lo, mid)
    return (lo + hi) / 2


def remap(c, t):
    """Piecewise linear coverage remap that sends level t to 0.5"""
    t = float(np.clip(t, 0.2, 0.8))
    return np.where(c < t, c * 0.5 / t, 0.5 + (c - t) * 0.5 / (1 - t)).astype(np.float32)


def _loo_iou(inst, mean):
    n = len(inst)
    return np.array([iou(inst[i], (mean * n - inst[i]) / (n - 1)) for i in range(n)]) if n > 2 else np.ones(n)


def stack_instances(inst, loo_min=None):
    """Leave-one-out screening and averaging of view-averaged instances, returns (mean, keep mask, loo IoU, loo IoU of the kept)"""
    loo_min = target.get()["stack"]["loo_min_iou"] if loo_min is None else loo_min
    mean = inst.mean(0)
    loo = _loo_iou(inst, mean)
    keep = loo >= loo_min
    if keep.all():
        return mean, keep, loo, loo
    inst = inst[keep]
    mean = inst.mean(0)
    return mean, keep, loo, _loo_iou(inst, mean)


def view_dev_um(views, cap=8):
    """Edge RMS between two views of one printed glyph, as a pair would show it, from any number of views

    Each view is compared with the mean of the others. With n views that distance carries n / (n - 1) times the
    single-view variance, against 2 for a pair, so the RMS is rescaled to the pair it stands for. At most `cap`
    views are compared
    """
    n = len(views)
    if n < 2:
        return None
    if n == 2:
        return edge_rms_um(views[0], views[1])
    total = np.sum(views, 0)
    d = [edge_rms_um(v, (total - v) / (n - 1)) for v in views[:cap]]
    d = np.array([x for x in d if x is not None])
    return float(np.sqrt((d**2).mean() * 2 * (n - 1) / n)) if len(d) else None


def _instances(R, keys, spans, levels, views=128):
    """Batches of (span index, raw views, mean of the remapped views), one per printed glyph, read from the store in order

    A batch holds whole printed glyphs and about `views` views, read with one take so each chunk is decompressed once
    """
    def inst(i, vs):
        a = spans[i][1]
        return np.mean([remap(v, levels[keys[a + j][0]]) for j, v in enumerate(vs)], 0)

    b = 0
    while b < len(spans):
        e = b + 1
        while e < len(spans) and spans[e][2] - spans[b][1] <= views:
            e += 1
        a0 = spans[b][1]
        block = R.take(range(a0, spans[e - 1][2]))
        idx = range(b, e)
        vs = [block[spans[i][1] - a0 : spans[i][2] - a0] for i in idx]
        yield list(zip(idx, vs, pmap(lambda iv: inst(*iv), list(zip(idx, vs)))))
        b = e


def stack_glyph(R, keys, spans, levels, asp, folder, loo_min=None):
    """Screens and stacks one glyph from its registered views in R, averaging each printed glyph's remapped views first

    Gives the same numbers as stack_instances on the view averages, since every sum runs in order in float32. Returns
    what stack_fused returns, with the view against view deviation of each printed glyph measured here
    """
    pair_dev, avgs = [None] * len(spans), []
    for batch in _instances(R, keys, spans, levels):
        # disagreement between the views of the same printed glyph, before the edge level remap
        for (i, _, _), d in zip(batch, pmap(lambda b: view_dev_um(b[1]), batch)):
            pair_dev[i] = d
        avgs.append([a for _, _, a in batch])
    return stack_fused([a for b in avgs for a in b], pair_dev, asp, folder, loo_min)


def stack_fused(inst, pair_dev, asp, folder, loo_min=None):
    """Screens and stacks one glyph from its fused instances, one float32 image per printed glyph in order

    Gives the same numbers as stack_instances, since every sum runs in order in float32. Writes stack.npz (mean, sd
    and the kept instances' leave-one-out IoU) and the kept instances as a chunks.Store under folder. Returns the
    stack, then per printed glyph the given view against view deviation, keep flag, leave-one-out IoU and deviation
    from the stack
    """
    loo_min = target.get()["stack"]["loo_min_iou"] if loo_min is None else loo_min
    n = len(inst)
    shape = inst[0].shape
    sc = lambda a: cv2.resize(a, None, fx=1, fy=asp, interpolation=cv2.INTER_CUBIC)
    acc = np.zeros(shape, np.float32)
    for a in inst:
        acc += a
    mean = acc / n
    loo = np.ones(n)
    if n > 2:
        loo = np.array(pmap(lambda a: iou(a, (mean * n - a) / (n - 1)), inst))
    keep = loo >= loo_min
    nk = int(keep.sum())
    if not keep.all():
        acc = np.zeros(shape, np.float32)
        for a, k in zip(inst, keep):
            if k:
                acc += a
        mean = acc / nk
    redo_loo = not keep.all() and nk > 2
    loo_kept = loo[keep] if keep.all() else np.ones(nk)
    mean_p = sc(mean)

    def kept(a):
        ap = sc(a)
        return ap, edge_rms_um(ap, mean_p), iou(a, (mean * nk - a) / (nk - 1)) if redo_loo else None

    devs, sq, k = [None] * n, np.zeros(shape, np.float32), 0
    idx = np.flatnonzero(keep)
    with chunks.Writer(folder / "inst") as w:
        for b in range(0, len(idx), 128):
            sel = idx[b : b + 128]
            for i, (ap, dev, li) in zip(sel, pmap(lambda i: kept(inst[i]), sel)):
                sq += (inst[i] - mean) ** 2
                devs[i] = dev
                if redo_loo:
                    loo_kept[k] = li
                w.add(ap)
                k += 1
    sd = sc(np.sqrt(sq / nk))
    np.savez_compressed(folder / "stack.npz", mean=mean_p, sd=sd, loo=loo_kept)
    return dict(mean=mean_p, sd=sd, n=nk, loo=loo_kept), pair_dev, keep, loo, devs


# The fused path runs in four stages, each one task per card or per glyph, so each fits a process pool.
# A registers each printed glyph's views against each other within its card and stores their average.
# B registers a glyph's printed glyph averages against each other, which gives each printed glyph a shift and the glyph its ink area.
# C renders every view once through both shifts, sets each scan's edge level and stores each printed glyph's fused instance.
# D screens and stacks each glyph's fused instances.


def render_view(scans, nk, M, size, offset, w):
    """One view rectified and placed on its glyph's canvas at an integer offset, then shifted by w, two resamples in all"""
    return registered(grid_unit(scans, nk, M), size, w, offset)


def _card_groups(scans, M_tie, groups, sizes, folder):
    """Stage A for one card: per glyph, each printed glyph's views registered against each other

    groups is {glyph: [[view keys of one printed glyph], ...]}. Each printed glyph's average is stored under
    folder/<glyph>/means. Returns {glyph: [[(offset, shift) per view] per printed glyph]}
    """
    out = {}
    for ch, gs in groups.items():
        size, res = sizes[ch], []
        with chunks.Writer(folder / fname(ch) / "means", np.float32) as w:
            for keys in gs:
                rect = [grid_unit(scans, nk, M_tie[nk[0]]) for nk in keys]
                _, shifts, mean = register_shifts(rect.__getitem__, len(rect), None, size=size)
                res.append([(com_offset(r, size), sh) for r, sh in zip(rect, shifts)])
                w.add(mean)
        out[ch] = res
    return out


def _glyph_shifts(folders, size):
    """Stage B for one glyph: its printed glyph averages from each card's store, in card order, registered against each other

    Returns (offset, shift) per printed glyph and the ink area of the glyph's mean
    """
    stores = [chunks.Store(f) for f in folders]
    at = [(s, i) for s in stores for i in range(len(s))]
    load = lambda j: at[j][0][at[j][1]]
    _, shifts, mean = register_shifts(load, len(at), None, size=size)
    offsets = [com_offset(load(j), size) for j in range(len(at))]
    return list(zip(offsets, shifts)), int((mean > 0.5).sum())


def _compose(view, group):
    """A view's placement and shift on the glyph canvas, from its place within its printed glyph and its printed glyph's place in the glyph"""
    (oy, ox), w = view
    (gy, gx), s = group
    w = w.copy()
    w[:, 2] += s[:, 2]
    return (oy + gy, ox + gx), w


def _card_fuse(scans, M_tie, groups, sizes, places, areas, folder):
    """Stage C for one card: every view rendered through both shifts, the card's per-scan edge levels, and each printed glyph's fused instance

    places is {glyph: [[(offset, shift) per view] per printed glyph]} composed from stages A and B. Each scan's level
    is the median over its views of the coverage level that gives the view its glyph's ink area. Fused instances, the
    mean of each printed glyph's views remapped by their scan's level, go to folder/<glyph>/inst. Returns the levels
    and {glyph: [view against view deviation per printed glyph]}
    """
    render = lambda ch, nk, pl: render_view(scans, nk, M_tie[nk[0]], sizes[ch], *pl)
    per_scan = defaultdict(list)
    for ch, gs in groups.items():
        jobs = [(nk, pl) for keys, pls in zip(gs, places[ch]) for nk, pl in zip(keys, pls)]
        for (nk, _), t in zip(jobs, pmap(lambda j: level_for_area(render(ch, *j), areas[ch]), jobs)):
            per_scan[nk[0]].append(t)
    levels = {n: float(np.median(v)) for n, v in per_scan.items()}

    def fuse(ch, keys, pls):
        views = [render(ch, nk, pl) for nk, pl in zip(keys, pls)]
        # disagreement between the views of the same printed glyph, before the edge level remap
        return view_dev_um(views), np.mean([remap(v, levels[nk[0]]) for nk, v in zip(keys, views)], 0)

    devs = {}
    for ch, gs in groups.items():
        res = pmap(lambda kp: fuse(ch, *kp), list(zip(gs, places[ch])))
        with chunks.Writer(folder / fname(ch) / "inst", np.float32) as w:
            for _, a in res:
                w.add(a)
        devs[ch] = [d for d, _ in res]
    return levels, devs


def _glyph_stack(folders, pair_dev, asp, folder):
    """Stage D for one glyph: its fused instances from each card's store, in card order, screened and stacked"""
    inst = [a for f in folders for a in chunks.Store(f).take(range(len(chunks.Store(f))))]
    return stack_fused(inst, pair_dev, asp, folder)


def _map(fn, tasks, workers, label):
    """fn(*task) for each task, in a process pool of `workers` with the largest task first, or here when workers is 1

    tasks is a list of (size, args). Returns the results in the order of tasks
    """
    out = [None] * len(tasks)
    prog = Progress(label, len(tasks))
    if workers > 1 and len(tasks) > 1:
        with pool(min(workers, len(tasks)), target.use, (target.get().path,)) as ex:
            futs = {ex.submit(fn, *args): i for i, (_, args) in sorted(enumerate(tasks), key=lambda t: -t[1][0])}
            for f in as_completed(futs):
                out[futs[f]] = f.result()
                prog.step()
    else:
        for i, (_, args) in enumerate(tasks):
            out[i] = fn(*args)
            prog.step()
    return out


def build(scans, workdir, iterations=2, chars=None, tie=True, stacks_dir=None, workers=None):
    """Runs the full stacking over {scan name: {units, scale, card, angle}}

    Returns {"stacks": {char: {...}}, "levels": {scan: level}, "aspect": float, "scans": per-scan geometry,
    "instances": manifest rows}. Each printed glyph's views are fused within their card into one instance under
    workdir/fused, and each glyph's stack goes to stacks_dir/<glyph>, default workdir/stacks, with its instances in a
    chunks.Store. The per-card and per-glyph stages run in `workers` processes, default one per core
    """
    workdir = Path(workdir)
    fused = workdir / "fused"
    if fused.exists():
        shutil.rmtree(fused)
    by_char = units_by_char(scans)
    achars = chars or affine_chars(by_char)
    log.info("%d scans, %d units, %d glyph kinds, geometry from %s", len(scans), sum(len(s["units"]) for s in scans.values()), len(by_char), achars)
    with timed("matching scans within cards"):
        maps, group_of = card_maps(scans)
    log.info("%d units in %d printed glyphs", len(group_of), len(set(group_of.values())))
    with timed("reference means"):
        refs = bootstrap_means(scans, by_char, achars)
    for it in range(iterations):
        with timed(f"geometry pass {it + 1}/{iterations}"):
            affines = scan_affines(scans, refs)
            M_tie, card_S, asp, info = tie_cards(scans, affines, maps, tie)
            log.info("pooled aspect %.4f", asp)
            if it + 1 < iterations:
                refs = {c: np.mean(register(rectify_glyph(scans, M_tie, sample(by_char[c], REF_SAMPLE))), 0) for c in achars}
    n_groups = {c: len({group_of.get(nk, nk) for nk in v}) for c, v in by_char.items()}
    min_units = target.get()["stack"]["min_units"]
    glyphs = [c for c, v in by_char.items() if len(v) >= min_units and n_groups[c] >= min_units // 2]
    workers = workers or os.cpu_count() or 1

    # each printed glyph's views, per card and glyph, in the order the views first appear
    cards = sorted({s["card"] for s in scans.values()})
    groups = {c: {ch: [] for ch in glyphs} for c in cards}
    for ch in glyphs:
        by_group = defaultdict(list)
        for key in by_char[ch]:
            by_group[group_of.get(key, key)].append(key)
        for keys in by_group.values():
            groups[scans[keys[0][0]]["card"]][ch].append(keys)
    sizes = {ch: canvas_size([rectified_shape(scans[n]["units"][k], scans[n]["scale"], M_tie[n]) for n, k in by_char[ch]]) for ch in glyphs}
    card_scans = {c: {n: s for n, s in scans.items() if s["card"] == c} for c in cards}
    card_M = {c: {n: M_tie[n] for n in card_scans[c]} for c in cards}
    views = lambda c: sum(len(k) for gs in groups[c].values() for k in gs)

    with timed("registering views within cards"):
        within = _map(_card_groups, [(views(c), (card_scans[c], card_M[c], groups[c], sizes, fused / c)) for c in cards], workers, "registering views within cards")
    folders = lambda ch, part: [fused / c / fname(ch) / part for c in cards if groups[c][ch]]
    with timed("registering printed glyphs"):
        between = _map(_glyph_shifts, [(n_groups[ch], (folders(ch, "means"), sizes[ch])) for ch in glyphs], workers, "registering printed glyphs")
    places = {c: {} for c in cards}
    for ch, (gplaces, _) in zip(glyphs, between):
        it = iter(gplaces)
        for c, w in zip(cards, within):
            places[c][ch] = [[_compose(v, g) for v in vs] for vs, g in zip(w[ch], it)]
    areas = {ch: area for ch, (_, area) in zip(glyphs, between)}
    # each view's canvas and placement, so a registered view can be rendered again later, as the site's pixel layers do
    (fused / "places.json").write_text(json.dumps({ch: dict(size=list(sizes[ch]), views=[
        [nk[0], nk[1], int(o[0]), int(o[1]), float(w[0, 2]), float(w[1, 2])]
        for c in cards for keys, pls in zip(groups[c][ch], places[c][ch]) for nk, (o, w) in zip(keys, pls)]) for ch in glyphs}))
    with timed("fusing views"):
        fusedres = _map(_card_fuse, [(views(c), (card_scans[c], card_M[c], groups[c], sizes, places[c], areas, fused / c)) for c in cards], workers, "fusing views")
    levels = {n: t for lv, _ in fusedres for n, t in lv.items()}
    log.info("edge levels %s", " ".join(f"{n}:{t:.3f}" for n, t in sorted(levels.items())))

    stacks_dir = Path(stacks_dir or workdir / "stacks")
    if stacks_dir.exists():
        shutil.rmtree(stacks_dir)
    # a glyph's printed glyphs in card order, each with its views
    order = {ch: [keys for c in cards for keys in groups[c][ch]] for ch in glyphs}
    pair_devs = {ch: [d for c, (_, dv) in zip(cards, fusedres) for d in dv.get(ch, [])] for ch in glyphs}
    with timed("stacking glyphs"):
        res = _map(_glyph_stack, [(n_groups[ch], (folders(ch, "inst"), pair_devs[ch], asp, stacks_dir / fname(ch))) for ch in glyphs], workers, "stacking glyphs")
    stacks, rows = {}, []
    for ch, (st, pair_dev, keep, loo, devs) in zip(glyphs, res):
        stack_cards = sorted({scans[keys[0][0]]["card"] for keys, kp in zip(order[ch], keep) if kp})
        for i, members in enumerate(order[ch]):
            kk = members[0]
            card = scans[kk[0]]["card"]
            rows.append(dict(
                char=ch, pair_id=f"{card}:{ch}:{kk[0]}:{kk[1]}", card=card, kept=bool(keep[i]),
                scans=[m[0] for m in members], units=[m[1] for m in members], edge_level=[levels[m[0]] for m in members],
                halo_px=[scans[m[0]]["units"][m[1]]["halo_px"] for m in members], card_scale=float(np.sqrt(card_S[card][0, 0] * card_S[card][1, 1])),
                print_scale=scans[kk[0]]["scale"], loo_iou=float(loo[i]), deviation_um=devs[i], pair_dev_um=pair_dev[i],
            ))
        stacks[ch] = dict(st, n_units=len(by_char[ch]), n_dropped=int((~keep).sum()), cards=stack_cards)
        log.debug("%r: %d instances from %d cards, %d dropped, LOO IoU %.3f", ch, st["n"], len(stack_cards), stacks[ch]["n_dropped"], st["loo"].mean())
    with_dev = [r for r in rows if r.get("pair_dev_um") is not None]
    devs = np.array([r["pair_dev_um"] for r in with_dev])
    nviews = np.array([len(r["scans"]) for r in with_dev])
    rms = float(np.sqrt((devs**2).mean())) if len(devs) else None
    # a printed glyph's mean over n views carries the single-view noise divided by the square root of n
    fused_um = float(np.sqrt((devs**2 / 2 / nviews).mean())) if len(devs) else None
    noise = dict(tied=bool(tie), n_pairs=len(devs), views_mean=float(nviews.mean()) if len(devs) else None, pair_rms_um=rms,
                 single_scan_um=None if rms is None else rms / np.sqrt(2), pair_mean_um=fused_um)
    if rms is not None:
        log.info("view against view %d printed glyphs (%.1f views each), edge RMS %.2f um, single scan noise %.2f um, mean of views %.2f um",
                 len(devs), noise["views_mean"], rms, noise["single_scan_um"], noise["pair_mean_um"])
    scan_rows = {n: dict(card=s["card"], angle=s["angle"], print_scale=s["scale"], edge_level=levels.get(n), **info[n]) for n, s in scans.items()}
    return dict(stacks=stacks, stacks_dir=str(stacks_dir), levels=levels, aspect=asp, scans=scan_rows, instances=rows, noise=noise, card_S={c: m.tolist() for c, m in card_S.items()})


def save(result, out):
    """Writes manifest.json under out and moves the per-glyph stacks to out/stacks when they were built elsewhere"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src, dst = Path(result["stacks_dir"]), out / "stacks"
    if src.resolve() != dst.resolve():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(src, dst)
        result["stacks_dir"] = str(dst)
    # a leftover stacks.npz no longer matches this run
    (out / "stacks.npz").unlink(missing_ok=True)
    meta = {ch: {k: v for k, v in s.items() if k not in ("mean", "sd", "inst", "loo")} for ch, s in result["stacks"].items()}
    manifest = dict(profile=result.get("profile"), aspect=result["aspect"], noise=result["noise"], scans=result["scans"], card_S=result["card_S"], glyphs=meta, instances=result["instances"], um_per_px=UM_PER_PX)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def load(folder):
    """Reads manifest.json and the per-glyph stacks of a session folder, each glyph's instances stay on disk as a chunks.Store"""
    folder = Path(folder)
    man = json.loads((folder / "manifest.json").read_text())
    if not (folder / "stacks").is_dir():
        raise SystemExit(f"{folder} has no stacks folder, rerun the session to write per-glyph stacks")
    stacks = {}
    for ch, m in man["glyphs"].items():
        d = folder / "stacks" / fname(ch)
        Z = np.load(d / "stack.npz")
        stacks[ch] = dict(m, mean=Z["mean"], sd=Z["sd"], loo=Z["loo"], inst=chunks.Store(d / "inst"))
    return stacks, man
