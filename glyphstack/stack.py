"""Stacks the instances of each glyph from many scans into one coverage map on a 4x grid

Each card's scans are matched to one reference scan from glyph centres, which gives every scan's linear map to
that reference and groups the views of each printed glyph. The card's tilt and scale against a running per-glyph
mean are estimated by ECC on a sample of units and carried to its other scans through those maps. Units are
registered, normalised to a common edge level, averaged over the views of each printed glyph, screened for
outliers and averaged
"""
import json
import pickle
import shutil
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import chunks
from .contours import UM_PER_PX, edge_rms_um
from .log import Progress, log, timed
from . import target
from .pairing import match_map
from .styles import stem as fname
from .register import ecc_affine, iou, place, pmap, rectified_shape, rectify, register, register_shifts, registered, rotation_scale

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
    ys, xs = np.nonzero(img > thr)
    if len(ys) == 0:
        return img
    return img[max(ys.min() - margin, 0) : ys.max() + margin + 1, max(xs.min() - margin, 0) : xs.max() + margin + 1]


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
    for _ in range(18):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (reg > mid).sum() > area else (lo, mid)
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


def register_glyph(scans, M_tie, keys, order, folder, batch=64):
    """Registers the given units of one glyph into a chunk store at folder, returns each unit's edge level in store order

    The running means are summed in the order of keys, which fixes the result, and the store holds keys[i] for i in order
    """
    load = lambda i: grid_unit(scans, keys[i], M_tie[keys[i][0]])
    shapes = [rectified_shape(scans[n]["units"][k], scans[n]["scale"], M_tie[n]) for n, k in keys]
    size, shifts, mean = register_shifts(load, len(keys), shapes)
    area = (mean > 0.5).sum()

    def one(i):
        r = registered(load(i), size, shifts[i])
        # the level is measured on the float32 image, before the store rounds it to float16
        return r, level_for_area(r, area)

    levels = []
    with chunks.Writer(folder, meta=dict(keys=[list(keys[i]) for i in order])) as w:
        for b in range(0, len(order), batch):
            for r, t in pmap(one, order[b : b + batch]):
                w.add(r)
                levels.append(t)
    return levels


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
    """Screens and stacks one glyph from its registered units in R, reading the store three or four times

    Gives the same numbers as stack_instances on the whole glyph, since every sum runs in order in float32. Writes
    stack.npz (mean, sd and the kept instances' leave-one-out IoU) and the kept instances as a chunks.Store under
    folder. Returns the stack, then per printed glyph the view against view deviation, keep flag, leave-one-out
    IoU and deviation from the stack
    """
    loo_min = target.get()["stack"]["loo_min_iou"] if loo_min is None else loo_min
    n = len(spans)
    sc = lambda a: cv2.resize(a, None, fx=1, fy=asp, interpolation=cv2.INTER_CUBIC)
    passes = lambda: _instances(R, keys, spans, levels)
    pair_dev, acc = [None] * n, np.zeros(R.shape, np.float32)
    for batch in passes():
        # disagreement between the views of the same printed glyph, before the edge level remap
        for (i, _, a), d in zip(batch, pmap(lambda b: view_dev_um(b[1]), batch)):
            acc += a
            pair_dev[i] = d
    mean = acc / n
    loo = np.ones(n)
    if n > 2:
        for batch in passes():
            for (i, _, _), v in zip(batch, pmap(lambda b: iou(b[2], (mean * n - b[2]) / (n - 1)), batch)):
                loo[i] = v
    keep = loo >= loo_min
    nk = int(keep.sum())
    if not keep.all():
        acc = np.zeros(R.shape, np.float32)
        for batch in passes():
            for i, _, a in batch:
                if keep[i]:
                    acc += a
        mean = acc / nk
    redo_loo = not keep.all() and nk > 2
    loo_kept = loo[keep] if keep.all() else np.ones(nk)
    mean_p = sc(mean)
    def kept(b):
        a = b[2]
        ap = sc(a)
        return ap, edge_rms_um(ap, mean_p), iou(a, (mean * nk - a) / (nk - 1)) if redo_loo else None

    devs, sq, k = [None] * n, np.zeros(R.shape, np.float32), 0
    with chunks.Writer(folder / "inst") as w:
        for batch in passes():
            batch = [b for b in batch if keep[b[0]]]
            for (i, _, a), (ap, dev, li) in zip(batch, pmap(kept, batch)):
                sq += (a - mean) ** 2
                devs[i] = dev
                if redo_loo:
                    loo_kept[k] = li
                w.add(ap)
                k += 1
    sd = sc(np.sqrt(sq / nk))
    np.savez_compressed(folder / "stack.npz", mean=mean_p, sd=sd, loo=loo_kept)
    return dict(mean=mean_p, sd=sd, n=nk, loo=loo_kept), pair_dev, keep, loo, devs


def build(scans, workdir, iterations=2, chars=None, tie=True, stacks_dir=None):
    """Runs the full stacking over {scan name: {units, scale, card, angle}}

    Returns {"stacks": {char: {...}}, "levels": {scan: level}, "aspect": float, "scans": per-scan geometry,
    "instances": manifest rows}. Registered per-unit maps go to workdir/regs and each glyph's stack to
    stacks_dir/<glyph>, default workdir/stacks, with its instances in a chunks.Store. Memory follows one batch of
    units rather than the largest glyph
    """
    workdir = Path(workdir)
    (workdir / "regs").mkdir(parents=True, exist_ok=True)
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
    # each glyph's store holds its units in printed glyph order, so the views of one printed glyph sit together
    spans_of, keys_of, per_scan, levels_of = {}, {}, defaultdict(list), {}
    prog = Progress("registering glyphs", len(glyphs))
    for ch in glyphs:
        groups = defaultdict(list)
        for key in by_char[ch]:
            groups[group_of.get(key, key)].append(key)
        keys_of[ch] = [key for g in groups.values() for key in g]
        ends = np.cumsum([len(g) for g in groups.values()])
        spans_of[ch] = [(kk, int(e - len(g)), int(e)) for (kk, g), e in zip(groups.items(), ends)]
        at = {key: i for i, key in enumerate(by_char[ch])}
        levels_of[ch] = register_glyph(scans, M_tie, by_char[ch], [at[key] for key in keys_of[ch]], workdir / "regs" / fname(ch))
        for nk, t in zip(keys_of[ch], levels_of[ch]):
            per_scan[nk[0]].append(t)
        prog.step(f"{ch!r}, {len(by_char[ch])} units")
    levels = {n: float(np.median(v)) for n, v in per_scan.items()}
    log.info("edge levels %s", " ".join(f"{n}:{t:.3f}" for n, t in sorted(levels.items())))
    stacks_dir = Path(stacks_dir or workdir / "stacks")
    if stacks_dir.exists():
        shutil.rmtree(stacks_dir)
    stacks, rows = {}, []
    prog = Progress("stacking glyphs", len(glyphs))
    for ch in glyphs:
        keys, spans = keys_of[ch], spans_of[ch]
        R = chunks.Store(workdir / "regs" / fname(ch))
        st, pair_dev, keep, loo, devs = stack_glyph(R, keys, spans, levels, asp, stacks_dir / fname(ch))
        cards = sorted({scans[kk[0]]["card"] for (kk, _, _), kp in zip(spans, keep) if kp})
        for i, (kk, a, b) in enumerate(spans):
            members = keys[a:b]
            card = scans[kk[0]]["card"]
            rows.append(dict(
                char=ch, pair_id=f"{card}:{ch}:{kk[0]}:{kk[1]}", card=card, kept=bool(keep[i]),
                scans=[m[0] for m in members], units=[m[1] for m in members], edge_level=[levels[m[0]] for m in members],
                halo_px=[scans[m[0]]["units"][m[1]]["halo_px"] for m in members], card_scale=float(np.sqrt(card_S[card][0, 0] * card_S[card][1, 1])),
                print_scale=scans[kk[0]]["scale"], loo_iou=float(loo[i]), deviation_um=devs[i], pair_dev_um=pair_dev[i],
            ))
        stacks[ch] = dict(st, n_units=len(keys), n_dropped=int((~keep).sum()), cards=cards)
        prog.step(f"{ch!r}: {st['n']} instances from {len(cards)} cards, {stacks[ch]['n_dropped']} dropped, LOO IoU {st['loo'].mean():.3f}")
    with_dev = [r for r in rows if r.get("pair_dev_um") is not None]
    devs = np.array([r["pair_dev_um"] for r in with_dev])
    nviews = np.array([len(r["scans"]) for r in with_dev])
    rms = float(np.sqrt((devs**2).mean())) if len(devs) else None
    # a printed glyph's mean over n views carries the single-view noise divided by the square root of n
    fused = float(np.sqrt((devs**2 / 2 / nviews).mean())) if len(devs) else None
    noise = dict(tied=bool(tie), n_pairs=len(devs), views_mean=float(nviews.mean()) if len(devs) else None, pair_rms_um=rms,
                 single_scan_um=None if rms is None else rms / np.sqrt(2), pair_mean_um=fused)
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
