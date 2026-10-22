"""Small raw crops, step images and registration transforms for the site's Scan to Glyph sequence and the Stack's pixel layers

Only pieces of a few letters are written: a rules-text crop about 240 x 144 scan pixels (5 x 3 mm) for the first
step, and one glyph-sized crop per glyph for the pixel layers. Everything else is derived coverage, never a whole card
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import styles, target
from . import units as un
from .contours import UM_PER_PX, contours_at
from .log import Progress, log, timed
from .register import UP, com_canvas, rectify, register
from .scan import load_gray
from . import chunks
from .stack import ScanUnits, fname

CONTEXT = (240, 144)
PIXEL_PAD = 24
REG_CHOICES = 6
PATH_TOL = 0.4
_ECC = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 400, 1e-8)


def save_gray(a, path):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path, lossless=True)


def save_ink(cov, path):
    """Black ink with the coverage as its alpha, so a page can recolour it with a CSS filter"""
    rgba = np.zeros(cov.shape + (4,), np.uint8)
    rgba[..., 3] = np.round(np.clip(cov, 0, 1) * 255)
    Image.fromarray(rgba).save(path, lossless=True)


def save_classes(cls, path):
    """RGBA map of 1 this glyph, 2 neighbouring ink, 3 exclusion halo, in the site palette"""
    colours = {1: (0x1F, 0x55, 0xA0, 255), 2: (0xD6, 0x23, 0x2A, 255), 3: (0xC9, 0xCE, 0xD6, 255)}
    rgba = np.zeros(cls.shape + (4,), np.uint8)
    for k, c in colours.items():
        rgba[cls == k] = c
    Image.fromarray(rgba).save(path, lossless=True)


def rel_path(polys, tol=PATH_TOL, step=0.25):
    """Closed relative paths of polylines in grid coordinates, simplified and rounded to a quarter pixel"""
    out = []
    for c in polys:
        q = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2), tol, True).reshape(-1, 2)
        if len(q) < 3:
            continue
        r = np.round(q / step).astype(int)
        out.append(f"M{r[0][0] * step:g},{r[0][1] * step:g}l" + " ".join(f"{(b[0] - a[0]) * step:g},{(b[1] - a[1]) * step:g}" for a, b in zip(r[:-1], r[1:])) + "z")
    return "".join(out)


def ink_box(img, thr=0.08, margin=6):
    ys, xs = np.where(img > thr)
    return max(ys.min() - margin, 0), min(ys.max() + margin + 1, img.shape[0]), max(xs.min() - margin, 0), min(xs.max() + margin + 1, img.shape[1])


def m_tie(man, scan):
    """The matrix that rectified a scan into the card's reference frame, as built by stack.tie_cards"""
    s = man["scans"][scan]
    if "M" in s:
        return np.array(s["M"])
    th = np.radians(s["rot_deg"])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return R @ (np.array(man["card_S"][s["card"]]) / s["print_scale"])


def native_to_reg(cov, reg, scale):
    """2x3 affine from the unit's native pixel coordinates to the registered 4x grid, found by ECC from a scale and centre of mass start"""
    f = UP / scale
    ref = reg.astype(np.float32)
    cr, cn = ndi.center_of_mass(ref), ndi.center_of_mass(cov)
    W = np.array([[1 / f, 0, cn[1] - cr[1] / f], [0, 1 / f, cn[0] - cr[0] / f]], np.float32)
    cc, W = cv2.findTransformECC(ref, cov.astype(np.float32), W, cv2.MOTION_AFFINE, _ECC, None, 5)
    return cv2.invertAffineTransform(W), float(cc)


def to_fu(p, asp):
    """2x3 affine from registered 4x grid pixel centres to font units, y down, including the stack's vertical aspect scaling"""
    s, ar, th, tx, ty = p[:5]
    Mi = np.linalg.inv(np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([s, s * ar]))
    A = Mi @ np.diag([1, asp])
    return np.hstack([A, (Mi @ (np.array([0, 0.5 * asp - 0.5]) - [tx, ty]))[:, None]])


def compose(outer, inner):
    """outer after inner, both 2x3"""
    o, i = np.vstack([outer, [0, 0, 1]]), np.vstack([inner, [0, 0, 1]])
    return (o @ i)[:2]


def svg_matrix(M):
    return [round(float(v), 5) for v in (M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2])]


class Session:
    def __init__(self, folder, scans_dir):
        self.folder, self.scans_dir = Path(folder), Path(scans_dir)
        self.man = json.loads((self.folder / "manifest.json").read_text())
        self.traces = json.loads((self.folder / "traces.json").read_text())
        self.scans = ScanUnits(self.folder)
        self._card, self._regs = None, {}

    def frame(self, scan):
        """The card crop of a scan in the upright frame the units were cut in, loaded once and kept for the last scan used"""
        if self._card is None or self._card[0] != scan:
            crop, _ = un.card_crop(load_gray(next(self.scans_dir.glob(f"{scan}.*"))))
            self._card = (scan, un.upright(crop, self.man["scans"][scan]["angle"])[0])
        return self._card[1]

    def reg(self, ch, scan, unit):
        """The registered 4x coverage of one unit of a glyph, from the glyph's chunk store"""
        # one glyph's store at a time, so the chunk it keeps is the only one in memory
        if self._regs.get("ch") != ch:
            R = chunks.Store(self.folder / "work" / "regs" / fname(ch))
            self._regs = dict(ch=ch, R=R, at={tuple(k): i for i, k in enumerate(R.index["keys"])})
        return self._regs["R"][self._regs["at"][(scan, unit)]]

    def instance_rows(self, ch):
        return [r for r in self.man["instances"] if r["char"] == ch and r["kept"]]

    def typical(self, ch, card=None):
        """Index (in the kept list, which is also the site id number) of an instance whose deviation is closest to the median, one with an upright and a turned view when there is one"""
        rows = self.instance_rows(ch)
        devs = [r["deviation_um"] for r in rows if r["deviation_um"] is not None]
        med = float(np.median(devs))
        ok = [(abs(r["deviation_um"] - med), k, has_pair(self, r)) for k, r in enumerate(rows) if r["deviation_um"] is not None and (card is None or r["card"] == card)]
        paired = [c for c in ok if c[2]]
        return min(paired or ok)[1], rows


def turn(S, scan):
    """A scan's angle in the scan table, from 0 up to 360"""
    return S.man["scans"][scan]["angle"] % 360


def unit_pair(S, row):
    """Upright and turned (scan, unit index) views of an instance, the first upright view and the view turned closest to a quarter turn from it"""
    views = list(zip(row["scans"], row["units"]))
    up = next((v for v in views if turn(S, v[0]) == 0), views[0])
    others = [v for v in views if v != up] or [up]
    return up, min(others, key=lambda v: abs((turn(S, v[0]) - turn(S, up[0])) % 180 - 90))


def has_pair(S, row):
    return any(turn(S, s) == 0 for s in row["scans"]) and any(turn(S, s) != 0 for s in row["scans"])


def export_pixels(S, out, p_by_char, asp, chars):
    """One raw crop and one registered 4x coverage image per glyph for the Stack's pixel layers, with the transforms that place them in font units"""
    d = Path(out) / "pixels"
    d.mkdir(parents=True, exist_ok=True)
    by_scan = {}
    for ch in chars:
        k, rows = S.typical(ch)
        by_scan.setdefault(unit_pair(S, rows[k])[0][0], []).append((ch, k, rows[k]))
    prog = Progress("pixel layer crops", len(chars))
    for scan, jobs in sorted(by_scan.items()):
        g = S.frame(scan)
        for ch, k, row in jobs:
            (sp, ku), _ = unit_pair(S, row)
            u = S.scans[sp]["units"][ku]
            y0, y1, x0, x1 = u["bbox"]
            y0, y1, x0, x1 = max(y0 - PIXEL_PAD, 0), min(y1 + PIXEL_PAD, g.shape[0]), max(x0 - PIXEL_PAD, 0), min(x1 + PIXEL_PAD, g.shape[1])
            raw = g[y0:y1, x0:x1]
            reg = S.reg(ch, sp, ku)
            by, bx = u["box"][0], u["box"][2]
            cov = u["cov"]
            Wn, cc = native_to_reg(cov, reg, S.scans[sp]["scale"])
            reg_a = cv2.resize(reg, None, fx=1, fy=asp, interpolation=cv2.INTER_CUBIC)
            r0, r1, c0, c1 = ink_box(reg_a, margin=10)
            F = to_fu(np.array(p_by_char[ch]), asp)
            # raw crop pixel centres -> unit box pixel centres -> registered grid -> font units
            shift = np.array([[1, 0, x0 - bx], [0, 1, y0 - by]], np.float32)
            raw_fu = compose(F, compose(np.vstack([Wn, [0, 0, 1]])[:2], shift))
            cov_fu = compose(F, np.array([[1, 0, c0], [0, 1, r0]], np.float64))
            name = styles.glyph_name(ch)
            save_gray(raw, d / f"{name}_raw.webp")
            save_ink(np.clip(reg_a[r0:r1, c0:c1], 0, 1), d / f"{name}_cov4.webp")
            card = row["card"]
            (d / f"{name}.json").write_text(json.dumps(dict(
                glyph=styles.char_of(ch), id=styles.instance_id(ch, k), card=card, scan=sp, angle=S.scans[sp]["angle"], ecc_cc=round(cc, 4),
                raw=dict(file=f"{name}_raw.webp", w=int(raw.shape[1]), h=int(raw.shape[0]), matrix=svg_matrix(raw_fu), um_per_px=UM_PER_PX),
                cov4=dict(file=f"{name}_cov4.webp", w=int(c1 - c0), h=int(r1 - r0), matrix=svg_matrix(cov_fu), um_per_px=UM_PER_PX / UP),
            ), separators=(",", ":")))
            prog.step(f"{ch!r} cc {cc:.3f}")
    return d


def export_steps(S, out, p_by_char, asp):
    """The images and numbers behind the ten-step figure for one `e`"""
    ch = target.get()["site"]["step_glyph"]
    d = Path(out) / "steps"
    d.mkdir(parents=True, exist_ok=True)
    k, rows = S.typical(ch, target.get()["site"]["step_card"])
    row = rows[k]
    (sp, kp), (sl, kl) = unit_pair(S, row)
    up = S.scans[sp]["units"][kp]
    g = S.frame(sp)
    # step 1 and 2, a rules-text crop with the glyph in it and the segmentation of its surroundings
    cy, cx = int(up["cy"]), int(up["cx"])
    cw, chh = CONTEXT
    y0, x0 = int(np.clip(cy - chh // 2, 0, g.shape[0] - chh)), int(np.clip(cx - cw // 2, 0, g.shape[1] - cw))
    raw = g[y0 : y0 + chh, x0 : x0 + cw]
    save_gray(raw, d / "s1_raw.webp")
    m = 100
    Y0, X0 = max(y0 - m, 0), max(x0 - m, 0)
    win = g[Y0 : y0 + chh + m, X0 : x0 + cw + m]
    lab, comps = un.segment(win)
    allink = np.isin(lab, [i for c in comps for i in c["ids"]])
    mine = [c for c in comps if abs((c["y0"] + c["y1"]) / 2 + Y0 - up["cy"]) < 2 and abs((c["x0"] + c["x1"]) / 2 + X0 - up["cx"]) < 2]
    assert len(mine) == 1, f"found {len(mine)} components for the unit"
    mask = np.isin(lab, mine[0]["ids"])
    others = allink & ~mask
    halo = ndi.binary_dilation(others, iterations=up["halo_px"]) & ~others & ~mask
    cls = np.zeros(win.shape, np.uint8)
    cls[halo], cls[others], cls[mask] = 3, 2, 1
    save_classes(cls[y0 - Y0 : y0 - Y0 + chh, x0 - X0 : x0 - X0 + cw], d / "s2_classes.webp")
    # step 3, linear coverage against the cleaned unit
    paper, ink = S.scans[sp]["paper"], S.scans[sp]["ink"]
    by0, by1, bx0, bx1 = up["box"]
    lin = np.clip((paper - g[by0:by1, bx0:bx1].astype(np.float32)) / (paper - ink), 0, 1)
    save_ink(lin, d / "s3_linear.webp")
    save_ink(up["cov"], d / "s3_clean.webp")
    # step 4, rectified 4x grid before registration
    rect = rectify(up, S.scans[sp]["scale"], m_tie(S.man, sp), S.scans[sp].get("to_up"))
    r0, r1, c0, c1 = ink_box(rect, margin=4 * 6)
    save_ink(rect[r0:r1, c0:c1], d / "s4_rect.webp")
    s4 = dict(file="s4_rect.webp", w=int(c1 - c0), h=int(r1 - r0))
    # step 5, centre of mass against registered contours of one e from each of several cards
    picks, seen = [], set()
    for rr in rows:
        if rr["card"] in seen or not rr["scans"] or rr["deviation_um"] is None:
            continue
        seen.add(rr["card"])
        s0 = [s for s in rr["scans"] if turn(S, s) == 0]
        if s0:
            picks.append((rr["card"], s0[0], rr["units"][rr["scans"].index(s0[0])]))
        if len(picks) == REG_CHOICES:
            break
    rects = [rectify(S.scans[s]["units"][u], S.scans[s]["scale"], m_tie(S.man, s), S.scans[s].get("to_up")) for _, s, u in picks]
    before = com_canvas(rects)
    after = register(rects)
    allc = [contours_at(a, 0.5) for a in before + after]
    pts = np.vstack([c for cs in allc for c in cs])
    ox, oy = int(pts[:, 0].min()) - 6, int(pts[:, 1].min()) - 6
    w5, h5 = int(pts[:, 0].max()) + 7 - ox, int(pts[:, 1].max()) + 7 - oy
    sh = lambda cs: [c - [ox, oy] for c in cs]
    s5 = dict(w=w5, h=h5, cards=[c for c, _, _ in picks], before=[rel_path(sh(c)) for c in allc[:len(picks)]], after=[rel_path(sh(c)) for c in allc[len(picks):]])
    # step 6, the same physical e from the upright scan and the scan turned closest to a quarter turn
    a, b = S.reg(ch, sp, kp), S.reg(ch, sl, kl)
    r0, r1, c0, c1 = ink_box(np.maximum(a, b), margin=4 * 6)
    a, b = a[r0:r1, c0:c1], b[r0:r1, c0:c1]
    save_ink(a, d / "s6_upright.webp")
    save_ink(b, d / "s6_turned.webp")
    save_ink((a + b) / 2, d / "s6_mean.webp")
    save_ink(np.clip(np.abs(a - b) * 2.5, 0, 1), d / "s6_diff.webp")
    p = p_by_char[ch]
    (d / f"{ch}.json").write_text(json.dumps(dict(
        glyph=ch, id=f"{ch}-{k:03d}", card=row["card"], upright=sp, turned=sl, upright_angle=S.scans[sp]["angle"], turned_angle=S.scans[sl]["angle"], um_per_px=UM_PER_PX, um_per_grid_px=UM_PER_PX / UP, um_per_fu=round(UM_PER_PX / UP * p[0], 4), ink_spread_um=round(UM_PER_PX / UP * p[5], 2),
        paper=paper, ink=ink, rot_deg=S.man["scans"][sp]["rot_deg"], print_scale=S.scans[sp]["scale"], edge_levels=row["edge_level"], pair_dev_um=row["pair_dev_um"], deviation_um=row["deviation_um"],
        s1=dict(file="s1_raw.webp", w=int(cw), h=int(chh), glyph_box=[int(up["bbox"][2] - x0), int(up["bbox"][0] - y0), int(up["bbox"][3] - up["bbox"][2]), int(up["bbox"][1] - up["bbox"][0])]),
        s2=dict(file="s2_classes.webp", halo_px=up["halo_px"]),
        s3=dict(linear="s3_linear.webp", clean="s3_clean.webp", w=int(lin.shape[1]), h=int(lin.shape[0])),
        s4=s4,
        s5=s5, s6=dict(files=["s6_upright.webp", "s6_turned.webp", "s6_mean.webp", "s6_diff.webp"], w=int(c1 - c0), h=int(r1 - r0)),
    ), separators=(",", ":")))
    return d


def export_pairs(S, out, chars):
    """The registered upright and turned views of one typical printed glyph per character, for the noise comparison"""
    d = Path(out) / "noise"
    d.mkdir(parents=True, exist_ok=True)
    for ch in chars:
        rows = [(k, r) for k, r in enumerate(S.instance_rows(ch)) if has_pair(S, r) and r["pair_dev_um"] is not None]
        med = float(np.median([r["pair_dev_um"] for _, r in rows]))
        k, row = min(rows, key=lambda kr: abs(kr[1]["pair_dev_um"] - med))
        (sp, kp), (sl, kl) = unit_pair(S, row)
        a, b = S.reg(ch, sp, kp), S.reg(ch, sl, kl)
        r0, r1, c0, c1 = ink_box(np.maximum(a, b), margin=4 * 6)
        name = styles.glyph_name(ch)
        save_ink(a[r0:r1, c0:c1], d / f"{name}_u.webp")
        save_ink(b[r0:r1, c0:c1], d / f"{name}_t.webp")
        (d / f"{name}.json").write_text(json.dumps(dict(
            glyph=styles.char_of(ch), id=styles.instance_id(ch, k), card=row["card"], upright=sp, turned=sl, upright_angle=S.scans[sp]["angle"], turned_angle=S.scans[sl]["angle"], pair_dev_um=round(row["pair_dev_um"], 3), median_pair_dev_um=round(med, 3),
            w=int(c1 - c0), h=int(r1 - r0), um_per_grid_px=UM_PER_PX / UP,
        ), separators=(",", ":")))
    return d


def export(session, scans_dir, out, chars=None):
    """Writes steps/ and pixels/ under out, from a session folder and the folder holding its scan PNGs"""
    S = Session(session, scans_dir)
    assert S.man["noise"]["tied"], "the step images rebuild the tied rectification, run with the tied session"
    p_by_char = {c: v["p"] for c, v in S.traces.items()}
    asp = S.man["aspect"]
    chars = [c for c in (chars or p_by_char) if c in p_by_char]
    with timed("step images"):
        export_steps(S, out, p_by_char, asp)
    with timed("pixel layer crops"):
        export_pixels(S, out, p_by_char, asp, chars)
    with timed("noise pairs"):
        export_pairs(S, out, chars)
    log.info("wrote crops and steps to %s", out)
