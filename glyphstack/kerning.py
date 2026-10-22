"""Measures the gap between neighbouring printed glyphs against the designed spacing and the font's pair kerning"""
from collections import defaultdict

import numpy as np

from . import styles
from .font import Kerning
from .log import log

MAX_GAP_PX = 100
MAX_DEV_FU = 120


def ink_edges(u, level):
    """Left and right ink edge of a unit in card crop columns at the given coverage level"""
    cols = np.nonzero((u["cov"] > level).any(0))[0]
    return u["box"][2] + cols.min(), u["box"][2] + cols.max() + 1


def right_neighbour(a, units):
    """The nearest unit to the right of a on the same baseline within MAX_GAP_PX, or None"""
    h = a["bbox"][1] - a["bbox"][0]
    cand = [b for b in units if b is not a and abs(b["bbox"][1] - a["bbox"][1]) < 0.15 * h + 12 and 0 < b["bbox"][2] - a["bbox"][3] < MAX_GAP_PX and b["cx"] > a["cx"]]
    return min(cand, key=lambda b: b["bbox"][2]) if cand else None


def measure_pairs(scans, levels, card_S, font, fu_px, spread_px, style=None):
    """One row per adjacent pair of printed glyphs of one face

    font is the font of `style`, only units of that face are measured, and fu_px is grid px per font unit at card
    scale 1 and spread_px the fitted ink spread per edge in grid px.
    Gaps are in font units between ink edges at each scan's own edge level, plus the ink spread of both edges.
    dev_nokern is the gap minus the designed gap, dev_kern also subtracts the font's pair kerning
    """
    style = style or styles.primary()
    kern = Kerning(font)
    spread_fu = spread_px / fu_px
    rows = []
    for n, s in scans.items():
        spf = fu_px / 4 * card_S[s["card"]][0][0]
        for a in s["units"]:
            b = right_neighbour(a, s["units"])
            if b is None or styles.style_of(a["ch"]) != style or styles.style_of(b["ch"]) != style:
                continue
            ca, cb = styles.char_of(a["ch"]), styles.char_of(b["ch"])
            if ord(ca) not in kern.cmap or ord(cb) not in kern.cmap:
                continue
            e1, e2 = ink_edges(a, levels[n]), ink_edges(b, levels[n])
            gap = (e2[0] - e1[1]) / spf + 2 * spread_fu
            base = kern.advance(ca) - kern.bounds(ca)[1] + kern.bounds(cb)[0]
            k = kern.pair(ca, cb)
            if abs(gap - base) >= MAX_DEV_FU:
                continue
            rows.append(dict(scan=n, card=s["card"], style=style, pair=ca + cb, gap=float(gap), base=float(base), kern=int(k), dev_nokern=float(gap - base), dev_kern=float(gap - base - k)))
    log.info("%d adjacent pairs measured from %d scans", len(rows), len(scans))
    return rows


def summarise(rows):
    """Per pair designed value and observed mean, spread and count, with the individual samples"""
    by = defaultdict(list)
    for r in rows:
        by[r["pair"]].append(r)
    out = {}
    for pair, rs in by.items():
        nk, dk = np.array([r["dev_nokern"] for r in rs]), np.array([r["dev_kern"] for r in rs])
        out[pair] = dict(
            n=len(rs), designed_gap=rs[0]["base"], kern=rs[0]["kern"], mean_dev_nokern=float(nk.mean()), mean_dev_kern=float(dk.mean()), sd=float(dk.std()),
            samples=[dict(scan=r["scan"], card=r["card"], dev_nokern=round(r["dev_nokern"], 1), dev_kern=round(r["dev_kern"], 1)) for r in rs],
        )
    return out
