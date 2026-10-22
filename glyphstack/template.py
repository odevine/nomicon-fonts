"""Labels units by overlap with glyphs rendered from a local font at the card's print scale, as a cross-check on alignment"""
import cv2
import numpy as np
from scipy import ndimage as ndi

from . import font as fnt
from . import styles, target
from .log import log
from .register import pmap

UP = 4


def render_templates(font, scale=1.0, chars=None, style=None):
    """Renders each character as a 4x binary mask with the ink spread added, keyed by glyph key

    The character set, em size, aspect and ink spread come from the profile, and the face defaults to its primary one
    """
    t = target.get()["templates"]
    chars, style = chars or t["charset"], style or styles.primary()
    sx = t["px_per_em"] / font["head"].unitsPerEm * scale
    out = {}
    cmap = font.getBestCmap()
    for ch in chars:
        if ord(ch) not in cmap:
            continue
        contours, _ = fnt.glyph_contours(font, ch)
        if not contours:
            continue
        cs = [np.stack([c[:, 0] * sx, -c[:, 1] * sx * t["aspect"]], 1) for c in contours]
        allp = np.vstack(cs)
        lo, hi = np.floor(allp.min(0)) - 8, np.ceil(allp.max(0)) + 8
        W, H = int(hi[0] - lo[0]), int(hi[1] - lo[1])
        mask = np.zeros((H, W), np.uint8)
        for p in cs:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.round((p - lo) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
            mask ^= m
        mask = (ndi.distance_transform_edt(mask == 0) <= t["ink_spread"]).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        out[styles.key(style, ch)] = dict(
            mask=mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32),
            w=(xs.max() - xs.min() + 1) / UP, h=(ys.max() - ys.min() + 1) / UP,
        )
    return out


def unit_mask4(cov):
    """Binary mask of a coverage map at 4x, cropped to its extent, or None when empty"""
    m = cv2.resize(cov, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC) > 0.5
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)


def best_iou(um, tm, search=12):
    """Best IoU of two binary masks over integer shifts within +-search of centre alignment"""
    H, W = max(um.shape[0], tm.shape[0]) + 2 * search, max(um.shape[1], tm.shape[1]) + 2 * search
    A = np.zeros((H, W), np.float32)
    y0, x0 = (H - um.shape[0]) // 2, (W - um.shape[1]) // 2
    A[y0 : y0 + um.shape[0], x0 : x0 + um.shape[1]] = um
    ov = cv2.matchTemplate(A, tm, cv2.TM_CCORR)
    ch, cw = (um.shape[0] - tm.shape[0]) / 2 + y0, (um.shape[1] - tm.shape[1]) / 2 + x0
    au, at = um.sum(), tm.sum()
    best = 0.0
    for dy in range(-search, search + 1):
        yy = int(round(ch + dy))
        if yy < 0 or yy >= ov.shape[0]:
            continue
        xs = np.arange(-search, search + 1) + int(round(cw))
        xs = xs[(xs >= 0) & (xs < ov.shape[1])]
        inter = ov[yy][xs].max()
        best = max(best, float(inter / (au + at - inter)))
    return best


def classify(cov, templates, top=3):
    """Returns the best (iou, char) candidates for one coverage map, best first"""
    um = unit_mask4(cov)
    if um is None:
        return []
    h, w = um.shape[0] / UP, um.shape[1] / UP
    res = []
    for ch, t in templates.items():
        if abs(t["h"] - h) > max(0.10 * t["h"], 3) or abs(t["w"] - w) > max(0.10 * t["w"], 3):
            continue
        res.append((best_iou(um, t["mask"]), ch))
    return sorted(res, reverse=True)[:top]


class Templates:
    """Caches rendered template sets by card scale, for one font per face given as {style: font}"""

    def __init__(self, fonts):
        self.fonts = fonts if isinstance(fonts, dict) else {styles.primary(): fonts}
        self._cache = {}

    def at(self, scale, only=None):
        """Templates of every face keyed by glyph key, or of the faces in `only`"""
        scale = round(scale, 3)
        out = {}
        for style, font in self.fonts.items():
            if only and style not in only:
                continue
            if (scale, style) not in self._cache:
                self._cache[(scale, style)] = render_templates(font, scale, style=style)
            out.update(self._cache[(scale, style)])
        return out


def find_scale(units, templates, seed=0, lo=None, hi=None, around=None):
    """Finds the print scale vs the font where templates overlap a sample of units best, no prior needed

    With `around`, only that scale and its neighbours 0.01 either side are tried, for further scans of a card
    whose scale is already known. The searched range comes from the profile unless lo and hi are given
    """
    rng_lo, rng_hi = target.get()["templates"]["scale_range"]
    lo, hi = rng_lo if lo is None else lo, rng_hi if hi is None else hi
    rng = np.random.default_rng(seed)
    sample = [units[i] for i in rng.choice(len(units), min(100, len(units)), replace=False)]

    def score(s):
        t = templates.at(s, only=(styles.primary(),))
        v = [(c or [(0, "")])[0][0] for c in pmap(lambda u: classify(u["cov"], t, top=1), sample)]
        return float(np.mean(sorted(v)[-25:]))

    if around is not None:
        return round(float(max(np.round(around + np.array([-0.01, 0, 0.01]), 3), key=score)), 3)
    coarse = max(np.arange(lo, hi + 1e-9, 0.02), key=score)
    fine = max(np.arange(coarse - 0.03, coarse + 0.0301, 0.01), key=score)
    return round(float(fine), 3)


def identify(units, templates, scale=None, around=None):
    """Labels units with the best template, returns (kept units, card scale)

    Units below the profile's min_iou or within its min_margin of a different character are dropped, which for the rules
    face also removes the bold type line and mana symbols, and any face without a font. A character that matches two faces
    equally takes the face of its neighbours. The scale is searched on the primary face only, or only next to `around`
    """
    if not scale:
        log.info("searching print scale from a sample of %d units%s", min(100, len(units)), "" if around is None else f" near {around:.3f}")
        scale = find_scale(units, templates, around=around)
        log.info("print scale %.3f", scale)
    t = templates.at(scale)
    min_iou, min_margin = target.get()["templates"]["min_iou"], target.get()["templates"]["min_margin"]
    kept, tied = [], []
    for u, c in zip(units, pmap(lambda u: classify(u["cov"], t), units)):
        if not c or c[0][0] < min_iou:
            continue
        # a runner-up that is the same character in the other face is a tie of faces, not of characters
        rival = next((r for r in c[1:] if styles.char_of(r[1]) != styles.char_of(c[0][1])), None)
        margin = c[0][0] - (rival[0] if rival else 0)
        if margin < min_margin:
            continue
        same = [r for r in c[1:] if styles.char_of(r[1]) == styles.char_of(c[0][1]) and c[0][0] - r[0] < min_margin]
        unit = dict(u, ch=c[0][1], iou=c[0][0], margin=margin, scale=scale)
        (tied if same else kept).append(unit)
    kept += resolve_faces(tied, kept)
    return kept, scale


def resolve_faces(tied, kept):
    """Gives units whose character is clear but whose face is not (a full stop looks the same in both) the face of the nearest unit on their line

    Units with no face-certain unit on their line take the primary face
    """
    out = []
    for u in tied:
        near = [k for k in kept if abs(k["cy"] - u["cy"]) < 0.5 * (u["bbox"][1] - u["bbox"][0]) + 20]
        face = styles.style_of(min(near, key=lambda k: abs(k["cx"] - u["cx"]))["ch"]) if near else styles.primary()
        out.append(dict(u, ch=styles.key(face, styles.char_of(u["ch"]))))
    return out
