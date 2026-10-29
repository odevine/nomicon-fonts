"""Labels units by overlap with glyphs rendered from a local font at the card's print scale, as a cross-check on alignment"""
import cv2
import numpy as np
from scipy import ndimage as ndi

from . import font as fnt
from . import styles, target
from .log import log
from .register import pmap

UP = 4
# combining accents and the spacing accent each face draws for them, whose template a mark printed over a letter takes
MARKS = {"\u0300": "`", "\u0301": "´", "\u0302": "ˆ", "\u0303": "˜", "\u0307": "˙", "\u0308": "¨", "\u030A": "˚", "\u030C": "ˇ"}
# the base an accent sits on, where it differs from the letter's own, as the dotless stem under an accented i
DOTLESS = {"i": "ı", "j": "ȷ"}
# how far a part's coverage reaches past its own ink when a dotted letter is split, in scan px
PART_PX = 4


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


def classify(cov, templates, top=3, tol=0.10):
    """Returns the best (iou, char) candidates for one coverage map, best first, from templates within tol of its size"""
    um = unit_mask4(cov)
    if um is None:
        return []
    h, w = um.shape[0] / UP, um.shape[1] / UP
    res = []
    for ch, t in templates.items():
        if abs(t["h"] - h) > max(tol * t["h"], 3) or abs(t["w"] - w) > max(tol * t["w"], 3):
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
        t = in_context_out(templates.at(s, only=(styles.primary(),)))
        v = [(c or [(0, "")])[0][0] for c in pmap(lambda u: classify(u["cov"], t, top=1), sample)]
        return float(np.mean(sorted(v)[-25:]))

    if around is not None:
        return round(float(max(np.round(around + np.array([-0.01, 0, 0.01]), 3), key=score)), 3)
    coarse = max(np.arange(lo, hi + 1e-9, 0.02), key=score)
    fine = max(np.arange(coarse - 0.03, coarse + 0.0301, 0.01), key=score)
    return round(float(fine), 3)


def count_scale(units, templates, lo=None, hi=None, around=None):
    """The print scale at which the most units are labelled, in steps of 0.02 and then 0.01 around the best

    This reads every unit, so it is slower than find_scale, and it stays on the peak when a textured box lowers every
    unit's overlap and flattens the sample's mean of its best overlaps. With `around`, only that scale and its
    neighbours 0.01 either side are tried
    """
    rng_lo, rng_hi = target.get()["templates"]["scale_range"]
    lo, hi = rng_lo if lo is None else lo, rng_hi if hi is None else hi
    n = lambda s: len(label(units, templates.at(round(float(s), 3)), round(float(s), 3)))
    if around is not None:
        return round(float(max(np.round(around + np.array([-0.01, 0, 0.01]), 3), key=n)), 3)
    coarse = max(np.arange(lo, hi + 1e-9, 0.02), key=n)
    return round(float(max(np.arange(coarse - 0.01, coarse + 0.0101, 0.01), key=n)), 3)


def card_templates(kept, base, min_units):
    """The templates with each character that has min_units or more kept units replaced by their mean mask

    Each unit's 4x mask is centred on its centre of mass and the masks are averaged and cut at one half, so the
    template takes the card's own letterforms and ink spread in place of the measuring stick font's
    """
    by = {}
    for u in kept:
        by.setdefault(u["ch"], []).append(u)
    out = dict(base)
    for ch, us in by.items():
        ms = [m for m in (unit_mask4(u["cov"]) for u in us) if m is not None]
        if len(ms) < min_units:
            continue
        H, W = max(m.shape[0] for m in ms) + 16, max(m.shape[1] for m in ms) + 16
        acc = np.zeros((H, W), np.float32)
        for m in ms:
            cy, cx = ndi.center_of_mass(m)
            acc += cv2.warpAffine(m, np.float32([[1, 0, W / 2 - cx], [0, 1, H / 2 - cy]]), (W, H), flags=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(acc / len(ms) > 0.5)
        if len(ys) == 0:
            continue
        mk = (acc / len(ms) > 0.5)[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
        out[ch] = dict(mask=mk, w=mk.shape[1] / UP, h=mk.shape[0] / UP)
    return out


def identify(units, templates, scale=None, around=None):
    """Labels units with the best template, returns (kept units, card scale)

    Units below the profile's min_iou or within its min_margin of a different character are dropped, which for the rules
    face also removes the bold type line and mana symbols, and any face without a font. A character that matches two faces
    equally takes the face of its neighbours. The scale is searched on the primary face only, or only next to `around`,
    by the profile's scale_search: top, the mean of a sample's best overlaps, or count, the most labelled units. With
    self_rounds set, the units are labelled again that many times against templates learned from the card's own kept units
    """
    cfg = target.get()["templates"]
    if not scale:
        mode = cfg.get("scale_search", "top")
        log.info("searching print scale by %s%s", "the best overlaps of a sample of %d units" % min(100, len(units)) if mode == "top" else "labelled units",
                 "" if around is None else f" near {around:.3f}")
        scale = (count_scale if mode == "count" else find_scale)(units, templates, around=around)
        log.info("print scale %.3f", scale)
    t = templates.at(scale)
    kept = label(units, t, scale)
    for _ in range(cfg.get("self_rounds", 0)):
        kept = label(units, card_templates(kept, t, cfg.get("self_min_units", 3)), scale)
    if "mark_min_iou" in cfg:
        kept = kept + label_marks(units, kept, t, scale)
        kept = kept + split_dotted(kept, templates, scale)
    return anchor_marks(kept, templates, scale), scale


def anchor(mark, base, templates, scale):
    """A mark's place on its base in x-heights of the base's face: its ink centre across less the base's, and the gap from its foot to the base's top"""
    em = target.get()["templates"]["px_per_em"] / UP * scale
    font = templates.fonts[styles.style_of(base["ch"])]
    xh = font["OS/2"].sxHeight / font["head"].unitsPerEm * em
    y0, y1, x0, x1 = mark["bbox"]
    dx = ((x0 + x1) - (base["bbox"][2] + base["bbox"][3])) / 2 / xh
    return dict(base=base["ch"], dx=round(float(dx), 3), gap=round(float((base["bbox"][0] - y1) / xh), 3))


def part(u, m, ch):
    """The part of a unit under mask m as a unit of its own, its coverage cut to the part's ink plus PART_PX"""
    win = ndi.binary_dilation(m, iterations=PART_PX)
    ys, xs = np.nonzero(win)
    a0, a1, b0, b1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = np.nonzero(m)
    by0, _, bx0, _ = u["box"]
    bbox = (by0 + my.min(), by0 + my.max() + 1, bx0 + mx.min(), bx0 + mx.max() + 1)
    out = {k: v for k, v in u.items() if k not in ("parts", "ncov", "nbox", "anchor")}
    out.update(cov=(u["cov"] * win)[a0:a1, b0:b1].astype(np.float32), mask=m[a0:a1, b0:b1], box=(by0 + a0, by0 + a1, bx0 + b0, bx0 + b1),
               bbox=bbox, cy=(bbox[0] + bbox[1]) / 2, cx=(bbox[2] + bbox[3]) / 2, ch=ch, part_of=u["ch"])
    return out


def split_dotted(kept, templates, scale):
    """Each labelled i and j split into its stem, keyed as the dotless letter, and its dot, keyed as the dot accent with its anchor

    The parts are kept beside the whole letter and carry part_of, so the letter still stacks whole while its stem and
    dot feed the stacks most accented letters are composed from. A part's coverage cannot be read from the native
    pixels of a turned scan, so parts take the upright coverage
    """
    out = []
    for u in kept:
        ch = styles.char_of(u["ch"])
        if ch not in DOTLESS or len(u.get("parts", [])) < 2:
            continue
        face = styles.style_of(u["ch"])
        stem = max(u["parts"], key=lambda m: m.sum())
        s = part(u, stem, styles.key(face, DOTLESS[ch]))
        out.append(s)
        for m in u["parts"]:
            if m is not stem:
                d = part(u, m, styles.key(face, MARKS["\u0307"]))
                out.append(dict(d, anchor=anchor(d, s, templates, scale)))
    return out


def in_context_out(t):
    """The templates without the accents and dotless letters, which are labelled only where a letter carries a mark"""
    skip = set(MARKS.values()) | set(DOTLESS.values())
    return {k: v for k, v in t.items() if styles.char_of(k) not in skip}


def label_marks(units, kept, t, scale):
    """Labels units left unlabelled that sit just over a letter as accents, against the accent templates alone

    A printed accent can differ from the measuring stick's more than a letter does, as a capital's acute is lower and
    a circumflex narrower, so these templates take sizes within mark_size_tolerance and overlaps from mark_min_iou.
    The letter under the mark, a unit at least 0.3 em tall within half an em, keeps a speck of texture or a stray mark
    from taking an accent. A letter under a mark that is itself unlabelled is tried against the dotless templates,
    so the stem under an ï takes ı
    """
    cfg = target.get()["templates"]
    accents, dotless = set(MARKS.values()), set(DOTLESS.values())
    marks = {k: v for k, v in t.items() if styles.char_of(k) in accents}
    bare = {k: v for k, v in t.items() if styles.char_of(k) in dotless}
    done = {tuple(k["bbox"]) for k in kept}
    em = cfg["px_per_em"] / UP * scale
    under = lambda u: [k for k in units if k["bbox"][1] - k["bbox"][0] >= 0.3 * em and 0 <= k["bbox"][0] - u["bbox"][1] < 0.5 * em
                       and k["bbox"][2] < u["bbox"][3] and k["bbox"][3] > u["bbox"][2]]
    todo = [u for u in units if tuple(u["bbox"]) not in done and u["bbox"][1] - u["bbox"][0] < 0.35 * em and under(u)]
    out = []
    for u, c in zip(todo, pmap(lambda u: classify(u["cov"], marks, top=4, tol=cfg["mark_size_tolerance"]), todo)):
        if not c or c[0][0] < cfg["mark_min_iou"]:
            continue
        rival = next((r for r in c[1:] if styles.char_of(r[1]) != styles.char_of(c[0][1])), None)
        margin = c[0][0] - (rival[0] if rival else 0)
        if margin < cfg["min_margin"]:
            continue
        out.append(dict(u, ch=c[0][1], iou=c[0][0], margin=margin, scale=scale))
        base = min(under(u), key=lambda k: k["bbox"][0] - u["bbox"][1])
        if tuple(base["bbox"]) not in done and bare:
            b = classify(base["cov"], bare, top=1)
            if b and b[0][0] >= cfg["min_iou"]:
                out.append(dict(base, ch=b[0][1], iou=b[0][0], margin=b[0][0], scale=scale))
                done.add(tuple(base["bbox"]))
    return out


def anchor_marks(kept, templates, scale):
    """Gives each unit labelled as an accent the base letter under it, that letter's face, and its anchor in x-heights of that face

    The base is the nearest labelled letter whose top lies under the mark within half an em and that overlaps it across.
    A mark over a capital takes its accent's capital form. The dots split from i and j already carry their anchors
    """
    accents = set(MARKS.values())
    em = target.get()["templates"]["px_per_em"] / UP * scale
    out = []
    for u in kept:
        if styles.char_of(u["ch"]) not in accents or "anchor" in u:
            out.append(u)
            continue
        y0, y1, x0, x1 = u["bbox"]
        under = [k for k in kept if styles.char_of(k["ch"]) not in accents and 0 <= k["bbox"][0] - y1 < 0.5 * em and k["bbox"][2] < x1 and k["bbox"][3] > x0]
        if not under:
            out.append(u)
            continue
        base = min(under, key=lambda k: k["bbox"][0] - y1)
        # accents look alike in both faces, so a mark takes the face of the letter it sits on, and over a capital its capital form
        acc = styles.char_of(u["ch"])
        ch = styles.key(styles.style_of(base["ch"]), styles.CASE[acc] if styles.char_of(base["ch"]).isupper() else acc)
        out.append(dict(u, ch=ch, anchor=anchor(u, base, templates, scale)))
    return out


def label(units, t, scale):
    """Labels units against one template set at the given scale, keeping those that clear min_iou and min_margin

    Two characters of the profile's twins, alike in shape and told apart by where they print, such as „ and ”, are
    not a tie of characters: the unit takes the low one of the pair when it sits on its line's baseline
    """
    cfg = target.get()["templates"]
    min_iou, min_margin = cfg["min_iou"], cfg["min_margin"]
    twins = cfg.get("twins", [])
    t = in_context_out(t)
    kept, tied = [], []
    for u, c in zip(units, pmap(lambda u: classify(u["cov"], t, top=5), units)):
        if not c or c[0][0] < min_iou:
            continue
        # a runner-up that is the same character in the other face is a tie of faces, not of characters
        rival = next((r for r in c[1:] if styles.char_of(r[1]) != styles.char_of(c[0][1])), None)
        margin = c[0][0] - (rival[0] if rival else 0)
        if margin < min_margin:
            pair = next((p for p in twins if rival and {styles.char_of(c[0][1]), styles.char_of(rival[1])} == set(p)), None)
            ch = pair and place_twin(u, units, scale, pair)
            if not ch:
                continue
            c = [r for r in c if styles.char_of(r[1]) == ch] + [r for r in c if styles.char_of(r[1]) not in pair]
            rival = next((r for r in c[1:] if styles.char_of(r[1]) != ch), None)
            margin = c[0][0] - (rival[0] if rival else 0)
            if c[0][0] < min_iou or margin < min_margin:
                continue
        same = [r for r in c[1:] if styles.char_of(r[1]) == styles.char_of(c[0][1]) and c[0][0] - r[0] < min_margin]
        unit = dict(u, ch=c[0][1], iou=c[0][0], margin=margin, scale=scale)
        (tied if same else kept).append(unit)
    return kept + resolve_faces(tied, kept)


def place_twin(u, units, scale, pair):
    """The low or the high character of a twin pair, from where the unit sits against the baseline of the letters beside it

    The baseline is the median foot of the units within six ems along the line that are at least 0.3 em tall. The unit is
    the low twin when its centre lies below 0.12 em over that baseline, None when it has no letters beside it
    """
    em = target.get()["templates"]["px_per_em"] / UP * scale
    y0, y1, x0, x1 = u["bbox"]
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    near = [o["bbox"][1] for o in units if o is not u and abs((o["bbox"][2] + o["bbox"][3]) / 2 - cx) < 6 * em
            and abs((o["bbox"][0] + o["bbox"][1]) / 2 - cy) < 0.45 * em and o["bbox"][1] - o["bbox"][0] >= 0.3 * em]
    if not near:
        return None
    return pair[0] if cy > np.median(near) - 0.12 * em else pair[1]


def resolve_faces(tied, kept):
    """Gives units whose character is clear but whose face is not (a full stop looks the same in both) the face of the nearest unit on their line

    A unit is on the line when its height overlaps the tied unit's widened by 0.3 em either way, so a mark that sits low
    or high, such as an underscore, still finds the letters beside it, and its centre is within 0.6 em of the tied
    unit's, so the ascender of a letter on the next line, which reaches into that band, is not taken for a neighbour.
    Units with no face-certain unit on their line take the primary face
    """
    out = []
    for u in tied:
        em = target.get()["templates"]["px_per_em"] / UP * u.get("scale", 1.0)
        y0, y1 = u["bbox"][0] - 0.3 * em, u["bbox"][1] + 0.3 * em
        mid = lambda k: (k["bbox"][0] + k["bbox"][1]) / 2
        near = [k for k in kept if k["bbox"][0] < y1 and k["bbox"][1] > y0 and abs(mid(k) - mid(u)) <= 0.6 * em]
        face = styles.style_of(min(near, key=lambda k: abs(k["cx"] - u["cx"]))["ch"]) if near else styles.primary()
        out.append(dict(u, ch=styles.key(face, styles.char_of(u["ch"]))))
    return out
