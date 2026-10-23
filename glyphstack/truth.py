"""Places every printed character of a card's rules text box from the card's known text, as ground truth for scoring

The text comes as paragraphs from the scan list's Card text sheet, which gives no printed line breaks and no italics.
Flavor text, reminder text in parentheses and an ability word before a dash are read as italic, and mana symbols such
as {T} are kept as gaps that are not placed. The box is turned level by its text skew, since a card can sit a degree
askew in a scan the pipeline does not turn. Each printed line is found from the rows of ink, words are given to lines
by matching the line's ink width, and the glyphs of each line are placed by a Viterbi pass that maximises the summed
normalised cross correlation of each rendered glyph with a darkness image while letting each advance stretch by a few
pixels. The print scale is then searched with the line breaks fixed. Every box is turned back into the card crop as a quad
"""
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import font as fnt
from . import target
from .scan import text_skew

# the rules text box as y0, y1, x0, x1 fractions of the upright card crop, for the standard frames
BOX = (0.615, 0.93, 0.07, 0.93)
SUPERSAMPLE = 4
SLACK_PX = 10
JITTER_PX = 3
# a mana symbol's width as a share of the em, for line widths only
SYMBOL_EM = 0.8
# ink further than this share of the em past the last word is not part of the line, such as a power and toughness box
GAP_EM = 0.8
FACES = {"r": "regular", "i": "italic"}


def paragraphs(text):
    """The card text as paragraphs, each a list of (character, face) with mana symbols as ("{X}", "symbol")

    Faces are regular or italic. Everything after a Flavor: line is italic, as are parentheses with what they hold and
    an ability word before a spaced dash at the start of a rules paragraph
    """
    out, flavor = [], False
    for line in text.split("\n"):
        if line.strip() in ("Rules:", "Flavor:"):
            flavor = line.strip() == "Flavor:"
            continue
        if not line.strip():
            continue
        chars, depth = [], 0
        ability = re.match(r"^([^(){}—]+ — )", line) if not flavor else None
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "{":
                j = line.index("}", i)
                chars.append((line[i : j + 1], "symbol"))
                i = j + 1
                continue
            if ch == "(":
                depth += 1
            italic = flavor or depth > 0 or (ability is not None and i < ability.end())
            chars.append((ch, "italic" if italic else "regular"))
            if ch == ")":
                depth = max(depth - 1, 0)
            i += 1
        out.append(chars)
    return out


def tokens(par):
    """A paragraph split at spaces into words, each a list of (character, face)"""
    words, cur = [], []
    for ch, face in par:
        if ch == " ":
            if cur:
                words.append(cur)
            cur = []
        else:
            cur.append((ch, face))
    if cur:
        words.append(cur)
    return words


def wrap_end(widths, space, target_w, start, stop):
    """The index of the last word on a line, the one whose line width from `start` comes nearest the ink width"""
    best, best_err = start, None
    w = 0.0
    for k in range(start, stop):
        w += widths[k] + (space if k > start else 0)
        err = abs(w - target_w)
        if best_err is None or err < best_err:
            best, best_err = k, err
    return best


def darkness(g):
    """Ink as 0 to 1 against the local paper level, the grey closing with a disc wider than any stroke"""
    g = g.astype(np.float32)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    return np.clip((bg - g) / np.maximum(bg, 1), 0, 1)


class Glyphs:
    """Anti-aliased glyphs of each face rendered at one print scale, with their offset from the pen and their advance"""

    def __init__(self, fonts, scale):
        t = target.get()["templates"]
        self.fonts, self.scale = fonts, scale
        self.px_em = t["px_per_em"] / 4 * scale
        self.aspect = t["aspect"]
        self.cache = {}

    def __call__(self, ch, face):
        key = (ch, face)
        if key not in self.cache:
            self.cache[key] = self._render(ch, face)
        return self.cache[key]

    def _render(self, ch, face):
        if face == "symbol":
            return dict(img=None, off=(0, 0), adv=SYMBOL_EM * self.px_em)
        font = self.fonts[face]
        sx = self.px_em / font["head"].unitsPerEm
        cmap = font.getBestCmap()
        if ord(ch) not in cmap:
            return dict(img=None, off=(0, 0), adv=0.5 * self.px_em)
        adv = font["hmtx"][cmap[ord(ch)]][0] * sx
        contours, _ = fnt.glyph_contours(font, ch)
        if not contours:
            return dict(img=None, off=(0, 0), adv=adv)
        cs = [np.stack([c[:, 0] * sx, -c[:, 1] * sx * self.aspect], 1) for c in contours]
        allp = np.vstack(cs)
        lo, hi = np.floor(allp.min(0)) - 2, np.ceil(allp.max(0)) + 2
        W, H = int(hi[0] - lo[0]), int(hi[1] - lo[1])
        m = np.zeros((H * SUPERSAMPLE, W * SUPERSAMPLE), np.uint8)
        for p in cs:
            mm = np.zeros_like(m)
            cv2.fillPoly(mm, [np.round((p - lo) * SUPERSAMPLE * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
            m ^= mm
        img = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        return dict(img=img, off=(int(lo[0]), int(lo[1])), adv=adv)

    def space(self):
        return self(" ", "regular")["adv"]


class Scores:
    """Each glyph's normalised cross correlation with the darkness image at every placement, computed once per glyph"""

    def __init__(self, D, glyphs):
        self.D, self.glyphs, self.maps = D, glyphs, {}

    def __getitem__(self, key):
        if key not in self.maps:
            t = self.glyphs(*key)
            self.maps[key] = None if t["img"] is None else cv2.matchTemplate(self.D, t["img"], cv2.TM_CCOEFF_NORMED)
        return self.maps[key]


def runs(chars):
    """A line's characters split at mana symbols into runs that are placed one after another, with each symbol's width"""
    out, cur, gaps = [], [], []
    for ch, face in chars:
        if face == "symbol":
            if cur:
                out.append(cur)
            cur = []
            gaps.append(len(out))
            continue
        cur.append((ch, face))
    if cur:
        out.append(cur)
    return out, gaps


def fit_run(run, scores, glyphs, b, xmin):
    """Viterbi placement of one run at baseline row b, returns (score, pen positions)"""
    W = scores.D.shape[1]
    NEG = np.float32(-1e9)
    score, back, prev_adv = None, [], 0.0
    for ch, face in run:
        t = glyphs(ch, face)
        R = scores[(ch, face)]
        s = np.zeros(W, np.float32)
        if R is not None:
            ox, oy = t["off"]
            rows = [r for r in range(b + oy - JITTER_PX, b + oy + JITTER_PX + 1) if 0 <= r < R.shape[0]]
            s = np.full(W, NEG, np.float32)
            if rows:
                best = R[rows].max(0)
                xs = np.arange(W) + ox
                ok = (xs >= 0) & (xs < len(best))
                s[ok] = best[xs[ok]]
        if score is None:
            score = s.copy()
            score[: max(xmin, 0)] = NEG
            back.append(None)
        else:
            step = int(round(prev_adv))
            best = np.full(W, NEG, np.float32)
            arg = np.zeros(W, np.int32)
            for d in range(-SLACK_PX, SLACK_PX + 1):
                sh = step + d
                if sh <= 0 or sh >= W:
                    continue
                cand = np.full(W, NEG, np.float32)
                cand[sh:] = score[: W - sh] - 0.004 * abs(d)
                better = cand > best
                best[better] = cand[better]
                arg[better] = sh
            score = best + s
            back.append(arg)
        prev_adv = t["adv"]
    end = int(np.argmax(score))
    pos = [end]
    for arg in reversed(back[1:]):
        pos.append(pos[-1] - int(arg[pos[-1]]))
    return float(score[end]), pos[::-1]


def fit_line(chars, scores, glyphs, b0, search=8):
    """Places a line's characters with its baseline searched within `search` rows of b0

    Returns (score, baseline, [(character, face, pen x)], right edge of the line's last ink or symbol)
    """
    best = None
    rs, gaps = runs(chars)
    for b in range(b0 - search, b0 + search + 1):
        if b < 0 or b >= scores.D.shape[0]:
            continue
        xmin, total, placed, right = 0, 0.0, [], 0.0
        for k, r in enumerate(rs):
            if k in gaps and placed:
                xmin = int(right + glyphs("{}", "symbol")["adv"])
            s, pos = fit_run(r, scores, glyphs, b, xmin)
            total += s
            placed += [(ch, face, p) for (ch, face), p in zip(r, pos)]
            right = pos[-1] + glyphs(*r[-1])["adv"]
            xmin = int(right) + 10
        if gaps and gaps[-1] == len(rs):
            right += glyphs("{}", "symbol")["adv"]
        if best is None or total > best[0]:
            best = (total, b, placed, right)
    return best


def text_lines(Ds, glyphs):
    """Each printed line's baseline row and the columns its ink spans, from the rows and columns of ink

    Ds is the darkness of the box blurred enough to average out a halftone screen, so a dark screen does not read as ink
    """
    H, W = Ds.shape
    ink = Ds > 0.5
    # frame bars along the box's edges are mostly ink along their length, text never is
    colfrac, rowfrac = ink.mean(0), ink.mean(1)
    ink = ink & (colfrac < 0.4)[None, :] & (rowfrac < 0.6)[:, None]
    m0, m1 = int(0.03 * W), int(0.97 * W)
    rows = np.convolve(ink[:, m0:m1].mean(1), np.ones(5) / 5, "same")
    # any row with a little ink belongs to a line, so a short last line of a paragraph is found too
    on = rows > 0.05 * np.percentile(rows[rows > 0], 95)
    bands, start = [], None
    for y, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append((start, y))
            start = None
    gap = GAP_EM * glyphs.px_em
    out = []
    for b0, b1 in bands:
        if b1 - b0 < 0.2 * glyphs.px_em:
            continue
        # the x-height zone runs from the band's steepest rise to its steepest fall, and its foot is the baseline
        d = np.diff(rows[b0:b1])
        top = b0 + int(np.argmax(d))
        bottom = top + 1 + int(np.argmin(d[top - b0 :]))
        h = bottom - top
        if h < 0.25 * glyphs.px_em:
            continue
        cols = np.nonzero(ink[max(top - h, 0) : bottom + h // 2, m0:m1].any(0))[0] + m0
        if len(cols) == 0:
            continue
        # the line ends where a gap wider than GAP_EM first opens after its first ink
        split = np.nonzero(np.diff(cols) > gap)[0]
        right = cols[split[0]] if len(split) else cols[-1]
        out.append(dict(base=int(bottom), x0=int(cols[0]), x1=int(right) + 1, height=int(h)))
    return out


def assign(pars, lines, glyphs, scores):
    """Gives each printed line its characters, paragraph by paragraph, by width and then by where the fitted ink ends"""
    out, li = [], 0
    space = glyphs.space()
    for par in pars:
        words = tokens(par)
        widths = [sum(glyphs(ch, face)["adv"] for ch, face in w) for w in words]
        start = 0
        while start < len(words):
            if li >= len(lines):
                raise ValueError("the text runs past the last printed line found")
            line = lines[li]
            k0 = wrap_end(widths, space, line["x1"] - line["x0"], start, len(words))
            cands = sorted({k for k in (k0 - 1, k0, k0 + 1) if start <= k < len(words)})
            best = None
            for k in cands:
                # spaces stay in the line so the pen advances over them, they have no image to place
                chars = [c for w in words[start : k + 1] for c in w + [(" ", "regular")]][:-1]
                score, b, placed, right = fit_line(chars, scores, glyphs, line["base"])
                err = abs(right - line["x1"])
                if best is None or err < best[0]:
                    best = (err, k, chars)
            out.append(dict(line, chars=best[2]))
            start = best[1] + 1
            li += 1
    return out


def level(g, box=BOX):
    """The rules box of an upright card crop turned level, with the map from the level box back to the card crop"""
    H, W = g.shape
    y0, y1, x0, x1 = int(box[0] * H), int(box[1] * H), int(box[2] * W), int(box[3] * W)
    crop = g[y0:y1, x0:x1]
    skew = text_skew(crop)
    h, w = crop.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    D = cv2.warpAffine(darkness(crop), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    Ds = cv2.warpAffine(darkness(cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 2.5)), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    back = cv2.invertAffineTransform(M)
    back[:, 2] += [x0, y0]
    return D, Ds, back, skew, (y0, y1, x0, x1)


def estimate_scale(lines):
    """Print scale from the height of the lines' x-height bands, against the regular face's x-height of 0.45 em"""
    t = target.get()["templates"]
    return float(np.median([l["height"] for l in lines])) / (0.45 * t["px_per_em"] / 4 * t["aspect"])


def fit(g, text, fonts, box=BOX, scales=None):
    """Truth for one upright card crop: every placed character with its face, quad and bounds in card crop pixels

    fonts is {"regular": font, "italic": font}. Returns a dict with the scale, the text skew, the box and the characters
    """
    D, Ds, back, skew, bounds = level(g, box)
    pars = paragraphs(text)
    s0 = estimate_scale(text_lines(Ds, Glyphs(fonts, 0.9)))
    glyphs = Glyphs(fonts, s0)
    lines = assign(pars, text_lines(Ds, glyphs), glyphs, Scores(D, glyphs))
    best = None
    for s in scales or np.round(np.arange(s0 - 0.06, s0 + 0.0601, 0.01), 3):
        gl = Glyphs(fonts, float(s))
        sc = Scores(D, gl)
        total, placed = 0.0, []
        for li, line in enumerate(lines):
            score, b, pl, _ = fit_line(line["chars"], sc, gl, line["base"])
            total += score
            placed.append((li, b, pl))
        if best is None or total > best[0]:
            best = (total, float(s), placed, gl, sc)
    total, s, placed, gl, sc = best
    chars = []
    for li, b, pl in placed:
        for ch, face, p in pl:
            t = gl(ch, face)
            if t["img"] is None:
                continue
            R = sc[(ch, face)]
            ox, oy = t["off"]
            x = min(max(p + ox, 0), R.shape[1] - 1)
            rows = [r for r in range(b + oy - JITTER_PX, b + oy + JITTER_PX + 1) if 0 <= r < R.shape[0]] or [min(max(b + oy, 0), R.shape[0] - 1)]
            ry = rows[int(np.argmax(R[rows, x]))]
            ys, xs = np.nonzero(t["img"] > 0.5)
            q = np.array([[x + xs.min(), ry + ys.min()], [x + xs.max() + 1, ry + ys.min()], [x + xs.max() + 1, ry + ys.max() + 1], [x + xs.min(), ry + ys.max() + 1]], float)
            q = q @ back[:, :2].T + back[:, 2]
            chars.append(dict(line=li, ch=ch, face=face, ncc=round(float(R[ry, x]), 4), quad=np.round(q, 1).tolist(),
                              x0=int(np.floor(q[:, 0].min())), x1=int(np.ceil(q[:, 0].max())), y0=int(np.floor(q[:, 1].min())), y1=int(np.ceil(q[:, 1].max()))))
    return dict(scale=s, skew_deg=round(float(skew), 3), box=[int(v) for v in bounds], lines=len(lines), chars=chars)


def overlay(g, truth, path):
    """Draws each character's quad on the card's rules box, blue where its correlation is 0.5 or more and red below"""
    y0, y1, x0, x1 = truth["box"]
    im = Image.fromarray(g[y0:y1, x0:x1]).convert("RGB")
    dr = ImageDraw.Draw(im)
    for c in truth["chars"]:
        dr.polygon([(px - x0, py - y0) for px, py in c["quad"]], outline=(0, 120, 255) if c["ncc"] >= 0.5 else (230, 0, 0), width=2)
    im.save(path)
