"""Places every printed character of a card's rules text box from the card's known text, as ground truth for scoring

The text comes as paragraphs from the cards file, in the scan list's Card text format, which gives no printed line
breaks and no italics.
Flavor text, reminder text in parentheses and an ability word before a dash are read as italic, text between underscores
takes the other face, and mana symbols such as {T} are kept as gaps that are not placed. A frame that splits the text
gives several boxes, which are fitted as one in reading order. The box is turned level by its text skew, since a card can sit a degree
askew in a scan the pipeline does not turn. Each printed line is found from the rows of ink, words are given to lines
by matching the line's ink width, and the glyphs of each line are placed by a Viterbi pass that maximises the summed
normalised cross correlation of each rendered glyph with a darkness image while letting each advance, kerned as the
font kerns it, stretch by a few pixels. The print scale is then searched with the line breaks fixed, and at that scale a line that
still fits poorly is fitted again over a wider range of baselines with each glyph free to sit further off it and to
take an exponent form, smaller and raised, as a formula's minus signs and exponents are printed. On a line that fits well, a glyph that
correlates poorly is then moved a little where it correlates well, since the print spaces some glyphs further apart. Every box is turned back into the card crop as a quad
"""
import hashlib
import inspect
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import cards
from . import font as fnt
from . import target
from .scan import text_skew

# the rules text box as y0, y1, x0, x1 fractions of the card's frame, for the standard frames
BOX = (0.62, 0.946, 0.05, 0.975)
SUPERSAMPLE = 4
SLACK_PX = 10
JITTER_PX = 3
# a mana symbol's width as a share of the em, for line widths only
SYMBOL_EM = 0.8
# ink further than this share of the em past the last word is not part of the line, such as a power and toughness box
GAP_EM = 0.8
# a printed line above the text whose best fit to the first words averages less correlation than this per glyph
# holds other ink
SKIP_NCC = 0.3
# a line's last word whose glyphs average less correlation than this was placed on the frame past the line's end
LAST_WORD_NCC = 0.35
# a glyph set as an exponent, such as the 90° of Niv-Mizzet's formula, prints at this share of its size and raised this
# share of the em, measured on that card
SMALL = 0.83
RAISE_EM = 0.17
# a line whose glyphs average less correlation than this is fitted again with its baseline searched this share of the
# em either way, since a line with few x-height letters, such as a formula, can take its baseline from the wrong zone,
# and with each glyph free to take its exponent form
WIDE_NCC = 0.5
WIDE_EM = 0.3
# and each glyph of that line may sit this share of the em off its baseline, as a formula's minus is printed below the
# measuring stick's
WIDE_JITTER_EM = 0.25
# a glyph that correlates less than this on its line is searched for this share of the em either way and moved where it
# reaches it, as the print sets some glyphs further from their neighbours than the measuring stick spaces them, so long as
# it lands on no neighbour's box
REFINE_NCC = 0.5
REFINE_EM = 0.3
REFINE_OVERLAP = 0.3
# a die roll's result before its dash, which is rules text rather than an ability word
DIE_RESULT = re.compile(r"\d+(?:\s*(?:–|-|or)\s*\d+)*\+? — ")
# the rows of blank paper left between a card's text boxes when they are fitted as one, in ems at the largest print scale
BOX_GAP_EM = 1.5


def paragraphs(text):
    """A cards file text as paragraphs, each a list of (character, face) with mana symbols as ("{X}", "symbol")

    Faces are regular or italic. All flavor text is italic, as are parentheses with what they hold and an ability word
    before a spaced dash at the start of a rules paragraph, though not a modal spell's Choose one or a die roll's
    result. Text the cards file sets between underscores takes the other face, as a ship's name in flavor text is roman
    """
    out = []
    for raw, flavor in cards.paragraphs(text):
        marks = [on for _, on in cards.marked(raw)]
        line = cards.printed(raw)
        if not line.strip():
            continue
        chars, depth = [], 0
        ability = re.match(r"^([^(){}—]+ — )", line) if not flavor else None
        # a modal spell's Choose one is rules text, not an ability word
        if ability and ability.group(1).startswith("Choose "):
            ability = None
        # so is a die roll's result, such as 1 or 2 or 10–19
        if ability and DIE_RESULT.fullmatch(ability.group(1)):
            ability = None
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
            italic = (flavor or depth > 0 or (ability is not None and i < ability.end())) != marks[i]
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

    def __call__(self, ch, face, small=False):
        """A glyph's image, offset and advance, in its exponent form raised and smaller when small"""
        key = (ch, face, small)
        if key not in self.cache:
            self.cache[key] = self._render(ch, face, small)
        return self.cache[key]

    def _render(self, ch, face, small=False):
        if face == "symbol":
            return dict(img=None, off=(0, 0), adv=SYMBOL_EM * self.px_em)
        font = self.fonts[face]
        sx = self.px_em * (SMALL if small else 1) / font["head"].unitsPerEm
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
        raise_px = int(round(RAISE_EM * self.px_em)) if small else 0
        return dict(img=img, off=(int(lo[0]), int(lo[1]) - raise_px), adv=adv)

    def space(self):
        return self(" ", "regular")["adv"]

    def kern(self, a, b):
        """The font's kerning in px between two characters of one face, 0 across faces or for a mana symbol"""
        (ca, fa), (cb, fb) = a, b
        if fa != fb or fa == "symbol" or ca == " " or cb == " ":
            return 0.0
        key = ("kern", fa)
        if key not in self.cache:
            self.cache[key] = fnt.Kerning(self.fonts[fa])
        cmap = self.fonts[fa].getBestCmap()
        if ord(ca) not in cmap or ord(cb) not in cmap:
            return 0.0
        return self.cache[key].pair(ca, cb) * self.px_em / self.fonts[fa]["head"].unitsPerEm


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


def fit_run(run, scores, glyphs, b, xmin, exponents=False, jitter=JITTER_PX):
    """Viterbi placement of one run at baseline row b, with exponents each glyph in its own form or its exponent form, returns (score, pen positions, forms)"""
    W = scores.D.shape[1]
    NEG = np.float32(-1e9)
    forms = (False, True) if exponents else (False,)
    score, back, prev = None, [], None
    for ch, face in run:
        em = []
        for small in forms:
            t = glyphs(ch, face, small)
            R = scores[(ch, face, small)]
            s = np.zeros(W, np.float32)
            if R is not None:
                ox, oy = t["off"]
                rows = [r for r in range(b + oy - jitter, b + oy + jitter + 1) if 0 <= r < R.shape[0]]
                s = np.full(W, NEG, np.float32)
                if rows:
                    best = R[rows].max(0)
                    xs = np.arange(W) + ox
                    ok = (xs >= 0) & (xs < len(best))
                    s[ok] = best[xs[ok]]
            em.append(s)
        if score is None:
            score = [s.copy() for s in em]
            for s in score:
                s[: max(xmin, 0)] = NEG
            back.append(None)
        else:
            new, links = [], []
            for f in range(len(forms)):
                best = np.full(W, NEG, np.float32)
                arg = np.zeros(W, np.int32)
                came = np.zeros(W, np.int8)
                for pf, psmall in enumerate(forms):
                    # kerning is the font's between glyphs of one size
                    adv = glyphs(*prev, psmall)["adv"] + (glyphs.kern(prev, (ch, face)) if not (psmall or forms[f]) else 0.0)
                    step = int(round(adv))
                    for d in range(-SLACK_PX, SLACK_PX + 1):
                        sh = step + d
                        if sh <= 0 or sh >= W:
                            continue
                        cand = np.full(W, NEG, np.float32)
                        cand[sh:] = score[pf][: W - sh] - 0.004 * abs(d)
                        better = cand > best
                        best[better] = cand[better]
                        arg[better] = sh
                        came[better] = pf
                new.append(best + em[f])
                links.append((arg, came))
            score = new
            back.append(links)
        prev = (ch, face)
    f = int(np.argmax([s.max() for s in score]))
    end = int(np.argmax(score[f]))
    total = float(score[f][end])
    pos, fs = [end], [f]
    for links in reversed(back[1:]):
        arg, came = links[fs[-1]]
        x = pos[-1]
        pos.append(x - int(arg[x]))
        fs.append(int(came[x]))
    return total, pos[::-1], [forms[k] for k in fs[::-1]]


def fit_line(chars, scores, glyphs, b0, search=8, exponents=False, jitter=JITTER_PX):
    """Places a line's characters with its baseline searched within `search` rows of b0, each within `jitter` rows of it, with exponents letting each take its exponent form

    Returns (score, baseline, [(character, face, pen x, exponent form)], right edge of the line's last ink or symbol)
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
            s, pos, small = fit_run(r, scores, glyphs, b, xmin, exponents, jitter)
            total += s
            placed += [(ch, face, p, sm) for (ch, face), p, sm in zip(r, pos, small)]
            right = pos[-1] + glyphs(*r[-1], small[-1])["adv"]
            xmin = int(right) + 10
        if gaps and gaps[-1] == len(rs):
            right += glyphs("{}", "symbol")["adv"]
        if best is None or total > best[0]:
            best = (total, b, placed, right)
    return best


def refine(found, reach, maps):
    """Moves each placed glyph that correlates less than REFINE_NCC on a line that fits well to the best place within reach px across and its jitter rows down or up, where that reaches REFINE_NCC and lands on no neighbour's box

    A neighbour that a moved glyph then lands on is searched the same way, as a full stop the line fit tucked into the
    letter before it. maps gives a placed glyph's correlation map
    """
    box = lambda f: (f["x"] + f["ink"][0], f["x"] + f["ink"][1], f["y"] + f["ink"][2], f["y"] + f["ink"][3])

    def overlap(a, b):
        w = min(a[1], b[1]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[2], b[2])
        return max(w, 0) * max(h, 0) / ((a[1] - a[0]) * (a[3] - a[2]))

    def move(line, k):
        """Moves glyph k of a line to its best clear place reaching REFINE_NCC, returns whether it moved"""
        f = line[k]
        R, j = maps(f), max(f["jitter"], JITTER_PX)
        y0, x0 = max(f["y"] - j, 0), max(f["x"] - reach, 0)
        win = R[y0 : f["y"] + j + 1, x0 : f["x"] + reach + 1]
        for flat in np.argsort(win, axis=None)[::-1][:50] if win.size else []:
            yy, xx = np.unravel_index(flat, win.shape)
            if win[yy, xx] < REFINE_NCC:
                break
            cand = dict(f, x=x0 + int(xx), y=y0 + int(yy))
            if all(overlap(box(cand), box(o)) <= REFINE_OVERLAP for o in line[max(k - 2, 0) : k + 3] if o is not f):
                f.update(x=cand["x"], y=cand["y"], ncc=float(win[yy, xx]))
                return True
        return False

    lines = {}
    for f in found:
        lines.setdefault(f["li"], []).append(f)
    for line in lines.values():
        if np.mean([f["ncc"] for f in line]) < REFINE_NCC:
            continue
        moved = [k for k, f in enumerate(line) if f["ncc"] < REFINE_NCC and move(line, k)]
        for k in moved:
            for n in (k - 1, k + 1):
                if 0 <= n < len(line) and n not in moved and overlap(box(line[n]), box(line[k])) > REFINE_OVERLAP:
                    move(line, n)


def placement_ncc(scores, glyphs, ch, face, p, b, small=False, jitter=JITTER_PX):
    """The correlation of one placed glyph at pen x p on baseline b, the best within the baseline jitter, None without an image"""
    t = glyphs(ch, face, small)
    R = scores[(ch, face, small)]
    if R is None:
        return None
    ox, oy = t["off"]
    x = min(max(p + ox, 0), R.shape[1] - 1)
    rows = [r for r in range(b + oy - jitter, b + oy + jitter + 1) if 0 <= r < R.shape[0]]
    return float(R[rows, x].max()) if rows else None


def best_row(R, rows, x):
    """The row among `rows` where a glyph's map at column x correlates best, with that correlation"""
    k = int(np.argmax(R[list(rows), x]))
    return rows[k], float(R[rows[k], x])


# the functions each step of a fit runs, whose source with the font module's and every constant of this module is
# hashed into the step's key, so a change to one step's code computes that step again and reuses the others
STEP_CODE = {"scale": ("estimate_scale",), "line": ("fit_line", "fit_run", "runs"), "ncc": ("placement_ncc",),
             "place": ("best_row",), "refine": ("refine",)}
_digests = {}


def step_digest(step):
    if step not in _digests:
        mod = globals()
        h = hashlib.sha256(inspect.getsource(fnt).encode())
        for name in STEP_CODE[step] + ("Glyphs", "Scores", "darkness"):
            h.update(inspect.getsource(mod[name]).encode())
        h.update(repr(sorted((k, repr(v)) for k, v in mod.items() if k.isupper())).encode())
        _digests[step] = h.hexdigest()
    return _digests[step]


class Memo:
    """The results of each step of one scan's fit, kept in a file between runs so a refit computes only what changed

    A result is keyed by its step, the digest of the step's code and its arguments, and the file holds results for one
    darkness image and one pair of fonts, so a changed box or scan starts it afresh. Only the results a fit used are
    saved, which keeps the file from growing over runs. Without a path it remembers results for one fit only
    """

    def __init__(self, path=None, fonts=""):
        self.path, self.fonts = (Path(path) if path else None), fonts
        self.image, self.saved, self.used, self.hits, self.misses = None, {}, {}, 0, 0
        self.loaded = {}
        if self.path and self.path.exists():
            try:
                self.loaded = pickle.loads(self.path.read_bytes())
            except Exception:
                self.loaded = {}

    def bind(self, D):
        """Ties the memo to the darkness image the fit reads, keeping saved results only when they came from the same one"""
        self.image = hashlib.sha256(np.ascontiguousarray(D).tobytes() + repr(D.shape).encode()).hexdigest()
        ok = self.loaded.get("image") == self.image and self.loaded.get("fonts") == self.fonts
        self.saved = self.loaded.get("steps", {}) if ok else {}

    def __call__(self, step, args, compute):
        key = (step, step_digest(step), args)
        if key in self.used:
            self.hits += 1
        elif key in self.saved:
            self.used[key] = self.saved[key]
            self.hits += 1
        else:
            self.used[key] = compute()
            self.misses += 1
        return self.used[key]

    def save(self):
        if self.path is None or self.image is None or (not self.misses and len(self.used) == len(self.saved)):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(pickle.dumps(dict(image=self.image, fonts=self.fonts, steps=self.used), pickle.HIGHEST_PROTOCOL))
        tmp.replace(self.path)


class Steps:
    """The steps of a fit over one darkness image, each kept in a Memo, with the glyphs and their maps at every scale used

    A correlation map is only computed when a step that needs it is not in the memo
    """

    def __init__(self, D, fonts, memo=None):
        self.D, self.fonts, self.memo = D, fonts, memo or Memo()
        self.memo.bind(D)
        self._glyphs, self._scores = {}, {}

    def glyphs(self, s):
        s = float(s)
        if s not in self._glyphs:
            self._glyphs[s] = Glyphs(self.fonts, s)
        return self._glyphs[s]

    def scores(self, s):
        s = float(s)
        if s not in self._scores:
            self._scores[s] = Scores(self.D, self.glyphs(s))
        return self._scores[s]

    def scale(self):
        return self.memo("scale", (), lambda: estimate_scale(self.D, self.fonts))

    def line(self, chars, s, b0, search=8, exponents=False, jitter=JITTER_PX):
        chars = tuple(chars)
        return self.memo("line", (float(s), chars, int(b0), int(search), bool(exponents), int(jitter)),
                         lambda: fit_line(list(chars), self.scores(s), self.glyphs(s), b0, search, exponents, jitter))

    def ncc(self, s, ch, face, p, b, small=False, jitter=JITTER_PX):
        return self.memo("ncc", (float(s), ch, face, int(p), int(b), bool(small), int(jitter)),
                         lambda: placement_ncc(self.scores(s), self.glyphs(s), ch, face, p, b, small, jitter))

    def place(self, s, ch, face, small, x, rows):
        rows = tuple(int(r) for r in rows)
        return self.memo("place", (float(s), ch, face, bool(small), int(x), rows), lambda: best_row(self.scores(s)[(ch, face, small)], rows, x))

    def refine(self, found, reach, s):
        """Runs `refine` over the placed glyphs and moves them where it moved them"""
        sig = tuple((f["li"], f["ch"], f["face"], bool(f["small"]), int(f["x"]), int(f["y"]), f["ncc"], int(f["jitter"]), tuple(int(v) for v in f["ink"])) for f in found)

        def run():
            work = [dict(f) for f in found]
            refine(work, reach, lambda f: self.scores(s)[(f["ch"], f["face"], f["small"])])
            return [(f["x"], f["y"], f["ncc"]) for f in work]

        for f, (x, y, v) in zip(found, self.memo("refine", (float(s), int(reach), sig), run)):
            f.update(x=x, y=y, ncc=v)


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
    gap = GAP_EM * glyphs.px_em
    em = glyphs.px_em

    def zones(rows, b0, b1):
        """The x-height zones in a band of rows, more than one where lines run together"""
        if b1 - b0 < 0.2 * em:
            return []
        # the x-height zone runs from a steep rise to a steep fall 0.3 to 0.6 em below it, the pair that is steepest
        # together, and its foot is the baseline. A dash or a hyphen gives a sharper but narrower step, which the
        # spacing rules out. The search reaches a few rows past the band, since a line with no ascenders rises at its edge
        p0, p1 = max(b0 - 6, 0), min(b1 + 6, len(rows))
        d = np.diff(rows[p0:p1])
        lo, hi = int(0.3 * em), int(0.6 * em)
        best = None
        for t in range(len(d)):
            seg = d[t + lo : t + hi + 1]
            if len(seg) == 0:
                break
            k = int(np.argmin(seg))
            v = d[t] - seg[k]
            if best is None or v > best[0]:
                best = (v, t, t + lo + k)
        out = []
        if best is not None:
            top, bottom = p0 + best[1], p0 + best[2] + 1
            out.append((top, bottom))
        else:
            top = bottom = (b0 + b1) // 2
        # a band holds one line's ascenders and descenders, about a quarter em either side, so more means another line
        if top - b0 > 0.6 * em:
            out = zones(rows, b0, top - int(0.3 * em)) + out
        if b1 - bottom > 0.6 * em:
            out = out + zones(rows, bottom + int(0.3 * em), b1)
        return out

    def find(c1):
        """Zones from the row profile of the columns up to c1"""
        rows = np.convolve(ink[:, m0:c1].mean(1), np.ones(5) / 5, "same")
        # any row with a little ink belongs to a line, so a short last line of a paragraph is found too
        on = np.append(rows > 0.05 * np.percentile(rows[rows > 0], 95), False)
        edges = np.flatnonzero(np.diff(np.concatenate([[False], on]).astype(int)))
        bands = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]
        # a band cut by the box's top or bottom edge is the frame or a line the box does not hold
        return [z for b0, b1 in bands if b0 > 0 and b1 < H for z in zones(rows, b0, b1)]

    # lines start at the left margin while a power and toughness box or a set emblem sits further right, and an
    # indented or centred line can start past the left third, so a line counts when either profile finds it
    found = sorted(find(int(0.33 * W)) + find(int(0.75 * W)), key=lambda z: z[1])
    merged = []
    for z in found:
        if merged and abs(z[1] - merged[-1][1]) < 0.3 * em:
            continue
        merged.append(z)

    out = []
    for top, bottom in merged:
        h = bottom - top
        cols = np.nonzero(ink[max(top - h, 0) : bottom + h // 2, m0:m1].any(0))[0] + m0
        if len(cols) == 0:
            continue
        # the line ends where a gap wider than GAP_EM first opens after its first ink
        split = np.nonzero(np.diff(cols) > gap)[0]
        right = cols[split[0]] if len(split) else cols[-1]
        out.append(dict(base=int(bottom), x0=int(cols[0]), x1=int(right) + 1, height=int(h)))
    return out


def assign(pars, lines, steps, s):
    """Gives each printed line its characters, paragraph by paragraph, by width and then by where the fitted ink ends"""
    out, li = [], 0
    glyphs = steps.glyphs(s)
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
            fits = []
            for k in cands:
                # spaces stay in the line so the pen advances over them, they have no image to place
                chars = [c for w in words[start : k + 1] for c in w + [(" ", "regular")]][:-1]
                score, b, placed, right = steps.line(chars, s, line["base"])
                n = sum(glyphs(ch, face)["img"] is not None for ch, face in chars)
                last = [steps.ncc(s, ch, face, p, b, sm) for ch, face, p, sm in placed[-len(words[k]) :]]
                last = [v for v in last if v is not None]
                fits.append((abs(right - line["x1"]), k, chars, score / max(n, 1), np.mean(last) if last else 1.0))
            best = min(fits)
            # a last word whose glyphs land on no ink was pushed onto the frame and belongs to the next line
            while best[4] < LAST_WORD_NCC and best[1] > start:
                shorter = [f for f in fits if f[1] == best[1] - 1]
                if not shorter:
                    break
                best = shorter[0]
            li += 1
            # a line above the text that the first words do not fit at all is the type line or frame the box took in.
            # Lines below the text are never given words, so only these need skipping
            if not out and best[3] < SKIP_NCC:
                continue
            out.append(dict(line, chars=best[2]))
            start = best[1] + 1
    return out


def box_bounds(g, box, frame=None):
    """A box given as a share of the card's frame as y0, y1, x0, x1 in the crop's pixels, clipped to the crop

    The frame defaults to the whole crop and lies partly outside it for a card the scan cut off
    """
    H, W = g.shape
    fy0, fy1, fx0, fx1 = frame or (0, H, 0, W)
    y0, y1 = max(int(fy0 + box[0] * (fy1 - fy0)), 0), min(int(fy0 + box[1] * (fy1 - fy0)), H)
    x0, x1 = max(int(fx0 + box[2] * (fx1 - fx0)), 0), min(int(fx0 + box[3] * (fx1 - fx0)), W)
    return y0, y1, x0, x1


def level(g, box=BOX, frame=None, skew=None):
    """The rules box of an upright card crop turned level, with the map from the level box back to the card crop

    The box is a share of the card's frame as `box_bounds` takes it. The skew is measured on the box unless given
    """
    y0, y1, x0, x1 = box_bounds(g, box, frame)
    crop = g[y0:y1, x0:x1]
    skew = text_skew(crop) if skew is None else skew
    h, w = crop.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    D = cv2.warpAffine(darkness(crop), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    Ds = cv2.warpAffine(darkness(cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 2.5)), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    # the card frame beside the box is nearly as dark as ink all the way down, which even a dark textured box is not,
    # so columns outside the widest run of box columns are cleared before lines are found and glyphs placed
    lv = cv2.warpAffine(crop.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    med = np.median(lv, axis=0)
    ink, paper = np.percentile(lv, 1), np.percentile(med, 90)
    inside = med > ink + 0.25 * (paper - ink)
    runs = np.flatnonzero(np.diff(np.concatenate([[0], inside.astype(int), [0]])))
    if len(runs):
        k = int(np.argmax(runs[1::2] - runs[::2]))
        keep = np.zeros(w, bool)
        keep[runs[2 * k] : runs[2 * k + 1]] = True
        D[:, ~keep], Ds[:, ~keep] = 0, 0
    back = cv2.invertAffineTransform(M)
    back[:, 2] += [x0, y0]
    return D, Ds, back, skew, (y0, y1, x0, x1)


def estimate_scale(D, fonts, lo=0.62, hi=1.1, step=0.02, chars="eaon"):
    """The print scale at which common letters of the regular face correlate best with the box, in steps of `step`

    Each letter's score is the mean of its 40 best correlations over the box, which sit on its printed copies when the
    rendered size matches. Reading the size from line heights instead is thrown off by lines of capitals and by the
    italic face's smaller x-height. The range reaches down to the small print of a card crowded with text, as Comet,
    Stellar Pup is printed at about 0.7
    """
    def score(s):
        gl = Glyphs(fonts, float(s))
        out = []
        for ch in chars:
            t = gl(ch, "regular")
            if t["img"] is None:
                continue
            R = cv2.matchTemplate(D, t["img"], cv2.TM_CCOEFF_NORMED)
            out.append(float(np.mean(np.partition(R.ravel(), -40)[-40:])))
        return np.mean(out)

    return float(max(np.round(np.arange(lo, hi + 1e-9, step), 3), key=score))


def fit(g, text, fonts, box=BOX, scales=None, frame=None, memo=None):
    """Truth for one upright card crop: every placed character with its face, quad and bounds in card crop pixels

    fonts is {"regular": font, "italic": font}, and frame the card's full frame in the crop's pixels as `level` takes it.
    box is one box or a list of them in reading order, for a frame that splits the text, such as a Saga's reminder,
    chapter column and flavor. Every box is turned level by the skew of the largest and they are stacked into one image
    with blank rows between, so the text flows from each box's last line to the next box's first and one print scale
    fits them all. memo keeps each step's result between fits of the same scan, see `Memo`. Returns a dict with the
    scale, the text skew, the bounds of every box and the characters
    """
    boxes = [tuple(box)] if np.ndim(box) == 1 else [tuple(b) for b in box]
    areas = [(y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in (box_bounds(g, b, frame) for b in boxes)]
    big = int(np.argmax(areas))
    skew = level(g, boxes[big], frame)[3]
    parts = [level(g, b, frame, skew) for b in boxes]
    gap = int(BOX_GAP_EM * Glyphs(fonts, 1.1).px_em) if len(parts) > 1 else 0
    W = max(p[0].shape[1] for p in parts)
    offsets = np.cumsum([0] + [p[0].shape[0] + gap for p in parts[:-1]]).tolist()
    D = np.zeros((offsets[-1] + parts[-1][0].shape[0], W), np.float32)
    Ds = np.zeros_like(D)
    for off, (d, ds, *_) in zip(offsets, parts):
        D[off : off + d.shape[0], : d.shape[1]] = d
        Ds[off : off + d.shape[0], : d.shape[1]] = ds
    pars = paragraphs(text)
    steps = Steps(D, fonts, memo)
    s0 = steps.scale()
    glyphs = steps.glyphs(s0)
    # lines are found box by box, so a frame bar along a box's edge is still told apart from text
    found_lines = [dict(l, base=l["base"] + off) for off, p in zip(offsets, parts) for l in text_lines(p[1], glyphs)]
    lines = assign(pars, found_lines, steps, s0)
    best = None
    for s in scales or np.round(np.arange(s0 - 0.03, s0 + 0.0301, 0.01), 3):
        total, placed = 0.0, []
        for li, line in enumerate(lines):
            score, b, pl, _ = steps.line(line["chars"], s, line["base"])
            total += score
            placed.append((li, b, pl, JITTER_PX))
        if best is None or total > best[0]:
            best = (total, float(s), placed)
    total, s, placed = best
    gl = steps.glyphs(s)
    mean_ncc = lambda b, pl, j: np.mean([v for v in (steps.ncc(s, ch, face, p, b, sm, j) for ch, face, p, sm in pl) if v is not None] or [1.0])
    wide = int(WIDE_JITTER_EM * gl.px_em)
    for k, (li, b, pl, j) in enumerate(placed):
        before = mean_ncc(b, pl, j)
        if before < WIDE_NCC:
            _, wb, wpl, _ = steps.line(lines[li]["chars"], s, lines[li]["base"], int(WIDE_EM * gl.px_em), exponents=True, jitter=wide)
            if mean_ncc(wb, wpl, wide) > before:
                placed[k] = (li, wb, wpl, wide)
    # a map's shape is the darkness image less the glyph, so placements are clipped to it without computing the map
    found = []
    for li, b, pl, jitter in placed:
        for ch, face, p, small in pl:
            t = gl(ch, face, small)
            if t["img"] is None:
                continue
            h, w = t["img"].shape
            mh, mw = D.shape[0] - h + 1, D.shape[1] - w + 1
            ox, oy = t["off"]
            x = min(max(p + ox, 0), mw - 1)
            rows = [r for r in range(b + oy - jitter, b + oy + jitter + 1) if 0 <= r < mh] or [min(max(b + oy, 0), mh - 1)]
            ry, v = steps.place(s, ch, face, small, x, rows)
            ys, xs = np.nonzero(t["img"] > 0.5)
            found.append(dict(li=li, ch=ch, face=face, small=small, x=x, y=ry, ncc=v, ink=(xs.min(), xs.max() + 1, ys.min(), ys.max() + 1), jitter=jitter))
    steps.refine(found, int(REFINE_EM * gl.px_em), s)
    steps.memo.save()
    chars = []
    for f in found:
        x0i, x1i, y0i, y1i = f["ink"]
        x, ry = f["x"], f["y"]
        k = max(i for i, off in enumerate(offsets) if off <= lines[f["li"]]["base"])
        back = parts[k][2]
        ry -= offsets[k]
        q = np.array([[x + x0i, ry + y0i], [x + x1i, ry + y0i], [x + x1i, ry + y1i], [x + x0i, ry + y1i]], float)
        q = q @ back[:, :2].T + back[:, 2]
        chars.append(dict(line=f["li"], ch=f["ch"], face=f["face"], ncc=round(f["ncc"], 4), quad=np.round(q, 1).tolist(),
                          x0=int(np.floor(q[:, 0].min())), x1=int(np.ceil(q[:, 0].max())), y0=int(np.floor(q[:, 1].min())), y1=int(np.ceil(q[:, 1].max())))
                     | (dict(small=True) if f["small"] else {}))
    bounds = [[int(v) for v in p[4]] for p in parts]
    union = [min(b[0] for b in bounds), max(b[1] for b in bounds), min(b[2] for b in bounds), max(b[3] for b in bounds)]
    return dict(scale=s, skew_deg=round(float(skew), 3), box=union, boxes=bounds, lines=len(lines), chars=chars)


def overlay(g, truth, path):
    """Draws each character's quad on the card's rules box, blue where its correlation is 0.5 or more and red below

    A card whose text sits in several boxes is drawn over the area holding them all, each box outlined in grey
    """
    y0, y1, x0, x1 = truth["box"]
    im = Image.fromarray(g[y0:y1, x0:x1]).convert("RGB")
    dr = ImageDraw.Draw(im)
    boxes = truth.get("boxes") or []
    for by0, by1, bx0, bx1 in boxes if len(boxes) > 1 else []:
        dr.rectangle([bx0 - x0, by0 - y0, bx1 - x0 - 1, by1 - y0 - 1], outline=(150, 150, 150), width=2)
    for c in truth["chars"]:
        dr.polygon([(px - x0, py - y0) for px, py in c["quad"]], outline=(0, 120, 255) if c["ncc"] >= 0.5 else (230, 0, 0), width=2)
    im.save(path)
