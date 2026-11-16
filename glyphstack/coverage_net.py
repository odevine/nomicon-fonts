"""Reads a glyph's coverage over a strong halftone screen with a small network trained on composites of a forward model

    python -m glyphstack.coverage_net train SESSION CARDS -o WEIGHTS.npz [--texture-card CARD ...]

Black text is overprinted on the text box's screen. `print_over` is the forward model of how a glyph and the screen
combine on the card and in the scan: the ink lets a share 1 - ALPHA of the screen's density through, its edge moves out
over the screen's dots and in over the gaps between them by GAMMA times the dots' local contrast, which leaves its
weight as it was, and the scanner blurs ink and screen together by SIGMA px. Glyphs read from plain background are
printed by it onto bare strong screen, and the network learns to take the composite back to the glyph's plain coverage.

The network is five 3 by 3 convolutions of 32 channels, dilated 1, 1, 2, 2 and 4, and a 1 by 1 output through a
sigmoid, a receptive field of 21 px, under half a letter. Its input is a unit's raw grey window over the texture level
around the glyph, with that level as a second channel. `train` builds the pairs from a session's cached units, its
textures from every card with bare strong screen in the span of its text, or from the cards given, and its sources from
every unit on plain background of the other cards, and needs torch, which nothing else does. The weights come from card scans and stay outside git, and a
session reads them with --coverage-model. Inference runs in numpy, so extraction loads neither torch nor onnxruntime.
Each unit whose centre has screen contrast inside the profile's texture `coverage_contrast` takes the network's
coverage in proportion to where it sits in that range, and a unit on plain background keeps its coverage
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import cards, target, texture, units
from .log import add_args, announce, log, pool, start, timed
from .scan import load_gray

# the forward model fitted to Dismal Backwater's glyphs on both sides of its gradient
ALPHA, SIGMA, GAMMA = 0.1, 0.5, 1.25
DILATIONS = (1, 1, 2, 2, 4)
CHANNELS = 32
# padding around a unit's box that the network reads, at least its receptive field's reach
PAD = 8
# a unit window narrower than this pads to it for training crops
CROP = 48

_weights = None


def use(path):
    """Loads the network's weights for every later extraction in this process, None for none"""
    global _weights
    _weights = None if path is None else dict(np.load(path))


def active():
    return _weights is not None


def digest(path):
    """sha256 of a weights file, which the extraction cache is keyed on"""
    return None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalise(win, mask):
    """The grey over the texture level, the median of the window away from the glyph, and that level as a second channel"""
    win = win.astype(np.float32)
    away = ~ndi.binary_dilation(mask, iterations=6)
    t = float(np.median(win[away])) if away.sum() > 20 else float(np.median(win))
    return np.stack([win / max(t, 1), np.full_like(win, t / 255)])


def _conv(x, w, b, d):
    c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (d, d), (d, d)))
    cols = np.stack([xp[:, i * d : i * d + h, j * d : j * d + wd] for i in range(3) for j in range(3)], 1)
    return np.tensordot(w.reshape(w.shape[0], c * 9), cols.reshape(c * 9, h * wd), 1).reshape(-1, h, wd) + b[:, None, None]


def forward(x, W=None):
    """The network's coverage for a normalised (2, h, w) window"""
    W = W or _weights
    for k, d in enumerate(DILATIONS):
        x = np.maximum(_conv(x, W[f"conv{k}_w"], W[f"conv{k}_b"], d), 0)
    x = np.tensordot(W["out_w"][:, :, 0, 0], x, 1) + W["out_b"][:, None, None]
    return (1 / (1 + np.exp(-x[0]))).astype(np.float32)


def reader(raw):
    """A function of a unit's box and own mask giving the network's coverage over the box and its weight, or None on plain background

    raw is the card crop before the profile's texture preparation, the grey the network was trained on
    """
    lo, hi = target.get().settings.get("texture", {})["coverage_contrast"]
    rc = texture.screen_contrast(raw)

    def read(box, mask):
        y0, y1, x0, x1 = box
        cy, cx = min((y0 + y1) // 2, raw.shape[0] - 1), min((x0 + x1) // 2, raw.shape[1] - 1)
        w = float(np.clip((rc[cy, cx] - lo) / (hi - lo), 0, 1))
        if w == 0:
            return None
        Y0, X0, Y1, X1 = max(y0 - PAD, 0), max(x0 - PAD, 0), min(y1 + PAD, raw.shape[0]), min(x1 + PAD, raw.shape[1])
        m = np.zeros((Y1 - Y0, X1 - X0), bool)
        m[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0] = mask
        return forward(normalise(raw[Y0:Y1, X0:X1], m))[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0], w

    return read


def _wiener(img, sigma, k=0.01):
    """Undoes a Gaussian blur of sigma px, edges padded by reflection"""
    p = int(4 * sigma) + 4
    f = np.pad(img.astype(np.float32), p, mode="reflect")
    fy, fx = np.fft.fftfreq(f.shape[0])[:, None], np.fft.fftfreq(f.shape[1])[None, :]
    H = np.exp(-2 * (np.pi * sigma) ** 2 * (fy**2 + fx**2))
    return np.real(np.fft.ifft2(np.fft.fft2(f) * H / (H**2 + k)))[p:-p, p:-p]


def print_over(c, screen, ink, paper, alpha=ALPHA, sigma=SIGMA, gamma=GAMMA):
    """A glyph of scanned coverage c printed over a scanned screen, as the scanner would read it

    ink and paper are the grey levels of solid ink and of the paper the coverage was read against
    """
    blurred = sigma > 0
    cs = np.clip(_wiener(c, sigma) if blurred else c, 0, 1)
    ds = np.clip(_wiener(screen, sigma) if blurred else screen, 0, 255)
    dark = np.clip((paper - ds) / (paper - ink), 0, 1)
    dev = dark - cv2.GaussianBlur(dark.astype(np.float32), (0, 0), 4.0)
    cs = np.clip(cs + gamma * 4 * cs * (1 - cs) * dev, 0, 1)
    out = (1 - cs) * ds + cs * (alpha * ink + (1 - alpha) * ds * ink / paper)
    return cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigma) if blurred else out.astype(np.float32)


def _raw_crop(path, angle):
    crop, _ = units.card_crop(load_gray(path))
    return units.upright(crop, angle, Path(path).name)[0]


def _ink_away(raw):
    """Every pixel near ink, from the prepared crop as segmentation sees it"""
    prep = texture.prepare(raw).astype(np.float32)
    return cv2.dilate((cv2.GaussianBlur(prep, (0, 0), 1.2) < target.get()["segment"]["ink_level"]).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0


def _sources(job):
    """A scan's labelled units on plain background, each with its coverage read against the paper around it and its target"""
    path, angle, cache_path, profile, per_scan = job
    target.use(profile)
    lo = target.get().settings["texture"]["coverage_contrast"][0]
    cache = pickle.load(open(cache_path, "rb"))
    raw = _raw_crop(path, angle)
    rc = texture.screen_contrast(raw)
    near = _ink_away(raw)
    w = (~near).astype(np.float32)
    fill = cv2.GaussianBlur(raw.astype(np.float32) * w, (0, 0), 12) / np.maximum(cv2.GaussianBlur(w, (0, 0), 12), 1e-3)
    us = [u for u in cache["units"] if "part_of" not in u and rc[int(u["cy"]), int(u["cx"])] < lo]
    if not us:
        return []
    ink = float(np.median(np.concatenate([raw[u["box"][0] : u["box"][1], u["box"][2] : u["box"][3]][ndi.binary_erosion(u["mask"], iterations=3)] for u in us])))
    rng = np.random.default_rng(int(hashlib.sha256(str(path).encode()).hexdigest()[:8], 16))
    out = []
    for i in rng.permutation(len(us))[:per_scan]:
        u = us[i]
        y0, y1, x0, x1 = u["box"]
        T = fill[y0:y1, x0:x1]
        c = np.clip((T - raw[y0:y1, x0:x1].astype(np.float32)) / np.maximum(T - ink, 1), 0, 1) * ndi.binary_dilation(u["mask"], iterations=4)
        out.append(dict(c=np.pad(c, PAD).astype(np.float32), cov=np.pad(u["cov"], PAD).astype(np.float32), mask=np.pad(u["mask"], PAD),
                        ink=ink, paper=float(np.median(T))))
    return out


def _texture(job):
    """A scan's raw crop and where bare strong screen lies in it, within the span of its labelled text so card art is left out"""
    path, angle, cache_path, profile = job
    target.use(profile)
    raw = _raw_crop(path, angle)
    hi = target.get().settings["texture"]["coverage_contrast"][1]
    zone = np.zeros(raw.shape, bool)
    b = np.array([u["bbox"] for u in pickle.load(open(cache_path, "rb"))["units"]])
    if len(b):
        zone[b[:, 0].min() : b[:, 1].max(), b[:, 2].min() : b[:, 3].max()] = True
    return dict(g=raw, free=(texture.screen_contrast(raw) >= hi) & ~_ink_away(raw) & zone, scan=Path(path).stem)


def pairs(sources, textures, n, seed=0, spread=(0.5, 0.8, 0.8)):
    """n composites of a random source printed onto a free patch of a random texture, the model's parameters drawn
    within the given share of their fit so the network does not learn one fit exactly
    """
    rng = np.random.default_rng(seed)
    integ = [cv2.integral((~t["free"]).astype(np.uint8)) for t in textures]
    out = []
    while len(out) < n:
        s = sources[rng.integers(len(sources))]
        k = rng.integers(len(textures))
        g, I = textures[k]["g"], integ[k]
        h, w = s["c"].shape
        for _ in range(200):
            y, x = rng.integers(0, g.shape[0] - h), rng.integers(0, g.shape[1] - w)
            if I[y + h, x + w] - I[y, x + w] - I[y + h, x] + I[y, x] == 0:
                break
        else:
            continue
        a, sg, gm = (p * rng.uniform(1 - d, 1 + d) for p, d in zip((ALPHA, SIGMA, GAMMA), spread))
        x_ = print_over(s["c"], g[y : y + h, x : x + w].astype(np.float32), s["ink"], s["paper"], a, sg, gm)
        out.append((normalise(np.clip(np.round(x_), 0, 255), s["mask"]), s["cov"]))
    return out


def fit(data, steps=6000, seed=0, batch=64):
    """Trains the network on (normalised window, target coverage) pairs with an L1 loss, returns its weights as numpy arrays"""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    layers = []
    for i, d in enumerate(DILATIONS):
        layers += [nn.Conv2d(2 if i == 0 else CHANNELS, CHANNELS, 3, padding=d, dilation=d), nn.ReLU()]
    net = nn.Sequential(*layers, nn.Conv2d(CHANNELS, 1, 1))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net.to(dev)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    nv = len(data) // 10
    val, tr = [data[i] for i in idx[:nv]], [data[i] for i in idx[nv:]]

    def crops(items, r, n):
        X, Y, V = [], [], []
        for _ in range(n):
            x, y = items[r.integers(len(items))]
            h, w = y.shape
            ph, pw = max(CROP - h, 0), max(CROP - w, 0)
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")
            v = np.pad(np.ones((h, w), np.float32), ((0, ph), (0, pw)))
            y = np.pad(y, ((0, ph), (0, pw)))
            a, b = r.integers(0, y.shape[0] - CROP + 1), r.integers(0, y.shape[1] - CROP + 1)
            X.append(x[:, a : a + CROP, b : b + CROP]); Y.append(y[a : a + CROP, b : b + CROP]); V.append(v[a : a + CROP, b : b + CROP])
        return [torch.tensor(np.stack(t)).to(dev) for t in (X, np.array(Y)[:, None], np.array(V)[:, None])]

    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    vx, vy, vw = crops(val, np.random.default_rng(seed + 1), 1024)
    for s in range(steps):
        x, y, w = crops(tr, rng, batch)
        loss = (torch.abs(torch.sigmoid(net(x)) - y) * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 1000 == 0 or s == steps - 1:
            with torch.no_grad():
                p = torch.sigmoid(net(vx))
                log.info("step %d, held out L1 %.4f, weight %.3f", s, float((torch.abs(p - vy) * vw).sum() / vw.sum()), float((p * vw).sum() / (vy * vw).sum()))
    convs = [m for m in net if isinstance(m, nn.Conv2d)]
    W = {}
    for k, m in enumerate(convs[:-1]):
        W[f"conv{k}_w"], W[f"conv{k}_b"] = m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy()
    W["out_w"], W["out_b"] = convs[-1].weight.detach().cpu().numpy(), convs[-1].bias.detach().cpu().numpy()
    return W


def cmd_train(a):
    session = Path(a.session)
    profile = json.loads((session / "manifest.json").read_text())["profile"]["path"]
    target.use(profile)
    if "coverage_contrast" not in target.get().settings.get("texture", {}):
        raise SystemExit(f"profile {profile} sets no texture coverage_contrast to train against")
    rows = cards.scans(a.cards)
    with timed("reading sources and textures"), pool(a.workers, target.use, (profile,)) as ex:
        tex_rows = [r for r in rows if not a.texture_card or r["card"] in a.texture_card]
        textures = [f.result() for f in [ex.submit(_texture, (r["path"], r["angle"], session / "work" / "units" / f"{r['path'].stem}.pkl", profile)) for r in tex_rows]]
        for t in textures:
            log.debug("%s has %d px of bare strong screen", t["scan"], t["free"].sum())
        textures = [t for t in textures if t["free"].sum() >= a.min_free_px]
        tex_cards = {r["card"] for r in tex_rows if r["path"].stem in {t["scan"] for t in textures}}
        src_rows = [r for r in rows if r["card"] not in tex_cards]
        futs = [ex.submit(_sources, (r["path"], r["angle"], session / "work" / "units" / f"{r['path'].stem}.pkl", profile, a.per_scan)) for r in src_rows]
        sources = [s for f in futs for s in f.result()]
    log.info("%d sources from %d scans, textures from %s", len(sources), len(src_rows), ", ".join(sorted(tex_cards)))
    with timed("printing composites"):
        data = pairs(sources, textures, a.pairs, a.seed)
    with timed("training"):
        W = fit(data, a.steps, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **W)
    announce("wrote %s, trained on the textures of %s", out, ", ".join(sorted(tex_cards)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train the network from a session's units and its cards' textures")
    t.add_argument("session", help="a session of a profile with a texture coverage_contrast, whose units give the sources")
    t.add_argument("cards", help="the session's cards file")
    t.add_argument("-o", "--out", required=True, help="weights file to write, kept outside git")
    t.add_argument("--texture-card", action="append", help="card whose screen the composites are printed on, repeat for each, default every card with bare strong screen")
    t.add_argument("--min-free-px", type=int, default=300000, help="bare strong screen a scan needs to give texture")
    t.add_argument("--per-scan", type=int, default=70, help="sources taken from each scan")
    t.add_argument("--pairs", type=int, default=8000, help="composites to train on")
    t.add_argument("--steps", type=int, default=6000, help="training steps of 64 crops")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, help="processes reading scans, default one per core")
    add_args(ap)
    a = ap.parse_args()
    start(a, Path(a.out).parent, "coverage_net")
    cmd_train(a)


if __name__ == "__main__":
    main()
