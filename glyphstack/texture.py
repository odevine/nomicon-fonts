"""Reads ink printed over a textured text box: the paper level is divided out and a strong halftone screen averaged away

A card's text box can carry a gradient, a tint or faint art behind the text, all printed as a halftone screen of about
6 px period at 1200 dpi. `prepare` returns the card crop as a grey image the rest of the pipeline reads as before. The
box is divided by its local paper level, the bright envelope of the screen, so a gradient or a tint reads at one paper
level and the fixed ink threshold keeps its meaning. Where the screen is strong next to the paper to ink range, as over
a dark gradient or art, each pixel is blended toward a Gaussian blur that averages the screen out. Where it is weak the
pixels are left sharp, so a plain box reads as it did. The profile's texture section sets all of this and is off by default
"""
import cv2
import numpy as np

from . import target

# the defaults leave a scan untouched, so a profile without a texture section reads as before
DEFAULTS = dict(flatten=False, envelope_px=41, envelope_blur=10.0, paper_level=216.0, ink_level=25.0, descreen_sigma=0.0,
                descreen_contrast=[0.09, 0.12], highpass_sigma=3.0, contrast_sigma=25.0, ink_grow_px=5, coverage_contrast=[0.09, 0.12])


def settings():
    return {**DEFAULTS, **target.get().settings.get("texture", {})}


def active():
    s = settings()
    return bool(s["flatten"] or s["descreen_sigma"])


def envelope(g, k, sigma):
    """Local paper level, the grey closing with a disc wider than any stroke, smoothed"""
    bg = cv2.morphologyEx(g.astype(np.float32), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.GaussianBlur(bg, (0, 0), sigma)


def flatten(g, s=None):
    """Grey divided by the local paper envelope and set to the profile's paper level"""
    s = s or settings()
    env = envelope(g, s["envelope_px"], s["envelope_blur"])
    return np.clip(s["paper_level"] * g.astype(np.float32) / np.maximum(env, 1), 0, 255)


def screen_contrast(g, s=None):
    """Local screen contrast as a fraction of the paper to ink range, from the high pass energy away from the ink

    The ink is found with the segment threshold on the flattened image and grown, and the energy of the rest is
    averaged with Gaussian weights, so the ink's own edges do not count as screen
    """
    s = s or settings()
    f = g.astype(np.float32)
    hp = f - cv2.GaussianBlur(f, (0, 0), s["highpass_sigma"])
    level = target.get()["segment"]["ink_level"]
    ink = cv2.GaussianBlur(flatten(g, s), (0, 0), 1.2) < level
    r = s["ink_grow_px"]
    ink = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    w = (~ink).astype(np.float32)
    num = cv2.GaussianBlur(hp * hp * w, (0, 0), s["contrast_sigma"])
    den = cv2.GaussianBlur(w, (0, 0), s["contrast_sigma"])
    rms = np.sqrt(num / np.maximum(den, 1e-3))
    return rms / np.maximum(envelope(g, s["envelope_px"], s["envelope_blur"]) - s["ink_level"], 10)


def descreen(g, s=None):
    """Blends each pixel toward a Gaussian blur in proportion to how strong the screen is around it"""
    s = s or settings()
    f = g.astype(np.float32)
    lo, hi = s["descreen_contrast"]
    w = np.clip((screen_contrast(g, s) - lo) / (hi - lo), 0, 1)
    return f + w * (cv2.GaussianBlur(f, (0, 0), s["descreen_sigma"]) - f)


def prepare(g):
    """The card crop as the pipeline reads it under the profile's texture settings, unchanged when they are off"""
    s = settings()
    if not active():
        return g
    out = g.astype(np.float32)
    if s["descreen_sigma"]:
        out = descreen(g, s)
    if s["flatten"]:
        out = flatten(out, s)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
