"""Loads a scan and turns the rules box into a flat, upright ink-density image"""
import cv2
import numpy as np
import pymupdf


def load_scan(path):
    """Returns the scan as a BGR float32 array in 0..1, from a PDF with one embedded image or a TIFF/PNG/JPEG"""
    if str(path).lower().endswith(".pdf"):
        doc = pymupdf.open(path)
        data = doc.extract_image(doc[0].get_images()[0][0])["image"]
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return img[..., :3].astype(np.float32) / scale


def _gray8(img):
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 255).astype(np.uint8)


def card_outline(img):
    """Fits a rotated rectangle to the dark card frame, as ((cx, cy), (w, h), angle) in scan pixels"""
    gray = _gray8(img)
    h, w = gray.shape
    small = cv2.resize(gray, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    mask = cv2.morphologyEx((small < 60).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(max(cs, key=cv2.contourArea))
    return (cx * 8, cy * 8), (rw * 8, rh * 8), ang


def square_card(img):
    """Rotates the card upright using its outline and returns the rotated image and its card bounding box"""
    (cx, cy), (rw, rh), ang = card_outline(img)
    if rw > rh:
        ang += 90
    ang = ((ang + 45) % 90) - 45
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    h, w = img.shape[:2]
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(1, 1, 1))
    cw, ch = min(rw, rh), max(rw, rh)
    return out, (int(cx - cw / 2), int(cy - ch / 2), int(cw), int(ch))


def find_text_box(img, card):
    """Finds the pale rules box as the largest bright region in the lower half of the card, in scan pixels"""
    cx, cy, cw, ch = card
    gray = _gray8(img)[cy : cy + ch, cx : cx + cw]
    small = cv2.resize(gray, (cw // 8, ch // 8), interpolation=cv2.INTER_AREA)
    mask = (small > 190).astype(np.uint8)
    mask[: int(small.shape[0] * 0.5)] = 0
    mask[int(small.shape[0] * 0.95) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise RuntimeError("no text box found, pass --box x,y,w,h")
    i = 1 + int(np.argmax(stats[1:, 4]))
    x, y, bw, bh = stats[i][:4]
    return cx + x * 8, cy + y * 8, bw * 8, bh * 8


def _rotate(img, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def text_skew(gray):
    """Finds the residual rotation in degrees that makes text rows sharpest in the horizontal projection"""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ink = (small < small.mean() * 0.6).astype(np.float32)

    def score(a):
        return float(np.var(_rotate(ink, a).sum(axis=1)))

    best = max(np.arange(-1.0, 1.0001, 0.1), key=score)
    return float(max(np.arange(best - 0.1, best + 0.1001, 0.01), key=score))


def ink_density(img, stem_px=14):
    """Returns paper as 0 and solid ink as 1 from the darkest channel, with the box texture divided out"""
    dark = img.min(axis=2)
    k = 3 * stem_px | 1
    paper = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    paper = cv2.GaussianBlur(paper, (0, 0), k / 4)
    density = 1.0 - dark / np.maximum(paper, 1e-3)
    solid = np.percentile(density, 99.7)
    return np.clip(density / solid, 0, 1).astype(np.float32)


def rules_box_density(img, box=None, margin=0.04):
    """Runs squaring, box crop, text deskew and flattening, returns the density image and the crop origin"""
    img, card = square_card(img)
    x, y, w, h = box or find_text_box(img, card)
    pad = int(w * margin)
    crop = img[y - pad : y + h + pad, x - pad : x + w + pad]
    ang = text_skew(_gray8(crop))
    crop = _rotate(crop, ang)
    crop = crop[pad : crop.shape[0] - pad, pad : crop.shape[1] - pad]
    return ink_density(crop), {"box": [int(x), int(y), int(w), int(h)], "deskew_deg": ang}


def load_gray(path):
    """Returns a scan as an 8-bit greyscale array, converting palette and colour PNGs the way the pilot did"""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    return np.array(Image.open(path).convert("L"))
