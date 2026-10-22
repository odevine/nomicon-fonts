"""Crops each scan in a scan table to its card plus a margin, so later runs read and store only the card

The crop slices whole pixels and keeps the file's format and bit depth, so the cropped scan holds exactly the
pixels of the original card region. Originals are never changed. The output folder gets the cropped scans under
their original names, a scan table pointing at them and `prepare.json`, which records each original's sha256,
size and the crop box in its pixels, so every cropped scan can be traced back to the file it came from
"""
import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .log import Progress, add_args, announce, log, pool, start
from .scan import load_gray
from .units import card_crop

MARGIN_PX = 150
# formats Pillow crops without touching pixel values, anything else goes through OpenCV at its native depth
PIL_MODES = {"1", "L", "P", "LA", "PA", "RGB", "RGBA", "I;16", "I;16B", "I"}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def card_box(path, margin=MARGIN_PX):
    """The card's (x, y, w, h) in the scan's pixels with the margin added, clipped to the scan"""
    g = load_gray(path)
    crop, (y0, x0) = card_crop(g)
    x1, y1 = min(x0 + crop.shape[1] + margin, g.shape[1]), min(y0 + crop.shape[0] + margin, g.shape[0])
    # an offset on the 8 px grid card_crop downsamples to, so a cropped scan yields the same units as its original
    x0, y0 = max(x0 - margin, 0) // 8 * 8, max(y0 - margin, 0) // 8 * 8
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0), g.shape[::-1]


def crop_file(src, dst, box):
    """Writes the box of src to dst in the same format and bit depth, returns how the pixels were read"""
    x, y, w, h = box
    ext = Path(src).suffix.lower()
    if ext not in (".png", ".tif", ".tiff"):
        raise SystemExit(f"{src}: only PNG and TIFF scans can be cropped without re-encoding")
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src) as im:
        if im.mode in PIL_MODES and ext == ".png":
            im.crop((x, y, x + w, y + h)).save(dst)
            return f"pillow {im.mode}"
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if not cv2.imwrite(str(dst), np.ascontiguousarray(img[y : y + h, x : x + w])):
        raise RuntimeError(f"could not write {dst}")
    return f"opencv {img.dtype} x{1 if img.ndim == 2 else img.shape[2]}"


def prepare_one(src, dst, margin):
    x, y, w, h, (W, H) = card_box(src, margin)
    how = crop_file(src, dst, (x, y, w, h))
    return dict(file=dst.name, source=str(src), source_sha256=sha256(src), source_bytes=src.stat().st_size, source_size=[int(W), int(H)],
                crop=[x, y, w, h], margin_px=margin, read=how, bytes=dst.stat().st_size, sha256=sha256(dst))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="scan table CSV, file paths relative to it")
    ap.add_argument("-o", "--out", required=True, help="folder for the cropped scans, their scan table and prepare.json")
    ap.add_argument("--margin", type=int, default=MARGIN_PX, help="pixels kept around the card on each side")
    ap.add_argument("--workers", type=int, help="processes, default one per core")
    add_args(ap)
    a = ap.parse_args()
    table, out = Path(a.table), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start(a, out, "prepare")
    rows = list(csv.DictReader(table.open()))
    man_path = out / "prepare.json"
    done = {r["file"]: r for r in json.loads(man_path.read_text())["scans"]} if man_path.exists() else {}
    jobs = []
    for r in rows:
        src = table.parent / r["file"]
        dst = out / src.name
        prev = done.get(dst.name)
        # a scan cropped before with the same margin from an unchanged original is kept
        if prev and dst.exists() and prev["margin_px"] == a.margin and prev["source_bytes"] == src.stat().st_size and prev["source_sha256"] == sha256(src):
            continue
        jobs.append((src, dst))
    if len({d.name for _, d in jobs}) != len(jobs):
        raise SystemExit("two scans in the table share a file name, the cropped copies would overwrite each other")
    log.info("%d scans in the table, %d to crop", len(rows), len(jobs))
    prog = Progress("cropping", len(jobs), every=5)
    with pool(a.workers or os.cpu_count()) as ex:
        for f in as_completed([ex.submit(prepare_one, s, d, a.margin) for s, d in jobs]):
            r = f.result()
            done[r["file"]] = r
            prog.step(f"{r['file']}, {r['source_bytes'] / 1e6:.0f} MB to {r['bytes'] / 1e6:.0f} MB")
    scans = [done[(table.parent / r["file"]).name] for r in rows]
    man_path.write_text(json.dumps(dict(table=str(table), margin_px=a.margin, scans=scans), indent=1))
    with (out / table.name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, file=Path(r["file"]).name))
    before, after = sum(s["source_bytes"] for s in scans), sum(s["bytes"] for s in scans)
    announce("wrote %s: %d scans, %.1f GB to %.1f GB, scan table %s", out, len(scans), before / 1e9, after / 1e9, out / table.name)


if __name__ == "__main__":
    main()
