import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent


def bed(dtype=np.uint8, turn=0):
    """A bright scanner bed with a dark card turned by `turn` degrees and some noise, so every pixel value matters"""
    top = 255 if dtype == np.uint8 else 65535
    g = np.full((1600, 1400), 0.97, np.float32)
    g[400:1150, 450:990] = 0.15
    g = cv2.warpAffine(g, cv2.getRotationMatrix2D((720, 775), turn, 1.0), (1400, 1600), borderValue=0.97)
    g += np.random.default_rng(0).normal(0, 0.01, g.shape).astype(np.float32)
    return (np.clip(g, 0, 1) * top).astype(dtype)


def run(table, out):
    subprocess.run([sys.executable, "-m", "glyphstack.prepare", str(table), "-o", str(out), "--margin", "40", "--workers", "1"], cwd=ROOT, check=True, capture_output=True)
    return json.loads((out / "prepare.json").read_text())


def test_crops_keep_pixels_format_and_provenance(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    gray = bed()
    Image.fromarray(gray).convert("P").save(src / "a.png")
    rgb16 = np.repeat(bed(np.uint16, turn=30)[..., None], 3, 2)
    cv2.imwrite(str(src / "b.tif"), rgb16)
    (src / "t.csv").write_text("file,card,angle\na.png,X,0\nb.tif,X,30\n")
    man = run(src / "t.csv", tmp_path / "out")

    a, b = man["scans"]
    x, y, w, h = a["crop"]
    assert w < 700 and h < 900 and a["source_size"] == [1400, 1600]
    with Image.open(src / "a.png") as im, Image.open(tmp_path / "out" / "a.png") as cr:
        assert cr.mode == "P"
        assert np.array_equal(np.array(cr), np.array(im)[y : y + h, x : x + w])

    x, y, w, h = b["crop"]
    got = cv2.imread(str(tmp_path / "out" / "b.tif"), cv2.IMREAD_UNCHANGED)
    assert got.dtype == np.uint16 and np.array_equal(got, rgb16[y : y + h, x : x + w])
    # a card turned 30 degrees keeps all four corners inside the crop
    assert w > 800 and h > 900

    assert len(a["source_sha256"]) == 64 and a["source_bytes"] == (src / "a.png").stat().st_size
    assert (tmp_path / "out" / "t.csv").read_text().splitlines()[1:] == ["a.png,X,0", "b.tif,X,30"]


def test_rerun_keeps_unchanged_crops(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    Image.fromarray(bed()).save(src / "a.png")
    (src / "t.csv").write_text("file,card,angle\na.png,X,0\n")
    run(src / "t.csv", tmp_path / "out")
    first = (tmp_path / "out" / "a.png").stat().st_mtime_ns
    run(src / "t.csv", tmp_path / "out")
    assert (tmp_path / "out" / "a.png").stat().st_mtime_ns == first
