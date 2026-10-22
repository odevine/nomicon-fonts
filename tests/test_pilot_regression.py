"""Regression checks against the private pilot scans, skipped when the scans or a font are not present"""
import os
from pathlib import Path

import pytest

from glyphstack import calibration, font, template, units

SCANS = Path(__file__).parent.parent / "scans" / "pilot_7card"
FONT = Path(os.environ.get("NOMICON_FONT", ""))

needs_scans = pytest.mark.skipif(not SCANS.exists(), reason="pilot scans not present")
needs_font = pytest.mark.skipif(not FONT.is_file(), reason="set NOMICON_FONT to a TrueType font")


@needs_scans
def test_calibration_matches_pilot():
    res = calibration.analyse(SCANS / "cal_a.png", SCANS / "cal_b.png")
    assert abs(res["summary"]["px_per_mm"] - 47.32) < 0.02
    assert res["summary"]["scan_anisotropy"] < 0.001
    assert calibration.gate(res) == []


@needs_scans
@needs_font
@pytest.mark.parametrize("name,angle,kept,scale", [("scan__008", 0, 79, 1.0), ("scan__009", 90, 78, 1.0)])
def test_extraction_matches_pilot_unit_counts(name, angle, kept, scale):
    r = units.extract_scan(SCANS / f"{name}.png", angle)
    got, s = template.identify(r["units"], template.Templates(font.load_font(str(FONT))))
    assert len(got) == kept
    assert s == scale
    assert round(r["paper"]) == 189
