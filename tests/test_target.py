import argparse

import numpy as np
import pytest

from glyphstack import styles, target
from glyphstack.units import coverage, coverage_units, in_regions, scan_levels, segment


@pytest.fixture
def profile(tmp_path):
    """Writes a variant of the default profile with the given replacements and selects it, restoring the default after"""
    base = (target.BUILTIN / "rules.toml").read_text()

    def make(*reps):
        text = base
        for a, b in reps:
            assert a in text
            text = text.replace(a, b)
        p = tmp_path / "variant.toml"
        p.write_text(text)
        return target.use(p)

    yield make
    target.use(target.DEFAULT)


ONE_FACE = ('''[[faces]]
name = "regular"
full_name = "Nomicon Rules"

[[faces]]
name = "italic"
full_name = "Nomicon Rules Italic"
prefix = "i:"
suffix = "_it"
stem = "it_"''', '''[[faces]]
name = "bold"''')
TOP_HALF = ("box = [0.0, 0.0, 1.0, 1.0]", 'box = [0.0, 0.0, 1.0, 0.5]')
LOCAL = ('mode = "scan"', 'mode = "local"')


def test_default_profile_is_rules_with_regular_first():
    p = target.get()
    assert p.name == "rules" and p.primary.name == "regular" and [f.name for f in p.faces] == ["regular", "italic"]
    assert [f.full_name for f in p.faces] == ["Nomicon Rules", "Nomicon Rules Italic"]
    assert p["templates"]["min_iou"] == 0.85 and p["fit"]["seed_glyph"] == "e"


def test_a_one_face_profile_keys_glyphs_by_character(profile):
    profile(ONE_FACE)
    assert styles.primary() == "bold" and styles.key("bold", "A") == "A" and styles.style_of("A") == "bold"
    assert styles.glyph_name("A") == "A_uc" and styles.instance_id("a", 2) == "a-002"
    with pytest.raises(KeyError):
        styles.style_of("i:a")


def test_profiles_with_clashing_faces_are_refused(profile):
    with pytest.raises(ValueError):
        profile(('prefix = "i:"', 'prefix = ""'))


def test_coverage_clip_sets_the_floor_and_ceiling(profile):
    g = np.array([212.0, 170.0, 120.0, 28.0])
    assert np.allclose(coverage(g, 212, 28), np.clip(((212 - g) / 184 - 0.25) / 0.5, 0, 1))
    profile(("coverage_clip = [0.25, 0.75]", "coverage_clip = [0.1, 0.75]"))
    assert np.allclose(coverage(g, 212, 28), np.clip(((212 - g) / 184 - 0.1) / 0.65, 0, 1))
    with pytest.raises(ValueError):
        profile(("coverage_clip = [0.25, 0.75]", "coverage_clip = [0.75, 0.25]"))


def test_face_fonts_are_checked_against_the_profile():
    ap = argparse.ArgumentParser()
    target.add_args(ap)
    a = ap.parse_args(["--font", "r.ttf", "--face-font", "italic=i.ttf"])
    assert target.font_paths(a) == {"regular": "r.ttf", "italic": "i.ttf"}
    with pytest.raises(KeyError):
        target.font_paths(ap.parse_args(["--font", "r.ttf", "--face-font", "bold=b.ttf"]))


def two_bands():
    """A dark-framed top half with one solid glyph-sized block and a pale bottom half with two, so the pale frame is the scan's paper"""
    g = np.full((400, 300), 230, np.uint8)
    g[:200] = 170
    g[60:120, 100:140] = 30
    g[260:320, 60:100] = 30
    g[260:320, 180:220] = 30
    return g


def test_regions_keep_only_their_components_but_every_component_still_masks(profile):
    profile(TOP_HALF)
    g = two_bands()
    lab, comps = segment(g)
    kept = in_regions(comps, g.shape)
    assert len(comps) == 3 and len(kept) == 1 and kept[0]["y0"] < 200 and kept[0]["region"] == "card"
    paper, ink, allink, ring = scan_levels(g, lab, comps, kept)
    # the levels come from inside the region only
    assert paper == 170 and ink == 30 and allink.sum() > kept[0]["area"]


def test_local_paper_follows_the_frame_under_each_glyph(profile):
    profile(LOCAL)
    g = two_bands()
    lab, comps = segment(g)
    paper, ink, allink, ring = scan_levels(g, lab, comps)
    units = coverage_units(g, lab, comps, paper, ink, allink, ring=ring)
    by_y = sorted(units, key=lambda u: u["cy"])
    assert [u["paper"] for u in by_y] == [170, 230, 230]
    # one scan-wide level is the pale frame's, which reads the dark frame as ink, but each unit's own level reads no ink
    assert paper == 230
    for u in by_y:
        edge = u["cov"][:, :3]
        assert edge.max() == 0
