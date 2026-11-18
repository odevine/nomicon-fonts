"""The synthetic font baseline, each face of the measuring stick drawn at a Bleed of 0, a quarter of the print look, which glyphstack.fontbuild builds the fonts from

It stands in for the glyphs that have too few scanned instances until more cards are scanned. Its glyphs are built the
way fontbuild builds its masters, from the session's traces and the measuring stick's outlines with a quarter of the
print look, and saved as one static font per face. The measuring stick's outlines are in it, so it stays under build/
and is never committed or released

    python -m glyphstack.baseline build/session --font REGULAR.ttf --italic-font ITALIC.ttf -o build/synthetic_baseline
"""
import argparse
import tempfile
from pathlib import Path

from . import font as fnt
from . import fontbuild, log, target

BLEED = 0


def make(font, traces, face, family, style, chars, out):
    """Writes one face's synthetic font baseline from the measuring stick and returns its path and how each glyph was made"""
    with tempfile.TemporaryDirectory() as tmp:
        vf, made = fontbuild.build_face(font, traces, face, family, style, chars, Path(tmp))
        return fontbuild.static(vf, BLEED, out), made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="folder written by glyphstack.session")
    ap.add_argument("-o", "--out", default="build/synthetic_baseline")
    target.add_args(ap)
    log.add_args(ap)
    a = ap.parse_args()
    session, out = Path(a.session), Path(a.out)
    log.start(a, session, "baseline")
    prof, traces, chars = fontbuild.load_session(session, a.profile)
    out.mkdir(parents=True, exist_ok=True)
    for face, path in target.font_paths(a).items():
        f, made = make(fnt.load_font(path), traces, face, prof.primary.full_name, fontbuild.style_of(prof, face), chars, out)
        fontbuild.report(f, made, chars)


if __name__ == "__main__":
    main()
