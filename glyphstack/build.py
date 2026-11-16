"""Glyphs a session has no stack for, made from others the way the measuring stick makes them

A composite of the measuring stick, such as É from E and acute, is its parts' traces at the font's component offsets,
and a combining accent is its spacing accent moved onto the origin with no advance
"""
import re
import unicodedata

# each combining accent of GF Latin Core and the measuring stick glyph of its spacing form
COMBINING = {"̀": "grave", "́": "acute", "̂": "circumflex", "̃": "tilde", "̄": "macron", "̆": "breve",
             "̇": "dotaccent", "̈": "dieresis", "̊": "ring", "̋": "hungarumlaut", "̌": "caron",
             "̦": "commaaccent", "̧": "cedilla", "̨": "ogonek"}


def part(font, name):
    """The character a glyph of the font is keyed by, None for a glyph with no code point or only a Private Use one"""
    for cp, n in sorted(font.getBestCmap().items()):
        if n == name and unicodedata.category(chr(cp)) != "Co":
            return chr(cp)
    return None


def parts(font, ch):
    """The parts a measuring stick builds a character from as (character, glyph name, offset in font units), None when it draws the character as one outline

    A combining accent is built from its spacing accent, moved left by half that accent's advance so it centres on the
    origin
    """
    cmap = font.getBestCmap()
    if ch in COMBINING:
        name = COMBINING[ch]
        return [(part(font, name), name, (-font["hmtx"][name][0] / 2, 0))] if name in font["hmtx"].metrics else None
    if ord(ch) not in cmap or not font["glyf"][cmap[ord(ch)]].isComposite():
        return None
    return [(part(font, c.glyphName), c.glyphName, (c.x, c.y)) for c in font["glyf"][cmap[ord(ch)]].components]


def shift(d, dx, dy):
    """An SVG path in font units, y down, moved dx across and dy up"""
    return re.sub(r"(-?[\d.]+),(-?[\d.]+)", lambda m: f"{float(m[1]) + dx:.1f},{float(m[2]) - dy:.1f}", d)


def source(entry):
    """How a glyph's trace was made, from its entry in traces.json"""
    return entry.get("built") or ("prior" if entry.get("prior") else "traced")


def compose(font, ch, ps, pieces):
    """The entry of a character built from its parts' entries at their offsets, recording each part and how it was made"""
    adv = 0 if ch in COMBINING else font["hmtx"][font.getBestCmap()[ord(ch)]][0]
    return dict(trace="".join(shift(e["trace"], o[0], o[1]) for (_, _, o), e in zip(ps, pieces)), adv=adv, n=0, nodes=sum(e["nodes"] for e in pieces),
                nc=sum(e["nc"] for e in pieces), built="derived" if ch in COMBINING else "composed",
                parts=[dict(char=c, glyph=name, offset=list(o), source=source(e)) for (c, name, o), e in zip(ps, pieces)])
