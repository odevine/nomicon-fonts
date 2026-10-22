"""Glyph keys that name a face and a character, so the faces of a profile share one pipeline without mixing

A glyph's key is its face's prefix followed by its character, so the primary face's keys are the bare character.
The faces come from the target profile in use
"""
from fontTools.agl import UV2AGL

from . import target

# the face names of the default profile, for callers that mean those faces
REGULAR, ITALIC = "regular", "italic"


def faces():
    return target.get().faces


def names():
    """Face names in profile order, the primary face first"""
    return [f.name for f in faces()]


def primary():
    return target.get().primary.name


def key(style, ch):
    """The key of a character in a face"""
    return target.get().face(style).prefix + ch


def face_of(k):
    """The face a key belongs to, the one whose prefix is everything before its character"""
    for f in faces():
        if f.prefix == k[:-1]:
            return f
    raise KeyError(f"no face of profile {target.get().name} has the key prefix {k[:-1]!r}")


def style_of(k):
    """The name of the face a key belongs to"""
    return face_of(k).name


def char_of(k):
    """The character of a key"""
    return k[-1]


def stem(k):
    """A file system safe stem for a key, with the face in it so files of different faces never collide

    Capitals get a `_uc` suffix so case-insensitive filesystems keep `a` and `A` apart
    """
    ch = char_of(k)
    base = (ch + "_uc" if ch.isupper() else ch) if ch.isalnum() else f"ord{ord(ch)}"
    return face_of(k).stem + base


def pair_key(a, b):
    """The kerning pair of two keys of one face, the face's prefix before the two characters"""
    return face_of(a).prefix + char_of(a) + char_of(b)


def glyph_folder(ch):
    """Standard glyph name for a character, with a suffix on capitals so case-insensitive filesystems keep them apart"""
    name = UV2AGL.get(ord(ch), f"uni{ord(ch):04X}")
    return name + "_uc" if ch.isupper() else name


def glyph_name(k):
    """The site name of a glyph, the standard glyph name with a suffix on capitals and the face's suffix"""
    return glyph_folder(char_of(k)) + face_of(k).suffix


def instance_id(k, n):
    """The id of the n-th exported instance of a glyph, with the face's suffix in it for faces other than the primary"""
    tag = face_of(k).suffix.lstrip("_")
    return f"{char_of(k)}-{tag}-{n:03d}" if tag else f"{char_of(k)}-{n:03d}"
