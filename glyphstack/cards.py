"""A scan folder's cards file: every card scanned in the folder, with its printed name and text, and the scans of it

A cards file is JSON with a list of cards, each with an `id`, its printed `name`, its `text` in the scan list's Card
text format (paragraphs under `Rules:` and `Flavor:`, with other italic text between underscores and a printed
underscore written as \\_), optionally its `set`, collector `number` and rules `box` as y0 y1
x0 x1 fractions of the card for a frame that puts the box elsewhere, and its `scans`, each a `file` relative to the cards
file's folder, the `angle` the card is turned counterclockwise in the scan image in degrees, and optional `notes`. The
id groups the scans of one physical card and names it on the site, so it is CamelCase without spaces, accents or colons
"""
import json
from pathlib import Path

from .log import log


def marked(line):
    """A line of card text as (character, marked italic) pairs, an underscore turning italics on or off and \\_ standing for a printed underscore"""
    out, on, i = [], False, 0
    while i < len(line):
        if line.startswith("\\_", i):
            out.append(("_", on))
            i += 2
        elif line[i] == "_":
            on = not on
            i += 1
        else:
            out.append((line[i], on))
            i += 1
    return out


def printed(line):
    """A line of card text as printed, without its italic markup"""
    return "".join(ch for ch, _ in marked(line))


def load(path):
    """The cards file's cards, checked, each scan's file resolved against the file's folder"""
    path = Path(path)
    d = json.loads(path.read_text())
    seen = set()
    for c in d["cards"]:
        cid = c.get("id", "")
        if not cid or ":" in cid or " " in cid:
            raise SystemExit(f"{path}: card id {cid!r} must be set and have no spaces or colons")
        if cid in seen:
            raise SystemExit(f"{path}: card {cid} is listed twice")
        seen.add(cid)
        if not c.get("scans"):
            raise SystemExit(f"{path}: card {cid} has no scans")
        for s in c["scans"]:
            if not isinstance(s.get("angle"), (int, float)):
                raise SystemExit(f"{path}: give an angle in degrees for {s.get('file')}")
            s["path"] = path.parent / s["file"]
    return d["cards"]


def scans(path):
    """Every scan of the cards file in order, as dicts with the scan path, card id, angle, printed name, text and box"""
    out = []
    for c in load(path):
        for s in c["scans"]:
            out.append(dict(path=s["path"], card=c["id"], angle=float(s["angle"]), name=c.get("name", ""), text=c.get("text"), box=c.get("box")))
    by = {}
    for r in out:
        by.setdefault(r["card"], []).append(r["angle"])
    for card, angles in by.items():
        if len(angles) < 2:
            log.warning("card %s has one scan, its glyphs have one view each", card)
        if not any(a % 360 == 0 for a in angles):
            log.warning("card %s has no upright scan, its first scan is the reference", card)
    return out


def card(path, cid):
    """One card of the cards file by its id"""
    for c in load(path):
        if c["id"] == cid:
            return c
    raise SystemExit(f"{path} has no card {cid}")


def write(path, cards):
    """Writes cards to a cards file, without the resolved scan paths"""
    clean = [dict(c, scans=[{k: v for k, v in s.items() if k != "path"} for s in c["scans"]]) for c in cards]
    Path(path).write_text(json.dumps(dict(cards=clean), indent=1, ensure_ascii=False) + "\n")
