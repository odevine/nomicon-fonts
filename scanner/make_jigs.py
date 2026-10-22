"""Writes printable scanning jigs, four corner pieces on one US Letter sheet that each stop a card at one angle

Two straight cuts through the middle of the sheet split it into quarters, each keeping one factory corner of the
paper, which goes into the scanner bed's corner. Each quarter then gets a notch, two straight cuts along the lines
of the card's two edges nearest that corner, so the card slides in from the open side and lies at the jig's angle.
Every jig puts the card's centre at the same place on the bed, which is searched so that each notch touches the
whole length of both card edges. Nothing is printed within the margin of the sheet's edge, so printers that cannot
print to the edge still print every line. Print at actual size on cardstock and keep the printed side up on the glass
"""
import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.path import Path as Poly

PAGE_MM = (215.9, 279.4)
CARD_MM = (63.5, 88.9)
# each jig in the four quarter turns gives 16 angles 22.5 degrees apart
ANGLES = (0, 22.5, 45, 67.5)
# unprinted border along the sheet's edges, wide enough for printers that cannot print near the edge
MARGIN_MM = 15
# a notch corner sits at least this far inside the margin, leaving a strip of card stock and room for the corner arrow
STRIP_MM = 12
INK, GREY, FILL = "#000000", "#9a9a9a", "#f2f2f2"
CORNERS = ("tl", "tr", "br", "bl")
TURN = {"tl": 0, "tr": -90, "br": 180, "bl": 90}


def quarter(corner):
    """Size of the quarter in jig coordinates, along the bed's top edge and then down its side"""
    W, H = PAGE_MM
    return (W / 2, H / 2) if corner in ("tl", "br") else (H / 2, W / 2)


def placed(pts, corner):
    """Jig coordinates, mm right of and down from the bed corner, on the page with the jig's bed corner on a page corner"""
    W, H = PAGE_MM
    pts = np.atleast_2d(pts)
    u, v = pts[:, 0], pts[:, 1]
    return {"tl": np.c_[u, H - v], "tr": np.c_[W - v, H - u], "br": np.c_[W - u, v], "bl": np.c_[v, u]}[corner]


def edges(deg):
    """Directions of the card's top edge and of its side edge for a counterclockwise turn as printed, down positive"""
    t = np.radians(deg)
    return np.array([np.cos(t), -np.sin(t)]), np.array([np.sin(t), np.cos(t)])


def exit_along(P, d, size, m=0.0):
    """Where a ray from P leaves the quarter, its two factory edges moved in by m, and whether it leaves through one of them"""
    U, V = size
    ts = [((U - P[0]) / d[0], False)] if d[0] > 1e-9 else []
    ts += [((V - P[1]) / d[1], False)] if d[1] > 1e-9 else []
    ts += [((m - P[0]) / d[0], True)] if d[0] < -1e-9 else []
    ts += [((m - P[1]) / d[1], True)] if d[1] < -1e-9 else []
    t, outer = min(ts)
    return P + d * t, outer


def notch(centre, deg, size, card, m):
    """The notch corner, where its cuts leave the printable part of the quarter, how many of them end at the margin
    and the share of each card edge the printed part of the notch touches"""
    e1, e3 = edges(deg)
    P = centre - e1 * card[0] / 2 - e3 * card[1] / 2
    (a, oa), (b, ob) = exit_along(P, e1, size, m), exit_along(P, e3, size, m)
    return P, a, b, oa + ob, min(np.hypot(*(a - P)) / card[0], 1), min(np.hypot(*(b - P)) / card[1], 1)


def layout(angles, card, m):
    """The card centre and the angle to corner assignment that give the most printed contact, then the fewest cuts
    running on past the printed line into the margin, then the widest strips"""
    best = None
    for assign in itertools.permutations(CORNERS, len(angles)):
        for cx in np.arange(20, 160, 1.0):
            for cy in np.arange(20, 160, 1.0):
                c = np.array([cx, cy])
                score = []
                for deg, k in zip(angles, assign):
                    U, V = quarter(k)
                    P, _, _, outer, f1, f3 = notch(c, deg, (U, V), card, m)
                    room = min(P[0] - m, P[1] - m, U - P[0], V - P[1])
                    score.append((min(f1, f3) if room >= STRIP_MM else -1, outer, room))
                key = (round(min(s[0] for s in score), 3), -sum(s[1] for s in score), min(s[2] for s in score))
                if best is None or key > best[0]:
                    best = (key, c, dict(zip(angles, assign)))
    (contact, _, _), centre, assign = best
    assert contact > 0, "no card position leaves every notch corner clear of the margin"
    return centre, assign, contact


def outline(P, a, b, size):
    """The piece, the quarter without the wedge beyond the notch, walking the quarter's edge the way round that holds the bed corner"""
    U, V = size
    ring = [np.array(p, float) for p in ((0, 0), (U, 0), (U, V), (0, V))]

    def at(p):
        # distance along the quarter's edge clockwise from the bed corner
        x, y = p
        if abs(y) < 1e-6:
            return x
        if abs(x - U) < 1e-6:
            return U + y
        if abs(y - V) < 1e-6:
            return U + V + (U - x)
        return 2 * U + V + (V - y)

    per = 2 * (U + V)
    sa, sb = at(a), at(b)
    # from b round to a through the bed corner at 0, keeping the corners passed on the way
    span = (sa - sb) % per
    passed = sorted((c for c in ring if 0 < (at(c) - sb) % per < span or np.allclose(c, (0, 0))), key=lambda c: (at(c) - sb) % per)
    return np.array([b] + passed + [a, P])


def check(poly, centre, deg, card):
    e1, e3 = edges(deg)
    k = np.array([centre + s * e1 * card[0] / 2 + t * e3 * card[1] / 2 for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    c = k.mean(0)
    g = np.mgrid[0:160:0.5, 0:160:0.5].reshape(2, -1).T
    inside = Poly(c + (k - c) * 0.995).contains_points(g)
    assert not Poly(poly).contains_points(g[inside]).any(), f"the {deg:g} degree jig covers the card"


def label_spot(poly, m):
    """The point of the piece's printable part farthest from its edges and from the margin, and that distance"""
    lo, hi = poly.min(0), poly.max(0)
    g = np.mgrid[lo[0] : hi[0] : 1.0, lo[1] : hi[1] : 1.0].reshape(2, -1).T
    g = g[Poly(poly).contains_points(g) & (g[:, 0] > m) & (g[:, 1] > m)]
    seg = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]

    def dist(p):
        out = []
        for s, e in seg:
            d = e - s
            t = np.clip(np.dot(p - s, d) / np.dot(d, d), 0, 1)
            out.append(np.hypot(*(p - s - t * d)))
        return min(min(out), p[0] - m, p[1] - m)

    ds = np.array([dist(p) for p in g])
    return g[ds.argmax()], ds.max()


def draw(ax, deg, corner, centre, card, m, printable):
    size = quarter(corner)
    P, a, b, outer, f1, f3 = notch(centre, deg, size, card, m)
    e1, e3 = edges(deg)
    # the piece itself reaches the paper's edge, only its printed lines stop at the margin
    poly = outline(P, exit_along(P, e1, size)[0], exit_along(P, e3, size)[0], size)
    check(poly, centre, deg, card)
    fill = ax.fill(*placed(poly, corner).T, facecolor=FILL, edgecolor="none")[0]
    fill.set_clip_path(printable)
    ax.plot(*placed(np.array([a, P, b]), corner).T, color=INK, linewidth=1.6)
    rot = TURN[corner]
    spot, room = label_spot(poly, m)
    # the label runs along the piece's long side so it fits a narrow strip
    lo, hi = poly.min(0), poly.max(0)
    along = 0 if hi[0] - lo[0] >= hi[1] - lo[1] else -90
    ax.text(*placed(spot, corner)[0], f"{deg:g}°", rotation=rot + along, ha="center", va="center", fontsize=min(30, room * 2.2), weight="bold")
    tip, tail = np.array([m + 2, m + 2]), np.array([m + 10, m + 10])
    assert Poly(poly).contains_points([tip, tail]).all(), f"no room for the corner arrow on the {deg:g} degree jig"
    ax.annotate("", xy=placed(tip, corner)[0], xytext=placed(tail, corner)[0], arrowprops=dict(arrowstyle="->", linewidth=0.8))
    return placed(poly, corner), outer, f1, f3


NOTES = ("print at actual size on cardstock", "printed side up on the glass", "cut the thin lines on to the edges", "then cut each notch",
         "arrow points to the bed corner")


def notes(ax, pieces, m):
    """The notes in one offcut inside the margin, clear of every piece and of the middle cuts by a few mm"""
    W, H = PAGE_MM
    w, h = 62, 4.5 * len(NOTES)
    for cx in np.arange(m + w / 2 + 3, W - m - w / 2 - 3, 2.0):
        for cy in np.arange(m + h / 2 + 3, H - m - h / 2 - 3, 2.0):
            box = np.mgrid[cx - w / 2 - 3 : cx + w / 2 + 3 : 2.0, cy - h / 2 - 3 : cy + h / 2 + 3 : 2.0].reshape(2, -1).T
            if np.ptp(np.sign(box[:, 0] - W / 2)) or np.ptp(np.sign(box[:, 1] - H / 2)):
                continue
            if not any(Poly(p).contains_points(box).any() for p in pieces):
                for i, t in enumerate(NOTES):
                    ax.text(cx, cy + h / 2 - 2.25 - 4.5 * i, t, ha="center", va="center", fontsize=7, color=GREY)
                return
    raise AssertionError("no room on the offcuts for the notes")


def blank_margin(name, m):
    """Fails when anything is drawn in the margin of the written PDF, read back as an image"""
    import pymupdf

    px = pymupdf.open(name)[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width)
    k = int(np.floor((m - 0.5) / 25.4 * 150))
    edge = np.concatenate([img[:k].ravel(), img[-k:].ravel(), img[:, :k].ravel(), img[:, -k:].ravel()])
    assert edge.min() >= 250, f"something is drawn within {m:g} mm of the sheet's edge"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=Path(__file__).parent, type=Path, help="folder for the PDF")
    ap.add_argument("--angles", type=float, nargs="+", default=ANGLES, help="up to four jig angles in degrees")
    ap.add_argument("--card", type=float, nargs=2, default=CARD_MM, metavar=("W", "H"), help="card size in mm, default a 2.5 by 3.5 inch playing card")
    ap.add_argument("--margin", type=float, default=MARGIN_MM, help="unprinted border along the sheet's edges in mm")
    a = ap.parse_args()
    assert len(a.angles) <= 4, "one sheet has four corners, so at most four jigs"
    card, m = np.array(a.card), a.margin
    centre, assign, contact = layout(tuple(a.angles), card, m)
    W, H = PAGE_MM
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    printable = Rectangle((m, m), W - 2 * m, H - 2 * m, transform=ax.transData)
    ax.plot([W / 2, W / 2], [m, H - m], color=INK, linewidth=0.5)
    ax.plot([m, W - m], [H / 2, H / 2], color=INK, linewidth=0.5)
    pieces = []
    for deg, corner in assign.items():
        poly, outer, f1, f3 = draw(ax, deg, corner, centre, card, m, printable)
        pieces.append(poly)
        run_on = f", {outer} notch cut runs on past its line to the edge" if outer else ""
        print(f"{deg:g} degrees in the {corner} corner, notch touches {f1:.0%} and {f3:.0%} of the card edges{run_on}")
    notes(ax, pieces, m)
    a.out.mkdir(parents=True, exist_ok=True)
    name = a.out / "jigs.pdf"
    with PdfPages(name) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    blank_margin(name, m)
    print(f"wrote {len(assign)} jigs on one sheet to {name}, card centre {centre.tolist()} mm from the bed corner")


if __name__ == "__main__":
    main()
