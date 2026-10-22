"""Writes printable scanning jigs, four corner pieces on one US Letter sheet that each stop a card at one angle

Two straight cuts through the middle of the sheet split it into quarters, each keeping one factory corner of the
paper, which goes into the scanner bed's corner. Each quarter then gets a notch, two straight cuts along the lines
of the card's two edges nearest that corner, so the card slides in from the open side and lies at the jig's angle.
Every jig puts the card's centre at the same place on the bed, which is searched so that each notch touches the
whole length of both card edges. Print at actual size on cardstock and keep the printed side up on the glass
"""
import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as Poly

PAGE_MM = (215.9, 279.4)
CARD_MM = (63.5, 88.9)
ANGLES = (0, 15, 30, 45)
# a notch corner closer than this to a factory edge leaves too thin a strip of card stock
MARGIN_MM = 12
SCALE_MM = 100
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


def exit_along(P, d, size):
    """Where a ray from P inside the quarter leaves it"""
    U, V = size
    ts = [(U - P[0]) / d[0]] if d[0] > 1e-9 else []
    ts += [(V - P[1]) / d[1]] if d[1] > 1e-9 else []
    ts += [-P[1] / d[1]] if d[1] < -1e-9 else []
    return P + d * min(ts)


def notch(centre, deg, size, card):
    """The notch corner, where its two cuts leave the quarter, and the share of each card edge the notch touches"""
    e1, e3 = edges(deg)
    P = centre - e1 * card[0] / 2 - e3 * card[1] / 2
    a, b = exit_along(P, e1, size), exit_along(P, e3, size)
    return P, a, b, min(np.hypot(*(a - P)) / card[0], 1), min(np.hypot(*(b - P)) / card[1], 1)


def layout(angles, card):
    """The card centre and the angle to corner assignment that give the most contact, then the widest strips"""
    best = None
    for assign in itertools.permutations(CORNERS, len(angles)):
        for cx in np.arange(20, 150, 1.0):
            for cy in np.arange(20, 150, 1.0):
                c = np.array([cx, cy])
                score = []
                for deg, k in zip(angles, assign):
                    U, V = quarter(k)
                    P, _, _, f1, f3 = notch(c, deg, (U, V), card)
                    m = min(P[0], P[1], U - P[0], V - P[1])
                    score.append((min(f1, f3) if m >= MARGIN_MM else -1, m))
                key = (round(min(s[0] for s in score), 3), min(s[1] for s in score))
                if best is None or key > best[0]:
                    best = (key, c, dict(zip(angles, assign)))
    (contact, margin), centre, assign = best
    assert contact > 0, "no card position leaves every notch corner clear of the factory edges"
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


def label_spot(poly):
    """The point inside the piece farthest from its edges, and that distance"""
    lo, hi = poly.min(0), poly.max(0)
    g = np.mgrid[lo[0] : hi[0] : 1.0, lo[1] : hi[1] : 1.0].reshape(2, -1).T
    g = g[Poly(poly).contains_points(g)]
    seg = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]

    def dist(p):
        out = []
        for s, e in seg:
            d = e - s
            t = np.clip(np.dot(p - s, d) / np.dot(d, d), 0, 1)
            out.append(np.hypot(*(p - s - t * d)))
        return min(out)

    ds = np.array([dist(p) for p in g])
    return g[ds.argmax()], ds.max()


def draw(ax, deg, corner, centre, card):
    size = quarter(corner)
    P, a, b, f1, f3 = notch(centre, deg, size, card)
    poly = outline(P, a, b, size)
    check(poly, centre, deg, card)
    ax.fill(*placed(poly, corner).T, facecolor=FILL, edgecolor="none")
    ax.plot(*placed(np.array([a, P, b]), corner).T, color=INK, linewidth=1.6)
    rot = TURN[corner]
    spot, room = label_spot(poly)
    # the label runs along the piece's long side so it fits a narrow strip
    lo, hi = poly.min(0), poly.max(0)
    along = 0 if hi[0] - lo[0] >= hi[1] - lo[1] else -90
    ax.text(*placed(spot, corner)[0], f"{deg:g}°", rotation=rot + along, ha="center", va="center", fontsize=min(30, room * 2.2), weight="bold")
    ax.annotate("", xy=placed((2.5, 2.5), corner)[0], xytext=placed((9, 9), corner)[0], arrowprops=dict(arrowstyle="->", linewidth=0.8))
    return placed(poly, corner), f1, f3


NOTES = ("print at actual size on cardstock", "printed side up on the glass", "cut the thin lines, then each notch", "arrow corner into the bed corner")


def scale_bar(ax, pieces):
    """A 100 mm bar with the notes beside it, on the offcuts where it is clear of every piece by a few mm"""
    W, H = PAGE_MM
    for x0 in np.arange(8, W - 8, 2.0):
        for y0 in np.arange(8, H - 8, 2.0):
            for d in (np.array([1.0, 0]), np.array([0, 1.0])):
                end = np.array([x0, y0]) + d * SCALE_MM
                if end[0] > W - 8 or end[1] > H - 8:
                    continue
                pts = np.linspace((x0, y0), end, 60)
                n = np.array([-d[1], d[0]])
                halo = np.vstack([pts + n * k for k in np.arange(-26, 6, 2.0)])
                if not any(Poly(p).contains_points(halo).any() for p in pieces):
                    ax.plot(*np.array([(x0, y0), end]).T, color=INK, linewidth=0.8)
                    for k in range(0, SCALE_MM + 1, 10):
                        p = np.array([x0, y0]) + d * k
                        ax.plot(*np.array([p, p + n * (2 if k % 50 else 4)]).T, color=INK, linewidth=0.5)
                    mid = (np.array([x0, y0]) + end) / 2
                    rot = 90 if d[1] else 0
                    ax.text(*(mid - n * 3.5), f"this bar measures {SCALE_MM} mm", rotation=rot, ha="center", va="center", fontsize=7)
                    for i, t in enumerate(NOTES):
                        ax.text(*(mid - n * (9 + 4 * i)), t, rotation=rot, ha="center", va="center", fontsize=6, color=GREY)
                    return
    raise AssertionError("no room on the offcuts for the scale bar")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=Path(__file__).parent, type=Path, help="folder for the PDF")
    ap.add_argument("--angles", type=float, nargs="+", default=ANGLES, help="up to four jig angles in degrees")
    ap.add_argument("--card", type=float, nargs=2, default=CARD_MM, metavar=("W", "H"), help="card size in mm, default a 2.5 by 3.5 inch playing card")
    a = ap.parse_args()
    assert len(a.angles) <= 4, "one sheet has four corners, so at most four jigs"
    card = np.array(a.card)
    centre, assign, contact = layout(tuple(a.angles), card)
    W, H = PAGE_MM
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([W / 2, W / 2], [0, H], color=INK, linewidth=0.5)
    ax.plot([0, W], [H / 2, H / 2], color=INK, linewidth=0.5)
    pieces = []
    for deg, corner in assign.items():
        poly, f1, f3 = draw(ax, deg, corner, centre, card)
        pieces.append(poly)
        print(f"{deg:g} degrees in the {corner} corner, notch touches {f1:.0%} and {f3:.0%} of the card edges")
    scale_bar(ax, pieces)
    a.out.mkdir(parents=True, exist_ok=True)
    name = a.out / "jigs.pdf"
    with PdfPages(name) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"wrote {len(assign)} jigs on one sheet to {name}, card centre {centre.tolist()} mm from the bed corner")


if __name__ == "__main__":
    main()
