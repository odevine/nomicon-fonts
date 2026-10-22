"""Matches the copies of each printed glyph between two scans of one card from glyph positions"""
from collections import defaultdict

import numpy as np

VOTE_BIN = 20.0
TOLERANCES = (40, 25, 14, 10)


def match_map(first, second):
    """Finds the rigid map between two scans of one card from glyph centres, returns (first idx, second idx) pairs

    Inputs are lists of (char, cx, cy). A translation vote over same-character candidates gives a first
    shift and a few rounds of Kabsch fits with shrinking tolerance refine it to a rotation and shift.
    Returns the (first idx, second idx) pairs and the rotation R and shift t that map the first onto the second
    """
    by_char = defaultdict(list)
    for k, (ch, cx, cy) in enumerate(second):
        by_char[ch].append((cx, cy, k))
    votes = defaultdict(int)
    for ch, cx, cy in first:
        for x2, y2, _ in by_char[ch]:
            votes[(round((x2 - cx) / VOTE_BIN), round((y2 - cy) / VOTE_BIN))] += 1
    (bx, by), _ = max(votes.items(), key=lambda kv: kv[1])
    t = np.array([bx * VOTE_BIN, by * VOTE_BIN])
    R = np.eye(2)
    pairs = []
    for tol in TOLERANCES:
        cand = []
        for ka, (ch, cx, cy) in enumerate(first):
            p = R @ np.array([cx, cy]) + t
            for x2, y2, kb in by_char[ch]:
                d = np.hypot(x2 - p[0], y2 - p[1])
                if d < tol:
                    cand.append((d, ka, kb))
        # one glyph of the second scan answers to at most one of the first, nearest first
        pairs, used_a, used_b = [], set(), set()
        for d, ka, kb in sorted(cand):
            if ka not in used_a and kb not in used_b:
                pairs.append((ka, kb))
                used_a.add(ka)
                used_b.add(kb)
        A = np.array([first[ka][1:] for ka, _ in pairs])
        B = np.array([second[kb][1:] for _, kb in pairs])
        ca, cb = A.mean(0), B.mean(0)
        u, _, vt = np.linalg.svd((A - ca).T @ (B - cb))
        R = vt.T @ u.T
        t = cb - R @ ca
    return pairs, R, t


def match_pair(first, second):
    """The pairs of match_map and the rotation between the scans in degrees"""
    pairs, R, _ = match_map(first, second)
    return pairs, float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
