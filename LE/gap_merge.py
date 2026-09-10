# ============================================================
# gap_merge.py
# Merge same-pipe line groups that are split by a gap, using the
# raw raster ink in the gap corridor to decide whether the gap is
# justified -- instead of relying on CE bounding boxes (which
# miss most small valve/fitting symbols, see conversation notes).
#
# For every pair of "simple" (2-free-end) groups that are aligned
# and close together, we sample a narrow band matched to the
# pipe's own stroke width, run connected-component analysis on
# the ink inside it, and classify what's causing the gap:
#
#   NONE       -> no ink at all in the corridor  -> do NOT merge
#   TEXT_ONLY  -> only small character-sized blobs (a tag label
#                 sitting near the pipe, not on it) -> do NOT merge
#   CONTINUOUS -> a blob spans (almost) the full gap at line width
#                 -> this was never really two lines; it's a Hough
#                 splitting artifact -> MERGE (and flag separately
#                 so you know to also loosen PPHT_MAX_LINE_GAP)
#   SYMBOL     -> a compact blob clearly bigger than a text
#                 character sits in the corridor -> MERGE (real
#                 valve/fitting causing the true gap)
#
# Usage:
#   python gap_merge.py \
#       --lines 02_Unmarked_geometric_filtered_lines.csv \
#       --pdf   02_Unmarked_original_highlighted.pdf \
#       --out   02_Unmarked_gap_merged_lines.csv
# ============================================================

import argparse
import numpy as np
import pandas as pd
import cv2
import fitz  # PyMuPDF


# ------------------------------------------------------------------
# TUNABLE PARAMETERS
# ------------------------------------------------------------------
FREE_END_TOL       = 6     # px -- endpoints within this are "the same point"
AXIS_ALIGN_TOL      = 15    # px -- perpendicular offset allowed between two
                            #        candidate segments to call them collinear
MAX_GAP             = 250   # px -- never consider bridging further than this
STROKE_WIDTH        = 11    # px -- typical pipe stroke width (measure your
                            #        own drawing with estimate_stroke_width()
                            #        below if this looks off)
BAND_MULTIPLIER     = 2.5   # corridor sampling band = STROKE_WIDTH * this
TEXT_BLOB_MAX_SIDE  = 26    # px -- blobs with both dimensions <= this are
                            #        treated as text-character noise
CONTINUOUS_FRACTION = 0.85  # a blob spanning >= this fraction of the gap
                            #        length, at roughly stroke width, counts
                            #        as "already continuous" (Hough artifact)
DPI                 = 300


# ------------------------------------------------------------------
# Step 0: rasterise the original PDF into a binary ink mask
# ------------------------------------------------------------------
def build_binary_mask(pdf_path, dpi=DPI):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0]
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    doc.close()
    return binary


def estimate_stroke_width(binary, df, n_samples=30):
    """Optional helper -- run this once on a new drawing to calibrate
    STROKE_WIDTH instead of guessing."""
    widths = []
    verticals = df[df.Orientation == 'Vertical'].sample(
        min(n_samples, len(df)), random_state=1
    )
    for _, r in verticals.iterrows():
        x = int((r.x_start + r.x_end) / 2)
        y = int((r.y_start + r.y_end) / 2)
        if not (0 <= y < binary.shape[0] and 20 <= x < binary.shape[1] - 20):
            continue
        row = binary[y, x - 20:x + 20]
        idx = np.where(row > 0)[0]
        if len(idx):
            widths.append(idx.max() - idx.min() + 1)
    return int(np.median(widths)) if widths else STROKE_WIDTH


# ------------------------------------------------------------------
# Step 1: find each group's "free ends" (endpoints not shared with
# another segment in the same group) -- only groups with exactly 2
# free ends are simple chains and eligible for cross-group merging.
# Branching/closed groups are left untouched.
# ------------------------------------------------------------------
def find_free_ends(g, tol=FREE_END_TOL):
    pts = []
    for _, r in g.iterrows():
        pts.append((r.x_start, r.y_start))
        pts.append((r.x_end, r.y_end))
    free = []
    for i, p in enumerate(pts):
        matches = sum(
            1 for j, q in enumerate(pts)
            if i != j and abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol
        )
        if matches == 0:
            free.append(p)
    return free


def build_simple_groups(df):
    groups = {}
    for gid, g in df.groupby('Group_ID'):
        free = find_free_ends(g)
        if len(free) != 2:
            continue
        (x1, y1), (x2, y2) = free
        orient = 'V' if abs(x1 - x2) < abs(y1 - y2) else 'H'
        groups[gid] = {'ends': [(x1, y1), (x2, y2)], 'orient': orient}
    return groups


# ------------------------------------------------------------------
# Step 2: find candidate pairs -- same orientation, collinear within
# tolerance, gap under MAX_GAP.
# ------------------------------------------------------------------
def find_candidate_pairs(groups):
    gids = list(groups.keys())
    candidates = []
    for i in range(len(gids)):
        for j in range(i + 1, len(gids)):
            g1, g2 = groups[gids[i]], groups[gids[j]]
            if g1['orient'] != g2['orient']:
                continue
            best = None
            for p1 in g1['ends']:
                for p2 in g2['ends']:
                    if g1['orient'] == 'V' and abs(p1[0] - p2[0]) > AXIS_ALIGN_TOL:
                        continue
                    if g1['orient'] == 'H' and abs(p1[1] - p2[1]) > AXIS_ALIGN_TOL:
                        continue
                    d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                    if d < 3 or d > MAX_GAP:
                        continue
                    if best is None or d < best[0]:
                        best = (d, p1, p2)
            if best:
                candidates.append((gids[i], gids[j], best[0], best[1], best[2], g1['orient']))
    return candidates


# ------------------------------------------------------------------
# Step 3: classify the gap by sampling a narrow band (matched to
# stroke width) between the two facing endpoints and running
# connected-component analysis on the ink inside it.
# ------------------------------------------------------------------
def classify_gap(binary, p1, p2, orient, stroke_width=STROKE_WIDTH):
    band = int(stroke_width * BAND_MULTIPLIER)
    H, W = binary.shape
    x1, y1 = p1
    x2, y2 = p2

    if orient == 'V':
        lo, hi = sorted([int(y1), int(y2)])
        gap_len = hi - lo
        xc = int((x1 + x2) / 2)
        x0, x1b = max(0, xc - band // 2), min(W, xc + band // 2)
        y0, y1b = max(0, lo), min(H, hi)
        patch = binary[y0:y1b, x0:x1b]
        long_axis = 0   # rows
    else:
        lo, hi = sorted([int(x1), int(x2)])
        gap_len = hi - lo
        yc = int((y1 + y2) / 2)
        y0, y1b = max(0, yc - band // 2), min(H, yc + band // 2)
        x0, x1b = max(0, lo), min(W, hi)
        patch = binary[y0:y1b, x0:x1b]
        long_axis = 1   # cols

    if patch.size == 0 or gap_len <= 0:
        return 'NONE', None

    n, labels, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=8)
    if n <= 1:
        return 'NONE', None

    best_blob = None
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        span = h if long_axis == 0 else w   # extent along the gap direction
        if best_blob is None or area > best_blob[2]:
            best_blob = (w, h, area, span)

    if best_blob is None:
        return 'NONE', None

    w, h, area, span = best_blob
    if span >= gap_len * CONTINUOUS_FRACTION and max(w, h) <= band * 1.5:
        return 'CONTINUOUS', best_blob
    if w <= TEXT_BLOB_MAX_SIDE and h <= TEXT_BLOB_MAX_SIDE:
        return 'TEXT_ONLY', best_blob
    return 'SYMBOL', best_blob


# ------------------------------------------------------------------
# Step 4: union-find merge over groups whose gap classified as
# CONTINUOUS or SYMBOL.
# ------------------------------------------------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def run(lines_csv, pdf_path, out_csv, stroke_width=None):
    df = pd.read_csv(lines_csv)
    binary = build_binary_mask(pdf_path)

    if stroke_width is None:
        stroke_width = estimate_stroke_width(binary, df)
        print(f"Estimated stroke width: {stroke_width}px")

    groups = build_simple_groups(df)
    print(f"Simple (2-free-end) groups eligible for merging: {len(groups)} / {df.Group_ID.nunique()}")

    candidates = find_candidate_pairs(groups)
    print(f"Candidate aligned close pairs: {len(candidates)}")

    uf = UnionFind(df.Group_ID.unique().tolist())
    log = []
    for gid1, gid2, gap, p1, p2, orient in candidates:
        cls, blob = classify_gap(binary, p1, p2, orient, stroke_width)
        merged = cls in ('CONTINUOUS', 'SYMBOL')
        if merged:
            uf.union(gid1, gid2)
        log.append({
            'group_1': gid1, 'group_2': gid2, 'gap_px': round(gap, 1),
            'orientation': orient, 'classification': cls, 'merged': merged
        })

    log_df = pd.DataFrame(log)
    print("\nGap classification breakdown:")
    print(log_df.classification.value_counts())
    print(f"\nMerged pairs: {log_df.merged.sum()} / {len(log_df)}")

    df['Merged_Line_ID'] = df.Group_ID.map(uf.find)
    print(f"\nFinal logical lines after merge: {df.Merged_Line_ID.nunique()} "
          f"(was {df.Group_ID.nunique()})")

    df.to_csv(out_csv, index=False)
    log_df.to_csv(out_csv.replace('.csv', '_decisions.csv'), index=False)
    print(f"\nSaved merged lines to {out_csv}")
    print(f"Saved per-pair decisions to {out_csv.replace('.csv', '_decisions.csv')} "
          f"-- review this to sanity-check the classifier before trusting it blindly.")

    return df, log_df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lines', required=True)
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stroke-width', type=int, default=None)
    args = ap.parse_args()
    run(args.lines, args.pdf, args.out, args.stroke_width)
