import os
import sys
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment
import fitz

# Ensure imports work from LE directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from line_tag_merge_4 import LineTagMerger
import config_line_tag_merge as cfg

# For ink-blob logic
from gap_merge import build_binary_mask

# Parameters
PDF_PATH = r"C:\Users\user\Desktop\Work\p&id\data\raw\02.Unmarked.pdf"
FRAGMENTS_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_gap_merged_lines.csv"
TAGS_CSV = r"C:\Users\user\Desktop\Work\p&id\TE\Results\TE_csv\process_tags.csv"
OUT_LOG = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\decision_log.csv"
OUT_UNMATCHED = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\unmatched_fragments.csv"
OUT_ORPHANS = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\orphan_tags.csv"
OUT_RESULTS = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_line_tag_associated.csv"

# Thresholds (in PDF points, will be scaled)
TIGHT_RADIUS_PT = 40
WIDE_RADIUS_PT = 100
LEADER_WIDTH = 4  # px, estimated width of a thin leader line
BAND_MULTIPLIER = 4 # Corridor width for leader lines

def _dist_point_to_segment(px, py, x1, y1, x2, y2):
    """Shortest distance from point (px, py) to line segment (x1, y1) - (x2, y2)."""
    seg_len_sq = (x2 - x1)**2 + (y2 - y1)**2
    if seg_len_sq == 0:
        return np.hypot(px - x1, py - y1)
    
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / seg_len_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return np.hypot(px - proj_x, py - proj_y)

def detect_leader_line(binary, p1, p2):
    """
    Sample corridor between p1 and p2 and check for continuous ink
    at leader-line width.
    """
    x1, y1 = p1
    x2, y2 = p2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 5 or dist > 300:
        return False
        
    band = int(LEADER_WIDTH * BAND_MULTIPLIER)
    H, W = binary.shape
    
    # Calculate angle
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # Extract ROI to rotate
    cx, cy = int(center[0]), int(center[1])
    roi_w = int(dist) + 20
    roi_h = roi_w
    x0, y0 = max(0, cx - roi_w), max(0, cy - roi_h)
    x1b, y1b = min(W, cx + roi_w), min(H, cy + roi_h)
    
    if x1b <= x0 or y1b <= y0:
        return False
        
    roi = binary[y0:y1b, x0:x1b]
    roi_center = (cx - x0, cy - y0)
    
    M_roi = cv2.getRotationMatrix2D(roi_center, angle, 1.0)
    rotated = cv2.warpAffine(roi, M_roi, (roi.shape[1], roi.shape[0]))
    
    corridor_y0 = max(0, int(roi_center[1] - band / 2))
    corridor_y1 = min(rotated.shape[0], int(roi_center[1] + band / 2))
    corridor_x0 = max(0, int(roi_center[0] - dist / 2))
    corridor_x1 = min(rotated.shape[1], int(roi_center[0] + dist / 2))
    
    corridor = rotated[corridor_y0:corridor_y1, corridor_x0:corridor_x1]
    
    if corridor.size == 0:
        return False
        
    n, labels, stats, _ = cv2.connectedComponentsWithStats(corridor, connectivity=8)
    if n <= 1:
        return False
        
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w >= (dist * 0.75) and h <= (LEADER_WIDTH * 3):
            return True
            
    return False

def main():
    print("--- Line-to-Tag Association ---")
    
    # 1. Initialize & Coordinate Sync
    merger = LineTagMerger(cfg)
    merger._build_coordinate_transform()
    print(f"Page Rotation: {merger._page_rotation} degrees")
    scale = merger._dpi_scale
    print(f"DPI Scale: {scale:.4f}")
    
    tight_px = TIGHT_RADIUS_PT * scale
    wide_px = WIDE_RADIUS_PT * scale
    print(f"Thresholds: {TIGHT_RADIUS_PT}pt -> {tight_px:.1f}px | {WIDE_RADIUS_PT}pt -> {wide_px:.1f}px")
    
    # 2. Load Data
    print("Loading fragments and tags...")
    lines_df = pd.read_csv(FRAGMENTS_CSV)
    tags_df = pd.read_csv(TAGS_CSV)
    
    # 3. Filter Tags (Rule 2)
    tags_df = tags_df[tags_df['tag_type'] == 'Line Tag']
    tags_df = tags_df[tags_df['file_name'] == '02.Unmarked'].copy()
    
    # Transform Tag Bounding Boxes
    def parse_bbox(b):
        return tuple(map(float, b.strip("()").split(",")))
        
    tags_df["_bbox_parsed"] = tags_df["bbox"].apply(lambda b: merger._transform_bbox(parse_bbox(b)))
    tags_df["_cx"] = tags_df["_bbox_parsed"].apply(lambda b: (b[0] + b[2]) / 2)
    tags_df["_cy"] = tags_df["_bbox_parsed"].apply(lambda b: (b[1] + b[3]) / 2)
    
    # Load Binary Mask for Arrow-Link Detection
    print("Building binary mask for leader line detection...")
    binary = build_binary_mask(PDF_PATH)
    
    group_col = 'Merged_Line_ID' if 'Merged_Line_ID' in lines_df.columns else 'Group_ID'
    fragments = lines_df.groupby(group_col)
    
    frag_ids = list(fragments.groups.keys())
    tag_ids = tags_df.index.tolist()
    
    N = len(frag_ids)
    M = len(tag_ids)
    
    dist_matrix = np.full((N, M), np.inf)
    arrow_matrix = np.full((N, M), False)
    
    print(f"Calculating distance matrix for {N} fragments and {M} tags...")
    for i, f_id in enumerate(frag_ids):
        frag_data = fragments.get_group(f_id)
        
        for j, t_idx in enumerate(tag_ids):
            tx, ty = tags_df.at[t_idx, '_cx'], tags_df.at[t_idx, '_cy']
            
            min_d = np.inf
            closest_pt_on_line = None
            
            for _, seg in frag_data.iterrows():
                d = _dist_point_to_segment(tx, ty, seg['x_start'], seg['y_start'], seg['x_end'], seg['y_end'])
                if d < min_d:
                    min_d = d
                    closest_pt_on_line = ( (seg['x_start']+seg['x_end'])/2, (seg['y_start']+seg['y_end'])/2 )
            
            dist_matrix[i, j] = min_d
            
            # Rule 9: Arrow Link Detection
            if 15 < min_d < wide_px + 50:
                if detect_leader_line(binary, (tx, ty), closest_pt_on_line):
                    arrow_matrix[i, j] = True

    matched_fragments = set()
    matched_tags = set()
    decision_log = []
    
    # --- ARROW LINK PASS ---
    for i in range(N):
        for j in range(M):
            if arrow_matrix[i, j] and i not in matched_fragments and j not in matched_tags:
                matched_fragments.add(i)
                matched_tags.add(j)
                decision_log.append({
                    'Fragment_ID': frag_ids[i],
                    'Matched_Tag': tags_df.at[tag_ids[j], 'text'],
                    'Distance_px': dist_matrix[i, j],
                    'Pass': 'Arrow',
                    'Confidence': 'Confirmed'
                })
    
    print(f"Matched via Arrow/Leader Line: {len([x for x in decision_log if x['Pass'] == 'Arrow'])}")
    
    # --- PASS 1 (TIGHT) ---
    cost_matrix_1 = np.copy(dist_matrix)
    cost_matrix_1[cost_matrix_1 > tight_px] = 1e9
    
    for i in matched_fragments: cost_matrix_1[i, :] = 1e9
    for j in matched_tags: cost_matrix_1[:, j] = 1e9
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix_1)
    pass1_count = 0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix_1[r, c] < 1e9:
            matched_fragments.add(r)
            matched_tags.add(c)
            decision_log.append({
                'Fragment_ID': frag_ids[r],
                'Matched_Tag': tags_df.at[tag_ids[c], 'text'],
                'Distance_px': dist_matrix[r, c],
                'Pass': 'Pass 1 (Tight)',
                'Confidence': 'High'
            })
            pass1_count += 1
            
    print(f"Matched via Pass 1: {pass1_count}")
    
    # --- PASS 2 (WIDE) ---
    cost_matrix_2 = np.copy(dist_matrix)
    cost_matrix_2[cost_matrix_2 > wide_px] = 1e9
    
    for i in matched_fragments: cost_matrix_2[i, :] = 1e9
    for j in matched_tags: cost_matrix_2[:, j] = 1e9
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix_2)
    pass2_count = 0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix_2[r, c] < 1e9:
            matched_fragments.add(r)
            matched_tags.add(c)
            decision_log.append({
                'Fragment_ID': frag_ids[r],
                'Matched_Tag': tags_df.at[tag_ids[c], 'text'],
                'Distance_px': dist_matrix[r, c],
                'Pass': 'Pass 2 (Wide)',
                'Confidence': 'Medium'
            })
            pass2_count += 1
            
    print(f"Matched via Pass 2: {pass2_count}")
    
    # --- UNMATCHED & ORPHANS ---
    unmatched_frags = [frag_ids[i] for i in range(N) if i not in matched_fragments]
    orphan_tags_idx = [tag_ids[j] for j in range(M) if j not in matched_tags]
    orphan_tags_list = [tags_df.at[j, 'text'] for j in orphan_tags_idx]
    
    print(f"Unmatched Fragments: {len(unmatched_frags)}")
    print(f"Orphan Tags: {len(orphan_tags_list)}")
    
    # --- OUTPUT LOGS ---
    pd.DataFrame(decision_log).to_csv(OUT_LOG, index=False)
    pd.DataFrame({'Fragment_ID': unmatched_frags}).to_csv(OUT_UNMATCHED, index=False)
    pd.DataFrame({'Orphan_Tag': orphan_tags_list}).to_csv(OUT_ORPHANS, index=False)
    
    print(f"Saved decision log to {OUT_LOG}")
    print(f"Saved unmatched to {OUT_UNMATCHED}")
    print(f"Saved orphans to {OUT_ORPHANS}")

    # Generate Final Results CSV
    lines_with_tags = lines_df.copy()
    tag_lookup = {row['Fragment_ID']: row['Matched_Tag'] for row in decision_log}
    lines_with_tags['Matched_Line_Tag'] = lines_with_tags[group_col].map(tag_lookup)
    lines_with_tags.to_csv(OUT_RESULTS, index=False)

if __name__ == '__main__':
    main()
