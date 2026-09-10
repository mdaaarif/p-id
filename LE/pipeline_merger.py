import pandas as pd
import numpy as np
import networkx as nx
import ast
import sys

sys.path.append(r'C:\Users\user\Desktop\Work\p&id\LE')
from line_tag_merge_src import LineTagMerger
import config_line_tag_merge as cfg

SEG_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_line_tag_associated.csv"
CE_CSV = r"C:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\02.Unmarked.csv"
OUT_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_SUPER_pipelines.csv"

MAJOR_TYPES = {
    'Vessel', 'Tank', 'Pump / Compressor', 'Heat Exchanger', 
    'Drum', 'Compressor', 'Complex Equipment'
}

def point_near_bbox(x, y, bbox, padding):
    x0, y0, x1, y1 = bbox
    return (x >= min(x0, x1) - padding) and (x <= max(x0, x1) + padding) and \
           (y >= min(y0, y1) - padding) and (y <= max(y0, y1) + padding)

def get_orientation(row):
    w = abs(row['x_end'] - row['x_start'])
    h = abs(row['y_end'] - row['y_start'])
    if w > h * 2: return 'H'
    if h > w * 2: return 'V'
    return 'U'

def are_collinear_and_close(row_a, row_b, snap_tol, gap_tol):
    ori_a = get_orientation(row_a)
    ori_b = get_orientation(row_b)
    
    if ori_a != ori_b or ori_a == 'U':
        return False
        
    if ori_a == 'H':
        y_diff = abs(row_a['y_start'] - row_b['y_start'])
        if y_diff > snap_tol: return False
        
        min_x = min(row_a['x_start'], row_a['x_end'], row_b['x_start'], row_b['x_end'])
        max_x = max(row_a['x_start'], row_a['x_end'], row_b['x_start'], row_b['x_end'])
        len_a = abs(row_a['x_end'] - row_a['x_start'])
        len_b = abs(row_b['x_end'] - row_b['x_start'])
        gap = max_x - min_x - len_a - len_b
        return gap < gap_tol
        
    if ori_a == 'V':
        x_diff = abs(row_a['x_start'] - row_b['x_start'])
        if x_diff > snap_tol: return False
        
        min_y = min(row_a['y_start'], row_a['y_end'], row_b['y_start'], row_b['y_end'])
        max_y = max(row_a['y_start'], row_a['y_end'], row_b['y_start'], row_b['y_end'])
        len_a = abs(row_a['y_end'] - row_a['y_start'])
        len_b = abs(row_b['y_end'] - row_b['y_start'])
        gap = max_y - min_y - len_a - len_b
        return gap < gap_tol

    return False

def endpoint_touches_segment(end_pt, seg_start, seg_end, tol):
    px, py = end_pt
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    # Check bounding box
    if not (min(x1, x2) - tol <= px <= max(x1, x2) + tol and min(y1, y2) - tol <= py <= max(y1, y2) + tol):
        return False
        
    # Check distance to line segment
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2) <= tol
        
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2) <= tol

def points_touch(row_a, row_b, threshold=20):
    pts_a = [(row_a['x_start'], row_a['y_start']), (row_a['x_end'], row_a['y_end'])]
    pts_b = [(row_b['x_start'], row_b['y_start']), (row_b['x_end'], row_b['y_end'])]
    for pa in pts_a:
        for pb in pts_b:
            dist = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
            if dist < threshold:
                return True
    return False

def main():
    print("Loading valid pipelines...")
    # Build group tags dictionary
    group_tags = {}
    
    seg_df = pd.read_csv(SEG_CSV)
    
    # We now group by Merged_Line_ID
    if 'Merged_Line_ID' in seg_df.columns:
        group_col = 'Merged_Line_ID'
    else:
        group_col = 'Group_ID'
        
    valid_gids = seg_df[group_col].unique()

    for gid in valid_gids:
        group_data = seg_df[seg_df[group_col] == gid]
        tag = group_data['Matched_Line_Tag'].iloc[0] if 'Matched_Line_Tag' in group_data.columns else None
        if pd.notna(tag) and str(tag).strip() != '':
            group_tags[gid] = str(tag).strip()
        else:
            group_tags[gid] = None
    def tags_compatible(gid_a, gid_b):
        tag_a = group_tags.get(gid_a)
        tag_b = group_tags.get(gid_b)
        if tag_a is None or tag_b is None:
            return True # Safe to merge if one or both are untagged
        return tag_a == tag_b # Refuse to merge if tags contradict
    
    # Filter valid lines
    seg_df = seg_df[seg_df[group_col].isin(valid_gids)].copy()
    
    max_x = max(seg_df['x_start'].max(), seg_df['x_end'].max())
    max_y = max(seg_df['y_start'].max(), seg_df['y_end'].max())
    diag = np.sqrt(max_x**2 + max_y**2)
    
    PADDING = 0.005 * diag
    SNAP_TOL = 0.0015 * diag
    GAP_TOL = 0.0 * diag  # Disabled: gap_merge.py handles this via raster ink
    TOUCH_TOL = 0.002 * diag
    
    # Load CE boxes
    print("Loading INLINE Equipment boxes...")
    merger = LineTagMerger(cfg)
    merger._build_coordinate_transform()
    ce_df = pd.read_csv(CE_CSV)
    inline_boxes = []
    
    for _, row in ce_df.iterrows():
        ce_type = row.get('type', 'Unknown')
        if pd.isna(ce_type) or ce_type in MAJOR_TYPES:
            continue
        try:
            bbox_str = row['bounding_box']
            if pd.isna(bbox_str): continue
            if isinstance(bbox_str, str): b = ast.literal_eval(bbox_str)
            else: b = bbox_str
            px_b = merger._transform_bbox(b)
            inline_boxes.append(px_b)
        except Exception:
            continue
            
    print(f"Found {len(inline_boxes)} INLINE equipment bounding boxes.")

    # Map groups to inline boxes
    group_to_boxes = {gid: set() for gid in valid_gids}
    for _, row in seg_df.iterrows():
        gid = row[group_col]
        x1, y1 = row['x_start'], row['y_start']
        x2, y2 = row['x_end'], row['y_end']
        for i, bbox in enumerate(inline_boxes):
            if point_near_bbox(x1, y1, bbox, PADDING) or point_near_bbox(x2, y2, bbox, PADDING):
                group_to_boxes[gid].add(i)

    # We will collect all valid geometric edges, then merge them safely using DSU
    candidate_edges = []
    
    # 1. CE Box Bridging
    for i in range(len(inline_boxes)):
        touching_groups = [gid for gid, boxes in group_to_boxes.items() if i in boxes]
        for j in range(len(touching_groups)):
            for k in range(j + 1, len(touching_groups)):
                bx0, by0, bx1, by1 = inline_boxes[i]
                cx, cy = (bx0 + bx1)/2, (by0 + by1)/2
                candidate_edges.append({
                    'u': touching_groups[j], 'v': touching_groups[k],
                    'x': cx, 'y': cy
                })

    # 2. Geometric Collinearity, Touch, & T-Junction Bridging
    print("Finding geometric intersections (collinear, touching, & T-junctions)...")
    groups = seg_df.groupby(group_col)
    group_list = list(groups)
    
    for i in range(len(group_list)):
        gid_a, df_a = group_list[i]
        for j in range(i + 1, len(group_list)):
            gid_b, df_b = group_list[j]
            
            merged = False
            for _, row_a in df_a.iterrows():
                pts_a = [(row_a['x_start'], row_a['y_start']), (row_a['x_end'], row_a['y_end'])]
                for _, row_b in df_b.iterrows():
                    pt_b1 = (row_b['x_start'], row_b['y_start'])
                    pt_b2 = (row_b['x_end'], row_b['y_end'])
                    
                    is_touching = False
                    ref_x, ref_y = 0.0, 0.0
                    
                    if points_touch(row_a, row_b, TOUCH_TOL):
                        is_touching = True
                        ref_x = (row_a['x_start'] + row_b['x_start']) / 2
                        ref_y = (row_a['y_start'] + row_b['y_start']) / 2
                    else:
                        # Check T-junctions
                        for pa in pts_a:
                            if endpoint_touches_segment(pa, pt_b1, pt_b2, TOUCH_TOL):
                                is_touching = True
                                ref_x, ref_y = pa
                        if not is_touching:
                            for pb in [pt_b1, pt_b2]:
                                if endpoint_touches_segment(pb, pts_a[0], pts_a[1], TOUCH_TOL):
                                    is_touching = True
                                    ref_x, ref_y = pb
                                
                    if is_touching:
                        candidate_edges.append({
                            'u': gid_a, 'v': gid_b,
                            'x': ref_x, 'y': ref_y
                        })
                        merged = True
                        break
                if merged: break
                
    # --- Constraint-based DSU ---
    parent = {n: n for n in valid_gids}
    comp_tag = {n: group_tags[n] for n in valid_gids}

    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            tag_i = comp_tag[root_i]
            tag_j = comp_tag[root_j]
            
            if tag_i is not None and tag_j is not None and tag_i != tag_j:
                return False
                
            parent[root_i] = root_j
            if tag_j is None and tag_i is not None:
                comp_tag[root_j] = tag_i
            return True
        return True

    rejected_edges = []
    accepted_edges = []
    
    def edge_priority(edge):
        u, v = edge['u'], edge['v']
        t1, t2 = group_tags[u], group_tags[v]
        if t1 is not None and t2 is not None:
            return 0 if t1 == t2 else 3
        if t1 is None and t2 is None:
            return 2
        return 1

    candidate_edges.sort(key=edge_priority)
    
    for edge in candidate_edges:
        if not union(edge['u'], edge['v']):
            rejected_edges.append(edge)
        else:
            accepted_edges.append(edge)
            
    components_dict = {}
    for n in valid_gids:
        root = find(n)
        if root not in components_dict: components_dict[root] = []
        components_dict[root].append(n)
        
    components = list(components_dict.values())
    print(f"Merged {len(valid_gids)} groups into {len(components)} SUPER pipelines!")
    
    group_to_super = {}
    sp_confidences = {}
    
    for i, comp in enumerate(components):
        super_id = f"SP_{i:03d}"
        for gid in comp:
            group_to_super[gid] = super_id
            
        comp_set = set(comp)
        has_confirmed = False
        for edge in accepted_edges:
            u, v = edge['u'], edge['v']
            if u in comp_set and v in comp_set:
                t1, t2 = group_tags[u], group_tags[v]
                if t1 is not None and t2 is not None and t1 == t2:
                    has_confirmed = True
                    break
                    
        if len(comp) == 1:
            conf = "Single Group"
        elif has_confirmed:
            conf = "Confirmed"
        else:
            conf = "Unverified"
            
        sp_confidences[super_id] = conf
            
    seg_df['Super_Pipeline_ID'] = seg_df[group_col].map(group_to_super)
    seg_df['Merge_Confidence'] = seg_df['Super_Pipeline_ID'].map(sp_confidences)
    
    branch_conn_data = []
    seen = set()
    for edge in rejected_edges:
        gid_a, gid_b = edge['u'], edge['v']
        sp_a = group_to_super.get(gid_a)
        sp_b = group_to_super.get(gid_b)
        
        pair_key = tuple(sorted([sp_a, sp_b]))
        if sp_a and sp_b and sp_a != sp_b and pair_key not in seen:
            seen.add(pair_key)
            branch_conn_data.append({
                'SP_1': sp_a,
                'SP_2': sp_b,
                'Tag_1': group_tags.get(gid_a),
                'Tag_2': group_tags.get(gid_b),
                'Refusal_X': round(edge['x'], 1),
                'Refusal_Y': round(edge['y'], 1)
            })
    
    branch_df = pd.DataFrame(branch_conn_data)
    if len(branch_df) > 0:
        branch_df = branch_df.drop_duplicates()
        branch_csv = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_branch_connections.csv"
        branch_df.to_csv(branch_csv, index=False)
        print(f"Saved {len(branch_df)} rejected branch connections to {branch_csv}")
    
    seg_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

if __name__ == '__main__':
    main()
