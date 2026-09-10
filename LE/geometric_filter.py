import os
import ast
import pandas as pd
import sys

sys.path.append(r'C:\Users\user\Desktop\Work\p&id\LE')
from line_tag_merge_4 import LineTagMerger
import config_line_tag_merge as cfg

CE_CSV_PATH = r'C:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\02.Unmarked.csv'
LINE_CSV_PATH = r'C:\Users\user\Desktop\Work\p&id\LE\Results\LE_csv\02.Unmarked_line_coordinates_grouped.csv'
OUT_CSV = r'C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_geometric_filtered_lines.csv'

PADDING = 5

def point_in_bbox(x, y, bbox):
    x0, y0, x1, y1 = bbox
    return (x >= min(x0, x1) - PADDING) and (x <= max(x0, x1) + PADDING) and \
           (y >= min(y0, y1) - PADDING) and (y <= max(y0, y1) + PADDING)

def main():
    print("Initializing LineTagMerger to reuse coordinate transforms...")
    merger = LineTagMerger(cfg)
    merger._build_coordinate_transform()
    
    print("Loading TE Boxes...")
    tags_df = pd.read_csv(cfg.TAG_CSV_PATH)
    if getattr(cfg, "TARGET_FILE_NAME", None) and "file_name" in tags_df.columns:
        tags_df = tags_df[tags_df["file_name"] == cfg.TARGET_FILE_NAME]
    excluded = getattr(cfg, "EXCLUDED_TAG_TYPES", None) or []
    if excluded:
        tags_df = tags_df[~tags_df["tag_type"].isin(excluded)]
    
    te_bboxes = []
    for _, row in tags_df.iterrows():
        b = merger._parse_bbox(row["bbox"])
        px_b = merger._transform_bbox(b)
        te_bboxes.append(px_b)
        
    print(f"Loaded {len(te_bboxes)} text bounding boxes.")

    print("Loading CE Boxes...")
    ce_df = pd.read_csv(CE_CSV_PATH)
    ce_bboxes = []
    for _, row in ce_df.iterrows():
        try:
            bbox_str = row['bounding_box']
            if pd.isna(bbox_str): continue
            if isinstance(bbox_str, str): b = ast.literal_eval(bbox_str)
            else: b = bbox_str
            px_b = merger._transform_bbox(b)
            ce_bboxes.append(px_b)
        except Exception:
            continue
            
    print(f"Loaded {len(ce_bboxes)} equipment bounding boxes.")
    
    all_bboxes = te_bboxes + ce_bboxes
    
    print("Loading Lines...")
    lines_df = pd.read_csv(LINE_CSV_PATH)
    initial_segments = len(lines_df)
    
    # 1. Masking Step
    valid_indices = []
    for idx, row in lines_df.iterrows():
        x1, y1 = row['x_start'], row['y_start']
        x2, y2 = row['x_end'], row['y_end']
        
        is_inside = False
        for bbox in all_bboxes:
            if point_in_bbox(x1, y1, bbox) and point_in_bbox(x2, y2, bbox):
                is_inside = True
                break
                
        if not is_inside:
            valid_indices.append(idx)
            
    filtered_df = lines_df.loc[valid_indices].copy()
    dropped_segs = initial_segments - len(filtered_df)
    print(f"Dropped {dropped_segs} segments that were entirely inside components/text.")
    
    # 2. Border Filter
    valid_groups = []
    groups = filtered_df.groupby('Group_ID')
    for gid, group in groups:
        min_x = group[['x_start', 'x_end']].min().min()
        max_x = group[['x_start', 'x_end']].max().max()
        min_y = group[['y_start', 'y_end']].min().min()
        max_y = group[['y_start', 'y_end']].max().max()
        
        w = max_x - min_x
        h = max_y - min_y
        
        # Removed the broken border filter that was deleting main headers
        valid_groups.append(gid)
            
    final_df = filtered_df[filtered_df['Group_ID'].isin(valid_groups)].copy()
    print(f"Final valid groups after border drop: {len(valid_groups)}")
    
    final_df.to_csv(OUT_CSV, index=False)
    print(f"Saved geometrically masked lines to {OUT_CSV}")

if __name__ == '__main__':
    main()
