import pandas as pd
import numpy as np

CSV_PATH = r"C:\Users\user\Desktop\Work\p&id\LE\Results\LE_csv\02.Unmarked_line_tag_topology.csv"
OUT_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_FINAL_pipeline_topology.csv"

def is_valid_tag(val):
    if pd.isna(val) or val == 'OPEN':
        return False
    return True

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} merged pipelines.")
    
    # Load raw segments to compute perimeter ratio
    SEG_CSV = r"C:\Users\user\Desktop\Work\p&id\Results\LE_csv\02.Unmarked_geometric_filtered_lines.csv"
    seg_df = pd.read_csv(SEG_CSV)
    
    max_x = max(seg_df['x_start'].max(), seg_df['x_end'].max())
    max_y = max(seg_df['y_start'].max(), seg_df['y_end'].max())
    diag = np.sqrt(max_x**2 + max_y**2)
    
    # Dynamic thresholds based on drawing size
    SNAP_TOL = 0.002 * diag  # equivalent to 20px on an 8k drawing
    NOISE_LEN = 0.04 * diag  # equivalent to 400px on an 8k drawing
    
    # Precompute ratios using connected components of segments
    import networkx as nx
    
    G = nx.Graph()
    for idx, r in seg_df.iterrows():
        G.add_node(idx, row=r)
        
    for i, r1 in seg_df.iterrows():
        for j, r2 in seg_df.iterrows():
            if i >= j: continue
            
            if not (min(r1['x_start'], r1['x_end']) - SNAP_TOL <= max(r2['x_start'], r2['x_end']) and
                    max(r1['x_start'], r1['x_end']) + SNAP_TOL >= min(r2['x_start'], r2['x_end']) and
                    min(r1['y_start'], r1['y_end']) - SNAP_TOL <= max(r2['y_start'], r2['y_end']) and
                    max(r1['y_start'], r1['y_end']) + SNAP_TOL >= min(r2['y_start'], r2['y_end'])):
                continue

            pts1 = [(r1['x_start'], r1['y_start']), (r1['x_end'], r1['y_end'])]
            pts2 = [(r2['x_start'], r2['y_start']), (r2['x_end'], r2['y_end'])]
            
            close = False
            for p1 in pts1:
                for p2 in pts2:
                    if (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 < SNAP_TOL**2:
                        close = True
                        break
                if close: break
            if close:
                G.add_edge(i, j)
                
    components = list(nx.connected_components(G))
    
    group_ratios = {}
    for comp in components:
        comp_segs = seg_df.loc[list(comp)]
        min_x = min(comp_segs['x_start'].min(), comp_segs['x_end'].min())
        max_x = max(comp_segs['x_start'].max(), comp_segs['x_end'].max())
        min_y = min(comp_segs['y_start'].min(), comp_segs['y_end'].min())
        max_y = max(comp_segs['y_start'].max(), comp_segs['y_end'].max())
        
        w = max_x - min_x
        h = max_y - min_y
        if w == 0 or h == 0: 
            ratio = 0.5
        else:
            bbox_perim = 2 * (w + h)
            sum_len = 0
            for _, r in comp_segs.iterrows():
                sum_len += ((r['x_end']-r['x_start'])**2 + (r['y_end']-r['y_start'])**2)**0.5
            ratio = sum_len / bbox_perim
            
        for gid in comp_segs['Group_ID'].unique():
            group_ratios[gid] = max(group_ratios.get(gid, 0), ratio)
    
    valid_indices = []
    
    for idx, row in df.iterrows():
        # Check if it's an off-page connector box/U-shape unconditionally
        gid = row['Line_ID']
        ratio = group_ratios.get(gid, 0.5)
        
        if ratio > 0.75:
            # Drop boxes unconditionally, even if they accidentally matched a tag
            continue
            
        has_tag = False
        
        # Check for Line Tag
        if is_valid_tag(row.get('line_tag')):
            has_tag = True
            
        # Check for Equipment Tag at start/end
        if is_valid_tag(row.get('tag_start')) or is_valid_tag(row.get('tag_end')):
            has_tag = True
            
        if has_tag:
            valid_indices.append(idx)
            continue
            
        # Untagged line. Check length.
        dx = row['x_end'] - row['x_start']
        dy = row['y_end'] - row['y_start']
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Keep if it's long enough
        if displacement >= NOISE_LEN:
            valid_indices.append(idx)
            
    filtered_df = df.loc[valid_indices].copy()
    dropped = len(df) - len(filtered_df)
    
    print(f"Dropped {dropped} untagged noise lines (displacement < {NOISE_LEN:.1f}px or box ratio > 0.75).")
    print(f"Final true pipelines kept: {len(filtered_df)}")
    
    filtered_df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")

if __name__ == '__main__':
    main()
