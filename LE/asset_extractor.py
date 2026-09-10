import pandas as pd
import numpy as np
import ast
import re
import sys

sys.path.append(r'C:\Users\user\Desktop\Work\p&id\LE')
from line_tag_merge_src import LineTagMerger
import config_line_tag_merge as cfg

SP_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_SUPER_pipelines.csv"
TOPO_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_FINAL_pipeline_topology.csv"
BRANCH_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_branch_connections.csv"
TE_CSV = r"C:\Users\user\Desktop\Work\p&id\TE\Results\TE_csv\process_tags.csv"
CE_CSV = r"C:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\02.Unmarked.csv"
OUT_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_Asset_List.csv"

MAJOR_TYPES = {
    'Vessel', 'Tank', 'Pump / Compressor', 'Heat Exchanger', 
    'Drum', 'Compressor', 'Complex Equipment'
}

def point_in_bbox(x, y, bbox, padding=50):
    x0, y0, x1, y1 = bbox
    return (x >= min(x0, x1) - padding) and (x <= max(x0, x1) + padding) and \
           (y >= min(y0, y1) - padding) and (y <= max(y0, y1) + padding)

def bbox_intersects(bbox1, bbox2, padding=0):
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    min_x1, max_x1 = min(x0_1, x1_1) - padding, max(x0_1, x1_1) + padding
    min_y1, max_y1 = min(y0_1, y1_1) - padding, max(y0_1, y1_1) + padding
    min_x2, max_x2 = min(x0_2, x1_2), max(x0_2, x1_2)
    min_y2, max_y2 = min(y0_2, y1_2), max(y0_2, y1_2)
    if max_x1 < min_x2 or min_x1 > max_x2: return False
    if max_y1 < min_y2 or min_y1 > max_y2: return False
    return True

def bbox_intersects_segment(bbox, x1, y1, x2, y2, padding=30):
    bx1, by1, bx2, by2 = bbox
    min_x, max_x = min(bx1, bx2) - padding, max(bx1, bx2) + padding
    min_y, max_y = min(by1, by2) - padding, max(by1, by2) + padding
    
    # Simple bounding box check first
    if min(x1, x2) > max_x or max(x1, x2) < min_x: return False
    if min(y1, y2) > max_y or max(y1, y2) < min_y: return False
        
    if x1 == x2: 
        if min_x <= x1 <= max_x and min(y1, y2) <= max_y and max(y1, y2) >= min_y: return True
    if y1 == y2:
        if min_y <= y1 <= max_y and min(x1, x2) <= max_x and max(x1, x2) >= min_x: return True
        
    return False

def transform_bbox(bbox, scale, w_pts, h_pts, rot):
    x0, y0, x1, y1 = bbox
    if rot == 90:
        tx0 = (h_pts - y1) * scale
        ty0 = x0 * scale
        tx1 = (h_pts - y0) * scale
        ty1 = x1 * scale
        return tx0, ty0, tx1, ty1
    return x0*scale, y0*scale, x1*scale, y1*scale

def get_equipment_name_near_bbox(ce_bbox, te_df):
    candidates = []
    for _, row in te_df.iterrows():
        if row.get('tag_type') != 'General': continue
        te_bbox = row['px_bbox']
        
        if bbox_intersects(te_bbox, ce_bbox, padding=0):
            text = str(row['text']).strip()
            if re.search(r'\d{3}-[A-Z]-\d{2,3}', text) or re.search(r'\d{3}-[A-Z]-\d+[A-Z]?', text):
                cx, cy = (min(ce_bbox[0], ce_bbox[2]) + max(ce_bbox[0], ce_bbox[2]))/2, (min(ce_bbox[1], ce_bbox[3]) + max(ce_bbox[1], ce_bbox[3]))/2
                tcx, tcy = (min(te_bbox[0], te_bbox[2]) + max(te_bbox[0], te_bbox[2]))/2, (min(te_bbox[1], te_bbox[3]) + max(te_bbox[1], te_bbox[3]))/2
                dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                candidates.append((dist, text))
                
    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None

def get_offpage_connector_text(x, y, te_df):
    candidates = []
    for _, row in te_df.iterrows():
        text = str(row['text']).strip()
        tx0, ty0, tx1, ty1 = row['px_bbox']
        tcx, tcy = (tx0 + tx1)/2, (ty0 + ty1)/2
        
        if point_in_bbox(tcx, tcy, (x, y, x, y), padding=400):
            if text.upper().startswith("TO ") or text.upper().startswith("FROM "):
                dist = np.sqrt((x - tcx)**2 + (y - tcy)**2)
                candidates.append((dist, text))
                
    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None

def get_component_name(ce_bbox, te_df):
    candidates = []
    for _, row in te_df.iterrows():
        if row.get('tag_type') in ['Line Tag', 'Equipment']: continue
        te_bbox = row['px_bbox']
        
        if bbox_intersects(te_bbox, ce_bbox, padding=0):
            text = str(row['text']).strip()
            cx, cy = (min(ce_bbox[0], ce_bbox[2]) + max(ce_bbox[0], ce_bbox[2]))/2, (min(ce_bbox[1], ce_bbox[3]) + max(ce_bbox[1], ce_bbox[3]))/2
            tcx, tcy = (min(te_bbox[0], te_bbox[2]) + max(te_bbox[0], te_bbox[2]))/2, (min(te_bbox[1], te_bbox[3]) + max(te_bbox[1], te_bbox[3]))/2
            dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
            candidates.append((dist, text))
                
    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None

def main():
    print("Loading datasets...")
    sp_df = pd.read_csv(SP_CSV)
    topo_df = pd.read_csv(TOPO_CSV)
    
    try:
        branch_df = pd.read_csv(BRANCH_CSV)
    except:
        branch_df = pd.DataFrame()
    
    te_raw = pd.read_csv(TE_CSV)
    te_df = te_raw[te_raw['file_name'] == '02.Unmarked'].copy()
    ce_df = pd.read_csv(CE_CSV)
    
    import fitz
    doc = fitz.open(cfg.PDF_PATH)
    page = doc[0]
    w_pts = page.cropbox.width
    h_pts = page.cropbox.height
    rot = page.rotation
    scale = 300 / 72.0
    doc.close()

    # Pre-parse and transform TE bounding boxes using fixed CW rotation mapping
    te_bboxes = []
    for idx, row in te_df.iterrows():
        try:
            b = ast.literal_eval(row['bbox'])
            px_bbox = transform_bbox(b, scale, w_pts, h_pts, rot)
            te_bboxes.append(px_bbox)
        except:
            te_bboxes.append((0,0,0,0))
    te_df['px_bbox'] = te_bboxes
            
    ce_data = []
    for idx, row in ce_df.iterrows():
        try:
            b = row['bounding_box']
            if isinstance(b, str):
                b = ast.literal_eval(b)
            
            # Safe NA check for tuples/lists
            is_na = False
            if isinstance(b, (list, tuple)):
                is_na = any(pd.isna(x) for x in b)
            else:
                is_na = pd.isna(b)
                
            if is_na:
                continue
            
            ce_bbox = transform_bbox(b, scale, w_pts, h_pts, rot)
            ce_data.append({
                'type': row.get('type', 'Unknown'),
                'bbox': ce_bbox
            })
        except Exception as e:
            continue
            
    group_tags = {}
    for _, r in topo_df.iterrows():
        if pd.notna(r['line_tag']):
            group_tags[r['Line_ID']] = str(r['line_tag']).strip()
            
    asset_list = []
    
    sp_groups = sp_df.groupby('Super_Pipeline_ID')
    for sp_id, df_group in sp_groups:
        sp_tag = None
        for gid in df_group['Group_ID'].unique():
            if gid in group_tags:
                sp_tag = group_tags[gid]
                break
        if not sp_tag:
            sp_tag = '(Missing Tag)'
            
        pts = []
        for _, r in df_group.iterrows():
            pts.append((r['x_start'], r['y_start']))
            pts.append((r['x_end'], r['y_end']))
            
        pt_counts = {}
        for p in pts:
            pt_counts[p] = pt_counts.get(p, 0) + 1
            
        endpoints = [p for p, c in pt_counts.items() if c == 1]
        if not endpoints:
            endpoints = [pts[0]]
            
        source = None
        dest = None
        
        terms = []
        for ep in endpoints:
            ex, ey = ep
            term_name = None
            
            for ce in ce_data:
                if ce['type'] in MAJOR_TYPES and point_in_bbox(ex, ey, ce['bbox'], padding=50):
                    name = get_equipment_name_near_bbox(ce['bbox'], te_df)
                    term_name = name if name else f"(unidentified {ce['type']})"
                    break
                    
            if not term_name:
                off_page = get_offpage_connector_text(ex, ey, te_df)
                if off_page:
                    term_name = off_page
                    
            if not term_name:
                if not branch_df.empty:
                    for _, b_row in branch_df.iterrows():
                        rx, ry = b_row['Refusal_X'], b_row['Refusal_Y']
                        if np.sqrt((ex - rx)**2 + (ey - ry)**2) < 60:
                            if b_row['SP_1'] in df_group['Group_ID'].values:
                                header_gid = b_row['SP_2']
                                header_sp = sp_df[sp_df['Group_ID'] == header_gid]['Super_Pipeline_ID'].values
                                h_id = header_sp[0] if len(header_sp) > 0 else b_row['Tag_2']
                                term_name = f"Branch off {h_id}"
                                break
                            elif b_row['SP_2'] in df_group['Group_ID'].values:
                                header_gid = b_row['SP_1']
                                header_sp = sp_df[sp_df['Group_ID'] == header_gid]['Super_Pipeline_ID'].values
                                h_id = header_sp[0] if len(header_sp) > 0 else b_row['Tag_1']
                                term_name = f"Branch off {h_id}"
                                break
                        
            if not term_name:
                term_name = "(open end)"
                
            terms.append(term_name)
            
        if len(terms) > 0: source = terms[0]
        if len(terms) > 1: 
            dest = ", ".join(list(set(terms[1:])))
            
        inline_comps = []
        for _, r in df_group.iterrows():
            for ce in ce_data:
                if ce['type'] not in MAJOR_TYPES:
                    if bbox_intersects_segment(ce['bbox'], r['x_start'], r['y_start'], r['x_end'], r['y_end']):
                        name = get_component_name(ce['bbox'], te_df)
                        c_name = name if name else ce['type']
                        inline_comps.append(c_name)
                        
        inline_comps = list(set(inline_comps))
        if not inline_comps:
            inline_comps_str = "None"
        else:
            inline_comps_str = ", ".join(inline_comps)
            
        confidence = df_group['Merge_Confidence'].iloc[0] if 'Merge_Confidence' in df_group.columns else 'Unknown'
            
        asset_list.append({
            'Super_Pipeline_ID': sp_id,
            'Line_Tag': sp_tag,
            'Confidence': confidence,
            'From': source,
            'To': dest,
            'Inline_Components': inline_comps_str
        })
        
    final_df = pd.DataFrame(asset_list)
    final_df.to_csv(OUT_CSV, index=False)
    print(f"Asset List generated: {OUT_CSV}")

if __name__ == '__main__':
    main()
