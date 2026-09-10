import cv2
import numpy as np
import pandas as pd
import ast
import re
import sys
import fitz

sys.path.append(r'C:\Users\user\Desktop\Work\p&id\LE')
from line_tag_merge_src import LineTagMerger
import config_line_tag_merge as cfg

PDF_PATH = r"C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf"
SP_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_SUPER_pipelines.csv"
TOPO_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_FINAL_pipeline_topology.csv"
TE_CSV = r"C:\Users\user\Desktop\Work\p&id\TE\Results\TE_csv\process_tags.csv"
CE_CSV = r"C:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\02.Unmarked.csv"
OUT_IMG = r"C:\Users\user\Desktop\Work\p&id\results\LE_images\02.Unmarked_Tags_Components.jpg"
OUT_PDF = r"C:\Users\user\Desktop\Work\p&id\results\LE_images\02.Unmarked_Tags_Components.pdf"

MAJOR_TYPES = {
    'Vessel', 'Tank', 'Pump / Compressor', 'Heat Exchanger', 
    'Drum', 'Compressor', 'Complex Equipment'
}

def rasterise_pdf(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()

def point_in_bbox(x, y, bbox, padding=50):
    x0, y0, x1, y1 = bbox
    return (x >= min(x0, x1) - padding) and (x <= max(x0, x1) + padding) and \
           (y >= min(y0, y1) - padding) and (y <= max(y0, y1) + padding)

def bbox_intersects_segment(bbox, x1, y1, x2, y2, padding=20):
    x0, y0, bx1, by1 = bbox
    min_x, max_x = min(x0, bx1) - padding, max(x0, bx1) + padding
    min_y, max_y = min(y0, by1) - padding, max(y0, by1) + padding
    
    if (min_x <= x1 <= max_x and min_y <= y1 <= max_y) or (min_x <= x2 <= max_x and min_y <= y2 <= max_y): return True
    if x1 == x2: 
        if min_x <= x1 <= max_x and min(y1, y2) <= max_y and max(y1, y2) >= min_y: return True
    if y1 == y2:
        if min_y <= y1 <= max_y and min(x1, x2) <= max_x and max(x1, x2) >= min_x: return True
    return False

def draw_box(img, bbox, color, label):
    x0, y0, x1, y1 = [int(v) for v in bbox]
    cv2.rectangle(img, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), color, 8)
    cv2.putText(img, label, (min(x0, x1), max(y0, y1) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4, cv2.LINE_AA)

def main():
    print("Rendering PDF...")
    img = rasterise_pdf(PDF_PATH)
    # Fade background
    img = cv2.addWeighted(img, 0.4, np.full(img.shape, 255, dtype=np.uint8), 0.6, 0)
    
    sp_df = pd.read_csv(SP_CSV)
    topo_df = pd.read_csv(TOPO_CSV)
    te_raw = pd.read_csv(TE_CSV)
    te_df = te_raw[te_raw['file_name'] == '02.Unmarked'].copy()
    ce_df = pd.read_csv(CE_CSV)
    
    merger = LineTagMerger(cfg)
    merger._build_coordinate_transform()
    
    te_bboxes = []
    for _, r in te_df.iterrows():
        try:
            b = ast.literal_eval(r['bbox'])
            te_bboxes.append(merger._transform_bbox(b))
        except:
            te_bboxes.append((0,0,0,0))
    te_df['px_bbox'] = te_bboxes
    
    ce_data = []
    for _, r in ce_df.iterrows():
        try:
            b = ast.literal_eval(r['bounding_box']) if isinstance(r['bounding_box'], str) else r['bounding_box']
            if pd.isna(b): continue
            ce_data.append({
                'type': r.get('type', 'Unknown'),
                'bbox': merger._transform_bbox(b)
            })
        except: continue

    group_tags = {}
    for _, r in topo_df.iterrows():
        if pd.notna(r['line_tag']):
            group_tags[r['Line_ID']] = str(r['line_tag']).strip()
            
    # Focus only on SP_000 and SP_001 as requested
    sp_groups = sp_df[sp_df['Super_Pipeline_ID'].isin(['SP_000', 'SP_001'])].groupby('Super_Pipeline_ID')
    
    for sp_id, df_group in sp_groups:
        sp_tag = None
        for gid in df_group['Group_ID'].unique():
            if gid in group_tags:
                sp_tag = group_tags[gid]
                break
                
        # 1. Highlight Line Tag in TE (Blue)
        if sp_tag:
            for _, r in te_df.iterrows():
                if str(r['text']).strip() == sp_tag:
                    draw_box(img, r['px_bbox'], (255, 0, 0), f"{sp_id} Tag")
                    
        # Find endpoints and draw the pipeline lines (Yellow)
        pts = []
        for _, r in df_group.iterrows():
            p1 = (int(r['x_start']), int(r['y_start']))
            p2 = (int(r['x_end']), int(r['y_end']))
            cv2.line(img, p1, p2, (0, 255, 255), 10, cv2.LINE_AA) # Yellow line
            pts.append(p1)
            pts.append(p2)
        pt_counts = {}
        for p in pts: pt_counts[p] = pt_counts.get(p, 0) + 1
        endpoints = [p for p, c in pt_counts.items() if c == 1]
        
        # 2. Highlight Source/Dest Equipment (Green)
        for ep in endpoints:
            ex, ey = ep
            for ce in ce_data:
                if ce['type'] in MAJOR_TYPES and point_in_bbox(ex, ey, ce['bbox'], padding=50):
                    # Found major equipment CE box
                    draw_box(img, ce['bbox'], (0, 255, 0), f"Equipment: {ce['type']}")
                    # Highlight its name in TE
                    for _, r in te_df.iterrows():
                        if r.get('tag_type') == 'General':
                            tx0, ty0, tx1, ty1 = r['px_bbox']
                            tcx, tcy = (tx0 + tx1)/2, (ty0 + ty1)/2
                            if point_in_bbox(tcx, tcy, ce['bbox'], padding=800):
                                text = str(r['text']).strip()
                                if re.search(r'\d{3}-[A-Z]-\d+', text):
                                    draw_box(img, r['px_bbox'], (0, 255, 0), "Equip Name")
                                    
            # Off page connectors
            for _, r in te_df.iterrows():
                text = str(r['text']).strip()
                tx0, ty0, tx1, ty1 = r['px_bbox']
                tcx, tcy = (tx0 + tx1)/2, (ty0 + ty1)/2
                if point_in_bbox(tcx, tcy, (ex, ey, ex, ey), padding=400):
                    if text.upper().startswith("TO ") or text.upper().startswith("FROM "):
                        draw_box(img, r['px_bbox'], (0, 255, 0), "Off-Page")
                        
        # 3. Highlight Inline Components (Red)
        for _, r in df_group.iterrows():
            for ce in ce_data:
                if ce['type'] not in MAJOR_TYPES:
                    if bbox_intersects_segment(ce['bbox'], r['x_start'], r['y_start'], r['x_end'], r['y_end']):
                        draw_box(img, ce['bbox'], (0, 0, 255), f"Inline: {ce['type']}")

    cv2.imwrite(OUT_IMG, img)
    print(f"Saved {OUT_IMG}")
    
    print("Converting image to PDF...")
    doc = fitz.open()
    h, w = img.shape[:2]
    rect = fitz.Rect(0, 0, w * 72 / 300, h * 72 / 300)
    page = doc.new_page(width=rect.width, height=rect.height)
    page.insert_image(rect, filename=OUT_IMG)
    doc.save(OUT_PDF)
    doc.close()
    print(f"Saved {OUT_PDF}")

if __name__ == '__main__':
    main()
