import os
import ast
import pandas as pd
import networkx as nx
import math
import cv2
import numpy as np
import fitz

import sys
sys.path.append(r'c:\Users\user\Desktop\Work\p&id\LE')
from line_tag_merge_4 import LineTagMerger
import config_line_tag_merge as cfg

CE_CSV_PATH = r'c:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\02.Unmarked.csv'
cfg.LINE_CSV_PATH = r'C:\Users\user\Desktop\Work\p&id\results\LE_csv\line_coordinates_grouped_v4.csv'

PDF_PATH = r'C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf'
OUT_IMG = r'C:\Users\user\Desktop\Work\p&id\results\LE_images\temp_render_graph.jpg'
OUT_PDF = r'C:\Users\user\Desktop\Work\p&id\results\LE_images\02.Unmarked_FINAL_graph.pdf'

DPI = 300
LINE_W = 10

def rasterise_pdf(pdf_path, dpi=DPI):
    doc  = fitz.open(pdf_path)
    page = doc[0]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()

def main():
    print("Initializing LineTagMerger to reuse coordinate transforms...")
    merger = LineTagMerger(cfg)
    merger._build_coordinate_transform()
    
    print("Loading LE lines...")
    merger.load_data()
    df_lines = merger.line_df
    segments_df = df_lines
    
    te_tags = []
    if merger.tags_df is not None:
        for _, row in merger.tags_df.iterrows():
            te_tags.append({'text': row['text'], 'bbox': row['_bbox_parsed'], 'type': row['tag_type']})
            
    ce_df = pd.read_csv(CE_CSV_PATH)
    ce_components = []
    for _, row in ce_df.iterrows():
        try:
            bbox_str = row['bounding_box']
            if pd.isna(bbox_str): continue
            if isinstance(bbox_str, str): bbox = ast.literal_eval(bbox_str)
            else: bbox = bbox_str
            px_bbox = merger._transform_bbox(bbox)
            ce_components.append({'text': row.get('tag', 'UNKNOWN_CE'), 'bbox': px_bbox, 'type': row.get('type', 'Equipment')})
        except Exception:
            continue
            
    all_anchors = te_tags + ce_components
    
    print("Building Graph...")
    G = nx.Graph()
    
    for i, anchor in enumerate(all_anchors):
        G.add_node(f"ANCHOR_{i}", type='anchor', data=anchor)
        
    for i, row in segments_df.iterrows():
        line_id = f"SEG_{i}" 
        group_id = row['Group_ID']
        start_node = f"{line_id}_START"
        end_node   = f"{line_id}_END"
        
        G.add_node(start_node, type='endpoint', coord=(row['x_start'], row['y_start']), group_id=group_id)
        G.add_node(end_node,   type='endpoint', coord=(row['x_end'], row['y_end']), group_id=group_id)
        G.add_edge(start_node, end_node, type='pipe', group_id=group_id)
        
    GAP_TOLERANCE = 15
    endpoints = [n for n, attr in G.nodes(data=True) if attr['type'] == 'endpoint']
    
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            n1 = endpoints[i]
            n2 = endpoints[j]
            c1 = G.nodes[n1]['coord']
            c2 = G.nodes[n2]['coord']
            dist = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
            if dist <= GAP_TOLERANCE:
                G.add_edge(n1, n2, type='connection', dist=dist)
                
    ANCHOR_PROXIMITY = 50
    for ep in endpoints:
        ex, ey = G.nodes[ep]['coord']
        for i, anchor in enumerate(all_anchors):
            ax0, ay0, ax1, ay1 = anchor['bbox']
            dx = max(0, ax0 - ex, ex - ax1)
            dy = max(0, ay0 - ey, ey - ay1)
            dist = math.hypot(dx, dy)
            if dist <= ANCHOR_PROXIMITY:
                G.add_edge(ep, f"ANCHOR_{i}", type='anchor_connection', dist=dist)
                
    connected_components = list(nx.connected_components(G))
    valid_groups = set()
    for comp in connected_components:
        if any(G.nodes[n]['type'] == 'anchor' for n in comp):
            for n in comp:
                if G.nodes[n]['type'] == 'endpoint':
                    valid_groups.add(G.nodes[n]['group_id'])
                    
    print(f"Graph Filter: Validated {len(valid_groups)} groups.")
    
    # Save Validated CSV
    csv_out_path = r'C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_graph_validated_lines.csv'
    valid_df = df_lines[df_lines['Group_ID'].isin(valid_groups)].copy()
    valid_df.to_csv(csv_out_path, index=False)
    print(f'Saved validated lines to {csv_out_path}')
    
    # Render PDF
    print('Rendering PDF to image...')
    img = rasterise_pdf(PDF_PATH)
    
    print('Drawing lines...')
    img = cv2.addWeighted(img, 0.2, np.full(img.shape, 255, dtype=np.uint8), 0.8, 0)
    
    # We will draw ALL segments that belong to a valid group!
    # Instead of drawing the grouped straight line, drawing the segments ensures we trace elbows properly.
    for i, row in segments_df.iterrows():
        if row['Group_ID'] in valid_groups:
            color = (0, 0, 200) # RED for Graph approach to distinguish from Green V5
            p1 = (int(row['x_start']), int(row['y_start']))
            p2 = (int(row['x_end']),   int(row['y_end']))
            cv2.line(img, p1, p2, color, LINE_W, cv2.LINE_AA)
            
    print('Writing image file...')
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    cv2.imwrite(OUT_IMG, img)
    
    print('Converting image to PDF...')
    doc = fitz.open()
    h, w = img.shape[:2]
    rect = fitz.Rect(0, 0, w * 72 / DPI, h * 72 / DPI)
    page = doc.new_page(width=rect.width, height=rect.height)
    page.insert_image(rect, filename=OUT_IMG)
    doc.save(OUT_PDF)
    doc.close()
    
    import shutil
    shutil.copy(OUT_PDF, r'C:\Users\user\.gemini\antigravity-ide\brain\8a29a639-db4c-40ef-b5ad-707ebe941305\02.Unmarked_FINAL_graph.pdf')
    print(f'Saved {OUT_PDF}')

if __name__ == '__main__':
    main()
