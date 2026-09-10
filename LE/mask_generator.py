
import os
import ast
import cv2
import numpy as np
import fitz
import pandas as pd
import sys

sys.path.append(r'C:\Users\user\Desktop\Work\p&id\LE')
from line_tag_merge_4 import LineTagMerger
import config_line_tag_merge as cfg

PDF_PATH = r'C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf'
CE_CSV_PATH = r'C:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\02.Unmarked.csv'
OUT_IMG = r'C:\Users\user\Desktop\Work\p&id\data\02.Unmarked_masked.jpg'

DPI = 300
PADDING = 3  # 3 pixels padding around the bounding boxes to ensure complete erasure

def rasterise_pdf(pdf_path, dpi=DPI):
    doc  = fitz.open(pdf_path)
    page = doc[0]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()

def main():
    print("Rasterizing PDF...")
    img = rasterise_pdf(PDF_PATH)
    
    print("Initializing LineTagMerger to reuse coordinate transforms...")
    merger = LineTagMerger(cfg)
    merger._build_coordinate_transform()
    
    # 1. Load TE Tags (already transformed by LineTagMerger load_data)
    # We just need to initialize the tags_df
    # We will do a custom light load to avoid requiring grouped line CSVs
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

    # 2. Load CE Components
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
    
    print(f"Masking out {len(all_bboxes)} regions...")
    # Draw white filled rectangles over the image
    for bbox in all_bboxes:
        x0, y0, x1, y1 = bbox
        
        # Ensure proper order and apply padding
        start_x = max(0, int(min(x0, x1)) - PADDING)
        start_y = max(0, int(min(y0, y1)) - PADDING)
        end_x = min(img.shape[1], int(max(x0, x1)) + PADDING)
        end_y = min(img.shape[0], int(max(y0, y1)) + PADDING)
        
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 255, 255), thickness=cv2.FILLED)
        
    print(f"Saving masked image to {OUT_IMG}")
    cv2.imwrite(OUT_IMG, img)

if __name__ == '__main__':
    main()
