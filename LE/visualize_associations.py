import cv2
import pandas as pd
import numpy as np
import fitz
import ast
import re
import sys

PDF_PATH = r"C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf"
CE_CSV = r"C:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\02.Unmarked.csv"
TE_CSV = r"C:\Users\user\Desktop\Work\p&id\TE\Results\TE_csv\process_tags.csv"
OUT_JPG = r"C:\Users\user\Desktop\Work\p&id\results\LE_images\02.Unmarked_Associations.jpg"

sys.path.append(r"C:\Users\user\Desktop\Work\p&id\LE")
from line_tag_merge_src import LineTagMerger
import config_line_tag_merge as cfg
from asset_extractor import bbox_intersects, MAJOR_TYPES

def rasterise_pdf(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()

def main():
    print("Rasterizing PDF...")
    img = rasterise_pdf(PDF_PATH)
    
    print("Loading datasets...")
    ce_df = pd.read_csv(CE_CSV)
    te_raw = pd.read_csv(TE_CSV)
    te_df = te_raw[te_raw['file_name'] == '02.Unmarked'].copy()
    
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
    
    print("Processing CE bounding boxes...")
    for _, r in ce_df.iterrows():
        try:
            b = ast.literal_eval(r['bounding_box']) if isinstance(r['bounding_box'], str) else r['bounding_box']
            if pd.isna(b): continue
            ce_bbox = merger._transform_bbox(b)
            ctype = r.get('type', 'Unknown')
            
            cv2.rectangle(img, (int(ce_bbox[0]), int(ce_bbox[1])), (int(ce_bbox[2]), int(ce_bbox[3])), (255, 0, 0), 2)
            
            found_te = False
            for _, te_r in te_df.iterrows():
                te_bbox = te_r['px_bbox']
                text = str(te_r['text']).strip()
                if ctype in MAJOR_TYPES:
                    if te_r.get('tag_type') == 'General' and bbox_intersects(te_bbox, ce_bbox, padding=50):
                        if re.search(r'\d{3}-[A-Z]-\d{2,3}', text) or re.search(r'\d{3}-[A-Z]-\d+[A-Z]?', text):
                            cv2.rectangle(img, (int(te_bbox[0]), int(te_bbox[1])), (int(te_bbox[2]), int(te_bbox[3])), (0, 255, 0), 3)
                            cv2.line(img, (int(ce_bbox[0]), int(ce_bbox[1])), (int(te_bbox[0]), int(te_bbox[1])), (0, 255, 0), 2)
                            found_te = True
                else:
                    if te_r.get('tag_type') not in ['Line Tag', 'Equipment'] and bbox_intersects(te_bbox, ce_bbox, padding=100):
                        if re.search(r'[A-Z]{2,}-\d+', text) or text in ['PSV', 'CV', 'FV'] or re.search(r'[A-Z]{2,}\s+\d+', text):
                            cv2.rectangle(img, (int(te_bbox[0]), int(te_bbox[1])), (int(te_bbox[2]), int(te_bbox[3])), (0, 255, 0), 3)
                            cv2.line(img, (int(ce_bbox[0]), int(ce_bbox[1])), (int(te_bbox[0]), int(te_bbox[1])), (0, 255, 0), 2)
                            found_te = True
                            
            if not found_te:
                cv2.putText(img, '?', (int(ce_bbox[0]), int(ce_bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        except Exception as e:
            continue
            
    print("Saving JPG...")
    cv2.imwrite(OUT_JPG, img)
    
    print("Saving PDF...")
    from PIL import Image
    im = Image.open(OUT_JPG)
    im.save(OUT_JPG.replace('.jpg', '.pdf'), 'PDF', resolution=300.0)
    print('Done!')

if __name__ == '__main__':
    main()
