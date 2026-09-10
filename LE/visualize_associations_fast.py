import cv2
import pandas as pd
import numpy as np
import fitz
import ast
import re

PDF_PATH = r"C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf"
CE_CSV = r"C:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\02.Unmarked.csv"
TE_CSV = r"C:\Users\user\Desktop\Work\p&id\TE\Results\TE_csv\process_tags.csv"
OUT_JPG = r"C:\Users\user\Desktop\Work\p&id\results\LE_images\02.Unmarked_Associations.jpg"

MAJOR_TYPES = {
    'Vessel', 'Tank', 'Pump / Compressor', 'Heat Exchanger', 
    'Drum', 'Compressor', 'Complex Equipment'
}

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

def rasterise_pdf(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()

def transform_bbox(bbox, scale, w_pts, h_pts, rot):
    x0, y0, x1, y1 = bbox
    if rot == 90:
        # CORRECT: CW rotation where Portrait (X,Y) -> Landscape (H_portrait - Y, X)
        tx0 = (h_pts - y1) * scale
        ty0 = x0 * scale
        tx1 = (h_pts - y0) * scale
        ty1 = x1 * scale
        return tx0, ty0, tx1, ty1
    return x0*scale, y0*scale, x1*scale, y1*scale

def main():
    print("Rasterizing PDF...")
    img = rasterise_pdf(PDF_PATH)
    
    print("Loading datasets...")
    ce_df = pd.read_csv(CE_CSV)
    te_raw = pd.read_csv(TE_CSV)
    te_df = te_raw[te_raw['file_name'] == '02.Unmarked'].copy()
    
    doc = fitz.open(PDF_PATH)
    page = doc[0]
    w_pts = page.cropbox.width
    h_pts = page.cropbox.height
    rot = page.rotation
    scale = 300 / 72.0
    doc.close()
    
    te_bboxes = []
    for _, r in te_df.iterrows():
        try:
            b = ast.literal_eval(r['bbox'])
            te_bboxes.append(transform_bbox(b, scale, w_pts, h_pts, rot))
        except:
            te_bboxes.append((0,0,0,0))
    te_df['px_bbox'] = te_bboxes
    
    print("Processing CE bounding boxes...")
    for _, r in ce_df.iterrows():
        try:
            b = ast.literal_eval(r['bounding_box']) if isinstance(r['bounding_box'], str) else r['bounding_box']
            if pd.isna(b): continue
            ce_bbox = transform_bbox(b, scale, w_pts, h_pts, rot)
            ctype = r.get('type', 'Unknown')
            
            cv2.rectangle(img, (int(ce_bbox[0]), int(ce_bbox[1])), (int(ce_bbox[2]), int(ce_bbox[3])), (255, 0, 0), 2)
            
            found_te = False
            for _, te_r in te_df.iterrows():
                te_bbox = te_r['px_bbox']
                text = str(te_r['text']).strip()
                if ctype in MAJOR_TYPES:
                    if te_r.get('tag_type') == 'General' and bbox_intersects(te_bbox, ce_bbox, padding=0):
                        if re.search(r'\d{3}-[A-Z]-\d{2,3}', text) or re.search(r'\d{3}-[A-Z]-\d+[A-Z]?', text):
                            cv2.rectangle(img, (int(te_bbox[0]), int(te_bbox[1])), (int(te_bbox[2]), int(te_bbox[3])), (0, 255, 0), 3)
                            cv2.line(img, (int((ce_bbox[0]+ce_bbox[2])/2), int((ce_bbox[1]+ce_bbox[3])/2)), (int((te_bbox[0]+te_bbox[2])/2), int((te_bbox[1]+te_bbox[3])/2)), (0, 255, 0), 2)
                            found_te = True
                else:
                    if te_r.get('tag_type') not in ['Line Tag', 'Equipment'] and bbox_intersects(te_bbox, ce_bbox, padding=0):
                        if re.search(r'[A-Z]{2,}-\d+', text) or text in ['PSV', 'CV', 'FV'] or re.search(r'[A-Z]{2,}\s+\d+', text):
                            cv2.rectangle(img, (int(te_bbox[0]), int(te_bbox[1])), (int(te_bbox[2]), int(te_bbox[3])), (0, 255, 0), 3)
                            cv2.line(img, (int((ce_bbox[0]+ce_bbox[2])/2), int((ce_bbox[1]+ce_bbox[3])/2)), (int((te_bbox[0]+te_bbox[2])/2), int((te_bbox[1]+te_bbox[3])/2)), (0, 255, 0), 2)
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
