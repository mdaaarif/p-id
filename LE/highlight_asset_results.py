import cv2
import numpy as np
import fitz
import pandas as pd
import os

PDF_PATH = r"C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf"
SUPER_CSV  = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_SUPER_pipelines.csv"

OUT_IMG  = r"C:\Users\user\Desktop\Work\p&id\results\LE_images\02.Unmarked_Asset_Results.jpg"
OUT_PDF  = r"C:\Users\user\Desktop\Work\p&id\results\LE_images\02.Unmarked_Asset_Results.pdf"

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
    print("Rendering PDF to image...")
    img = rasterise_pdf(PDF_PATH)
    
    super_df = pd.read_csv(SUPER_CSV)
    
    # Fade background heavily so highlights pop
    img = cv2.addWeighted(img, 0.3, np.full(img.shape, 255, dtype=np.uint8), 0.7, 0)
    
    # Distinct colors for the first 9 SPs (SP_000 to SP_008)
    # BGR format
    colors = {
        'SP_000': (0, 0, 255),       # Red
        'SP_001': (255, 0, 0),       # Blue (Main Header)
        'SP_002': (0, 255, 0),       # Green
        'SP_003': (0, 165, 255),     # Orange
        'SP_004': (255, 0, 255),     # Magenta
        'SP_005': (255, 255, 0),     # Cyan
        'SP_006': (128, 0, 128),     # Purple
        'SP_007': (0, 128, 255),     # Amber
        'SP_008': (0, 200, 200),     # Teal
    }
    
    super_groups = super_df.groupby('Super_Pipeline_ID')
    
    for sid, df_group in super_groups:
        if sid in colors:
            color = colors[sid]
            lw = 12
        else:
            color = (200, 200, 200) # Faint gray for others
            lw = 6
            
        pts = []
        for _, seg in df_group.iterrows():
            p1 = (int(seg["x_start"]), int(seg["y_start"]))
            p2 = (int(seg["x_end"]),   int(seg["y_end"]))
            cv2.line(img, p1, p2, color, lw, cv2.LINE_AA)
            pts.extend([p1, p2])
            
        if sid in colors and pts:
            # Draw label near the center
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            
            # For SP_001 which is very long, label near the left side
            if sid == 'SP_001':
                cx, cy = int(np.min(xs)) + 500, int(np.min(ys)) - 100
                
            cv2.putText(img, sid, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(img, sid, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5, cv2.LINE_AA)

    print("Writing image file...")
    cv2.imwrite(OUT_IMG, img)

    print("Converting image to PDF...")
    doc = fitz.open()
    h, w = img.shape[:2]
    rect = fitz.Rect(0, 0, w * 72 / DPI, h * 72 / DPI)
    page = doc.new_page(width=rect.width, height=rect.height)
    page.insert_image(rect, filename=OUT_IMG)
    doc.save(OUT_PDF)
    doc.close()

    print(f"Saved {OUT_IMG} and {OUT_PDF}")

if __name__ == '__main__':
    main()
