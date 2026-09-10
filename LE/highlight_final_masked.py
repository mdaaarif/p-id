import cv2
import numpy as np
import fitz
import pandas as pd
import random
import os

PDF_PATH = r"C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf"
SEG_CSV  = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_masked_lines.csv"
OUT_IMG  = r"C:\Users\user\Desktop\Work\p&id\results\LE_images\temp_render_masked.jpg"
OUT_PDF  = r"C:\Users\user\Desktop\Work\p&id\results\LE_images\02.Unmarked_FINAL_masked_result.pdf"

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
    seg_df  = pd.read_csv(SEG_CSV)
    
    print(f"Loaded {len(seg_df)} line segments.")

    print("Drawing lines...")
    # Fade background slightly for better visibility
    img = cv2.addWeighted(img, 0.4, np.full(img.shape, 255, dtype=np.uint8), 0.6, 0)
    
    random.seed(42)
    colors = {}
    
    groups = seg_df['Group_ID'].unique()
    for gid in groups:
        colors[gid] = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        
    for _, seg in seg_df.iterrows():
        gid = int(seg["Group_ID"])
        p1 = (int(seg["x_start"]), int(seg["y_start"]))
        p2 = (int(seg["x_end"]),   int(seg["y_end"]))
        cv2.line(img, p1, p2, colors[gid], LINE_W, cv2.LINE_AA)

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
    
    # Cleanup temp image
    if os.path.exists(OUT_IMG):
        os.remove(OUT_IMG)

    print(f"Saved {OUT_PDF}")

if __name__ == '__main__':
    main()
