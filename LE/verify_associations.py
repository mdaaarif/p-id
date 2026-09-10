import sys
import os
import cv2
import pandas as pd
import numpy as np
import fitz

# Load original image or convert PDF to image
PDF_PATH = r"C:\Users\user\Desktop\Work\p&id\data\raw\02.Unmarked.pdf"
FRAGMENTS_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_gap_merged_lines.csv"
TAGS_CSV = r"C:\Users\user\Desktop\Work\p&id\TE\Results\TE_csv\process_tags.csv"
LOG_CSV = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\decision_log.csv"
OUT_IMG = r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\verification_associations.png"

def main():
    doc = fitz.open(PDF_PATH)
    page = doc[0]
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    lines_df = pd.read_csv(FRAGMENTS_CSV)
    tags_df = pd.read_csv(TAGS_CSV)
    log_df = pd.read_csv(LOG_CSV)

    # Pick 5 matches (3 from Pass 1, 2 from Pass 2)
    pass1 = log_df[log_df['Pass'] == 'Pass 1 (Tight)'].head(3)
    pass2 = log_df[log_df['Pass'] == 'Pass 2 (Wide)'].head(2)
    samples = pd.concat([pass1, pass2])

    group_col = 'Merged_Line_ID' if 'Merged_Line_ID' in lines_df.columns else 'Group_ID'

    # Need LineTagMerger logic to parse tag bboxes again
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from line_tag_merge_4 import LineTagMerger
    import config_line_tag_merge as cfg
    merger = LineTagMerger(cfg)
    merger._build_coordinate_transform()
    def parse_bbox(b): return tuple(map(float, b.strip("()").split(",")))

    for _, row in samples.iterrows():
        f_id = row['Fragment_ID']
        t_text = row['Matched_Tag']
        
        # Get fragment coords
        frag_data = lines_df[lines_df[group_col] == f_id]
        
        # Draw fragment lines (blue)
        for _, seg in frag_data.iterrows():
            pt1 = (int(seg['x_start']), int(seg['y_start']))
            pt2 = (int(seg['x_end']), int(seg['y_end']))
            cv2.line(img, pt1, pt2, (255, 0, 0), 4)

        # Get tag coords
        tag_row = tags_df[(tags_df['text'] == t_text) & (tags_df['tag_type'] == 'Line Tag')].iloc[0]
        bbox = merger._transform_bbox(parse_bbox(tag_row['bbox']))
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        # Draw tag box (red)
        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 4)

        # Draw a line connecting the centers (green)
        cx, cy = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
        seg = frag_data.iloc[0]
        lx, ly = int((seg['x_start'] + seg['x_end']) / 2), int((seg['y_start'] + seg['y_end']) / 2)
        cv2.line(img, (cx, cy), (lx, ly), (0, 255, 0), 2)

    cv2.imwrite(OUT_IMG, img)
    print(f"Saved verification image to {OUT_IMG}")

if __name__ == '__main__':
    main()
