import config_line_ext_4 as cfg
from Line_ext_4 import LineExtractor
import os
import pandas as pd

def main():
    print("--- Running V5 Masked Pipeline ---")
    le = LineExtractor(cfg)
    masked_img_path = r'C:\Users\user\Desktop\Work\p&id\data\02.Unmarked_masked.jpg'
    le.load_file(masked_img_path)
    le.crop_image()
    gray, binary = le.preprocess()
    line_mask = le.extract_morph_lines(binary)
    edges = le.detect_edges(line_mask)
    raw_segments = le.detect_lines_ppht(edges)
    deduped = le.dedupe_segments(raw_segments)
    filtered = le.filter_isolated_fragments(deduped)
    le.group_lines(filtered)
    
    # Save the original outputs (just in case)
    le.save_csv()
    le.save_grouped_csv()
    
    # Now run the generic filter
    csv_path = r'C:\Users\user\Desktop\Work\p&id\results\LE_csv\line_coordinates_grouped_v4.csv'
    df = pd.read_csv(csv_path)
    
    def process_group(g):
        min_x = min(g['x_start'].min(), g['x_end'].min())
        max_x = max(g['x_start'].max(), g['x_end'].max())
        min_y = min(g['y_start'].min(), g['y_end'].min())
        max_y = max(g['y_start'].max(), g['y_end'].max())
        w, h = max_x - min_x, max_y - min_y
        
        # Keep dashed lines! We only drop the borders.
        is_drawing_border = (w > 2000 or h > 2000) and (min_x < 500 or max_x > 9400 or min_y < 800 or max_y > 6500)
        
        return pd.Series({
            'is_junk': is_drawing_border
        })
        
    grouped = df.groupby('Group_ID').apply(process_group).reset_index()
    junk_ids = grouped[grouped['is_junk']]['Group_ID'].tolist()
    print(f"Generic Filter: dropping {len(junk_ids)} junk groups (borders only, keeping instruments)")
    
    df = df[~df['Group_ID'].isin(junk_ids)]
    out_path = r'C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_masked_lines.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved perfectly filtered lines to {out_path}")

if __name__ == '__main__':
    main()
