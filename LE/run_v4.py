import config_line_ext_4 as cfg
from Line_ext_4 import LineExtractor
import os

le = LineExtractor(cfg)
le.load_file(r'C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf')
le.crop_image()
gray, binary = le.preprocess()
line_mask = le.extract_morph_lines(binary)
edges = le.detect_edges(line_mask)
raw_segments = le.detect_lines_ppht(edges)
deduped = le.dedupe_segments(raw_segments)
filtered = le.filter_isolated_fragments(deduped)
le.group_lines(filtered)
le.save_csv()
le.save_grouped_csv()
import pandas as pd
csv_path = r'C:\Users\user\Desktop\Work\p&id\results\LE_csv\line_coordinates_grouped_v4.csv'
df = pd.read_csv(csv_path)
df['length'] = ((df['x_end']-df['x_start'])**2 + (df['y_end']-df['y_start'])**2)**0.5

def process_group(g):
    total_len = g['length'].sum()
    dashed_len = g[g['Line_Style'] == 'Dashed']['length'].sum()
    dashed_ratio = dashed_len / total_len if total_len > 0 else 0
    
    min_x = min(g['x_start'].min(), g['x_end'].min())
    max_x = max(g['x_start'].max(), g['x_end'].max())
    min_y = min(g['y_start'].min(), g['y_end'].min())
    max_y = max(g['y_start'].max(), g['y_end'].max())
    w, h = max_x - min_x, max_y - min_y
    
    is_drawing_border = (w > 2000 or h > 2000) and (min_x < 500 or max_x > 9400 or min_y < 800 or max_y > 6500)
    is_dashed_signal = dashed_ratio >= 0.8
    
    return pd.Series({
        'is_junk': is_drawing_border or is_dashed_signal
    })

grouped = df.groupby('Group_ID').apply(process_group).reset_index()
junk_ids = grouped[grouped['is_junk']]['Group_ID'].tolist()
print(f"Generic Filter: dropping {len(junk_ids)} junk groups (dashed lines and borders)")

df = df[~df['Group_ID'].isin(junk_ids)]
df = df.drop(columns=['length'])
df.to_csv(csv_path.replace('.csv', '_filtered.csv'), index=False)

from line_tag_merge_4 import LineTagMerger
import config_line_tag_merge as merge_cfg
assoc = LineTagMerger(merge_cfg)
assoc.run()
