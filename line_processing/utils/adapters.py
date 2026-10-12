# Copied from line_association/adapters.py so line_processing is self-contained
import ast
import pandas as pd
from typing import List, Optional, Set, Tuple, Any
from ..src.models import Point, BBox, Segment, LineFragment, InlineComponent

class CoordinateTransformer:
    """
    Handles coordinate synchronization between rasterized line pixels (LE)
    and vector PDF stream points (TE tags & CE components).
    """
    def __init__(self, page_w_pts: float = 1684.0, page_h_pts: float = 2384.0, rotation: int = 90, dpi: float = 300.0):
        self.page_w_pts = page_w_pts
        self.page_h_pts = page_h_pts
        self.rotation = rotation
        self.dpi_scale = dpi / 72.0

    @classmethod
    def from_pdf(cls, pdf_path: str, dpi: float = 300.0) -> "CoordinateTransformer":
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[0]
        h = max(page.rect.width, page.rect.height)
        w = min(page.rect.width, page.rect.height)
        rot = page.rotation
        doc.close()
        return cls(page_w_pts=w, page_h_pts=h, rotation=rot, dpi=dpi)

    def le_to_pt(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Converts raster line pixel coordinates (x_px, y_px) to PDF stream point coordinates."""
        s = self.dpi_scale
        if self.rotation == 90:
            return y_px / s, self.page_h_pts - (x_px / s)
        elif self.rotation == 0:
            return x_px / s, y_px / s
        elif self.rotation == 180:
            return self.page_w_pts - (x_px / s), self.page_h_pts - (y_px / s)
        else: # 270
            return self.page_w_pts - (y_px / s), x_px / s

def parse_bbox_string(s: Any) -> Tuple[float, float, float, float]:
    if isinstance(s, (tuple, list)):
        return tuple(float(x) for x in s)
    s_clean = str(s).strip(" ()[]")
    parts = s_clean.split(",")
    return tuple(float(p.strip()) for p in parts[:4])

def load_line_fragments(
    csv_path: str,
    needed_rows: Optional[Set[int]] = None,
    transformer: Optional[CoordinateTransformer] = None
) -> List[LineFragment]:
    """
    Loads raw line fragments from line_coordinates_grouped.csv.
    Converts segment coordinates to PDF points if transformer is provided.
    Filters to needed_rows if provided.
    """
    df = pd.read_csv(csv_path)
    group_col = 'Group_ID' if 'Group_ID' in df.columns else 'Merged_Line_ID'
    
    fragments = []
    for gid, group_df in df.groupby(group_col):
        gid_int = int(gid)
        if needed_rows is not None and gid_int not in needed_rows:
            continue

        line_type = str(group_df['Type'].iloc[0]) if 'Type' in group_df.columns else "Line"
        segments = []
        for _, row in group_df.iterrows():
            x1, y1 = float(row['x_start']), float(row['y_start'])
            x2, y2 = float(row['x_end']), float(row['y_end'])

            if transformer:
                x1, y1 = transformer.le_to_pt(x1, y1)
                x2, y2 = transformer.le_to_pt(x2, y2)

            seg = Segment(
                x_start=x1,
                y_start=y1,
                x_end=x2,
                y_end=y2,
                orientation=str(row.get('Orientation', 'U')),
                segment_in_group=int(row.get('Segment_In_Group', 1))
            )
            segments.append(seg)

        fragment = LineFragment(
            id=gid_int,
            line_type=line_type,
            segments=segments
        )
        fragments.append(fragment)

    return fragments

def load_components(
    csv_path: str
) -> List[InlineComponent]:
    """Loads associated components from results/CE_csv/<stem>_component_tag_associated.csv (already in PDF points)."""
    df = pd.read_csv(csv_path)
    components = []
    for _, row in df.iterrows():
        raw_bbox = parse_bbox_string(row['bounding_box'])
        bbox = BBox(*raw_bbox)

        # Prefer final_type if present and non-empty, otherwise type
        comp_type = str(row['final_type']).strip() if pd.notna(row.get('final_type')) and str(row.get('final_type')).strip() else str(row.get('type', 'Unknown')).strip()

        # Clean matched_tag (ignore 'None' or empty)
        raw_tag = row.get('matched_tag')
        if pd.notna(raw_tag) and str(raw_tag).strip() not in ('', 'None', 'nan'):
            matched_tag = str(raw_tag).strip()
        else:
            matched_tag = None

        comp = InlineComponent(
            symbol_id=str(row['symbol_id']),
            type=comp_type,
            group=str(row.get('group', 'Unknown')),
            bbox=bbox,
            mounting=str(row.get('mounting', '')) if pd.notna(row.get('mounting')) else None,
            matched_tag=matched_tag
        )
        components.append(comp)

    return components

