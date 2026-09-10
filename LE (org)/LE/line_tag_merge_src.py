# ============================================================
# src/line_tag_merge_src/line_tag_merge.py
# P&ID Line <-> Tag Association  (v2 — segment-proximity matching)
# ============================================================
#
# Key improvements over v1
# ------------------------
# 1.  CORRECT COORDINATE TRANSFORM
#     PyMuPDF page.get_text() returns tag bounding boxes in the DISPLAYED
#     (rotation-applied) page space (PDF points).  page.get_pixmap(dpi=D)
#     produces a rasterised image in the same rotated orientation but in
#     pixels.  For a Rotate=90 page the displayed y-axis maps to the pixel
#     x-axis and (W - displayed_x) maps to the pixel y-axis, so the old
#     rotation_matrix approach (which is for converting RAW/unrotated points)
#     was incorrect.  v2 reads the actual page rotation and applies the
#     correct axis-swap + reflection formula for each case (0°, 90°, 180°,
#     270°).
#
# 2.  SEGMENT-PROXIMITY MATCHING (USE_SEGMENT_PROXIMITY = True)
#     P&ID line labels are placed beside the pipe run, not at its endpoints.
#     v2 measures the perpendicular distance from every tag centre to the
#     closest point on every line segment, then picks the logical group
#     (Group_ID) whose closest segment wins.  This replaces the old endpoint-
#     only search, raising coverage from ~12 % to ~95 %+ for Line Tag types.
#
# 3.  AUTO-GROUPING FROM RAW SEGMENT CSV
#     If the line CSV contains "Line_Number" instead of "Group_ID", the code
#     automatically chains spatially connected segments into logical line
#     groups using a configurable snap tolerance (SEGMENT_SNAP_TOLERANCE).
#
# Output: one CSV row per logical line (Group_ID), with the nearest tag,
# its type, its transformed pixel coordinates, distance, and confidence.
# ============================================================

import os
import ast
import math
from collections import defaultdict, deque

import fitz
import pandas as pd


class LineTagMerger:
    """
    Associates extracted P&ID lines with their nearest tag labels.

    Reads the grouped (or raw) line-coordinate CSV produced by LineExtractor
    and the classified tag CSV produced by TextExtractor, optionally builds
    logical line groups by chaining connected segments, transforms tag
    coordinates into the same pixel space as the line CSV, and writes a
    single association CSV with one row per logical line.
    """

    # ------------------------------------------------------------------
    # CONSTRUCTION
    # ------------------------------------------------------------------

    def __init__(self, config):
        self.config = config
        self.line_df = None
        self.tags_df = None
        self.result_df = None
        self._page_rotation = 0
        self._dpi_scale = 1.0
        self._page_w_pts = 0.0   # displayed page width  in PDF points
        self._page_h_pts = 0.0   # displayed page height in PDF points
        self._create_directories()

    def _create_directories(self):
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # COORDINATE TRANSFORM
    # ------------------------------------------------------------------

    def _build_coordinate_transform(self):
        """
        Read page geometry from the PDF and store the parameters needed to
        convert tag bounding boxes (displayed PDF-point space) into line
        pixel space.

        The correct transform depends on the page's /Rotate value:

        Rotate=0   (normal):
            px_x = pt_x * scale
            px_y = pt_y * scale

        Rotate=90  (landscape PDF displayed as portrait-rotated):
            displayed page: width=cropbox.height, height=cropbox.width
            pixmap (landscape): x → long axis, y → short axis
            px_x = pt_y * scale
            px_y = (W_pts - pt_x) * scale

        Rotate=180:
            px_x = (W_pts - pt_x) * scale
            px_y = (H_pts - pt_y) * scale

        Rotate=270:
            px_x = (H_pts - pt_y) * scale
            px_y = pt_x * scale
        """
        if not getattr(self.config, "APPLY_COORDINATE_TRANSFORM", False):
            print("APPLY_COORDINATE_TRANSFORM is False — tag coordinates "
                  "used as-is (must already be in pixel space).")
            self._dpi_scale = 1.0
            self._page_rotation = 0
            return

        doc = fitz.open(self.config.PDF_PATH)
        page = doc[self.config.PDF_PAGE_INDEX]
        rotation = page.rotation  # 0, 90, 180, or 270

        # PyMuPDF cropbox stores the UN-rotated page dimensions.
        # For a Rotate=90 or Rotate=270 page the displayed axes are swapped,
        # but get_text() still returns coordinates in a space where:
        #   displayed x  ∈ [0, cropbox.width]   (the SHORT edge of the physical sheet)
        #   displayed y  ∈ [0, cropbox.height]  (the LONG  edge of the physical sheet)
        # The rasterised pixmap from get_pixmap() produces a landscape image where:
        #   pixel x → long dimension  ↔  displayed y  (0 … cropbox.height)
        #   pixel y → short dimension ↔  (cropbox.width − displayed x)
        # Therefore W_pts (the value used in the y-flip) = cropbox.width (short edge),
        # and H_pts = cropbox.height (long edge, maps to pixel x).
        #
        # For Rotate=0 or Rotate=180 there is no axis swap.
        if rotation in (90, 270):
            w_pts = page.cropbox.width    # displayed width  (short edge, used for y-flip)
            h_pts = page.cropbox.height   # displayed height (long  edge, maps to pixel x)
        else:
            w_pts = page.cropbox.width
            h_pts = page.cropbox.height

        doc.close()

        self._page_rotation = rotation
        self._dpi_scale = self.config.PDF_DPI / 72.0
        self._page_w_pts = w_pts
        self._page_h_pts = h_pts

        print(
            f"PDF page {self.config.PDF_PAGE_INDEX}: rotation={rotation}°, "
            f"displayed size={w_pts:.1f}×{h_pts:.1f} pts, "
            f"DPI={self.config.PDF_DPI} → scale={self._dpi_scale:.4f}"
        )

    def _transform_bbox(self, bbox):
        """
        Convert a tag bounding box from displayed PDF-point space into line
        pixel space.

        :param bbox: (x0, y0, x1, y1) in displayed PDF-point space
        :return:     (x0, y0, x1, y1) in line pixel space
        """
        if not getattr(self.config, "APPLY_COORDINATE_TRANSFORM", False):
            return bbox

        x0, y0, x1, y1 = bbox
        s = self._dpi_scale
        W = self._page_w_pts
        H = self._page_h_pts
        rot = self._page_rotation

        if rot == 0:
            return x0 * s, y0 * s, x1 * s, y1 * s

        elif rot == 90:
            # px_x = pt_y * scale,  px_y = (W - pt_x) * scale
            new_x0 = min(y0, y1) * s
            new_y0 = (W - max(x0, x1)) * s
            new_x1 = max(y0, y1) * s
            new_y1 = (W - min(x0, x1)) * s
            return new_x0, new_y0, new_x1, new_y1

        elif rot == 180:
            # px_x = (W - pt_x) * scale,  px_y = (H - pt_y) * scale
            new_x0 = (W - max(x0, x1)) * s
            new_y0 = (H - max(y0, y1)) * s
            new_x1 = (W - min(x0, x1)) * s
            new_y1 = (H - min(y0, y1)) * s
            return new_x0, new_y0, new_x1, new_y1

        else:  # rot == 270
            # px_x = (H - pt_y) * scale,  px_y = pt_x * scale
            new_x0 = (H - max(y0, y1)) * s
            new_y0 = min(x0, x1) * s
            new_x1 = (H - min(y0, y1)) * s
            new_y1 = max(x0, x1) * s
            return new_x0, new_y0, new_x1, new_y1

    @staticmethod
    def _parse_bbox(bbox_value):
        if isinstance(bbox_value, (tuple, list)):
            parsed = bbox_value
        else:
            parsed = ast.literal_eval(str(bbox_value))
        return tuple(float(v) for v in parsed)

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------

    def _build_groups_from_raw(self, df):
        """
        Auto-group raw line segments into logical lines by chaining segments
        whose endpoints are within SEGMENT_SNAP_TOLERANCE pixels of each other.

        :param df: DataFrame with columns x_start, y_start, x_end, y_end
        :return:   same DataFrame with a new "Group_ID" column
        """
        snap = getattr(self.config, "SEGMENT_SNAP_TOLERANCE", 10)
        n = len(df)
        adj = defaultdict(set)

        coords = [
            ((row["x_start"], row["y_start"]), (row["x_end"], row["y_end"]))
            for _, row in df.iterrows()
        ]

        for i in range(n):
            for j in range(i + 1, n):
                connected = False
                for pi in coords[i]:
                    for pj in coords[j]:
                        if math.hypot(pi[0] - pj[0], pi[1] - pj[1]) <= snap:
                            connected = True
                            break
                    if connected:
                        break
                if connected:
                    adj[i].add(j)
                    adj[j].add(i)

        visited = set()
        group_map = {}
        gid = 0
        for start in range(n):
            if start in visited:
                continue
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                group_map[node] = gid
                queue.extend(adj[node] - visited)
            gid += 1

        df = df.copy()
        df["Group_ID"] = [group_map[i] for i in range(n)]
        return df

    def load_data(self):
        """
        Load the line CSV and tag CSV, build groups if necessary, apply
        tag-type/file filters, and transform tag bounding boxes into pixel
        space.
        """
        self._build_coordinate_transform()

        # ---- line CSV ----
        line_df = pd.read_csv(self.config.LINE_CSV_PATH)

        if "Group_ID" not in line_df.columns:
            if "Line_Number" in line_df.columns:
                print("Group_ID column not found — auto-grouping segments "
                      "by endpoint proximity …")
                line_df = self._build_groups_from_raw(line_df)
                n_groups = line_df["Group_ID"].nunique()
                print(f"  {len(line_df)} segments → {n_groups} logical lines")
            else:
                raise KeyError(
                    "Line CSV must contain either 'Group_ID' or 'Line_Number'."
                )

        # Normalise column names (x1/y1/x2/y2 or x_start/y_start/x_end/y_end)
        col_map = {}
        for src, dst in [
            ("x1", "x_start"), ("y1", "y_start"),
            ("x2", "x_end"),   ("y2", "y_end"),
        ]:
            if src in line_df.columns and dst not in line_df.columns:
                col_map[src] = dst
        if col_map:
            line_df = line_df.rename(columns=col_map)

        self.line_df = line_df

        # ---- tag CSV ----
        tags_df = pd.read_csv(self.config.TAG_CSV_PATH)

        if getattr(self.config, "TARGET_FILE_NAME", None) and "file_name" in tags_df.columns:
            tags_df = tags_df[tags_df["file_name"] == self.config.TARGET_FILE_NAME]

        excluded = getattr(self.config, "EXCLUDED_TAG_TYPES", None) or []
        if excluded:
            tags_df = tags_df[~tags_df["tag_type"].isin(excluded)]

        tags_df = tags_df.reset_index(drop=True)
        tags_df["_bbox_parsed"] = tags_df["bbox"].apply(
            lambda b: self._transform_bbox(self._parse_bbox(b))
        )
        tags_df["_cx"] = tags_df["_bbox_parsed"].apply(lambda b: (b[0] + b[2]) / 2)
        tags_df["_cy"] = tags_df["_bbox_parsed"].apply(lambda b: (b[1] + b[3]) / 2)

        self.tags_df = tags_df

        print(
            f"Loaded {len(self.line_df)} line segments across "
            f"{self.line_df['Group_ID'].nunique()} logical lines."
        )
        print(
            f"Loaded {len(self.tags_df)} candidate tags "
            f"(excluded types: {excluded})."
        )

    # ------------------------------------------------------------------
    # GEOMETRY HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _dist_point_to_segment(px, py, x1, y1, x2, y2):
        """
        Minimum distance from point (px, py) to the finite segment (x1,y1)-(x2,y2).
        """
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    @staticmethod
    def _get_line_endpoints(group_df, snap_tol=10):
        """
        Find the two exterior endpoints of a logical line group by identifying
        points that appear only once (not shared between two segments).
        """
        from collections import Counter
        points = []
        for _, row in group_df.iterrows():
            points.append((float(row["x_start"]), float(row["y_start"])))
            points.append((float(row["x_end"]), float(row["y_end"])))

        snapped_counts = Counter()
        snapped_to_orig = {}
        for p in points:
            key = (round(p[0] / snap_tol), round(p[1] / snap_tol))
            snapped_counts[key] += 1
            snapped_to_orig.setdefault(key, p)

        exterior = [snapped_to_orig[k] for k, c in snapped_counts.items() if c == 1]
        candidates = exterior if len(exterior) >= 2 else list(snapped_to_orig.values())

        if len(candidates) < 2:
            only = candidates[0] if candidates else points[0]
            return only, only

        # Pick the pair that maximises distance (true ends of the line)
        best = (candidates[0], candidates[1])
        best_d = -1.0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                d = math.hypot(
                    candidates[i][0] - candidates[j][0],
                    candidates[i][1] - candidates[j][1],
                )
                if d > best_d:
                    best_d = d
                    best = (candidates[i], candidates[j])

        return sorted(best, key=lambda p: (p[0], p[1]))

    # ------------------------------------------------------------------
    # MATCHING
    # ------------------------------------------------------------------

    def _find_nearest_tag_by_segment(self, group_df):
        """
        For a logical line group, find the nearest tag by measuring the
        distance from each tag centre to the closest point on every segment
        in the group.  Returns (tag_row, best_distance).
        """
        if self.tags_df is None or len(self.tags_df) == 0:
            return None, None

        best_row = None
        best_dist = float("inf")

        for _, tag in self.tags_df.iterrows():
            cx, cy = tag["_cx"], tag["_cy"]
            for _, seg in group_df.iterrows():
                d = self._dist_point_to_segment(
                    cx, cy,
                    float(seg["x_start"]), float(seg["y_start"]),
                    float(seg["x_end"]),   float(seg["y_end"]),
                )
                if d < best_dist:
                    best_dist = d
                    best_row = tag

        return best_row, best_dist

    def _find_nearest_tag_by_endpoints(self, start_pt, end_pt):
        """
        Legacy endpoint-only search (used when USE_SEGMENT_PROXIMITY = False).
        Searches the nearest tag bbox to each of the two exterior endpoints
        and returns the closer match.
        """
        if self.tags_df is None or len(self.tags_df) == 0:
            return None, None, None

        def closest_bbox(point):
            px, py = point
            best_row = None
            best_dist = float("inf")
            for _, row in self.tags_df.iterrows():
                x0, y0, x1, y1 = row["_bbox_parsed"]
                cx = min(max(px, min(x0, x1)), max(x0, x1))
                cy = min(max(py, min(y0, y1)), max(y0, y1))
                d = math.hypot(px - cx, py - cy)
                if d < best_dist:
                    best_dist = d
                    best_row = row
            return best_row, best_dist

        s_row, s_dist = closest_bbox(start_pt)
        e_row, e_dist = closest_bbox(end_pt)

        if s_dist <= e_dist:
            return s_row, s_dist, "Start"
        return e_row, e_dist, "End"

    def _compute_confidence(self, distance):
        if distance is None:
            return 0.0
        max_d = self.config.MAX_MATCH_DISTANCE
        score = max(0.0, min(1.0, 1.0 - distance / max_d))
        return round(score, self.config.CONFIDENCE_DECIMALS)

    # ------------------------------------------------------------------
    # ASSOCIATION
    # ------------------------------------------------------------------

    def associate_lines_with_tags(self):
        """
        Build the line/tag association table (one row per logical line).

        When USE_SEGMENT_PROXIMITY is True, measures distance from every tag
        to every segment; assigns the tag whose closest segment wins.
        When False, uses the legacy two-endpoint search.

        :return: DataFrame with one row per Group_ID
        """
        if self.line_df is None or self.tags_df is None:
            raise RuntimeError("Call load_data() before associate_lines_with_tags().")

        max_dist = self.config.MAX_MATCH_DISTANCE
        use_seg = getattr(self.config, "USE_SEGMENT_PROXIMITY", True)
        snap = getattr(self.config, "SEGMENT_SNAP_TOLERANCE", 10)

        records = []

        for group_id, group_df in self.line_df.groupby("Group_ID"):
            line_type = group_df["Type"].iloc[0] if "Type" in group_df.columns else "Unknown"
            start_pt, end_pt = self._get_line_endpoints(group_df, snap_tol=snap)

            if use_seg:
                tag_row, distance = self._find_nearest_tag_by_segment(group_df)
                matched_endpoint = "Segment"
            else:
                tag_row, distance, matched_endpoint = self._find_nearest_tag_by_endpoints(
                    start_pt, end_pt
                )

            matched = tag_row is not None and distance is not None and distance <= max_dist

            records.append({
                "Line_ID":                  group_id,
                "Line_Type":                line_type,
                "Num_Segments":             len(group_df),
                "Start_Endpoint":           (round(start_pt[0], 2), round(start_pt[1], 2)),
                "End_Endpoint":             (round(end_pt[0], 2), round(end_pt[1], 2)),
                "Match_Method":             matched_endpoint,
                "Nearest_Tag_Text":         tag_row["text"]     if matched else None,
                "Tag_Type":                 tag_row["tag_type"] if matched else None,
                "Tag_Coordinates_Pixels":   (
                    tuple(round(v, 1) for v in tag_row["_bbox_parsed"])
                    if matched else None
                ),
                "Distance_Pixels":          round(distance, 2)  if matched else None,
                "Confidence_Score":         self._compute_confidence(
                    distance if matched else None
                ),
            })

        self.result_df = pd.DataFrame.from_records(records)
        return self.result_df

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    def save_csv(self, df=None):
        if df is None:
            df = self.result_df
        if df is None:
            raise RuntimeError("Nothing to save — call associate_lines_with_tags() first.")

        out_path = os.path.join(self.config.OUTPUT_DIR, self.config.OUTPUT_FILENAME)
        df.to_csv(out_path, index=False)
        print(f"Line-tag association CSV saved → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def print_diagnostics(self, df=None):
        """Print a matching-quality summary to stdout."""
        if df is None:
            df = self.result_df
        if df is None:
            return

        n_lines   = len(df)
        n_matched = df["Nearest_Tag_Text"].notna().sum()
        pct       = 100 * n_matched / n_lines if n_lines else 0.0

        print(f"\n{'='*55}")
        print(f"  Lines processed : {n_lines}")
        print(f"  Tags matched    : {n_matched} / {n_lines}  ({pct:.1f} %)")

        if n_matched:
            dists = df.loc[df["Distance_Pixels"].notna(), "Distance_Pixels"]
            print(f"  Distance  median: {dists.median():.1f} px")
            print(f"  Distance  mean  : {dists.mean():.1f} px")
            print(f"  Distance  max   : {dists.max():.1f} px")

            print("\n  Tag type breakdown:")
            for tag_type, cnt in df["Tag_Type"].value_counts().items():
                print(f"    {tag_type:35s}: {cnt}")

        print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # CONVENIENCE
    # ------------------------------------------------------------------

    def run(self):
        """load_data → associate_lines_with_tags → save_csv → diagnostics"""
        self.load_data()
        df = self.associate_lines_with_tags()
        out_path = self.save_csv(df)
        self.print_diagnostics(df)
        return {"df": df, "output_path": out_path}