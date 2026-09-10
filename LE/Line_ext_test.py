# ============================================================
# src/Line_ext_3.py
# P&ID Line Extraction (v3) – morphology -> Canny -> PPHT ->
#                              de-dup -> gap-preserving grouping
# ============================================================
#
# What changed vs the previous pipeline (Line_extraction.py / LE_2.py):
#
#   1. SMALL-SEGMENT REMOVAL
#      Unchanged in spirit but centralised: every raw PPHT segment
#      shorter than cfg.MIN_LINE_LENGTH is dropped. This removes the
#      large number of tiny, unwanted stray fragments.
#
#   2. CONTINUOUS-PATH IDENTIFICATION (the main fix)
#      The old pipeline "merged" fragments by literally redrawing a
#      single straight segment across the whole span — which erased
#      the real gap left by a valve/instrument/reducer symbol, and
#      for H+V chains it replaced the bend with a fake diagonal line.
#
#      v3 NEVER changes a segment's coordinates. Instead it:
#        a) fuses only TRUE duplicate/overlapping Hough detections of
#           the same physical stroke (tiny tolerance) — this is pure
#           clean-up, not gap bridging.
#        b) groups the remaining real segments that belong to the same
#           physical pipe run — either (i) same-orientation segments
#           broken by a component gap, or (ii) a horizontal segment
#           that continues as a vertical segment (a bend) — under one
#           shared Line/Group number and a "Type" of Horizontal /
#           Vertical / Straight / Elbow.
#        c) every segment is still drawn at its own true position, so
#           the component gap keeps its original length and location.
#
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz                         # PyMuPDF


class LineExtractor:
    """
    Extracts pipe / process lines from a P&ID PDF or image using:

    1. Margin-based cropping (all four borders) to exclude title blocks,
       notes areas, and drawing borders
    2. Morphological filtering (horizontal + vertical kernels)
    3. Canny edge detection on the morph mask
    4. Probabilistic Hough Line Transform (PPHT)
    5. Minimum-length filtering (drop tiny stray fragments)
    6. De-duplication of overlapping Hough detections of one stroke
    7. Gap-preserving grouping of fragments into logical pipe runs
       (straight-with-gap, or elbow/bend) WITHOUT altering geometry

    Line coordinates in all outputs (CSV and images) are expressed in the
    **original, pre-crop pixel coordinate system**. The crop offsets are
    added back to every detected endpoint automatically.

    Outputs
    -------
    - ``LE_images/morphological_line_mask.png``   – morph-only line mask
    - ``LE_images/numbered_lines.png``             – every real segment
      drawn at its true position, one label per logical group
    - ``LE_images/numbered_lines_with_axis.png``   – same, with pixel axis
    - ``LE_csv/line_coordinates_raw.csv``          – one row per real,
      de-duplicated, filtered segment
    - ``LE_csv/line_coordinates_grouped.csv``      – same segments, with
      Group_ID / Type / Segment_In_Group columns added
    """

    # ------------------------------------------------------------------
    # CONSTRUCTION
    # ------------------------------------------------------------------

    def __init__(self, cfg):
        """
    Initialise the LineExtractor with a configuration module and reset all
    internal pipeline state to empty/None.

    :param cfg: Configuration module (e.g. ``import config_line_ext_3 as cfg``)
                supplying all tunable parameters (DPI, margins, thresholds, paths)
    :type  cfg: module

    :return: None

    :raises TypeError: If ``cfg`` is not a Python module object
    """
        if not hasattr(cfg, "__name__"):
            raise TypeError("cfg must be a Python module (e.g. import config_line_ext_3 as cfg).")
        self.cfg = cfg

        # internal state populated during the pipeline
        self._img          = None   # cropped BGR image (margins removed)
        self._orig_img     = None   # full rasterised BGR image (pre-crop)
        self._gray         = None
        self._binary       = None
        self._line_mask    = None   # morph result
        self._edges        = None   # canny result

        self._segments     = []     # de-duplicated, filtered segments:
                                     #   [seg_id, orientation, x1, y1, x2, y2]
        self._df            = None  # raw-segment DataFrame

        self._grouped       = []    # grouped rows: [Group_ID, Segment_In_Group,
                                     #   Type, Orientation, x1, y1, x2, y2]
        self._grouped_df     = None
        self._group_labels   = []   # [(Group_ID, Type, label_x, label_y)]

        # pixel offsets applied to restore original coordinate space
        self._crop_x_offset = 0
        self._crop_y_offset = 0

    # ------------------------------------------------------------------
    # STEP 1 – LOAD
    # ------------------------------------------------------------------

    def load_file(self, path=None):
        """
        Load a PDF (page 0 only) or a raster image file into an internal BGR
        NumPy array. Supported image formats: PNG, JPG/JPEG, BMP, TIF/TIFF.

        :param path: Path to the PDF or image file to load; defaults to
                     ``cfg.INPUT_PATH`` when not supplied
        :type  path: str or None

        :return: Loaded BGR image as a NumPy array
        :rtype : np.ndarray

        :raises FileNotFoundError: If the file does not exist at the given path
        :raises ValueError       : If the file extension is unsupported or OpenCV
                                   fails to decode the image
        """
        if path is None:
            path = self.cfg.INPUT_PATH

        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            print("PDF detected")
            pdf  = fitz.open(path)
            page = pdf[0]
            pix  = page.get_pixmap(dpi=self.cfg.PDF_DPI)
            img  = np.frombuffer(pix.samples, dtype=np.uint8)
            img  = img.reshape(pix.height, pix.width, pix.n)
            img  = (cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    if pix.n == 4
                    else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            print("Image detected")
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"OpenCV could not read the image: {path}")
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                "Expected .pdf or a common image format (.png, .jpg, …)."
            )

        self._img      = img
        self._orig_img = img
        h, w = img.shape[:2]
        print(f"Loaded image size: {w} × {h}")
        return img

    # ------------------------------------------------------------------
    # STEP 2 – CROP  (4-border margin removal)
    # ------------------------------------------------------------------

    def crop_image(self, img=None,
                   margin_left=None, margin_right=None,
                   margin_top=None,  margin_bottom=None):
        """
        Remove fixed-pixel margins from all four borders of the loaded image so
        that title blocks, drawing frames, and notes areas are excluded from line
        detection. The crop offsets are stored internally and automatically added
        back to every detected endpoint, keeping all outputs in the original
        pre-crop pixel coordinate space.

        :param img          : BGR image to crop; defaults to the internally loaded image
        :type  img          : np.ndarray or None
        :param margin_left  : Pixels to remove from the left border; defaults to ``cfg.MARGIN_LEFT``
        :type  margin_left  : int or None
        :param margin_right : Pixels to remove from the right border; defaults to ``cfg.MARGIN_RIGHT``
        :type  margin_right : int or None
        :param margin_top   : Pixels to remove from the top border; defaults to ``cfg.MARGIN_TOP``
        :type  margin_top   : int or None
        :param margin_bottom: Pixels to remove from the bottom border; defaults to ``cfg.MARGIN_BOTTOM``
        :type  margin_bottom: int or None

        :return: Cropped BGR image as a NumPy array
        :rtype : np.ndarray

        :raises ValueError: If the combined horizontal or vertical margins equal or
                            exceed the corresponding image dimension
        """
        if img is None:
            if self._img is None:
                raise ValueError("No image loaded. Call load_file() first.")
            img = self._img

        if margin_left   is None: margin_left   = self.cfg.MARGIN_LEFT
        if margin_right  is None: margin_right  = self.cfg.MARGIN_RIGHT
        if margin_top    is None: margin_top    = self.cfg.MARGIN_TOP
        if margin_bottom is None: margin_bottom = self.cfg.MARGIN_BOTTOM

        h, w = img.shape[:2]

        x_start = margin_left
        x_end   = w - margin_right
        y_start = margin_top
        y_end   = h - margin_bottom

        if x_start >= x_end:
            raise ValueError(
                f"MARGIN_LEFT ({margin_left}) + MARGIN_RIGHT ({margin_right}) "
                f">= image width ({w}). Reduce the horizontal margins."
            )
        if y_start >= y_end:
            raise ValueError(
                f"MARGIN_TOP ({margin_top}) + MARGIN_BOTTOM ({margin_bottom}) "
                f">= image height ({h}). Reduce the vertical margins."
            )

        cropped = img[y_start:y_end, x_start:x_end]

        self._crop_x_offset = x_start
        self._crop_y_offset = y_start

        self._img = cropped
        print(
            f"Margins applied — left:{margin_left}  right:{margin_right}  "
            f"top:{margin_top}  bottom:{margin_bottom}"
        )
        print(
            f"Cropped size: {cropped.shape[1]} × {cropped.shape[0]}  "
            f"(offset x={self._crop_x_offset}, y={self._crop_y_offset})"
        )
        return cropped

    # ------------------------------------------------------------------
    # STEP 3 – GRAYSCALE + BINARY
    # ------------------------------------------------------------------

    def preprocess(self, img=None, thresh=None):
        """
        Convert the loaded BGR image to grayscale and then to an inverted binary
        mask using a fixed threshold. Dark ink strokes become foreground (255)
        on a black background (0).

        :param img   : BGR image to process; defaults to the internally stored image
        :type  img   : np.ndarray or None
        :param thresh: Pixel intensity threshold for binarisation (0–255);
                       defaults to ``cfg.BINARY_THRESH``
        :type  thresh: int or None

        :return: Tuple of (grayscale image, inverted binary mask)
        :rtype : Tuple[np.ndarray, np.ndarray]

        :raises ValueError: If no image is loaded or ``thresh`` is outside [0, 255]
        """
        if img is None:
            if self._img is None:
                raise ValueError("No image available. Call load_file() first.")
            img = self._img
        if thresh is None:
            thresh = self.cfg.BINARY_THRESH

        if not (0 <= thresh <= 255):
            raise ValueError(f"thresh must be in [0, 255], got {thresh}.")

        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]

        self._gray   = gray
        self._binary = binary
        return gray, binary

    # ------------------------------------------------------------------
    # STEP 4 – MORPHOLOGICAL LINE EXTRACTION
    # ------------------------------------------------------------------

    def extract_morph_lines(self, binary=None, h_kernel=None, v_kernel=None,
                            close_kernel=None, close_iter=None):
        """
        Isolate horizontal and vertical ink strokes from the binary mask using
        morphological opening with directional rectangular kernels, then merge
        the two results and close small gaps with a square closing kernel.

        :param binary      : Inverted binary mask to process; defaults to the internally stored mask
        :type  binary      : np.ndarray or None
        :param h_kernel    : Size tuple for the horizontal opening kernel; defaults to ``cfg.MORPH_H_KERNEL``
        :type  h_kernel    : tuple or None
        :param v_kernel    : Size tuple for the vertical opening kernel; defaults to ``cfg.MORPH_V_KERNEL``
        :type  v_kernel    : tuple or None
        :param close_kernel: Size tuple for the closing kernel; defaults to ``cfg.MORPH_CLOSE_K``
        :type  close_kernel: tuple or None
        :param close_iter  : Number of closing iterations; defaults to ``cfg.MORPH_CLOSE_ITER``
        :type  close_iter  : int or None

        :return: Binary morphological line mask with H/V strokes isolated
        :rtype : np.ndarray

        :raises ValueError: If no binary mask is available and ``binary`` is not supplied
        """
        if binary is None:
            if self._binary is None:
                raise ValueError("No binary mask. Call preprocess() first.")
            binary = self._binary

        if h_kernel     is None: h_kernel     = self.cfg.MORPH_H_KERNEL
        if v_kernel     is None: v_kernel     = self.cfg.MORPH_V_KERNEL
        if close_kernel is None: close_kernel = self.cfg.MORPH_CLOSE_K
        if close_iter   is None: close_iter   = self.cfg.MORPH_CLOSE_ITER

        hk = cv2.getStructuringElement(cv2.MORPH_RECT, h_kernel)
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, v_kernel)
        ck = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
        vertical   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
        line_mask  = cv2.bitwise_or(horizontal, vertical)
        line_mask  = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, ck,
                                      iterations=close_iter)

        self._line_mask = line_mask
        return line_mask

    # ------------------------------------------------------------------
    # STEP 5 – CANNY EDGE DETECTION
    # ------------------------------------------------------------------

    def detect_edges(self, line_mask=None, thresh1=None, thresh2=None,
                     aperture=None):
        """
        Apply Canny edge detection to the morphological line mask to produce a
        thin-edge map suitable for the Hough transform.

        :param line_mask: Binary morphological mask to process; defaults to the internally stored mask
        :type  line_mask: np.ndarray or None
        :param thresh1  : Lower hysteresis threshold for Canny; defaults to ``cfg.CANNY_THRESH1``
        :type  thresh1  : int or None
        :param thresh2  : Upper hysteresis threshold for Canny; defaults to ``cfg.CANNY_THRESH2``
        :type  thresh2  : int or None
        :param aperture : Sobel kernel size for gradient computation (must be 3, 5, or 7);
                          defaults to ``cfg.CANNY_APERTURE``
        :type  aperture : int or None

        :return: Canny edge map as a binary NumPy array
        :rtype : np.ndarray

        :raises ValueError: If no line mask is available, or ``aperture`` is not 3, 5, or 7
        """
        if line_mask is None:
            if self._line_mask is None:
                raise ValueError("No line mask. Call extract_morph_lines() first.")
            line_mask = self._line_mask

        if thresh1  is None: thresh1  = self.cfg.CANNY_THRESH1
        if thresh2  is None: thresh2  = self.cfg.CANNY_THRESH2
        if aperture is None: aperture = self.cfg.CANNY_APERTURE

        if aperture not in (3, 5, 7):
            raise ValueError(f"aperture must be 3, 5, or 7; got {aperture}.")

        edges = cv2.Canny(line_mask, thresh1, thresh2, apertureSize=aperture)
        self._edges = edges
        return edges

    # ------------------------------------------------------------------
    # HELPER – fraction of "on" pixels along a line (solid vs dashed)
    # ------------------------------------------------------------------

    @staticmethod
    def _line_fill_ratio(mask, x1, y1, x2, y2):
        """
        Compute the fraction of foreground (non-zero) pixels sampled along the
        straight path between two endpoints in a binary mask. Solid lines score
        close to 1.0; dashed or dotted lines score significantly lower due to
        the real gaps between dashes.

        :param mask: Binary mask image in which foreground pixels are non-zero
        :type  mask: np.ndarray
        :param x1  : X coordinate of the first endpoint (in mask pixel space)
        :type  x1  : int
        :param y1  : Y coordinate of the first endpoint (in mask pixel space)
        :type  y1  : int
        :param x2  : X coordinate of the second endpoint (in mask pixel space)
        :type  x2  : int
        :param y2  : Y coordinate of the second endpoint (in mask pixel space)
        :type  y2  : int

        :return: Fill ratio in [0.0, 1.0]
        :rtype : float
        """
        length = int(max(abs(x2 - x1), abs(y2 - y1), 1))
        num_samples = max(length, 2)

        h, w = mask.shape[:2]
        xs = np.clip(np.linspace(x1, x2, num_samples).round().astype(int), 0, w - 1)
        ys = np.clip(np.linspace(y1, y2, num_samples).round().astype(int), 0, h - 1)

        vals = mask[ys, xs]
        return float(np.count_nonzero(vals)) / num_samples

    # ------------------------------------------------------------------
    # STEP 6 – PPHT LINE DETECTION + SMALL-SEGMENT / DASH FILTER
    # (coordinates restored to original pre-crop space here)
    # ------------------------------------------------------------------

    def detect_lines_ppht(self, edges=None, rho=None, theta_denom=None,
                          threshold=None, min_line_len=None, max_line_gap=None):
        """
        Run the Probabilistic Hough Line Transform (PPHT) on the Canny edge map,
        then filter out dashed/dotted detections (fill ratio below
        ``cfg.MIN_LINE_SOLID_RATIO``) and short stray fragments (length below
        ``cfg.MIN_LINE_LENGTH``). Crop offsets are added back to every endpoint
        so all returned coordinates are in the original pre-crop pixel space.

        :param edges        : Edge map to process; falls back to stored Canny edges
                              or line mask if not supplied
        :type  edges        : np.ndarray or None
        :param rho          : Distance resolution of the accumulator in pixels;
                              defaults to ``cfg.PPHT_RHO``
        :type  rho          : float or None
        :param theta_denom  : Denominator for the angular resolution
                              (theta = π / theta_denom); defaults to ``cfg.PPHT_THETA``
        :type  theta_denom  : int or None
        :param threshold    : Minimum accumulator votes for a line to be accepted;
                              defaults to ``cfg.PPHT_THRESHOLD``
        :type  threshold    : int or None
        :param min_line_len : Minimum segment length in pixels to retain;
                              defaults to ``cfg.PPHT_MIN_LINE_LEN``
        :type  min_line_len : int or None
        :param max_line_gap : Maximum gap in pixels to bridge within one PPHT segment;
                              defaults to ``cfg.PPHT_MAX_LINE_GAP``
        :type  max_line_gap : int or None

        :return: List of ``[seg_id, orientation, x1, y1, x2, y2]`` in original pixel coordinates
        :rtype : list[list]

        :raises ValueError: If no edge map or line mask is available
        """
        if edges is None:
            if self._edges is not None:
                edges = self._edges
            elif self._line_mask is not None:
                edges = self._line_mask
            else:
                raise ValueError("No edge map. Call detect_edges() first.")

        if rho is None: rho = self.cfg.PPHT_RHO
        if theta_denom is None: theta_denom = self.cfg.PPHT_THETA
        if threshold is None: threshold = self.cfg.PPHT_THRESHOLD
        if min_line_len is None: min_line_len = self.cfg.PPHT_MIN_LINE_LEN
        if max_line_gap is None: max_line_gap = self.cfg.PPHT_MAX_LINE_GAP

        raw = cv2.HoughLinesP(
            edges,
            rho=rho,
            theta=np.pi / theta_denom,
            threshold=threshold,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap,
        )

        ox = self._crop_x_offset
        oy = self._crop_y_offset

        min_solid_ratio = getattr(self.cfg, "MIN_LINE_SOLID_RATIO", 0.70)
        min_dashed_ratio = getattr(self.cfg, "MIN_LINE_DASHED_RATIO", 0.20)
        min_length = self.cfg.MIN_LINE_LENGTH

        segments = []
        idx = 0
        dropped_dashed = 0
        dropped_short = 0
        if raw is not None:
            for line in raw:
                x1, y1, x2, y2 = line[0]  # cropped-image coordinates

                fill_ratio = self._line_fill_ratio(self._binary, x1, y1, x2, y2)
                if fill_ratio >= min_solid_ratio:
                    style = "Solid"
                elif fill_ratio >= min_dashed_ratio:
                    style = "Dashed"
                else:
                    dropped_dashed += 1
                    continue

                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length < min_length:
                    dropped_short += 1
                    continue

                # restore to original coordinate space
                x1 += ox; y1 += oy
                x2 += ox; y2 += oy
                orientation = ("Horizontal" if abs(x2 - x1) > abs(y2 - y1)
                               else "Vertical")
                idx += 1
                segments.append([idx, orientation, x1, y1, x2, y2, style])

        self._segments = segments
        print(
            f"PPHT raw detections: {0 if raw is None else len(raw)}  |  "
            f"dropped (dashed): {dropped_dashed}  |  "
            f"dropped (< {min_length}px): {dropped_short}  |  "
            f"kept: {len(segments)}  (offset x={ox}, y={oy})"
        )
        return segments

    # ------------------------------------------------------------------
    # GEOMETRY HELPERS shared by dedupe + grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_val(seg):
        """
        Return the coordinate on the axis perpendicular to the segment's
        orientation: the Y midpoint for Horizontal segments, or the X midpoint
        for Vertical segments. Used to test whether two segments share the same
        row or column.

        :param seg: Segment row in the form ``[seg_id, orientation, x1, y1, x2, y2]``
        :type  seg: list

        :return: Perpendicular-axis coordinate as a float
        :rtype : float
        """
        _, ori, x1, y1, x2, y2 = seg[:6]
        return (y1 + y2) / 2.0 if ori == "Horizontal" else (x1 + x2) / 2.0

    @staticmethod
    def _span(seg):
        """
    Return the (start, end) coordinate pair along the segment's own primary
    axis: (min_x, max_x) for Horizontal segments, (min_y, max_y) for
    Vertical segments. Used to compute overlap and gap distances.

    :param seg: Segment row in the form ``[seg_id, orientation, x1, y1, x2, y2]``
    :type  seg: list

    :return: Tuple of (start, end) along the segment's own axis
    :rtype : Tuple[float, float]
    """
        _, ori, x1, y1, x2, y2 = seg[:6]
        return (min(x1, x2), max(x1, x2)) if ori == "Horizontal" else (min(y1, y2), max(y1, y2))

    @staticmethod
    def _endpoints(seg):
        """
        Return both endpoint coordinate pairs of a segment.

        :param seg: Segment row in the form ``[seg_id, orientation, x1, y1, x2, y2]``
        :type  seg: list

        :return: Tuple of ((x1, y1), (x2, y2))
        :rtype : Tuple[Tuple[int, int], Tuple[int, int]]
        """
        _, _, x1, y1, x2, y2 = seg[:6]
        return (x1, y1), (x2, y2)

    @staticmethod
    def _dist(p, q):
        """
        Compute the Euclidean distance between two 2-D points.

        :param p: First point as (x, y)
        :type  p: Tuple[float, float]
        :param q: Second point as (x, y)
        :type  q: Tuple[float, float]

        :return: Euclidean distance between p and q
        :rtype : float
        """
        return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5

    # ------------------------------------------------------------------
    # STEP 6b – DE-DUPLICATE OVERLAPPING DETECTIONS OF THE SAME STROKE
    # ------------------------------------------------------------------

    def dedupe_segments(self, segments=None, axis_tol=None, touch_tol=None):
        """
        Fuse segments that are duplicate PPHT detections of the same physical ink
        stroke: same orientation, near-identical perpendicular-axis value, and
        touching or overlapping spans within a small pixel tolerance. The fused
        segment spans the full extent of all merged members. This step only
        unifies truly co-located detections and never bridges real component gaps,
        which are handled non-destructively in ``group_lines``.

        :param segments: List of ``[seg_id, orientation, x1, y1, x2, y2]`` rows
                         to deduplicate; defaults to ``self._segments``
        :type  segments: list[list] or None
        :param axis_tol : Maximum perpendicular-axis deviation in pixels for two
                          segments to be considered on the same stroke;
                          defaults to ``cfg.DEDUPE_AXIS_TOLERANCE``
        :type  axis_tol : int or None
        :param touch_tol: Maximum span gap in pixels between two segments for them
                          to be considered touching/overlapping;
                          defaults to ``cfg.DEDUPE_TOUCH_TOLERANCE``
        :type  touch_tol: int or None

        :return: Deduplicated list of ``[seg_id, orientation, x1, y1, x2, y2]`` rows
                 with IDs reassigned from 1
        :rtype : list[list]
        """
        if segments is None:
            segments = self._segments
        if axis_tol is None:
            axis_tol = self.cfg.DEDUPE_AXIS_TOLERANCE
        if touch_tol is None:
            touch_tol = self.cfg.DEDUPE_TOUCH_TOLERANCE

        h_segs_solid = [s for s in segments if s[1] == "Horizontal" and (s[6] == "Solid" if len(s) > 6 else True)]
        v_segs_solid = [s for s in segments if s[1] == "Vertical" and (s[6] == "Solid" if len(s) > 6 else True)]
        h_segs_dashed = [s for s in segments if s[1] == "Horizontal" and len(s) > 6 and s[6] == "Dashed"]
        v_segs_dashed = [s for s in segments if s[1] == "Vertical" and len(s) > 6 and s[6] == "Dashed"]

        def fuse_group(segs, style="Solid"):
            if not segs:
                return []
            segs = sorted(segs, key=lambda s: (self._axis_val(s), self._span(s)[0]))
            n = len(segs)
            parent = list(range(n))

            def find(a):
                while parent[a] != a:
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(n):
                for j in range(i + 1, n):
                    if abs(self._axis_val(segs[i]) - self._axis_val(segs[j])) > axis_tol:
                        continue
                    si, ei = self._span(segs[i])
                    sj, ej = self._span(segs[j])
                    gap = max(0, max(si, sj) - min(ei, ej))
                    if gap <= touch_tol:
                        union(i, j)

            clusters = {}
            for i in range(n):
                clusters.setdefault(find(i), []).append(i)

            fused = []
            for members in clusters.values():
                if len(members) == 1:
                    fused.append(list(segs[members[0]]))
                    continue
                ax_avg = sum(self._axis_val(segs[m]) for m in members) / len(members)
                starts = [self._span(segs[m])[0] for m in members]
                ends   = [self._span(segs[m])[1] for m in members]
                s, e = min(starts), max(ends)
                ori = segs[members[0]][1]
                ax = round(ax_avg)
                if ori == "Horizontal":
                    fused.append([0, ori, s, ax, e, ax, style])
                else:
                    fused.append([0, ori, ax, s, ax, e, style])
            return fused

        fused_h_solid = fuse_group(h_segs_solid, "Solid")
        fused_v_solid = fuse_group(v_segs_solid, "Solid")
        fused_h_dashed = fuse_group(h_segs_dashed, "Dashed")
        fused_v_dashed = fuse_group(v_segs_dashed, "Dashed")

        all_fused = fused_h_solid + fused_v_solid + fused_h_dashed + fused_v_dashed
        for idx, row in enumerate(all_fused, start=1):
            row[0] = idx

        before, after = len(segments), len(all_fused)
        print(f"De-duplication: {before} raw segments → {after} unique segments "
              f"({before - after} duplicate detections fused)")

        self._segments = all_fused
        return all_fused

    # ------------------------------------------------------------------
    # STEP 6d – DROP SHORT, UNGROUPED STRAY FRAGMENTS
    #
    # Runs the same connectivity test as group_lines() purely to check
    # whether each segment has a real partner. Any segment left in a
    # single-member cluster AND shorter than ISOLATED_MIN_LENGTH is
    # dropped outright — not drawn, not numbered, not in any CSV.
    # ------------------------------------------------------------------

    def filter_isolated_fragments(self, segments=None, axis_tol=None,
                                  gap_tol=None, elbow_tol=None,
                                  isolated_min_len=None):
        """
        Drop short segments that are completely isolated — i.e., they form a
        single-member cluster when the same connectivity test used in
        ``group_lines`` is applied — and whose length falls below
        ``cfg.ISOLATED_MIN_LENGTH``. Segments that connect to at least one other
        segment (straight-with-gap or elbow) are always retained regardless of
        length. The filter is a no-op when ``ISOLATED_MIN_LENGTH`` is 0 or unset.

        :param segments        : List of ``[seg_id, orientation, x1, y1, x2, y2]`` rows
                                 to filter; defaults to ``self._segments``
        :type  segments        : list[list] or None
        :param axis_tol        : Perpendicular-axis tolerance for straight grouping;
                                 defaults to ``cfg.GROUP_AXIS_TOLERANCE``
        :type  axis_tol        : int or None
        :param gap_tol         : Maximum span gap to still consider segments connected;
                                 defaults to ``cfg.GROUP_GAP_TOLERANCE``
        :type  gap_tol         : int or None
        :param elbow_tol       : Maximum endpoint distance for elbow detection;
                                 defaults to ``cfg.ELBOW_MERGE_TOLERANCE``
        :type  elbow_tol       : int or None
        :param isolated_min_len: Segments shorter than this value are dropped if
                                 isolated; defaults to ``cfg.ISOLATED_MIN_LENGTH``
        :type  isolated_min_len: int or None

        :return: Filtered list of ``[seg_id, orientation, x1, y1, x2, y2]`` rows
                 with IDs reassigned from 1
        :rtype : list[list]
        """
        if segments is None:
            segments = self._segments
        if axis_tol is None:
            axis_tol = self.cfg.GROUP_AXIS_TOLERANCE
        if gap_tol is None:
            gap_tol = self.cfg.GROUP_GAP_TOLERANCE
        if elbow_tol is None:
            elbow_tol = self.cfg.ELBOW_MERGE_TOLERANCE
        if isolated_min_len is None:
            isolated_min_len = getattr(self.cfg, "ISOLATED_MIN_LENGTH", 0)

        if not isolated_min_len:
            return segments  # filter disabled

        n = len(segments)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            si = segments[i]
            for j in range(i + 1, n):
                sj = segments[j]
                if si[1] == sj[1] and (si[6] if len(si)>6 else "Solid") == (sj[6] if len(sj)>6 else "Solid"):
                    if abs(self._axis_val(si) - self._axis_val(sj)) > axis_tol:
                        continue
                    s0, e0 = self._span(si)
                    s1, e1 = self._span(sj)
                    gap = max(0, max(s0, s1) - min(e0, e1))
                    if gap <= gap_tol:
                        union(i, j)
                elif (si[6] if len(si)>6 else "Solid") == (sj[6] if len(sj)>6 else "Solid"):
                    pts_i = self._endpoints(si)
                    pts_j = self._endpoints(sj)
                    if any(self._dist(a, b) <= elbow_tol for a in pts_i for b in pts_j):
                        union(i, j)

        clusters = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        kept, dropped = [], 0
        for members in clusters.values():
            if len(members) == 1:
                _, ori, x1, y1, x2, y2 = segments[members[0]][:6]
                style = segments[members[0]][6] if len(segments[members[0]]) > 6 else "Solid"
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length < isolated_min_len:
                    dropped += 1
                    continue
            for m in members:
                kept.append(segments[m])

        for idx, row in enumerate(kept, start=1):
            row[0] = idx

        print(f"Isolated-fragment filter: dropped {dropped} short, ungrouped "
              f"stray segments (< {isolated_min_len}px); kept {len(kept)}")

        self._segments = kept
        return kept

    # ------------------------------------------------------------------
    # STEP 6c – GAP-PRESERVING GROUPING  — REQUIREMENT 2
    #
    # Groups real segments into logical pipe runs WITHOUT altering any
    # coordinate. A group is either:
    #   - a single segment                       -> Type = Horizontal/Vertical
    #   - same-orientation segments interrupted
    #     by a component (gap <= GROUP_GAP_TOLERANCE)   -> Type = "Straight"
    #   - segments that change orientation at a
    #     shared/near-touching joint (a bend)           -> Type = "Elbow"
    # Every member segment keeps its own true coordinates, so component
    # gaps remain exactly the size/location they were originally.
    # ------------------------------------------------------------------

    def group_lines(self, segments=None, axis_tol=None, gap_tol=None, elbow_tol=None):
        """
        Group de-duplicated segments into logical pipe runs using a Union-Find
        structure without altering any segment coordinate. Two segments are
        connected if they share the same orientation and their spans overlap or
        are separated by a gap no larger than ``cfg.GROUP_GAP_TOLERANCE``
        (straight-with-gap), or if they have different orientations and their
        endpoints are within ``cfg.ELBOW_MERGE_TOLERANCE`` pixels of each other
        (elbow/bend). Groups larger than ``cfg.MAX_GROUP_SIZE`` are split back
        into single-segment groups to prevent transitive over-merging.

        Group types assigned:
          - Single segment → ``Horizontal`` or ``Vertical``
          - Multiple same-orientation segments with gap → ``Straight``
          - Mixed-orientation segments at a bend → ``Elbow``

        :param segments : List of ``[seg_id, orientation, x1, y1, x2, y2]`` rows
                          to group; defaults to ``self._segments``
        :type  segments : list[list] or None
        :param axis_tol : Maximum perpendicular-axis deviation in pixels for
                          straight grouping; defaults to ``cfg.GROUP_AXIS_TOLERANCE``
        :type  axis_tol : int or None
        :param gap_tol  : Maximum span gap in pixels to still connect two same-orientation
                          segments; defaults to ``cfg.GROUP_GAP_TOLERANCE``
        :type  gap_tol  : int or None
        :param elbow_tol: Maximum endpoint distance in pixels for elbow connection;
                          defaults to ``cfg.ELBOW_MERGE_TOLERANCE``
        :type  elbow_tol: int or None

        :return: List of ``[Group_ID, Segment_In_Group, Type, Orientation, x1, y1, x2, y2]`` rows;
                 also stored in ``self._grouped``
        :rtype : list[list]

        :raises ValueError: If no segments are available
        """
        if segments is None:
            if not self._segments:
                raise ValueError("No segments. Call detect_lines_ppht() (and dedupe_segments()) first.")
            segments = self._segments
        if axis_tol is None:
            axis_tol = self.cfg.GROUP_AXIS_TOLERANCE
        if gap_tol is None:
            gap_tol = self.cfg.GROUP_GAP_TOLERANCE
        if elbow_tol is None:
            elbow_tol = self.cfg.ELBOW_MERGE_TOLERANCE

        n = len(segments)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            si = segments[i]
            for j in range(i + 1, n):
                sj = segments[j]
                if si[1] == sj[1] and (si[6] if len(si)>6 else "Solid") == (sj[6] if len(sj)>6 else "Solid"):
                    # same orientation -> straight-with-gap connection
                    if abs(self._axis_val(si) - self._axis_val(sj)) > axis_tol:
                        continue
                    s0, e0 = self._span(si)
                    s1, e1 = self._span(sj)
                    gap = max(0, max(s0, s1) - min(e0, e1))
                    if gap <= gap_tol:
                        union(i, j)
                else:
                    # different orientation -> elbow (bend) connection
                    pts_i = self._endpoints(si)
                    pts_j = self._endpoints(sj)
                    if any(self._dist(a, b) <= elbow_tol for a in pts_i for b in pts_j):
                        union(i, j)

        clusters = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        # stable ordering: by the smallest original segment id in the cluster
        ordered_clusters = sorted(
            clusters.values(), key=lambda members: min(segments[m][0] for m in members)
        )

        # AFTER
        max_group_size = getattr(self.cfg, "MAX_GROUP_SIZE", 6)

        grouped = []
        group_labels = []
        group_id = 0
        for members in ordered_clusters:
            # Break transitive over-merges: any group larger than max_group_size
            # is split back into individual single-segment groups
            if len(members) > max_group_size:
                for m in members:
                    group_id += 1
                    _, ori, x1, y1, x2, y2 = segments[m][:6]
                    style = segments[m][6] if len(segments[m]) > 6 else "Solid"
                    grouped.append([group_id, 1, ori, ori, x1, y1, x2, y2])
                    # label at midpoint of the segment itself
                    group_labels.append((group_id, ori,
                                         int((x1 + x2) / 2),
                                         int((y1 + y2) / 2)))
                continue

            group_id += 1
            orientations = {segments[m][1] for m in members}

            if len(members) == 1:
                line_type = segments[members[0]][1]
            elif len(orientations) > 1:
                line_type = "Elbow"
            else:
                line_type = "Straight"

            xs, ys = [], []
            for seg_order, m in enumerate(members, start=1):
                _, ori, x1, y1, x2, y2 = segments[m][:6]
                style = segments[m][6] if len(segments[m]) > 6 else "Solid"
                grouped.append([group_id, seg_order, line_type, ori, x1, y1, x2, y2, style])
                xs.extend([x1, x2])
                ys.extend([y1, y2])

            label_x = int(sum(xs) / len(xs))
            label_y = int(sum(ys) / len(ys))
            group_labels.append((group_id, line_type, label_x, label_y))

        n_elbow = sum(1 for r in group_labels if r[1] == "Elbow")
        n_straight = sum(1 for r in group_labels if r[1] == "Straight")
        n_single = len(group_labels) - n_elbow - n_straight

        print(
            f"Grouping: {n} segments → {len(group_labels)} logical lines "
            f"(single: {n_single}, straight-with-gap: {n_straight}, elbow: {n_elbow})"
        )

        self._grouped = grouped
        self._group_labels = group_labels
        return grouped

    # ------------------------------------------------------------------
    # STEP 7 – SAVE RAW (de-duplicated, filtered) SEGMENTS CSV
    # ------------------------------------------------------------------

    def save_csv(self, segments=None, out_dir=None, filename=None):
        """
        Write de-duplicated, length-filtered segments to a CSV file with one row
        per segment. Columns: Line_Number, Orientation, x_start, y_start, x_end,
        y_end. Output directory is created if it does not exist.

        :param segments: List of ``[seg_id, orientation, x1, y1, x2, y2]`` rows to write;
                         defaults to ``self._segments``
        :type  segments: list[list] or None
        :param out_dir : Directory to write the CSV into; defaults to ``cfg.LE_CSV_DIR``
        :type  out_dir : str or None
        :param filename: Output filename; defaults to ``cfg.CSV_FILENAME``
        :type  filename: str or None

        :return: DataFrame containing the written segment data
        :rtype : pd.DataFrame

        :raises ValueError: If no segments are available
        :raises OSError   : If the output directory cannot be created
        """
        if segments is None:
            if not self._segments:
                raise ValueError("No segments. Call detect_lines_ppht() first.")
            segments = self._segments

        if out_dir  is None: out_dir  = self.cfg.LE_CSV_DIR
        if filename is None: filename = self.cfg.CSV_FILENAME

        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(
            segments,
            columns=["Line_Number", "Orientation",
                     "x_start", "y_start", "x_end", "y_end", "Line_Style"],
        )

        csv_path = os.path.join(out_dir, filename)
        df.to_csv(csv_path, index=False)
        print(f"Raw segment CSV saved → {csv_path}")

        self._df = df
        return df

    # ------------------------------------------------------------------
    # STEP 7b – SAVE GROUPED CSV
    # ------------------------------------------------------------------

    def save_grouped_csv(self, grouped=None, out_dir=None, filename=None):
        """
        Write grouped segment rows to a CSV file with one row per real segment,
        tagged with Group_ID, Segment_In_Group, and Type so segments belonging
        to the same logical pipe run can be identified without their coordinates
        having been altered. Output directory is created if it does not exist.

        :param grouped  : List of ``[Group_ID, Segment_In_Group, Type, Orientation,
                          x1, y1, x2, y2]`` rows to write; regenerated from
                          ``self._segments`` if not supplied
        :type  grouped  : list[list] or None
        :param out_dir  : Directory to write the CSV into; defaults to ``cfg.LE_CSV_DIR``
        :type  out_dir  : str or None
        :param filename : Output filename; defaults to ``cfg.GROUPED_CSV_FILENAME``
        :type  filename : str or None

        :return: DataFrame containing the written grouped segment data
        :rtype : pd.DataFrame

        :raises OSError: If the output directory cannot be created
        """
        if grouped is None:
            grouped = self.group_lines()

        if out_dir is None: out_dir = self.cfg.LE_CSV_DIR
        if filename is None: filename = self.cfg.GROUPED_CSV_FILENAME

        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(
            grouped,
            columns=["Group_ID", "Segment_In_Group", "Type", "Orientation",
                     "x_start", "y_start", "x_end", "y_end", "Line_Style"],
        )

        csv_path = os.path.join(out_dir, filename)
        df.to_csv(csv_path, index=False)
        print(f"Grouped CSV saved → {csv_path}")

        self._grouped_df = df
        return df

    # ------------------------------------------------------------------
    # STEP 8 – SAVE MORPH MASK IMAGE
    # ------------------------------------------------------------------

    def save_morph_image(self, line_mask=None, out_dir=None, filename=None):
        """
        Save the morphological line mask as a PNG image to the LE_images output
        directory, creating it if it does not exist.

        :param line_mask: Binary morphological mask to save; defaults to ``self._line_mask``
        :type  line_mask: np.ndarray or None
        :param out_dir  : Directory to save the image into; defaults to ``cfg.LE_IMAGES_DIR``
        :type  out_dir  : str or None
        :param filename : Output filename; defaults to ``cfg.MORPH_IMAGE_FILENAME``
        :type  filename : str or None

        :return: Full file path where the image was saved
        :rtype : str

        :raises ValueError: If no line mask is available
        :raises OSError   : If the output directory cannot be created
        """
        if line_mask is None:
            if self._line_mask is None:
                raise ValueError("No line mask. Call extract_morph_lines() first.")
            line_mask = self._line_mask

        if out_dir  is None: out_dir  = self.cfg.LE_IMAGES_DIR
        if filename is None: filename = self.cfg.MORPH_IMAGE_FILENAME

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, line_mask)
        print(f"Morph mask image saved → {path}")
        return path

    # ------------------------------------------------------------------
    # SHARED CANVAS BUILDER for STEP 9 / STEP 10
    #
    # Draws every REAL segment at its true position (gaps preserved)
    # and places exactly ONE label per logical group at that group's
    # centroid, nudged to avoid overlap.
    # ------------------------------------------------------------------

    def _build_labelled_canvas(self, img, segments, group_labels):
        """
        Construct a white canvas the same size as ``img``, draw every real
        segment at its true pixel position (preserving all component gaps), then
        place exactly one group-ID label per logical group at a nudged centroid
        position to avoid label overlap. Labels are drawn over a white rectangle
        background for legibility.

        :param img         : Reference image used only for canvas dimensions
        :type  img         : np.ndarray
        :param segments    : List of ``[seg_id, orientation, x1, y1, x2, y2]`` rows
                             to draw
        :type  segments    : list[list]
        :param group_labels: List of ``(Group_ID, Type, label_x, label_y)`` tuples
                             used to position group number labels
        :type  group_labels: list[tuple]

        :return: BGR canvas image with all segments and group labels drawn
        :rtype : np.ndarray
        """
        cfg = self.cfg
        canvas = np.ones_like(img) * 255

        # draw every real segment exactly where it was detected
        for seg in segments:
            _, _, x1, y1, x2, y2 = seg[:6]
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 2)

        label_positions = []
        # AFTER
        # Build a lookup of segment midpoints for each group (for label snapping)
        group_midpoints = {}
        for seg in segments:
            _, _, x1, y1, x2, y2 = seg[:6]
            # seg[0] is seg_id, not group_id — use group_labels order instead
        # Rebuild from grouped data if available (fall back to group_labels centroid)
        grouped_by_gid = {}
        for seg in (self._grouped if self._grouped else []):
            gid = seg[0]
            grouped_by_gid.setdefault(gid, []).append(seg)

        # AFTER
        for group_id, line_type, cx, cy in group_labels:
            if group_id in grouped_by_gid:
                first_seg = grouped_by_gid[group_id][0]
                _, _, _, _, fx1, fy1, fx2, fy2 = first_seg
                snap_x = int((fx1 + fx2) / 2)
                snap_y = int((fy1 + fy2) / 2)
            else:
                snap_x, snap_y = cx, cy

            if line_type == "Horizontal":
                tx, ty = snap_x, snap_y + cfg.LABEL_H_OFFSET_Y
            elif line_type == "Vertical":
                tx, ty = snap_x + cfg.LABEL_V_OFFSET_X, snap_y
            else:
                tx, ty = snap_x + cfg.LABEL_V_OFFSET_X, snap_y + cfg.LABEL_H_OFFSET_Y
            while any(
                    abs(tx - px) < cfg.LABEL_OVERLAP_DX and
                    abs(ty - py) < cfg.LABEL_OVERLAP_DY
                    for px, py in label_positions
            ):
                ty += cfg.LABEL_NUDGE_Y
                tx += cfg.LABEL_NUDGE_X

            label_positions.append((tx, ty))

            text = str(group_id)
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX,
                cfg.LABEL_FONT_SCALE, cfg.LABEL_THICKNESS
            )

            m = cfg.LABEL_MARGIN
            tx = max(m, min(tx, canvas.shape[1] - tw - m))
            ty = max(th + m, min(ty, canvas.shape[0] - m))

            cv2.rectangle(
                canvas,
                (tx - m, ty - th - m),
                (tx + tw + m, ty + m),
                (255, 255, 255), -1
            )
            cv2.putText(
                canvas, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.LABEL_FONT_SCALE,
                cfg.LABEL_COLOR_BGR,
                cfg.LABEL_THICKNESS,
            )
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX,
                cfg.LABEL_FONT_SCALE, cfg.LABEL_THICKNESS
            )

            m = cfg.LABEL_MARGIN
            tx = max(m, min(tx, canvas.shape[1] - tw - m))
            ty = max(th + m, min(ty, canvas.shape[0] - m))

            cv2.rectangle(
                canvas,
                (tx - m, ty - th - m),
                (tx + tw + m, ty + m),
                (255, 255, 255), -1
            )
            cv2.putText(
                canvas, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.LABEL_FONT_SCALE,
                cfg.LABEL_COLOR_BGR,
                cfg.LABEL_THICKNESS,
            )

        return canvas

    # ------------------------------------------------------------------
    # STEP 9 – SAVE NUMBERED LINES IMAGE (no-overlap group labels)
    # ------------------------------------------------------------------

    def save_numbered_image(self, img=None, segments=None, group_labels=None,
                            out_dir=None, filename=None):
        """
        Render every real segment onto a white canvas at its true pixel position
        (component gaps remain intact) with one group-ID label per logical group,
        and save the result as a PNG. Uses ``_build_labelled_canvas`` internally.

        :param img         : Reference image for canvas size; defaults to ``self._orig_img``
        :type  img         : np.ndarray or None
        :param segments    : Segment list to draw; defaults to ``self._segments``
        :type  segments    : list[list] or None
        :param group_labels: Group label list; defaults to ``self._group_labels``,
                             regenerated via ``group_lines`` if empty
        :type  group_labels: list[tuple] or None
        :param out_dir     : Output directory; defaults to ``cfg.LE_IMAGES_DIR``
        :type  out_dir     : str or None
        :param filename    : Output filename; defaults to ``cfg.NUMBERED_IMAGE_FILENAME``
        :type  filename    : str or None

        :return: Full file path where the image was saved
        :rtype : str

        :raises ValueError: If no image or segments are available
        :raises OSError   : If the output directory cannot be created
        """
        if img is None:
            if self._orig_img is None:
                raise ValueError("No image. Call load_file() first.")
            img = self._orig_img
        if segments is None:
            if not self._segments:
                raise ValueError("No segments. Call detect_lines_ppht() first.")
            segments = self._segments
        if group_labels is None:
            if not self._group_labels:
                self.group_lines(segments)
            group_labels = self._group_labels

        if out_dir  is None: out_dir  = self.cfg.LE_IMAGES_DIR
        if filename is None: filename = self.cfg.NUMBERED_IMAGE_FILENAME

        os.makedirs(out_dir, exist_ok=True)

        canvas = self._build_labelled_canvas(img, segments, group_labels)

        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, canvas)
        print(f"Numbered lines image saved → {path}")
        return path

    # ------------------------------------------------------------------
    # STEP 10 – SAVE AXIS IMAGE (matplotlib)
    # ------------------------------------------------------------------

    def save_axis_image(self, img=None, segments=None, group_labels=None,
                        out_dir=None, filename=None):
        """
        Produce the same labelled segment canvas as ``save_numbered_image`` but
        wrap it in a Matplotlib figure with a pixel-coordinate axis grid, tick
        marks, title, and axis labels, then save it as a PNG via Matplotlib.

        :param img         : Reference image for canvas size; defaults to ``self._orig_img``
        :type  img         : np.ndarray or None
        :param segments    : Segment list to draw; defaults to ``self._segments``
        :type  segments    : list[list] or None
        :param group_labels: Group label list; defaults to ``self._group_labels``,
                             regenerated via ``group_lines`` if empty
        :type  group_labels: list[tuple] or None
        :param out_dir     : Output directory; defaults to ``cfg.LE_IMAGES_DIR``
        :type  out_dir     : str or None
        :param filename    : Output filename; defaults to ``cfg.AXIS_IMAGE_FILENAME``
        :type  filename    : str or None

        :return: Full file path where the image was saved
        :rtype : str

        :raises ValueError: If no image or segments are available
        :raises OSError   : If the output directory cannot be created
        """
        if img is None:
            if self._orig_img is None:
                raise ValueError("No image. Call load_file() first.")
            img = self._orig_img
        if segments is None:
            if not self._segments:
                raise ValueError("No segments. Call detect_lines_ppht() first.")
            segments = self._segments
        if group_labels is None:
            if not self._group_labels:
                self.group_lines(segments)
            group_labels = self._group_labels

        if out_dir  is None: out_dir  = self.cfg.LE_IMAGES_DIR
        if filename is None: filename = self.cfg.AXIS_IMAGE_FILENAME

        os.makedirs(out_dir, exist_ok=True)

        canvas = self._build_labelled_canvas(img, segments, group_labels)
        cfg = self.cfg

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w = canvas.shape[:2]

        fig, ax = plt.subplots(figsize=cfg.AXIS_FIG_SIZE)
        ax.imshow(rgb)
        ax.set_title(cfg.AXIS_TITLE, fontsize=18, pad=20)
        ax.set_xlabel(cfg.AXIS_XLABEL, fontsize=14)
        ax.set_ylabel(cfg.AXIS_YLABEL, fontsize=14)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        step = cfg.AXIS_TICK_STEP
        ax.set_xticks(np.arange(0, w, step))
        ax.set_yticks(np.arange(0, h, step))
        ax.tick_params(axis="x", labelrotation=90, labelsize=10, pad=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, alpha=0.35)

        plt.subplots_adjust(bottom=0.18, left=0.12, right=0.95, top=0.92)

        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=cfg.AXIS_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Axis image saved → {path}")
        return path

    # ------------------------------------------------------------------
    # CONVENIENCE – run the full pipeline in one call
    # ------------------------------------------------------------------

    def run(self, input_path=None):
        """
        Execute the complete line extraction pipeline in the standard step order:
        load_file → crop_image → preprocess → extract_morph_lines → detect_edges
        → detect_lines_ppht → dedupe_segments → filter_isolated_fragments →
        group_lines → save_csv → save_grouped_csv → save_morph_image →
        save_numbered_image → save_axis_image.

        :param input_path: Optional path to override ``cfg.INPUT_PATH`` for this run
        :type  input_path: str or None

        :return: Dict with keys ``df`` (raw segment DataFrame), ``grouped_df``
                 (grouped segment DataFrame), ``morph_path`` (str),
                 ``numbered_path`` (str), and ``axis_path`` (str)
        :rtype : dict
        """
        if input_path:
            self.cfg.INPUT_PATH = input_path

        self.load_file()
        self.crop_image()
        self.preprocess()
        self.extract_morph_lines()
        self.detect_edges()
        self.detect_lines_ppht()
        self.dedupe_segments()
        self.filter_isolated_fragments()
        self.group_lines()

        df = self.save_csv()
        grouped_df = self.save_grouped_csv()
        morph_path = self.save_morph_image()
        numbered_path = self.save_numbered_image()
        axis_path = self.save_axis_image()

        return {
            "df": df,
            "grouped_df": grouped_df,
            "morph_path": morph_path,
            "numbered_path": numbered_path,
            "axis_path": axis_path,
        }