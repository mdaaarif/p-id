# ============================================================
# config/config_line_ext_3.py
# Configuration for P&ID Line Extraction Pipeline (v3)
#
# v3 changes vs LE_2 / CONFIG_LE_2:
#   - Adds a de-duplication pass (DEDUPE_*) that fuses truly
#     overlapping/duplicate Hough detections of the SAME physical
#     stroke into one segment (tiny tolerance only).
#   - Replaces the old "collinear merge" (which redrew one long
#     solid line straight through a valve/instrument gap) and the
#     old "elbow merge" (which redrew a bend as a fake diagonal
#     line) with a single GROUPING pass. Grouping never invents or
#     deletes geometry — every real detected segment is still drawn
#     exactly where it was found (so equipment/component gaps stay
#     visually intact), but segments that belong to the same
#     physical pipe run (straight-with-gap, or bending H->V) share
#     one Line/Group number and get a "Type" of Horizontal /
#     Vertical / Straight / Elbow.
# ============================================================

import os

# ------------------------------------------------------------------
# DIRECTORY PATHS
# ------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PDF_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
LE_CSV_DIR    = os.path.join(RESULTS_DIR, "LE_csv")
LE_IMAGES_DIR = os.path.join(RESULTS_DIR, "LE_images")

# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------
INPUT_PATH = INPUT_PDF_DIR      # folder — run/line_ext_3.py iterates all PDFs in it

# ------------------------------------------------------------------
# PDF RASTERISATION
# ------------------------------------------------------------------
PDF_DPI = 300

# ------------------------------------------------------------------
# MARGINS  (pixels, applied after rasterising at PDF_DPI)
#
# Pixels stripped from each border before line extraction, to
# exclude the title block, notes area, and drawing frame.
# Set a margin to 0 to keep the full edge on that side.
# All detected line coordinates are reported back in the ORIGINAL
# (pre-crop) pixel coordinate system automatically.
# ------------------------------------------------------------------
MARGIN_LEFT   = 250
MARGIN_RIGHT  = 1750
MARGIN_TOP    = 600
MARGIN_BOTTOM = 800

# ------------------------------------------------------------------
# BINARY THRESHOLD
# ------------------------------------------------------------------
BINARY_THRESH = 200

# ------------------------------------------------------------------
# MORPHOLOGICAL KERNELS
# ------------------------------------------------------------------
MORPH_H_KERNEL   = (20, 1)
MORPH_V_KERNEL   = (1, 20)
MORPH_CLOSE_K    = (3, 3)
MORPH_CLOSE_ITER = 1

# ------------------------------------------------------------------
# CANNY EDGE DETECTION
# ------------------------------------------------------------------
CANNY_THRESH1  = 50
CANNY_THRESH2  = 150
CANNY_APERTURE = 3

# ------------------------------------------------------------------
# PROBABILISTIC HOUGH TRANSFORM (PPHT)
# ------------------------------------------------------------------
PPHT_RHO          = 1
PPHT_THETA        = 180
PPHT_THRESHOLD    = 30
PPHT_MIN_LINE_LEN = 30
PPHT_MAX_LINE_GAP = 80

# ------------------------------------------------------------------
# DASHED / DOTTED LINE FILTER
#
# After a line is detected, we sample the ORIGINAL binary mask
# (before any gap-bridging) along its path. A solid line has ink
# almost the entire way along; a dashed/dotted line has real gaps.
# This is the minimum fraction of "on" pixels required to keep a
# line as solid. Lower it if real solid lines are being rejected;
# raise it if dashed lines are still slipping through.
# ------------------------------------------------------------------
MIN_LINE_SOLID_RATIO = 0.70

# ------------------------------------------------------------------
# MINIMUM LINE LENGTH FILTER  — REQUIREMENT 1  (pixels)
#
# Any detected raw segment shorter than this is discarded entirely.
# This is a hard final cutoff, separate from PPHT_MIN_LINE_LEN (which
# only controls Hough's own internal segment-building). This is what
# removes the many tiny, unwanted stray line fragments.
# ------------------------------------------------------------------
MIN_LINE_LENGTH = 25

# ------------------------------------------------------------------
# DE-DUPLICATION  (pixels)
#
# Hough frequently fires more than once on the SAME physical ink
# stroke, producing several near-identical / overlapping segments.
# These are fused into one segment BEFORE grouping so they are not
# double-counted. This tolerance must stay much smaller than any
# real component gap (see GROUP_GAP_TOLERANCE below), otherwise real
# gaps would be silently bridged.
#
# DEDUPE_AXIS_TOLERANCE — max perpendicular distance (px) between two
#   same-orientation segments to be considered the same stroke.
# DEDUPE_TOUCH_TOLERANCE — max gap (px) between the spans of two
#   same-orientation segments to be fused as literal duplicates.
#   Keep this small (a handful of pixels) — it is NOT meant to
#   bridge a valve/instrument gap, only sub-pixel Hough noise.
# ------------------------------------------------------------------
DEDUPE_AXIS_TOLERANCE  = 12
DEDUPE_TOUCH_TOLERANCE = 10

# ------------------------------------------------------------------
# ISOLATED-FRAGMENT FILTER
#
# A segment that ends up with NO partner (no same-orientation run with
# a gap, no elbow connection to anything) after the grouping test is
# almost always a leaked text stroke, tag-bubble edge, or table/box
# border rather than a real standalone pipe stub. Segments that DO get
# grouped with at least one partner are exempt — they're validated as
# part of a real continuous run and only need to pass MIN_LINE_LENGTH.
# Set to 0 to disable this filter.
# ------------------------------------------------------------------
ISOLATED_MIN_LENGTH = 50

# ------------------------------------------------------------------
# LINE GROUPING  — REQUIREMENT 2
#
# Grouping NEVER changes a segment's coordinates. It only decides
# which real, detected segments belong to the same logical pipe run
# so they can share one Line/Group number and a "Type" label. Every
# segment is still drawn at its own true position, so a gap left by
# a valve/instrument symbol remains a visible gap of the same size
# as in the original drawing.
#
# Two kinds of connections are recognised:
#
# 1) STRAIGHT-WITH-GAP  (same orientation, interrupted by a symbol)
#    GROUP_AXIS_TOLERANCE — max perpendicular deviation (px) for two
#      same-orientation segments to be considered on the same pipe
#      run (e.g. two vertical segments both at x≈500).
#    GROUP_GAP_TOLERANCE  — max gap (px) between the near ends of two
#      such segments to still group them as the same run. Set this
#      to comfortably exceed your largest symbol width/height at the
#      working DPI (a 1" symbol at 300 DPI ≈ 300 px).
#
# 2) ELBOW  (orientation changes: a run goes H then continues V)
#    ELBOW_MERGE_TOLERANCE — max distance (px) between an endpoint of
#      a horizontal segment and an endpoint of a vertical segment for
#      them to be treated as the same physical bend / joint.
# ------------------------------------------------------------------
GROUP_AXIS_TOLERANCE   = 5
GROUP_GAP_TOLERANCE    = 200
ELBOW_MERGE_TOLERANCE  = 10
MAX_GROUP_SIZE         = 6

# ------------------------------------------------------------------
# LABEL RENDERING
# ------------------------------------------------------------------
LABEL_FONT_SCALE  = 0.9
LABEL_THICKNESS   = 2
LABEL_COLOR_BGR   = (255, 0, 0)
LABEL_H_OFFSET_Y  = -25
LABEL_V_OFFSET_X  = 25
LABEL_NUDGE_Y     = 35
LABEL_NUDGE_X     = 20
LABEL_OVERLAP_DX  = 40
LABEL_OVERLAP_DY  = 30
LABEL_MARGIN      = 5

# ------------------------------------------------------------------
# AXIS IMAGE
# ------------------------------------------------------------------
AXIS_TICK_STEP = 200
AXIS_FIG_SIZE  = (22, 14)
AXIS_DPI       = 300
AXIS_TITLE     = "Extracted P&ID Lines with Coordinates (v3 — grouped)"
AXIS_XLABEL    = "X coordinate (pixels)"
AXIS_YLABEL    = "Y coordinate (pixels)"

# ------------------------------------------------------------------
# OUTPUT FILE NAMES  (overwritten per PDF in run/line_ext_3.py)
# ------------------------------------------------------------------
CSV_FILENAME            = "line_coordinates_raw.csv"
GROUPED_CSV_FILENAME    = "line_coordinates_grouped.csv"
MORPH_IMAGE_FILENAME    = "morphological_line_mask.png"
NUMBERED_IMAGE_FILENAME = "numbered_lines.png"
AXIS_IMAGE_FILENAME     = "numbered_lines_with_axis.png"