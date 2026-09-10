# ============================================================
# config_line_tag_merge.py
# Configuration for the Line ↔ Tag Association Pipeline
# ============================================================

import os

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BASE)

# Input: grouped line CSV from Stage 3 (LE)
LINE_CSV_PATH = os.path.join(
    _BASE, "Results", "LE_csv", "02.Unmarked_geometric_filtered_lines.csv"
)

# Input: classified tags CSV from Stage 1 (TE)
TAG_CSV_PATH = os.path.join(
    _REPO, "TE", "Results", "TE_csv", "process_tags.csv"
)

# Input: the original PDF (needed only for coordinate transform)
PDF_PATH = r"C:\Users\user\Desktop\Work\pid_digitization\data\raw\02.Unmarked.pdf"
PDF_PAGE_INDEX = 0       # 0-based page index

# Output directory
OUTPUT_DIR = os.path.join(_BASE, "Results", "LE_csv")
OUTPUT_FILENAME = "02.Unmarked_line_tag_topology.csv"

# ------------------------------------------------------------------
# COORDINATE TRANSFORM
# Set True if tag bboxes are in PDF-point space and need converting
# to pixel space (required when the PDF page has a Rotate value).
# ------------------------------------------------------------------
APPLY_COORDINATE_TRANSFORM = True
PDF_DPI = 300            # must match the DPI used in Line Extraction

# ------------------------------------------------------------------
# TAG FILTERING
# Only process tags that belong to the target file.
# ------------------------------------------------------------------
TARGET_FILE_NAME = "02.Unmarked"

# Tag types to exclude from matching (title block noise, etc.)
EXCLUDED_TAG_TYPES = ["General"]

# ------------------------------------------------------------------
# SQUARE / BOUNDING-BOX FILTER
# Groups whose segments form a closed rectangle are instrument boxes,
# not real pipe runs. They are removed before any association.
# ------------------------------------------------------------------
SQUARE_FILTER_ENABLED   = True
SQUARE_MAX_SEGMENTS     = 6       # groups with more segments are never squares
SQUARE_COORD_TOLERANCE  = 15      # px — how close corners must be to snap

# ------------------------------------------------------------------
# LINE MERGING
# After filtering, each group is collapsed into a single row showing
# its true start and end coordinates.
# ------------------------------------------------------------------
MERGE_ENABLED           = True
SEGMENT_SNAP_TOLERANCE  = 15      # px — endpoint snap used to find exterior ends

# ------------------------------------------------------------------
# TAG MATCHING
# ------------------------------------------------------------------
USE_SEGMENT_PROXIMITY = True       # True = perpendicular dist to segment (recommended)
MAX_MATCH_DISTANCE    = 500        # px — tags farther than this are not matched
CONFIDENCE_DECIMALS   = 3

# ------------------------------------------------------------------
# ENDPOINT INSTRUMENT ASSOCIATION
# After merging, find which instrument sits closest to each endpoint.
# ------------------------------------------------------------------
ENDPOINT_INSTRUMENT_TYPES = [
    "Instrument", "Sensor", "Equipment",
    "Pressure Safety Valve", "Thermowell",
    "Piping Component", "Special Piping Item",
    "Corrosion Monitoring System",
]
ENDPOINT_PROXIMITY = 400          # px — max distance to match an instrument to an endpoint
