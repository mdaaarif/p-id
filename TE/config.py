"""
Configuration file for P&ID Text Extraction

KEY FIX: NOTES_MAX_X / NOTES_MAX_Y
  The previous values (NOTES_MAX_X=1675, NOTES_MAX_Y=400) were far too
  aggressive and were filtering out almost all process tags.

  Correct values derived from actual PDF coordinate analysis:
    - Page size: 2384 x 1684 (rotated 90°)
    - Notes/title block occupies: cx > 1380, cy < 450
    - Process tags live in:        cx < 1350, cy range 20–2360
"""

import os

# ===================================================================
# DIRECTORY PATHS
# ===================================================================

# BASE_DIR is the project root; this module lives in <project>/TE.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TE_DIR = os.path.dirname(os.path.abspath(__file__))

# Source drawings are deliberately separate from the symbol legend PDF.
INPUT_PDF_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Downstream CE/LE scripts consume TE/Results.  TE_RESULTS_DIR can be set for
# an isolated trial run so existing baseline artefacts are not overwritten.
RESULTS_DIR = os.environ.get("TE_RESULTS_DIR", os.path.join(TE_DIR, "Results"))
RESULTS_CSV_DIR = os.path.join(RESULTS_DIR, "TE_csv")
RESULTS_JSON_DIR = os.path.join(RESULTS_DIR, "TE_json")
RESULTS_IMAGES_DIR = os.path.join(RESULTS_DIR, "TE_images")
PER_PDF_DIR = os.path.join(RESULTS_DIR, "TE_per_pdf")

# ===================================================================
# BORDER REMOVAL SETTINGS
# ===================================================================

TOP_BORDER    = 0
BOTTOM_BORDER = 0
LEFT_BORDER   = 0
RIGHT_BORDER  = 0

# ===================================================================
# NOTES REGION SETTINGS
# ===================================================================
# The notes/title block is in the top-right of the PDF coordinate space.
# Derived from actual coordinate analysis of the P&ID pages:
#   - Title block:   cx > 1380, cy < 450
#   - Grid letters:  cy < 30  (filtered separately by is_unwanted)
#
# NOTES_MAX_X: only filter if center_x is ALSO in notes zone
# NOTES_MAX_Y: only filter if center_y is ALSO in notes zone
#
# OLD (WRONG):  NOTES_MAX_X=1675, NOTES_MAX_Y=400
#   → removed everything with cx<1675 AND cy<400 — almost all tags!
#
# NEW (CORRECT): NOTES_MAX_X=1380, NOTES_MAX_Y=450
#   → removes only the actual title/notes block in top-right corner

NOTES_MAX_X = 1380    # Title block starts at cx > 1380
NOTES_MAX_Y = 450     # Title block occupies cy < 450

# ===================================================================
# EQUIPMENT INFO BOX SETTINGS
# ===================================================================
# Equipment info table is in the bottom-right corner of the drawing.
# In PDF coords (rotated page): high cx, low cy values.
#
# Adjust these per your actual drawing layout if needed.

INFO_MAX_X = 150      # Equipment info where center_x < 150
INFO_MIN_Y = 650      # Equipment info where center_y > 650

# ===================================================================
# TAG MERGING SETTINGS
# ===================================================================
# The PDF is rotated 90°, so instrument codes (TT, FI, etc.) and their
# numbers (1202, 1201 A) are stacked HORIZONTALLY in PDF coordinates.
#
# Measured X-gap between code and number: 3–7 pixels
# Measured Y-alignment between code and number: 0–6 pixels
#
# Thresholds in Text_Extraction.py are set accordingly:
#   X_GAP_MAX = 15   (generous margin above the measured 3–7px)
#   Y_ALIGN_MAX = 10 (generous margin above the measured 0–6px)

VERTICAL_MERGE_GAP_THRESHOLD   = 15   # Max X gap for horizontal merge
INSTRUMENT_NUMBER_MERGE_DISTANCE = 10  # Max Y misalignment for merge


# ===================================================================
# VALIDATION PDF SETTINGS
# ===================================================================

BBOX_COLOR_EQUIPMENT_INFO = (1, 0, 0)   # RED  for equipment info
BBOX_COLOR_PROCESS_TAGS   = (0, 0, 1)   # BLUE for process tags
BBOX_LINE_WIDTH           = 1.5

# ===================================================================
# LOGGING SETTINGS
# ===================================================================

VERBOSE = True
SAVE_INTERMEDIATE_FILES = True

# ===================================================================
# FILE NAMING CONVENTIONS
# ===================================================================

RAW_TEXT_FILE            = "01_raw_text.csv"
CLEANED_TEXT_FILE        = "02_cleaned_text.csv"
EQUIPMENT_INFO_CSV       = "equipment_info.csv"
EQUIPMENT_INFO_JSON      = "equipment_info.json"
PROCESS_TAGS_CSV         = "process_tags.csv"
PROCESS_TAGS_JSON        = "process_tags.json"
VALIDATION_PDF           = "validation_bbox.pdf"

COMBINED_PROCESS_TAGS_CSV    = "ALL_process_tags.csv"
COMBINED_PROCESS_TAGS_JSON   = "ALL_process_tags.json"
COMBINED_EQUIPMENT_INFO_CSV  = "ALL_equipment_info.csv"
COMBINED_EQUIPMENT_INFO_JSON = "ALL_equipment_info.json"

# ===================================================================
# INSTRUMENT CODES
# ===================================================================

INSTRUMENT_CODES = {
    "SI", "PI", "TI", "FI", "LI",
    "PDI", "PDT", "TG", "TW",
    "PT", "TT", "FT", "LT",
    "PSH", "PSL", "LS", "CM",
    "P", "SP", "TE", "CP",
    "SV", "XV", "HV",
    "FCV", "PCV", "PSV",
    "LV", "FV", "TV"
}

# ===================================================================
# TAG CLASSIFICATION SETTINGS
# ===================================================================

# Single-letter equipment prefixes (ISA / plant convention)
# Used by classify_tags() in Text_Extraction.py when prefix length == 1
EQUIPMENT_PREFIX_MAP = {
    "P":  "Pump",
    "C":  "Compressor / Column",
    "E":  "Heat Exchanger",
    "V":  "Vessel / Drum",
    "R":  "Reactor",
    "F":  "Filter / Furnace",
}

# Two-letter equipment prefixes (used as a reference — classification
# is handled by the ISA last-letter rule in classify_tags, but listed
# here for documentation and future lookup needs)
EQUIPMENT_TWO_LETTER_PREFIXES = {
    "HE": "Heat Exchanger",
    "TK": "Tank",
}

# ISA function-letter sets (last letter of 2-3 letter prefix)
# E  → Sensor / Primary Element
# T  → Transmitter       → Instrument
# I  → Indicator         → Instrument
# C  → Controller        → Instrument
# R  → Recorder          → Instrument
# S  → Switch            → Instrument
# V  → Valve             → Instrument
# (These are used directly in classify_tags logic, documented here
#  so thresholds can be tuned without touching the source file.)
ISA_SENSOR_LAST_LETTER     = "E"
ISA_INSTRUMENT_LAST_LETTERS = {"T", "I", "C", "R", "S", "V"}
