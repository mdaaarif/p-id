# ─────────────────────────────────────────────────────────────────────────────
# Config_CE.py  –  Component Extraction Pipeline Configuration
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CE_DIR          = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR       = os.path.join(BASE_DIR, "data", "raw")          # source PDFs
# Defaults retain compatibility with the established pipeline. Overrides make
# trial runs safe by keeping their inputs and outputs separate from baselines.
TAG_CSV_DIR     = os.environ.get(
    "CE_TAG_CSV_DIR", os.path.join(BASE_DIR, "TE", "Results", "TE_per_pdf")
)
LEGEND_PDF      = os.path.join(BASE_DIR, "data", "legend", "pid-legend_labels.pdf")

RESULTS_DIR     = os.environ.get("CE_RESULTS_DIR", os.path.join(CE_DIR, "Results"))
OUT_CSV_DIR     = os.path.join(RESULTS_DIR, "CE_csv_1")
OUT_JSON_DIR    = os.path.join(RESULTS_DIR, "CE_json_1")
OUT_IMG_DIR     = os.path.join(RESULTS_DIR, "CE_images_1")
OUT_PRECLF_DIR  = os.path.join(RESULTS_DIR, "CE_pre_classification_1")
OUT_COMPARE_DIR = os.path.join(RESULTS_DIR, "CE_comparison_1")

# ── PDF rendering ─────────────────────────────────────────────────────────────
RENDER_DPI      = 150          # DPI for rasterising P&ID pages for visualisation
LEGEND_DPI      = 200          # DPI for rasterising legend symbols

# ── Path / geometry ───────────────────────────────────────────────────────────
# Paths whose bounding box falls below these dims are considered sub-strokes
# (hatching, tick marks) and are skipped during clustering
MIN_PATH_W      = 3.0          # pts
MIN_PATH_H      = 3.0          # pts

# DBSCAN clustering: paths within CLUSTER_EPS pts of each other are grouped
CLUSTER_EPS     = 18.0         # pts  (≈ one symbol body at 150 DPI)
CLUSTER_MIN_PTS = 2            # minimum paths to form a cluster

# After clustering, clusters smaller than this bbox are noise
MIN_CLUSTER_W   = 12.0
MIN_CLUSTER_H   = 12.0

# How far (pts) outside a component bbox to look for tag text
TAG_SEARCH_RADIUS = 60.0       # pts

# ── Instrument-bubble geometry ────────────────────────────────────────────────
# ISA 5.1 – circles and squares up to this size are instrument bubbles
BUBBLE_MIN_SIZE = 20.0         # pts  (min width or height)
BUBBLE_MAX_SIZE = 90.0         # pts  (max width or height)
CIRCLE_ASPECT   = 0.65         # min ratio of (min/max) side to call a path a circle



# ── Template matching ─────────────────────────────────────────────────────────
TEMPLATE_MATCH_THRESHOLD = 0.40   # cv2.matchTemplate score  (0-1, lower = worse)
TEMPLATE_SCALE_FACTORS   = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
TEMPLATE_ROTATIONS       = [0, 90, 180, 270]  # degrees to try
NMS_OVERLAP_THRESH       = 0.30   # IoU threshold for non-max suppression

# ── Symbol group catalogue ────────────────────────────────────────────────────
# Top-level groups and their ISA letter codes (first letter of function code)
ISA_FUNCTION_GROUPS = {
    "F": "Flow",
    "L": "Level",
    "P": "Pressure",
    "T": "Temperature",
    "A": "Analyzer",
    "C": "Control",
    "S": "Speed/Safety",
    "X": "Unclassified",
    "Z": "Position/Safety",
    "H": "Hand",
    "E": "Voltage/Sensor-Primary",
    "V": "Vibration/Viscosity",
    "W": "Weight/Force",
    "D": "Density",
    "Y": "Relay/Computation",
}

# Second letter → modifier
ISA_MODIFIER = {
    "I": "Indicator",
    "T": "Transmitter",
    "C": "Controller",
    "R": "Recorder",
    "E": "Primary Element",
    "G": "Glass/Gauge",
    "A": "Alarm",
    "S": "Switch",
    "V": "Valve",
    "Y": "Relay/Converter",
    "K": "Control Station",
    "W": "Well/Thermowell",
}

# Valve keyword patterns (used to classify valve clusters)
VALVE_KEYWORDS = [
    "gate", "globe", "ball", "butterfly", "check", "control",
    "needle", "plug", "diaphragm", "relief", "safety", "psv",
    "angle", "3-way", "4-way", "knife", "pinch", "solenoid",
]

# ── Output ────────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "symbol_id", "type", "group", "tag",
    "mounting", "location_xy", "bounding_box", "nearby_text",
]

# Colour palette for annotated output image  (BGR for OpenCV)
BBOX_COLORS = {
    "Instrument":    (0,   200, 255),   # cyan
    "Valve":         (0,   128, 0),     # green
    "Pump":          (255, 128, 0),     # orange
    "Heat Exchanger":(0,   0,   220),   # red
    "Line":          (180, 180, 180),   # grey
    "Unknown":       (150, 0,   150),   # purple
}
BBOX_THICKNESS  = 2
LABEL_FONT_SCALE = 0.4
