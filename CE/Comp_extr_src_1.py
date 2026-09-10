# ─────────────────────────────────────────────────────────────────────────────
# Comp_extr_src.py  –  Component Extraction Source Library
#
# Approach : Template Matching (vector geometry + raster cross-check)
#
# Pipeline per PDF page
# ─────────────────────────────────────────────────────────────────────────────
#  1. extract_page_paths()      – pull all vector strokes from PDF via PyMuPDF
#  2. cluster_paths()           – DBSCAN on path centroids → candidate symbols
#  3. classify_cluster()        – rule-based classification per cluster
#       a. detect_instrument_bubble()  ISA balloon shape → Instrument
#       b. template_match_valve()      OpenCV TM against rasterised valve templates
#       c. heuristic_classify()        aspect-ratio / path-count fallback
#  4. load_tag_csv()            – read existing TE_per_pdf CSV
#  5. associate_tags()          – nearest-neighbour tag ↔ component
#  6. build_output_records()    – assemble CSV rows
#  7. draw_bboxes()             – annotate rasterised page image
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
import Config_CE_1 as CFG

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PathInfo:
    """Lightweight wrapper around a single PyMuPDF drawing path."""
    rect: fitz.Rect
    is_dashed: bool
    is_filled: bool
    fill_color: Optional[Tuple]
    has_curve: bool
    item_count: int
    centroid: Tuple[float, float]


@dataclass
class ComponentRecord:
    symbol_id: str
    type: str  # e.g. "Temperature Transmitter"
    group: str  # e.g. "Temperature"
    tag: str  # e.g. "TT-1206"
    mounting: str  # e.g. "Field-Mounted"
    location_xy: str  # "cx,cy" in PDF points
    bounding_box: str  # "x0,y0,x1,y1" in PDF points
    nearby_text: str  # raw text found in neighbourhood


# ═════════════════════════════════════════════════════════════════════════════
# 1. Vector path extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_page_paths(page: fitz.Page) -> List[PathInfo]:
    """
    Extract all drawable vector paths from a single PDF page, filtering out
    paths that are too small to represent meaningful symbols.

    :param page: A PyMuPDF page object from which drawings are extracted
    :type  page: fitz.Page

    :return: List of PathInfo objects, one per qualifying drawing path
    :rtype : List[PathInfo]
    """
    result: List[PathInfo] = []
    for p in page.get_drawings():
        r = p["rect"]
        if r.width < CFG.MIN_PATH_W or r.height < CFG.MIN_PATH_H:
            continue

        has_curve = any(item[0] == "c" for item in p["items"])
        is_dashed = p.get("dashes", "[] 0") not in ("[] 0", "")
        is_filled = p.get("fill") is not None
        fill_color = p.get("fill")

        cx = (r.x0 + r.x1) / 2.0
        cy = (r.y0 + r.y1) / 2.0

        result.append(PathInfo(
            rect=r,
            is_dashed=is_dashed,
            is_filled=is_filled,
            fill_color=fill_color,
            has_curve=has_curve,
            item_count=len(p["items"]),
            centroid=(cx, cy),
        ))
    return result


def draw_raw_paths(page: fitz.Page, paths: List[PathInfo], out_path: str) -> None:
    """
    Render a debug image of every extracted vector path onto a rasterised
    page, drawing curved paths in red and straight paths in blue.

    :param page    : PyMuPDF page to rasterise as background
    :type  page    : fitz.Page
    :param paths   : List of PathInfo objects to visualise
    :type  paths   : List[PathInfo]
    :param out_path: File path where the output PNG will be saved
    :type  out_path: str

    :return: None

    :raises cv2.error: If OpenCV fails to write the image to disk
    """
    mat = fitz.Matrix(CFG.RENDER_DPI / 72.0, CFG.RENDER_DPI / 72.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    scale = CFG.RENDER_DPI / 72.0
    for p in paths:
        x0 = int(p.rect.x0 * scale)
        y0 = int(p.rect.y0 * scale)
        x1 = int(p.rect.x1 * scale)
        y1 = int(p.rect.y1 * scale)
        color = (0, 0, 255) if p.has_curve else (255, 0, 0)  # red=curve, blue=line
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
    cv2.imwrite(out_path, img)
    log.info("Raw paths image saved → %s  (%d paths)", out_path, len(paths))


def draw_clusters(page: fitz.Page, clusters: Dict[int, List[PathInfo]],
                  out_path: str) -> None:
    """
    Render a debug image showing each DBSCAN cluster as a coloured bounding
    box on a rasterised page, with the cluster ID printed above each box.
    Noise cluster (id -1) is skipped.

    :param page    : PyMuPDF page to rasterise as background
    :type  page    : fitz.Page
    :param clusters: Mapping of cluster_id → list of PathInfo objects in that cluster
    :type  clusters: Dict[int, List[PathInfo]]
    :param out_path: File path where the output PNG will be saved
    :type  out_path: str

    :return: None

    :raises cv2.error: If OpenCV fails to write the image to disk
    """
    mat = fitz.Matrix(CFG.RENDER_DPI / 72.0, CFG.RENDER_DPI / 72.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    scale = CFG.RENDER_DPI / 72.0
    palette = [
        (255, 0, 0), (0, 200, 0), (0, 0, 255), (200, 200, 0), (0, 200, 200),
        (200, 0, 200), (255, 128, 0), (0, 128, 255), (128, 0, 255), (0, 255, 128),
    ]
    for cid, cpaths in clusters.items():
        if cid == -1:
            continue
        bbox = cluster_bbox(cpaths)
        color = palette[cid % len(palette)]
        x0, y0 = int(bbox.x0 * scale), int(bbox.y0 * scale)
        x1, y1 = int(bbox.x1 * scale), int(bbox.y1 * scale)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, str(cid), (x0, max(y0 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
    log.info("Clusters image saved → %s  (%d clusters)", out_path,
             sum(1 for k in clusters if k != -1))


# ═════════════════════════════════════════════════════════════════════════════
# 2. DBSCAN clustering
# ═════════════════════════════════════════════════════════════════════════════

def cluster_paths(paths: List[PathInfo]) -> Dict[int, List[PathInfo]]:
    """
    Group nearby vector paths into candidate symbol clusters using DBSCAN
    on path centroids. Cluster ID -1 represents noise (ungrouped paths).

    :param paths: List of PathInfo objects whose centroids are used for clustering
    :type  paths: List[PathInfo]

    :return: Mapping of cluster_id → list of PathInfo objects belonging to that cluster;
             cluster_id -1 holds all noise paths
    :rtype : Dict[int, List[PathInfo]]
    """
    if not paths:
        return {}

    centroids = np.array([p.centroid for p in paths], dtype=np.float32)
    labels = DBSCAN(eps=CFG.CLUSTER_EPS, min_samples=CFG.CLUSTER_MIN_PTS,
                    algorithm="ball_tree").fit_predict(centroids)

    clusters: Dict[int, List[PathInfo]] = {}
    for lbl, path in zip(labels, paths):
        clusters.setdefault(lbl, []).append(path)
    return clusters


def cluster_bbox(paths: List[PathInfo]) -> fitz.Rect:
    """
    Compute the union bounding box that tightly encloses all paths in a cluster.

    :param paths: List of PathInfo objects in a single cluster
    :type  paths: List[PathInfo]

    :return: Smallest fitz.Rect that contains every path's bounding rectangle
    :rtype : fitz.Rect
    """
    x0 = min(p.rect.x0 for p in paths)
    y0 = min(p.rect.y0 for p in paths)
    x1 = max(p.rect.x1 for p in paths)
    y1 = max(p.rect.y1 for p in paths)
    return fitz.Rect(x0, y0, x1, y1)


def build_pre_classification_records(
        clusters: Dict[int, List[PathInfo]],
        page: fitz.Page,
        pdf_stem: str,
        page_num: int) -> List[Dict]:
    """
    Build intermediate cluster records after DBSCAN grouping but before any
    classification step (template matching, bubble detection, or heuristics).
    Skips noise clusters and clusters below minimum size thresholds.
    These records are intended for comparison against the legend.

    :param clusters: Mapping of cluster_id → list of PathInfo objects
    :type  clusters: Dict[int, List[PathInfo]]
    :param page    : PyMuPDF page used for nearby text extraction
    :type  page    : fitz.Page
    :param pdf_stem: Stem of the source PDF filename, used to prefix cluster IDs
    :type  pdf_stem: str
    :param page_num: Zero-based page index, used in cluster ID construction
    :type  page_num: int

    :return: List of dicts, each describing one cluster with geometry features
             and nearby text; intended for CSV export
    :rtype : List[Dict]
    """
    records = []
    for cid, cpaths in clusters.items():
        if cid == -1:
            continue  # skip noise
        bbox = cluster_bbox(cpaths)
        if (bbox.width < CFG.MIN_CLUSTER_W or
                bbox.height < CFG.MIN_CLUSTER_H):
            continue

        cx = (bbox.x0 + bbox.x1) / 2.0
        cy = (bbox.y0 + bbox.y1) / 2.0
        nearby_text = extract_nearby_text(page, bbox)

        n_paths      = len(cpaths)
        n_curved     = sum(1 for p in cpaths if p.has_curve)
        n_dashed     = sum(1 for p in cpaths if p.is_dashed)
        n_filled     = sum(1 for p in cpaths if p.is_filled)
        aspect_ratio = round(
            min(bbox.width, bbox.height) / max(bbox.width, bbox.height), 4
        ) if max(bbox.width, bbox.height) > 0 else 0

        records.append({
            "cluster_id":    f"{pdf_stem}_p{page_num + 1}_c{cid:04d}",
            "page_num":      page_num + 1,
            "bbox_x0":       round(bbox.x0, 2),
            "bbox_y0":       round(bbox.y0, 2),
            "bbox_x1":       round(bbox.x1, 2),
            "bbox_y1":       round(bbox.y1, 2),
            "centroid_x":    round(cx, 2),
            "centroid_y":    round(cy, 2),
            "width_pts":     round(bbox.width, 2),
            "height_pts":    round(bbox.height, 2),
            "aspect_ratio":  aspect_ratio,
            "n_paths":       n_paths,
            "n_curved":      n_curved,
            "n_dashed":      n_dashed,
            "n_filled":      n_filled,
            "nearby_text":   nearby_text,
        })
    return records


def write_pre_classification_csv(records: List[Dict], out_path: str) -> None:
    """
    Write pre-classification cluster records to a CSV file at the specified path,
    creating output directories if they do not already exist.

    :param records : List of pre-classification cluster dicts to write
    :type  records : List[Dict]
    :param out_path: Destination file path for the output CSV
    :type  out_path: str

    :return: None

    :raises OSError: If the output directory cannot be created or file cannot be written
    """
    columns = [
        "cluster_id", "page_num",
        "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
        "centroid_x", "centroid_y",
        "width_pts", "height_pts", "aspect_ratio",
        "n_paths", "n_curved", "n_dashed", "n_filled",
        "nearby_text",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    log.info("Pre-classification CSV saved → %s  (%d rows)", out_path, len(records))


# ═════════════════════════════════════════════════════════════════════════════
# 3a. Instrument bubble detection  (ISA 5.1 balloon rules)
# ═════════════════════════════════════════════════════════════════════════════

def _is_circle_like(path: PathInfo) -> bool:
    """
    Return True if the path bounding box is roughly square, falls within the
    expected bubble size range, and contains at least one curve item.
    Used as a proxy for ISA circular instrument bubbles.

    :param path: PathInfo object representing a single vector drawing path
    :type  path: PathInfo

    :return: True if the path resembles a circle, False otherwise
    :rtype : bool
    """
    r = path.rect
    w, h = r.width, r.height
    if w < CFG.BUBBLE_MIN_SIZE or w > CFG.BUBBLE_MAX_SIZE:
        return False
    if h < CFG.BUBBLE_MIN_SIZE or h > CFG.BUBBLE_MAX_SIZE:
        return False
    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
    return aspect >= CFG.CIRCLE_ASPECT and path.has_curve


def _is_square_like(path: PathInfo) -> bool:
    """
    Return True if the path bounding box is roughly square, falls within the
    expected bubble size range, and contains no curve items.
    Used as a proxy for ISA square instrument bubbles.

    :param path: PathInfo object representing a single vector drawing path
    :type  path: PathInfo

    :return: True if the path resembles a square, False otherwise
    :rtype : bool
    """
    r = path.rect
    w, h = r.width, r.height
    if w < CFG.BUBBLE_MIN_SIZE or w > CFG.BUBBLE_MAX_SIZE:
        return False
    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
    return aspect >= 0.75 and not path.has_curve


def detect_instrument_bubble(paths: List[PathInfo]) -> Optional[Dict]:
    """
    Scan a cluster's paths for an ISA 5.1 instrument bubble shape (circle or
    square) and determine its mounting type by counting horizontal divider lines
    inside the bubble boundary.

    Mounting logic:
      - 0 solid, 0 dashed  → Field-Mounted
      - 1 solid, 0 dashed  → Panel-Mounted (Primary)
      - 0 solid, 1 dashed  → Behind-Panel (Primary)
      - 2+ solid, 0 dashed → Panel-Mounted (Secondary)
      - 0 solid, 2+ dashed → Behind-Panel (Secondary)
      - mixed              → Unknown

    :param paths: List of PathInfo objects comprising the cluster
    :type  paths: List[PathInfo]

    :return: Dict with keys ``bubble_path`` (PathInfo), ``shape`` ('circle'|'square'),
             and ``mounting`` (str); or None if no bubble is found
    :rtype : Optional[Dict]
    """
    bubble_path = None
    shape = None

    for p in paths:
        if _is_circle_like(p):
            bubble_path = p
            shape = "circle"
            break
        if _is_square_like(p):
            bubble_path = p
            shape = "square"
            break

    if not bubble_path:
        return None

    solid_hlines = 0
    dashed_hlines = 0

    bw, bh = bubble_path.rect.width, bubble_path.rect.height
    bx0, by0, bx1, by1 = bubble_path.rect.x0, bubble_path.rect.y0, bubble_path.rect.x1, bubble_path.rect.y1

    for p in paths:
        if p is bubble_path:
            continue
        if p.has_curve:
            continue

        r = p.rect
        # Check if horizontal line intersecting the bubble
        if r.height < 5.0 and r.width > bw * 0.4:
            if r.x0 >= bx0 - 5 and r.x1 <= bx1 + 5 and r.y0 >= by0 - 5 and r.y1 <= by1 + 5:
                if p.is_dashed:
                    dashed_hlines += 1
                else:
                    solid_hlines += 1

    if solid_hlines == 0 and dashed_hlines == 0:
        mounting = "Field-Mounted"
    elif solid_hlines == 1 and dashed_hlines == 0:
        mounting = "Panel-Mounted (Primary)"
    elif solid_hlines == 0 and dashed_hlines == 1:
        mounting = "Behind-Panel (Primary)"
    elif solid_hlines >= 2 and dashed_hlines == 0:
        mounting = "Panel-Mounted (Secondary)"
    elif solid_hlines == 0 and dashed_hlines >= 2:
        mounting = "Behind-Panel (Secondary)"
    else:
        mounting = "Unknown"

    return {"bubble_path": bubble_path, "shape": shape, "mounting": mounting}


# ═════════════════════════════════════════════════════════════════════════════
# 3b. Template library  (built once from the legend PDF)
# ═════════════════════════════════════════════════════════════════════════════

# Each entry: {"name": str, "group": str, "templates": [np.ndarray, ...]}
_TEMPLATE_LIBRARY: List[Dict] = []

# ── NEW: buffer to accumulate match scores across all clusters ────────────────
_match_score_buffer: List[Dict] = []


def flush_match_scores(out_path: str) -> None:
    """
    Write all accumulated per-cluster template match scores from the module-level
    buffer to a CSV file, then clear the buffer. Intended to be called once per
    PDF after all pages have been processed.

    CSV columns: cluster_id, template_name, template_group, best_score,
    best_scale, threshold, above_threshold, winner.

    :param out_path: Destination file path for the match scores CSV
    :type  out_path: str

    :return: None

    :raises OSError: If the output directory cannot be created or file cannot be written
    """
    global _match_score_buffer
    if not _match_score_buffer:
        log.info("No match scores to flush (template library may be empty).")
        return
    columns = [
        "cluster_id", "template_name", "template_group",
        "best_score", "best_scale", "threshold", "above_threshold", "winner",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in _match_score_buffer:
            writer.writerow(row)
    log.info("Match scores CSV saved → %s  (%d rows)", out_path, len(_match_score_buffer))
    _match_score_buffer = []


# Symbol definitions we want to extract from the legend PDF.
# key = substring to search in text near each legend symbol
# value = (display_name, group)
LEGEND_SYMBOL_MAP = {
    "gate valve": ("Gate Valve", "Valve"),
    "globe valve": ("Globe Valve", "Valve"),
    "ball valve": ("Ball Valve", "Valve"),  # 'ball' alone matches 'normally closed ball'
    "butterfly valve": ("Butterfly Valve", "Valve"),
    "check valve": ("Check Valve", "Valve"),
    "control valve": ("Control Valve", "Valve"),
    "needle valve": ("Needle Valve", "Valve"),
    "relief valve": ("Relief Valve", "Valve"),
    "safety valve": ("Safety Valve", "Valve"),
    "psv": ("PSV", "Valve"),
    "solenoid valve": ("Solenoid Valve", "Valve"),
    "3-way valve": ("3-Way Valve", "Valve"),
    "diaphragm valve": ("Diaphragm Valve", "Valve"),
    "centrifugal pump": ("Centrifugal Pump", "Pump"),
    "positive displacement pump": ("PD Pump", "Pump"),
    "reciprocating pump": ("Reciprocating Pump", "Pump"),
    "rotary pump": ("Rotary Pump", "Pump"),
    "heat exchanger": ("Heat Exchanger", "Heat Exchanger"),
    "condenser": ("Condenser", "Heat Exchanger"),
    "reboiler": ("Reboiler", "Heat Exchanger"),
    "cooler": ("Cooler", "Heat Exchanger"),
    "vessel": ("Vessel", "Vessel"),
    "tank": ("Tank", "Vessel"),
    "drum": ("Drum", "Vessel"),
    "filter": ("Filter", "Filter"),
    "strainer": ("Strainer", "Filter"),
    "compressor": ("Compressor", "Compressor"),
}


def _crop_symbol_from_legend(page: fitz.Page, text_rect: fitz.Rect,
                             pad: int = 5) -> Optional[np.ndarray]:
    """
    Render the legend symbol graphic that sits to the left of a given text label
    rect, returning it as a grayscale binary numpy array (strokes white on black).
    Returns None if the computed symbol region falls outside the page bounds or
    yields an empty image.

    :param page     : PyMuPDF page containing the legend layout
    :type  page     : fitz.Page
    :param text_rect: Bounding rectangle of the legend label text
    :type  text_rect: fitz.Rect
    :param pad      : Extra padding in PDF points added around the symbol region
    :type  pad      : int

    :return: Grayscale binary image array of the symbol, or None on failure
    :rtype : Optional[np.ndarray]
    """
    # Symbol is typically to the left of its label in the legend layout.
    # We search a square area whose right edge aligns with the label's left edge.
    symbol_size = 60  # pts – typical legend symbol size
    sym_rect = fitz.Rect(
        text_rect.x0 - symbol_size - pad,
        text_rect.y0 - pad,
        text_rect.x0 + pad,
        text_rect.y1 + pad,
    )
    # Expand vertically to catch symbols that are taller than the label
    sym_rect.y0 = min(sym_rect.y0, text_rect.y0 - 10)
    sym_rect.y1 = max(sym_rect.y1, text_rect.y1 + 10)

    if sym_rect.x0 < 0 or sym_rect.y0 < 0:
        return None

    mat = fitz.Matrix(CFG.LEGEND_DPI / 72.0, CFG.LEGEND_DPI / 72.0)
    clip = page.rect & sym_rect  # intersect with page bounds
    if clip.is_empty:
        return None

    pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    # Invert so symbol strokes are white on black (easier for matchTemplate)
    _, arr = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
    return arr if arr.size > 0 else None


def build_template_library(legend_pdf_path: str) -> None:
    """
    Parse the legend PDF to locate each symbol defined in LEGEND_SYMBOL_MAP,
    crop its graphic from the region left of its label, generate rotated variants,
    and store them in the module-level _TEMPLATE_LIBRARY. Also saves every variant
    as a PNG under results/CE_comparison/templates/ for visual inspection.
    Clears any previously loaded library before rebuilding.

    :param legend_pdf_path: File path to the legend PDF
    :type  legend_pdf_path: str

    :return: None

    :raises fitz.FileNotFoundError: If the legend PDF cannot be opened by PyMuPDF
    """
    global _TEMPLATE_LIBRARY
    _TEMPLATE_LIBRARY = []

    if not os.path.exists(legend_pdf_path):
        log.warning("Legend PDF not found: %s – template matching disabled", legend_pdf_path)
        return

    log.info("Building template library from %s …", legend_pdf_path)
    doc = fitz.open(legend_pdf_path)

    # ── Ensure template output dir exists ─────────────────────────────────────
    tmpl_dir = os.path.join(CFG.OUT_COMPARE_DIR, "templates")
    os.makedirs(tmpl_dir, exist_ok=True)

    for page_num, page in enumerate(doc):
        words = page.get_text("words")  # (x0,y0,x1,y1, word, block, line, word_idx)
        # Build line groups: collect words by approximate y-row
        lines: Dict[int, List] = {}
        for w in words:
            row_key = int(round(w[1] / 8.0))  # bucket by 8pt rows
            lines.setdefault(row_key, []).append(w)

        # For each candidate keyword, find matching label text on the page
        page_text_lower = page.get_text("text").lower()
        for keyword, (display_name, group) in LEGEND_SYMBOL_MAP.items():
            if keyword not in page_text_lower:
                continue

            # Find the specific word(s) matching the keyword
            for row_words in lines.values():
                row_str = " ".join(w[4] for w in row_words).lower()
                if keyword not in row_str:
                    continue
                # Bounding rect of this label row
                x0 = min(w[0] for w in row_words)
                y0 = min(w[1] for w in row_words)
                x1 = max(w[2] for w in row_words)
                y1 = max(w[3] for w in row_words)
                label_rect = fitz.Rect(x0, y0, x1, y1)

                template_img = _crop_symbol_from_legend(page, label_rect)
                if template_img is None or template_img.sum() < 100:
                    continue

                # Generate rotated variants and save each to disk
                variants = []
                safe_name = display_name.replace(" ", "_").replace("/", "-")
                for angle in CFG.TEMPLATE_ROTATIONS:
                    if angle == 0:
                        variant = template_img.copy()
                    else:
                        h, w = template_img.shape
                        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                        variant = cv2.warpAffine(template_img, M, (w, h))
                    variants.append(variant)

                    # ── Save legend template image (once per name+angle) ──────
                    tmpl_save_path = os.path.join(
                        tmpl_dir, f"{safe_name}_rot{angle}.png")
                    if not os.path.exists(tmpl_save_path):
                        cv2.imwrite(tmpl_save_path, variant)
                        log.debug("Legend template saved → %s", tmpl_save_path)
                    # ─────────────────────────────────────────────────────────

                # Check if this name already exists; if so, append
                existing = next((t for t in _TEMPLATE_LIBRARY
                                 if t["name"] == display_name), None)
                if existing:
                    existing["templates"].extend(variants)
                else:
                    _TEMPLATE_LIBRARY.append({
                        "name": display_name,
                        "group": group,
                        "templates": variants,
                    })

    doc.close()
    log.info("Template library: %d symbol types loaded.", len(_TEMPLATE_LIBRARY))


def get_template_library() -> List[Dict]:
    """
    Return the module-level template library built by build_template_library.

    :return: List of template entry dicts, each with keys
             ``name`` (str), ``group`` (str), and ``templates`` (List[np.ndarray])
    :rtype : List[Dict]
    """
    return _TEMPLATE_LIBRARY


# ═════════════════════════════════════════════════════════════════════════════
# 3c. Template matching on rasterised cluster region
# ═════════════════════════════════════════════════════════════════════════════

def _render_cluster_region(page: fitz.Page, bbox: fitz.Rect,
                           pad: int = 15) -> np.ndarray:
    """
    Rasterise a padded rectangular region of a PDF page to a grayscale binary
    image (strokes white on black), clipped to page boundaries.

    :param page: PyMuPDF page to render
    :type  page: fitz.Page
    :param bbox: Bounding rectangle of the cluster in PDF points
    :type  bbox: fitz.Rect
    :param pad : Padding in PDF points added on all four sides before rendering
    :type  pad : int

    :return: Grayscale binary numpy array of the rendered region
    :rtype : np.ndarray
    """
    padded = fitz.Rect(
        max(0, bbox.x0 - pad), max(0, bbox.y0 - pad),
        min(page.rect.width, bbox.x1 + pad),
        min(page.rect.height, bbox.y1 + pad),
    )
    mat = fitz.Matrix(CFG.RENDER_DPI / 72.0, CFG.RENDER_DPI / 72.0)
    pix = page.get_pixmap(matrix=mat, clip=padded, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    _, arr = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
    return arr


def template_match_cluster(page: fitz.Page,
                           bbox: fitz.Rect,
                           cluster_id: str = "") -> Optional[Tuple[str, str, float]]:
    """
    Run every template in _TEMPLATE_LIBRARY against the rasterised image of the
    given cluster region at multiple scale factors, returning the best-matching
    template above the configured threshold.

    Side effects:
      - Saves the cluster binary image to CE_comparison/clusters/<cluster_id>_cluster.png
      - Appends one score row per template type to the module-level _match_score_buffer,
        with winner=True on the row that produced the final classification

    :param page      : PyMuPDF page used to render the cluster region
    :type  page      : fitz.Page
    :param bbox      : Bounding rectangle of the cluster in PDF points
    :type  bbox      : fitz.Rect
    :param cluster_id: Unique cluster identifier used for image filename and score tracking
    :type  cluster_id: str

    :return: Tuple of (name, group, score) for the winning template, or None if
             no template exceeds the threshold or the library is empty
    :rtype : Optional[Tuple[str, str, float]]
    """
    global _match_score_buffer

    if not _TEMPLATE_LIBRARY:
        return None

    region = _render_cluster_region(page, bbox)
    if region.size == 0:
        return None

    # ── Save cluster binary image to disk ─────────────────────────────────────
    if cluster_id:
        clust_dir = os.path.join(CFG.OUT_COMPARE_DIR, "clusters")
        os.makedirs(clust_dir, exist_ok=True)
        clust_img_path = os.path.join(clust_dir, f"{cluster_id}_cluster.png")
        cv2.imwrite(clust_img_path, region)
        log.debug("Cluster image saved → %s", clust_img_path)
    # ─────────────────────────────────────────────────────────────────────────

    best_name  = ""
    best_group = ""
    best_score = 0.0
    score_rows = []   # one row per template type

    for entry in _TEMPLATE_LIBRARY:
        entry_best_score = 0.0
        entry_best_scale = 1.0

        for tmpl in entry["templates"]:
            for scale in CFG.TEMPLATE_SCALE_FACTORS:
                th = max(1, int(tmpl.shape[0] * scale))
                tw = max(1, int(tmpl.shape[1] * scale))
                if th > region.shape[0] or tw > region.shape[1]:
                    continue
                resized = cv2.resize(tmpl, (tw, th),
                                     interpolation=cv2.INTER_AREA)
                try:
                    res = cv2.matchTemplate(region, resized,
                                            cv2.TM_CCOEFF_NORMED)
                except cv2.error:
                    continue
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > entry_best_score:
                    entry_best_score = max_val
                    entry_best_scale = scale
                if max_val > best_score:
                    best_score = max_val
                    best_name  = entry["name"]
                    best_group = entry["group"]

        # Collect per-template-type best score
        score_rows.append({
            "cluster_id":      cluster_id,
            "template_name":   entry["name"],
            "template_group":  entry["group"],
            "best_score":      round(entry_best_score, 4),
            "best_scale":      entry_best_scale,
            "threshold":       CFG.TEMPLATE_MATCH_THRESHOLD,
            "above_threshold": entry_best_score >= CFG.TEMPLATE_MATCH_THRESHOLD,
            "winner":          False,   # filled in below
        })

    # Mark the winning template row
    for row in score_rows:
        if row["template_name"] == best_name and best_score >= CFG.TEMPLATE_MATCH_THRESHOLD:
            row["winner"] = True

    # Accumulate into module-level buffer (flushed once per PDF by the runner)
    _match_score_buffer.extend(score_rows)

    if best_score >= CFG.TEMPLATE_MATCH_THRESHOLD:
        return (best_name, best_group, best_score)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 3d. Heuristic fallback classifier
# ═════════════════════════════════════════════════════════════════════════════

def heuristic_classify(paths: List[PathInfo],
                       nearby_text: str) -> Tuple[str, str]:
    """
    Last-resort rule-based classifier that assigns a component type and group
    using nearby text keywords and path geometry features. Applied only when
    template matching and bubble detection both fail.

    Priority order:
      1. ISA tag pattern in nearby text (e.g. TT, FI, PC)
      2. Valve keyword match in nearby text
      3. Geometry heuristics (fill, curve count, path count, aspect ratio)

    :param paths      : List of PathInfo objects in the cluster
    :type  paths      : List[PathInfo]
    :param nearby_text: Raw text extracted from the neighbourhood of the cluster
    :type  nearby_text: str

    :return: Tuple of (type_name, group) strings
    :rtype : Tuple[str, str]
    """
    txt = nearby_text.lower()

    # Check nearby text for ISA tag patterns  e.g.  TT, FI, PC, LT …
    isa_tag_match = re.search(r'\b([FLPTACSZEYWDVH]{1,2})(I|T|C|R|E|G|A|S|V|Y|K|W)\b', txt.upper())
    if isa_tag_match:
        func_letter = isa_tag_match.group(1)[0]
        mod_letter = isa_tag_match.group(2)
        group = CFG.ISA_FUNCTION_GROUPS.get(func_letter, "Instrument")
        mod = CFG.ISA_MODIFIER.get(mod_letter, "")
        name = f"{group} {mod}".strip() if mod else group
        return name, "Instrument"

    # Valve keywords in nearby text
    for kw in CFG.VALVE_KEYWORDS:
        if kw in txt:
            return f"{kw.title()} Valve", "Valve"

    # Geometry heuristics
    n_paths = len(paths)
    has_fill = any(p.is_filled for p in paths)
    curved = sum(1 for p in paths if p.has_curve)

    if curved >= 2 and has_fill:
        return "Pump / Compressor", "Pump"
    if curved >= 1 and n_paths >= 3:
        return "Heat Exchanger", "Heat Exchanger"
    if n_paths >= 6:
        return "Complex Equipment", "Unknown"

    # Triangle-like shapes → valve body
    bboxes = [p.rect for p in paths]
    if any(abs(b.width - b.height) < 5 for b in bboxes):
        return "Valve (generic)", "Valve"

    return "Unknown", "Unknown"


# ═════════════════════════════════════════════════════════════════════════════
# 3. Master classify function
# ═════════════════════════════════════════════════════════════════════════════

def classify_cluster(paths: List[PathInfo],
                     page: fitz.Page,
                     nearby_text: str,
                     cluster_id: str = "") -> Tuple[str, str, str]:
    """
    Master classification function for a single cluster. Applies three strategies
    in priority order and returns the first successful result.

    Priority:
      1. Template matching  (template_match_cluster)
      2. Instrument bubble detection (detect_instrument_bubble) + ISA tag parse
      3. Heuristic fallback (heuristic_classify)

    :param paths      : List of PathInfo objects comprising the cluster
    :type  paths      : List[PathInfo]
    :param page       : PyMuPDF page, passed through to template matching and text extraction
    :type  page       : fitz.Page
    :param nearby_text: Raw text from the neighbourhood of the cluster centroid
    :type  nearby_text: str
    :param cluster_id : Unique cluster identifier forwarded to template_match_cluster
                        for image saving and score tracking
    :type  cluster_id : str

    :return: Tuple of (type_name, group, mounting) strings
    :rtype : Tuple[str, str, str]
    """
    # 1. Template matching
    bbox = cluster_bbox(paths)
    tm = template_match_cluster(page, bbox, cluster_id=cluster_id)
    if tm is not None:
        name, grp, _ = tm
        # Check if it's an instrument bubble to get mounting
        bubble = detect_instrument_bubble(paths)
        mounting = bubble["mounting"] if bubble else "Unknown"
        return name, grp, mounting

    # 2. Instrument bubble fallback
    bubble = detect_instrument_bubble(paths)
    if bubble is not None:
        tag_match = re.search(
            r'\b([FLPTACSZEYWDVH]{1,2})(I|T|C|R|E|G|A|S|V|Y|K|W)\b',
            nearby_text.upper())
        if tag_match:
            func = tag_match.group(1)[0]
            mod = tag_match.group(2)
            group = CFG.ISA_FUNCTION_GROUPS.get(func, "Instrument")
            mod_name = CFG.ISA_MODIFIER.get(mod, "Instrument")
            type_name = f"{group} {mod_name}".strip()
        else:
            type_name = "Instrument"
            group = "Instrument"
        return type_name, group, bubble["mounting"]

    # 3. Heuristic fallback
    type_name, group = heuristic_classify(paths, nearby_text)
    mounting = "Unknown"
    return type_name, group, mounting


# ═════════════════════════════════════════════════════════════════════════════
# 4. Tag CSV loader
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TagEntry:
    tag: str
    x: float
    y: float
    text: str


def load_tag_csv(pdf_stem: str) -> List[TagEntry]:
    """
    Load tag entries from the Text Extraction per-PDF CSV file corresponding to
    the given PDF stem. Accepts common column name variants for text, tag, and
    coordinate fields. Falls back to parsing coordinates from a bbox string if
    x/y columns are zero or absent.

    :param pdf_stem: Stem of the source PDF filename used to construct the CSV filename
    :type  pdf_stem: str

    :return: List of TagEntry objects parsed from the CSV; empty list if file not found
    :rtype : List[TagEntry]
    """
    fname = f"{pdf_stem}_process.csv"
    fpath = os.path.join(CFG.TAG_CSV_DIR, fname)
    if not os.path.exists(fpath):
        log.warning("Tag CSV not found: %s", fpath)
        return []

    entries: List[TagEntry] = []
    with open(fpath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Accept common column-name variants
            text = row.get("text", row.get("Text", row.get("raw_text", "")))
            tag = row.get("tag", row.get("Tag", row.get("label", text)))

            # Try parsing coordinates from x/y columns
            x = float(row.get("x", row.get("cx", row.get("X", 0))) or 0)
            y = float(row.get("y", row.get("cy", row.get("Y", 0))) or 0)

            # Or fall back to parsing from bbox string e.g. "(x0, y0, x1, y1)"
            if x == 0 and y == 0:
                bbox_str = row.get("bbox", "")
                if bbox_str:
                    try:
                        # Strip parentheses and split by comma
                        parts = bbox_str.replace("(", "").replace(")", "").split(",")
                        if len(parts) == 4:
                            x0, y0, x1, y1 = map(float, parts)
                            x = (x0 + x1) / 2.0
                            y = (y0 + y1) / 2.0
                    except ValueError:
                        pass

            entries.append(TagEntry(tag=tag, x=x, y=y, text=text))
    log.info("Loaded %d tag entries from %s", len(entries), fpath)
    return entries


# ═════════════════════════════════════════════════════════════════════════════
# 5. Tag association  (nearest neighbour)
# ═════════════════════════════════════════════════════════════════════════════

def associate_tags(components: List[Dict],
                   tags: List[TagEntry]) -> None:
    """
    Assign the nearest tag to each component dict in-place using a KD-tree
    nearest-neighbour search. Only assigns if the closest tag lies within
    CFG.TAG_SEARCH_RADIUS PDF points of the component centroid.

    :param components: List of component dicts, each containing ``_cx`` and ``_cy``
                       centroid coordinates; ``tag`` key is updated in-place
    :type  components: List[Dict]
    :param tags      : List of TagEntry objects providing candidate tag positions
    :type  tags      : List[TagEntry]

    :return: None
    """
    if not tags:
        return

    tag_coords = np.array([[t.x, t.y] for t in tags], dtype=np.float64)
    tree = cKDTree(tag_coords)

    for comp in components:
        cx, cy = comp["_cx"], comp["_cy"]
        dist, idx = tree.query([cx, cy], k=1)
        if dist <= CFG.TAG_SEARCH_RADIUS:
            comp["tag"] = tags[idx].tag or tags[idx].text


# ═════════════════════════════════════════════════════════════════════════════
# 6. Nearby text extraction from page
# ═════════════════════════════════════════════════════════════════════════════

def extract_nearby_text(page: fitz.Page, bbox: fitz.Rect,
                        radius: float = None) -> str:
    """
    Return a space-joined string of all text tokens found within a square search
    region centred on the bbox centroid. Search radius defaults to CFG.TAG_SEARCH_RADIUS
    if not explicitly provided.

    :param page  : PyMuPDF page from which text is extracted
    :type  page  : fitz.Page
    :param bbox  : Bounding rectangle whose centroid defines the search centre
    :type  bbox  : fitz.Rect
    :param radius: Half-width of the square search region in PDF points
    :type  radius: float

    :return: Space-separated string of all word tokens found within the search area
    :rtype : str
    """
    if radius is None:
        radius = CFG.TAG_SEARCH_RADIUS
    cx = (bbox.x0 + bbox.x1) / 2.0
    cy = (bbox.y0 + bbox.y1) / 2.0
    search_rect = fitz.Rect(cx - radius, cy - radius, cx + radius, cy + radius)
    words = page.get_text("words", clip=search_rect)
    tokens = [w[4] for w in words]
    return " ".join(tokens)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Build output records
# ═════════════════════════════════════════════════════════════════════════════

def classify_by_tag(tag: str) -> Optional[Tuple[str, str]]:
    """
    Derive an ISA instrument type name and function group from a tag string by
    extracting the first ISA letter pair (function letter + modifier letter).

    :param tag: ISA instrument tag string, e.g. 'PI-1209' or 'TI 1204'
    :type  tag: str

    :return: Tuple of (type_name, group) if a valid ISA letter pair is found,
             or None if the tag does not match the expected pattern
    :rtype : Optional[Tuple[str, str]]
    """
    # Regex to match ISA pattern in the tag
    tag_match = re.search(
        r'\b([FLPTACSZEYWDVH]{1,2})(I|T|C|R|E|G|A|S|V|Y|K|W)\b',
        tag.upper()
    )
    if tag_match:
        func = tag_match.group(1)[0]
        mod = tag_match.group(2)
        group = CFG.ISA_FUNCTION_GROUPS.get(func, "Instrument")
        mod_name = CFG.ISA_MODIFIER.get(mod, "Instrument")
        type_name = f"{group} {mod_name}".strip()
        return type_name, group
    return None


def process_page(page: fitz.Page,
                 pdf_stem: str,
                 page_num: int,
                 tags: List[TagEntry]) -> List[ComponentRecord]:
    """
    Run the full component extraction pipeline for a single PDF page and return
    a list of ComponentRecord objects.

    Pipeline steps executed in order:
      1. extract_page_paths   – pull all vector strokes
      2. cluster_paths        – DBSCAN grouping of path centroids
      3. classify_cluster     – per-cluster type/group/mounting classification
      4. associate_tags       – nearest-neighbour tag assignment from TE CSV
      5. Build ComponentRecord objects for each valid component

    :param page    : PyMuPDF page to process
    :type  page    : fitz.Page
    :param pdf_stem: Stem of the source PDF filename used for symbol IDs and cluster IDs
    :type  pdf_stem: str
    :param page_num: Zero-based page index used in symbol and cluster ID construction
    :type  page_num: int
    :param tags    : Pre-loaded list of TagEntry objects for tag association
    :type  tags    : List[TagEntry]

    :return: List of ComponentRecord objects, one per detected component on the page
    :rtype : List[ComponentRecord]
    """
    log.info("Processing page %d of %s …", page_num + 1, pdf_stem)

    paths = extract_page_paths(page)
    clusters = cluster_paths(paths)

    components: List[Dict] = []

    for cid, cpaths in clusters.items():
        if cid == -1:
            continue  # noise
        bbox = cluster_bbox(cpaths)
        if (bbox.width < CFG.MIN_CLUSTER_W or
                bbox.height < CFG.MIN_CLUSTER_H):
            continue

        cx = (bbox.x0 + bbox.x1) / 2.0
        cy = (bbox.y0 + bbox.y1) / 2.0

        nearby_text = extract_nearby_text(page, bbox)

        # Build the cluster_id used for image filename and score tracking
        cluster_id = f"{pdf_stem}_p{page_num + 1}_c{cid:04d}"

        type_name, group, mounting = classify_cluster(
            cpaths, page, nearby_text, cluster_id=cluster_id)

        components.append({
            "cid": cid,
            "_cx": cx,
            "_cy": cy,
            "bbox": bbox,
            "type": type_name,
            "group": group,
            "mounting": mounting,
            "tag": "",
            "nearby_text": nearby_text,
        })

    # Associate tags from existing TE CSV
    associate_tags(components, tags)

    # Build final records
    records: List[ComponentRecord] = []
    for i, comp in enumerate(components):
        symbol_id = f"{pdf_stem}_p{page_num + 1}_{i:04d}"
        bbox = comp["bbox"]
        records.append(ComponentRecord(
            symbol_id=symbol_id,
            type=comp["type"],
            group=comp["group"],
            tag=comp["tag"],
            mounting=comp["mounting"],
            location_xy=f"{comp['_cx']:.1f},{comp['_cy']:.1f}",
            bounding_box=f"{bbox.x0:.1f},{bbox.y0:.1f},{bbox.x1:.1f},{bbox.y1:.1f}",
            nearby_text=comp["nearby_text"],
        ))

    log.info("Page %d: %d components extracted.", page_num + 1, len(records))
    return records


# ═════════════════════════════════════════════════════════════════════════════
# 8. Annotation image
# ═════════════════════════════════════════════════════════════════════════════

def draw_bboxes(page: fitz.Page,
                records: List[ComponentRecord],
                out_path: str) -> None:
    """
    Rasterise the page and draw a labelled bounding box for every ComponentRecord,
    colouring each box according to its group using CFG.BBOX_COLORS. Labels show
    the tag if available, otherwise the first 12 characters of the type name.

    :param page    : PyMuPDF page to rasterise as background
    :type  page    : fitz.Page
    :param records : List of ComponentRecord objects whose bounding boxes are drawn
    :type  records : List[ComponentRecord]
    :param out_path: File path where the annotated output PNG will be saved
    :type  out_path: str

    :return: None

    :raises cv2.error: If OpenCV fails to write the image to disk
    """
    mat = fitz.Matrix(CFG.RENDER_DPI / 72.0, CFG.RENDER_DPI / 72.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    scale = CFG.RENDER_DPI / 72.0

    for rec in records:
        parts = [float(v) for v in rec.bounding_box.split(",")]
        x0, y0, x1, y1 = (int(v * scale) for v in parts)

        group = rec.group
        if group in CFG.ISA_FUNCTION_GROUPS.values():
            group = "Instrument"

        color = CFG.BBOX_COLORS.get(group,
                                    CFG.BBOX_COLORS.get(rec.type,
                                                        CFG.BBOX_COLORS["Unknown"]))

        cv2.rectangle(img, (x0, y0), (x1, y1), color, CFG.BBOX_THICKNESS)

        label = f"{rec.tag or rec.type[:12]}"
        cv2.putText(img, label, (x0, max(y0 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    CFG.LABEL_FONT_SCALE, color, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    log.info("Annotation image saved → %s", out_path)


# ═════════════════════════════════════════════════════════════════════════════
# 9. CSV / JSON writers
# ═════════════════════════════════════════════════════════════════════════════

def write_csv(records: List[ComponentRecord], out_path: str) -> None:
    """
    Serialise a list of ComponentRecord objects to a CSV file using the column
    order defined in CFG.CSV_COLUMNS, creating output directories if needed.

    :param records : List of ComponentRecord objects to write
    :type  records : List[ComponentRecord]
    :param out_path: Destination file path for the output CSV
    :type  out_path: str

    :return: None

    :raises OSError: If the output directory cannot be created or file cannot be written
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CFG.CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))
    log.info("CSV saved → %s  (%d rows)", out_path, len(records))


def write_json(records: List[ComponentRecord], out_path: str) -> None:
    """
    Serialise a list of ComponentRecord objects to a pretty-printed JSON file,
    creating output directories if needed.

    :param records : List of ComponentRecord objects to serialise
    :type  records : List[ComponentRecord]
    :param out_path: Destination file path for the output JSON
    :type  out_path: str

    :return: None

    :raises OSError: If the output directory cannot be created or file cannot be written
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = [asdict(r) for r in records]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("JSON saved → %s", out_path)