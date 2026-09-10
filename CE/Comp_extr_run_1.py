# ─────────────────────────────────────────────────────────────────────────────
# Comp_extr_run.py  –  Component Extraction Runner
#
# Usage (from project root):
#     python run/Comp_extr_run.py
#
# Loops over every .pdf in data/raw/, extracts components, and writes:
#     results/CE_csv/<stem>.csv
#     results/CE_csv/<stem>_raw_paths.csv
#     results/CE_json/<stem>.json
#     results/CE_images/<stem>_p<N>.png              (one per page)
#     results/CE_images/<stem>_p<N>_debug_paths.png  (one per page)
#     results/CE_images/<stem>_p<N>_debug_clusters.png (one per page)
#     results/CE_pre_classification/<stem>_pre_clf.csv
#     results/CE_comparison/templates/<name>_rot<angle>.png  (legend side)
#     results/CE_comparison/clusters/<cluster_id>_cluster.png (cluster side)
#     results/CE_comparison/<stem>_match_scores.csv
# ─────────────────────────────────────────────────────────────────────────────

import csv
import logging
import os
import sys
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "config"))
sys.path.insert(0, str(ROOT / "src"))

import Config_CE_1 as CFG
import Comp_extr_src_1 as CE

import fitz   # PyMuPDF

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Comp_extr_run")


# ─────────────────────────────────────────────────────────────────────────────
# Ensure output directories exist
# ─────────────────────────────────────────────────────────────────────────────
for d in (CFG.OUT_CSV_DIR, CFG.OUT_JSON_DIR, CFG.OUT_IMG_DIR,
          CFG.OUT_PRECLF_DIR, CFG.OUT_COMPARE_DIR):   # ← OUT_COMPARE_DIR added
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build template library once  (reads legend PDF + saves template PNGs)
# ─────────────────────────────────────────────────────────────────────────────
CE.build_template_library(CFG.LEGEND_PDF)
log.info("Template library ready: %d symbol types.",
         len(CE.get_template_library()))


# ─────────────────────────────────────────────────────────────────────────────
# Discover all PDFs in data/raw/
# ─────────────────────────────────────────────────────────────────────────────
input_pdfs = sorted(Path(CFG.INPUT_DIR).glob("*.pdf"))

if not input_pdfs:
    log.error("No PDF files found in %s – aborting.", CFG.INPUT_DIR)
    sys.exit(1)

log.info("Found %d PDF(s) to process: %s",
         len(input_pdfs), [p.name for p in input_pdfs])


# ─────────────────────────────────────────────────────────────────────────────
# Process each PDF
# ─────────────────────────────────────────────────────────────────────────────
for pdf_path in input_pdfs:
    stem = pdf_path.stem          # e.g. "02.Unmarked"   "01.Marked"
    log.info("=" * 60)
    log.info("Processing PDF: %s", pdf_path.name)

    # -- Load tag CSV produced by the text-extraction stage ------------------
    tags = CE.load_tag_csv(stem)

    # -- Open PDF ------------------------------------------------------------
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        log.error("Cannot open %s: %s", pdf_path.name, exc)
        continue

    all_records   = []
    all_raw_paths = []
    all_pre_clf   = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Rotate page if needed so coordinates are consistent
        # (the sample PDF has 90° rotation stored in metadata)
        if page.rotation != 0:
            page.set_rotation(0)   # normalise to 0 for coordinate consistency

        try:
            # --- debug image 1: raw paths -----------------------------------
            raw_paths = CE.extract_page_paths(page)
            raw_img_out = os.path.join(
                CFG.OUT_IMG_DIR, f"{stem}_p{page_num + 1}_debug_paths.png")
            CE.draw_raw_paths(page, raw_paths, raw_img_out)

            # Record raw paths info immediately after vector PDF extraction
            for i, p in enumerate(raw_paths):
                all_raw_paths.append({
                    "path_id":    f"{stem}_p{page_num + 1}_path_{i:04d}",
                    "page_num":   page_num + 1,
                    "x0":         f"{p.rect.x0:.3f}",
                    "y0":         f"{p.rect.y0:.3f}",
                    "x1":         f"{p.rect.x1:.3f}",
                    "y1":         f"{p.rect.y1:.3f}",
                    "centroid_x": f"{p.centroid[0]:.3f}",
                    "centroid_y": f"{p.centroid[1]:.3f}",
                    "is_dashed":  p.is_dashed,
                    "is_filled":  p.is_filled,
                    "fill_color": p.fill_color,
                    "has_curve":  p.has_curve,
                    "item_count": p.item_count,
                })

            # --- debug image 2: clusters ------------------------------------
            clusters = CE.cluster_paths(raw_paths)
            clust_img_out = os.path.join(
                CFG.OUT_IMG_DIR, f"{stem}_p{page_num + 1}_debug_clusters.png")
            CE.draw_clusters(page, clusters, clust_img_out)

            # --- pre-classification snapshot --------------------------------
            pre_clf_records = CE.build_pre_classification_records(
                clusters, page, stem, page_num)
            all_pre_clf.extend(pre_clf_records)

            # --- main extraction  (also saves cluster PNGs + score rows) ----
            #     process_page() now passes cluster_id into classify_cluster()
            #     → template_match_cluster() saves the cluster binary image
            #       and accumulates scores into CE._match_score_buffer
            records = CE.process_page(page, stem, page_num, tags)

        except Exception as exc:
            log.exception("Error on page %d of %s: %s", page_num + 1, stem, exc)
            continue

        all_records.extend(records)

        # -- Annotated image (one per page) ----------------------------------
        img_out = os.path.join(
            CFG.OUT_IMG_DIR, f"{stem}_p{page_num + 1}.png")
        try:
            CE.draw_bboxes(page, records, img_out)
        except Exception as exc:
            log.warning("Could not write image for page %d: %s", page_num + 1, exc)

    doc.close()

    # -- Pre-classification CSV ----------------------------------------------
    pre_clf_out = os.path.join(CFG.OUT_PRECLF_DIR, f"{stem}_pre_clf.csv")
    CE.write_pre_classification_csv(all_pre_clf, pre_clf_out)

    # -- Main components CSV -------------------------------------------------
    csv_out = os.path.join(CFG.OUT_CSV_DIR, f"{stem}.csv")
    CE.write_csv(all_records, csv_out)

    # -- Raw Paths CSV -------------------------------------------------------
    raw_csv_out = os.path.join(CFG.OUT_CSV_DIR, f"{stem}_raw_paths.csv")
    raw_columns = [
        "path_id", "page_num", "x0", "y0", "x1", "y1",
        "centroid_x", "centroid_y", "is_dashed", "is_filled",
        "fill_color", "has_curve", "item_count",
    ]
    try:
        with open(raw_csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=raw_columns)
            writer.writeheader()
            for r in all_raw_paths:
                writer.writerow(r)
        log.info("Raw paths CSV saved → %s  (%d rows)", raw_csv_out, len(all_raw_paths))
    except Exception as exc:
        log.error("Could not write raw paths CSV to %s: %s", raw_csv_out, exc)

    # -- JSON ----------------------------------------------------------------
    import json as _json   # already imported in src; re-import here for clarity
    json_out = os.path.join(CFG.OUT_JSON_DIR, f"{stem}.json")
    CE.write_json(all_records, json_out)

    # -- Match scores CSV  (one per PDF, written after all pages done) -------
    #    This is the file that shows exactly which cluster image was compared
    #    against which legend template image and what score each pair got.
    scores_out = os.path.join(CFG.OUT_COMPARE_DIR, f"{stem}_match_scores.csv")
    CE.flush_match_scores(scores_out)

    log.info("Done: %s  →  %d total components extracted.", stem, len(all_records))

log.info("=" * 60)
log.info("All PDFs processed.")