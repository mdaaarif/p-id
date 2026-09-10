# ============================================================
# run/line_ext_3.py
# Entry point – run the P&ID line-extraction pipeline (v3)
# ============================================================

import sys
import os

# Allow imports from project root regardless of where this
# script is launched from
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config.config_line_ext_3 as cfg
from src.Line_ext_3 import LineExtractor


def main():
    print("=" * 60)
    print("  P&ID Line Extraction Pipeline (v3 — grouped, gap-preserving)")
    print("=" * 60)

    input_folder = cfg.INPUT_PDF_DIR
    pdf_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return

    for pdf_file in pdf_files:
        input_path = os.path.join(input_folder, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        print("-" * 60)

        stem = os.path.splitext(pdf_file)[0]
        cfg.CSV_FILENAME            = f"{stem}_line_coordinates_raw.csv"
        cfg.GROUPED_CSV_FILENAME    = f"{stem}_line_coordinates_grouped.csv"
        cfg.MORPH_IMAGE_FILENAME    = f"{stem}_morphological_line_mask.png"
        cfg.NUMBERED_IMAGE_FILENAME = f"{stem}_numbered_lines.png"
        cfg.AXIS_IMAGE_FILENAME     = f"{stem}_numbered_lines_with_axis.png"

        extractor = LineExtractor(cfg)
        outputs   = extractor.run(input_path=input_path)

        print()
        print(f"  Raw segments CSV : {os.path.join(cfg.LE_CSV_DIR, cfg.CSV_FILENAME)}")
        print(f"  Grouped CSV      : {os.path.join(cfg.LE_CSV_DIR, cfg.GROUPED_CSV_FILENAME)}")
        print(f"  Morph mask       : {outputs['morph_path']}")
        print(f"  Numbered lines   : {outputs['numbered_path']}")
        print(f"  With axis        : {outputs['axis_path']}")
        print(f"  Segments kept    : {len(outputs['df'])}")
        print(f"  Logical lines    : {outputs['grouped_df']['Group_ID'].nunique()}")
        print("=" * 60)

if __name__ == "__main__":
    main()