import sys
import os

# Make LE folder importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_line_tag_merge as cfg
from line_tag_merge_src import LineTagMerger

# ---------------------------------------------------------------
# PDF definitions — add more entries here as needed
# ---------------------------------------------------------------
PDFS = [
#    {
#        "label":        "01.Marked",
#        "pdf_path":     r"C:\Users\user\Desktop\Work\p&id\data\01.Marked.pdf",
#        "line_csv":     r"C:\Users\user\Desktop\Work\p&id\LE\Results\LE_csv\01.Marked_line_coordinates_grouped.csv",
#        "tag_file":     "01.Marked",
#        "ce_csv":       r"C:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\01.Marked.csv",
#        "output_name":  "01.Marked_line_tag_topology.csv",
#    },
    {
        "label":        "02.Unmarked",
        "pdf_path":     r"C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf",
        "line_csv":     r"C:\Users\user\Desktop\Work\p&id\results\LE_csv\02.Unmarked_geometric_filtered_lines.csv",
        "tag_file":     "02.Unmarked",
        "ce_csv":       r"C:\Users\user\Desktop\Work\p&id\CE\Results\CE_csv_1\02.Unmarked.csv",
        "output_name":  "02.Unmarked_line_tag_topology.csv",
    },
]


def run_one(pdf_def):
    """Run the full pipeline for a single PDF definition."""

    # Patch config values for this specific PDF
    cfg.PDF_PATH          = pdf_def["pdf_path"]
    cfg.LINE_CSV_PATH     = pdf_def["line_csv"]
    cfg.TAG_CSV_PATH      = getattr(cfg, "TAG_CSV_PATH", None) # leave default
    cfg.CE_CSV_PATH       = pdf_def.get("ce_csv")
    cfg.TARGET_FILE_NAME  = pdf_def["tag_file"]
    cfg.OUTPUT_FILENAME   = pdf_def["output_name"]

    print()
    print("=" * 60)
    print(f"  Processing: {pdf_def['label']}")
    print("=" * 60)

    merger = LineTagMerger(cfg)
    result = merger.run()
    df     = result["df"]

    print(f"\n  SAMPLE OUTPUT — {pdf_def['label']}  (first 15 rows):")
    print(
        df[["Line_ID", "Line_Type", "line_tag",
            "x_start", "y_start", "x_end", "y_end",
            "tag_start", "tag_end"]]
        .head(15)
        .to_string(index=False)
    )
    print(f"\n  Saved → {result['output_path']}")
    return df


def main():
    for pdf_def in PDFS:
        run_one(pdf_def)

    print("\n\nAll PDFs processed.")


if __name__ == "__main__":
    main()