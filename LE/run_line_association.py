import os
import sys
import argparse

# Ensure project root is on sys.path
_LE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_LE_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from line_association import (
    LineTagAssociator,
    SuperPipelineMerger,
    LineAssociationConfig,
    CoordinateTransformer,
    load_line_tags,
    load_line_fragments,
    load_components,
    LineAssociationExporter,
    summarize_line_association,
    format_line_stats
)

# Curated process line rows for 02.Unmarked (excluding borders, revision tables, ticks)
# 111 user needed_rows + 6 restored rows (29, 109, 134, 138, 142, 143)
NEEDED_ROWS_02_UNMARKED = {
    2, 4, 6, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 33, 34, 37, 38, 39, 40, 41, 42, 54, 55, 56, 57, 59, 60, 61, 63, 65, 66,
    70, 73, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 124, 125, 126, 127, 132, 134, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 176, 177, 181, 183, 184,
    191, 192, 193, 195, 202, 218, 224
}

def run_association(stem: str = "02.Unmarked", out_dir: str = "results/LE_csv"):
    pdf_path = os.path.join(_ROOT, "data", "raw", f"{stem}.pdf")
    lines_csv = os.path.join(_ROOT, "LE", "Results", "LE_csv", f"{stem}_line_coordinates_grouped.csv")
    tags_csv = os.path.join(_ROOT, "TE", "Results", "TE_csv", "process_tags.csv")
    comp_csv = os.path.join(_ROOT, "results", "CE_csv", f"{stem}_component_tag_associated.csv")

    abs_out_dir = os.path.join(_ROOT, out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUNNING LINE ASSOCIATION & TOPOLOGY ENGINE: {stem}")
    print(f"{'='*60}")

    # 1. Coordinate Transformer
    print(f"[1/5] Building coordinate transformer from {pdf_path}...")
    transformer = CoordinateTransformer.from_pdf(pdf_path, dpi=300.0)
    print(f"      Rotation: {transformer.rotation}°, Scale: {transformer.dpi_scale:.4f}")

    # 2. Load Inputs
    print("[2/5] Loading input CSVs...")
    needed = NEEDED_ROWS_02_UNMARKED if stem == "02.Unmarked" else None
    fragments = load_line_fragments(lines_csv, needed_rows=needed, transformer=transformer)
    tags = load_line_tags(tags_csv, file_name=stem)
    components = load_components(comp_csv)
    print(f"      Loaded {len(fragments)} line fragments, {len(tags)} line tags, {len(components)} components.")

    # 3. Part 1: Line <-> Line Tag Association (Rules 1 to 6)
    print("[3/5] Running Part 1: Line-Tag Bipartite Association...")
    config = LineAssociationConfig()
    associator = LineTagAssociator(config=config)
    assoc_result = associator.associate(fragments, tags)
    print(f"      Matches: {len(assoc_result.matches)} (Pass 1: {assoc_result.pass1_count}, Pass 2: {assoc_result.pass2_count})")
    print(f"      Unmatched Fragments: {len(assoc_result.unmatched_fragments)}, Orphan Tags: {len(assoc_result.orphan_tags)}")

    # 4. Part 2 & Continuous Super-Pipeline Merging
    print("[4/5] Running Part 2 & Super-Pipeline Merging...")
    merger = SuperPipelineMerger(config=config)
    super_pipelines, branches = merger.merge(fragments, components)
    print(f"      Generated {len(super_pipelines)} continuous Super Pipelines and {len(branches)} branch connections.")

    # 5. Export Results
    print(f"[5/5] Exporting result CSVs to {abs_out_dir}...")
    exporter = LineAssociationExporter()
    exporter.export_association_results(assoc_result, fragments, abs_out_dir, stem)
    exporter.export_super_pipelines(super_pipelines, branches, fragments, abs_out_dir, stem, components=components)

    stats = summarize_line_association(assoc_result, len(fragments), super_pipelines, branches)
    print("\n" + format_line_stats(stats))
    print(f"\nOutputs written to: {abs_out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run Line Association and Super Pipeline Engine")
    parser.add_argument("--stem", default="02.Unmarked", help="Drawing stem (e.g. 02.Unmarked, 01.Marked)")
    parser.add_argument("--out-dir", default="results/LE_csv", help="Output directory for CSV files")
    args = parser.parse_args()

    run_association(stem=args.stem, out_dir=args.out_dir)

if __name__ == '__main__':
    main()
