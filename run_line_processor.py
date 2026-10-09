import os
import sys
import argparse

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from line_association.adapters import CoordinateTransformer, load_line_tags
from line_association import LineTagAssociator, LineAssociationConfig, LineAssociationExporter
from line_processing.line_processor import LineProcessor
from line_processing.line_processor_config import LineProcessorConfig
from line_association import summarize_line_association, format_line_stats

def run_association(stem: str = "02.Unmarked", out_dir: str = "results/LE_csv"):
    pdf_path = os.path.join(_ROOT, "data", "raw", f"{stem}.pdf")
    lines_csv = os.path.join(_ROOT, "LE", "Results", "LE_csv", f"{stem}_line_coordinates_grouped.csv")
    tags_csv = os.path.join(_ROOT, "TE", "Results", "TE_csv", "process_tags.csv")
    comp_csv = os.path.join(_ROOT, "results", "CE_csv", f"{stem}_component_tag_associated.csv")

    abs_out_dir = os.path.join(_ROOT, out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUNNING LINE PROCESSOR ENGINE: {stem}")
    print(f"{'='*60}")

    print(f"[1/5] Building coordinate transformer from {pdf_path}...")
    transformer = CoordinateTransformer.from_pdf(pdf_path, dpi=300.0)

    # 1. Initialize LineProcessor
    print("[2/5] Initializing LineProcessor and loading inputs...")
    config = LineProcessorConfig()
    processor = LineProcessor(config)

    fragments = processor.load_and_filter_fragments(lines_csv, needed_rows=None, transformer=transformer)
    tags = load_line_tags(tags_csv, file_name=stem)
    components = processor.load_components(comp_csv)
    print(f"      Loaded {len(fragments)} line fragments, {len(tags)} line tags, {len(components)} components.")

    # 2. (EXTERNAL) Line Tag Association
    print("[3/5] Running Part 1: Line-Tag Bipartite Association...")
    tag_config = LineAssociationConfig()
    associator = LineTagAssociator(config=tag_config)
    assoc_result = associator.associate(fragments, tags)
    print(f"      Matches: {len(assoc_result.matches)} (Pass 1: {assoc_result.pass1_count}, Pass 2: {assoc_result.pass2_count})")
    
    # 3. Merging (using LineProcessor)
    print("[4/5] Running Part 2 & Super-Pipeline Merging via LineProcessor...")
    super_pipelines, branches = processor.build_super_pipelines(fragments, components)
    print(f"      Generated {len(super_pipelines)} continuous Super Pipelines and {len(branches)} branch connections.")

    # 4. Exporting
    print(f"[5/5] Exporting result CSVs to {abs_out_dir}...")
    exporter = LineAssociationExporter()
    exporter.export_association_results(assoc_result, fragments, abs_out_dir, stem)
    processor.export_super_pipelines(super_pipelines, branches, fragments, abs_out_dir, stem, components=components)

    stats = summarize_line_association(assoc_result, len(fragments), super_pipelines, branches)
    print("\n" + format_line_stats(stats))
    print(f"\nOutputs written to: {abs_out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run Line Processor Engine")
    parser.add_argument("--stem", default="02.Unmarked", help="Drawing stem (e.g. 02.Unmarked, 01.Marked)")
    parser.add_argument("--out-dir", default="results/LE_csv", help="Output directory for CSV files")
    args = parser.parse_args()

    run_association(stem=args.stem, out_dir=args.out_dir)

if __name__ == '__main__':
    main()
