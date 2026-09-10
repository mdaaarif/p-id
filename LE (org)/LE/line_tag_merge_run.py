import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
from config import config_line_tag_merge as cfg
from src.line_tag_merge_src import LineTagMerger


def main():
    merger = LineTagMerger(cfg)
    result = merger.run()

    df = result["df"]
    print("\nSample output (first 15 matched rows):")
    matched = df[df["Nearest_Tag_Text"].notna()]
    print(
        matched[["Line_ID", "Nearest_Tag_Text", "Tag_Type",
                 "Distance_Pixels", "Confidence_Score"]]
        .head(15)
        .to_string(index=False)
    )
    print(f"\nFull results → {result['output_path']}")


if __name__ == "__main__":
    main()