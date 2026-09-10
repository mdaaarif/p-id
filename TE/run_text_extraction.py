"""Run the P&ID text-extraction stage from the current project layout."""

import sys

import config
from text_extraction import TextExtractor


def main() -> int:
    # The extraction module prints box-drawing status messages.  Windows
    # consoles commonly default to cp1252, which cannot encode those symbols.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("P&ID TEXT EXTRACTION")
    print("=" * 70)
    print(f"Input PDFs: {config.INPUT_PDF_DIR}")
    print(f"Output directory: {config.RESULTS_DIR}")

    TextExtractor(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
