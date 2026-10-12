# Prompt for Antigravity: restructure `line_processing/` into `src/ utils/ config/ tests/ run/` and add real tests

**Pure restructuring + tests. No behaviour change. Output must stay byte-for-byte identical.**

## Scope
Only `line_processing/` (and the root-level `run_line_processor.py`, which moves into it) may change. Do NOT touch `line_association/`, `association/`, `TE/`, `CE/`, `LE/`, `data/`, `results/` (except creating new result folders named below), or any `.md` file. The rest of the repository relies on a flat layout and must stay as it is.

## Step 0: safety snapshot
1. Run `git status`. Then make ONE local commit containing only the current `line_processing/` and `run_line_processor.py` (message: `Snapshot before restructuring line_processing`). Do not push. Do not add `scratch/`, `results/` or anything else.
2. Record the current baseline output: run `.venv\Scripts\python.exe run_line_processor.py --out-dir results/restructure_before` (this is the pre-change reference; `results/nr_after_no_file` is the older reference and must also match).

## Target layout
```
line_processing/
├── __init__.py                  # exports LineProcessor, LineProcessorConfig
├── src/
│   ├── __init__.py
│   ├── line_processor.py        # LineProcessor class + DisjointSetUnion  (core logic)
│   ├── models.py                # Point, BBox, Segment, LineFragment, InlineComponent, SuperPipeline
│   └── matcher.py               # InlineComponentMatcher
├── utils/
│   ├── __init__.py
│   ├── geometry.py              # geometry helper functions
│   ├── adapters.py              # CoordinateTransformer, load_line_fragments, load_components, parse_bbox_string
│   └── exporter.py              # export_super_pipelines
├── config/
│   ├── __init__.py
│   └── line_processor_config.py # LineProcessorConfig (constants only)
├── tests/
│   ├── __init__.py
│   ├── test_geometry.py
│   ├── test_line_processor.py
│   ├── test_regression.py
│   └── fixtures/                # baseline CSVs for the regression test (see below)
└── run/
    ├── __init__.py
    └── run_line_processor.py    # the runner (moved from the repo root)
```
Use `git mv` for files that are tracked so history is kept; new files are just created. Keep file CONTENTS identical except for import lines and the path logic described below. Update the line-1 "Copied from ..." header comments only if they now say something wrong.

## Import rules
- Inside the package use RELATIVE imports (`from ..utils.geometry import ...`, `from ..config.line_processor_config import ...`, `from .models import ...`). `line_processing/__init__.py` must still make `from line_processing import LineProcessor, LineProcessorConfig` work.
- Nothing inside `line_processing/` (src, utils, config, tests/helpers) may import `line_association`, `association`, `TE`, `CE` or `LE`. EXCEPTION: only `run/run_line_processor.py` may import `line_association` for the tag-association stage (`LineTagAssociator`, `LineAssociationConfig`, `load_line_tags`, `LineAssociationExporter.export_association_results`, `summarize_line_association`, `format_line_stats`), exactly as it does today.
- No circular imports. If one appears, move the shared item to the lower layer (models/config) rather than adding lazy imports.

## Runner (`run/run_line_processor.py`)
- It must work both as `.venv\Scripts\python.exe line_processing\run\run_line_processor.py ...` and as `.venv\Scripts\python.exe -m line_processing.run.run_line_processor ...`. Use absolute imports (`from line_processing.src.line_processor import LineProcessor`, etc.) and put the project root on `sys.path` (root = three levels above this file: `Path(__file__).resolve().parents[2]`).
- All input paths (`data/raw`, `LE/Results/LE_csv`, `TE/Results/TE_csv`, `results/CE_csv`) are resolved from that project root exactly as before. Keep `--stem` and `--out-dir` and every other behaviour unchanged. No absolute paths.
- Leave NO copy of the runner at the repo root (move it), but if another file in the repo imports or calls `run_line_processor.py`, tell me instead of editing that file.

## Tests (standard library `unittest` only; do NOT add pytest or any dependency)
Run with `.venv\Scripts\python.exe -m unittest discover -s line_processing/tests -t .`
- `test_geometry.py`: small hand-computed cases for `point_to_segment_distance`, `segments_touch`, `endpoint_touches_body` (T-junction yes, near-endpoint no), `is_point_near_bbox` (with padding), `segments_collinear_gap` (horizontal gap, vertical gap, too-large gap, misaligned axis), `CoordinateTransformer.le_to_pt` for rotation 0 and 90 with a known DPI.
- `test_line_processor.py`: build synthetic `LineFragment`s in code (no files) and check `LineProcessor().build_super_pipelines(...)`:
  1. two fragments touching at a corner -> 1 super pipeline;
  2. T-junction (endpoint on another segment's body) -> merged;
  3. two collinear horizontal fragments with a 20 pt gap -> merged; with an 80 pt gap (> 65) -> not merged;
  4. collinear gap whose gap endpoint lies inside a major-equipment component bbox (e.g. type `Tank`) -> NOT merged;
  5. two touching fragments tagged `A` and `B` -> two super pipelines and one branch connection with `Tag_1`/`Tag_2` set;
  6. fragment tagged `A` touching an untagged one -> merged and the super pipeline tag is `A`; confidence labels: single fragment -> `Single Group`, two fragments both tagged `A` -> `Confirmed`, one tagged + one untagged -> `Unverified`;
  7. two refused edges between the same two pipelines -> exactly ONE branch connection row;
  8. `DisjointSetUnion`: union returns False on tag conflict, True otherwise, find compresses paths.
  Compute expected values by hand from the code and the config defaults; do not just copy whatever the code prints. If a test fails because the code behaves differently from what the config/docstrings say, DO NOT change the code: report it.
- `test_regression.py`: runs the whole pipeline on `02.Unmarked` (via the runner's `run_association` function) into a temp directory and compares the 5 CSVs byte for byte with the files in `tests/fixtures/`. Skip the test with a clear message (`self.skipTest`) if the real input files are missing. Create `tests/fixtures/` by copying the 5 CSVs from `results/restructure_before`; if their total size is over 2 MB, store only `02.Unmarked_SUPER_pipelines.csv` and `02.Unmarked_branch_connections.csv`.
- Also add `test_independence.py`-style check inside `test_line_processor.py`: in a subprocess with `-I`, block `line_association`, `association`, `TE`, `CE`, `LE` via `sys.modules[...] = None` and import `line_processing.src`, `line_processing.utils`, `line_processing.config`; it must succeed.

## Hard rules
- No logic changes, no renames of functions/classes/fields, no reformatting of moved code, no new dependencies, no absolute paths.
- Do not delete existing result folders.
- If anything cannot be done without changing behaviour, stop and report.

## Verification (run all; report each result)
1. `.venv\Scripts\python.exe line_processing\run\run_line_processor.py --out-dir results/restructure_after`. Compare each of the 5 CSVs byte for byte (`fc /b`) with `results/restructure_before` AND with `results/nr_after_no_file`. All must be identical.
2. Same via module form: `.venv\Scripts\python.exe -m line_processing.run.run_line_processor --out-dir results/restructure_after_m` and compare again.
3. `.venv\Scripts\python.exe -m unittest discover -s line_processing/tests -t . -v`. Show the full output (count of tests, passes, failures, skips).
4. Independence: `.venv\Scripts\python.exe -I -c "import sys; [sys.modules.__setitem__(m, None) for m in ('line_association','association','TE','CE','LE')]; sys.path.insert(0,'.'); from line_processing import LineProcessor, LineProcessorConfig; print('OK')"`.
5. `grep -rn "line_association\|from association\|import association" line_processing --include=*.py`: the only allowed non-comment hits are inside `line_processing/run/run_line_processor.py`. Show the output.
6. `git status` and `git diff --stat HEAD`: only paths under `line_processing/` (moved/new files) and the removed root-level `run_line_processor.py` may differ; nothing in `line_association/` or other folders.
7. Show the final `tree`/`Get-ChildItem -Recurse line_processing` (excluding `__pycache__`) and a table `old path -> new path`.
8. Report anything you could not keep identical, any test that failed, and any file elsewhere in the repo that referenced the old root-level runner or the old module paths.
