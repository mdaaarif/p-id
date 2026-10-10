# LineProcessor Code Walkthrough

## 1. Cheat Sheet

**What depends on what:**
* `line_processing/` is a 100% self-contained module. It has its own models, geometry, matchers, and exporter.
* The runner (`run_line_processor.py`) imports `line_processing` for pipeline merging, but also imports `line_association` solely for the legacy Tag Association stage.
* Third-party dependencies: `pandas` (for CSV reading/writing) and `PyMuPDF` / `fitz` (for reading PDF page dimensions). The pipeline does NOT use machine learning or semantic segmentation.

**Pipeline Steps (7-step process):**
1. Load & Transform: Read fragments and components, convert pixels to PDF points.
2. Tag Association (External): Match `LineTag`s to `LineFragment`s.
3. Candidate Generation (Inline): Find fragments touching the same inline component.
4. Candidate Generation (Touch): Find fragments physically touching at corners or T-junctions.
5. Candidate Generation (Collinear): Find collinear fragments with a small gap (e.g. unextracted valves).
6. DSU Merging: Group fragments using Disjoint Set Union, refusing merges with conflicting tags.
7. Export: Output Super Pipelines and branch connections.

**ASCII Diagram:**
```text
[Line Fragments (LE)] + [Components (CE)] + [Tags (TE)]
       |
       v
CoordinateTransformer (pixels to points)
       |
       v
LineTagAssociator (Pass 1 & Pass 2) --> Sets matched_tag
       |
       v
LineProcessor (Candidate Generation)
  - touch_or_tjunc
  - inline_component
  - collinear_gap
       |
       v
DisjointSetUnion (Merges fragments into SuperPipelines, refuses conflicts)
       |
       v
exporter (CSV outputs)
```

**Method Summary (Say it in one sentence):**
* `CoordinateTransformer.le_to_pt`: Converts raster pixels to PDF vector points based on page rotation and DPI.
* `load_line_fragments`: Reads line fragments, groups them by ID, and converts their coordinates to PDF points.
* `InlineComponentMatcher.find_inline_components`: Finds bridging and on-path components belonging to a super pipeline.
* `_find_inline_component_edges`: Creates merge candidates for fragments that touch the same inline component's bounding box.
* `_find_touching_edges`: Creates merge candidates for fragments that physically touch at endpoints (corners or T-junctions).
* `_find_collinear_edges`: Creates merge candidates for fragments that align horizontally or vertically across a small gap.
* `DisjointSetUnion.union`: Combines two fragments into the same group only if they don't have conflicting text tags.
* `_build_branch_connections`: Turns refused edges into a deduplicated list of branch connections.

---

## 2. Concepts in Plain Words

* **Line Fragment**: A single continuous line or set of connected segments output by the line extractor.
* **Super Pipeline**: A logical pipeline composed of multiple fragments merged together (e.g. fragments connected by a valve).
* **Tag Conflict Refusal**: If fragment A has tag "100-P-01" and fragment B has tag "200-P-02", they will never merge into the same Super Pipeline, even if they physically touch.
* **Collinear Gap**: If fragment A ends at `(100, 50)` and fragment B starts at `(120, 50)`, they form a collinear gap of 20 points along the X-axis.

---

## 3. Full Line-by-Line Walkthrough

### `line_processing/line_processor_config.py`
This holds the magical thresholds.

Lines 11-12:
```python
    # Inline component threshold (Rule 7b: center within 15 pt of line path)
    on_path_threshold_pt: float = 15.0
```
Say it in one sentence: Defines how close a component's center must be to a line to be considered "on" it.

Lines 14-16:
```python
    # Geometric tolerances for corners and bends (in PDF points)
    touch_tolerance_pt: float = 5.0
    padding_pt: float = 12.0 # Bounding box padding for inline valves/fittings
```
Say it in one sentence: `padding_pt` expands a component's bounding box by 12 points in all directions when checking if a line segment touches it. Making it larger merges lines further away; making it smaller requires tighter connections.

Lines 18-20:
```python
    # Collinear gap bridging (bridges across unextracted inline valves and dashed lines)
    collinear_max_gap_pt: float = 65.0
    collinear_axis_tolerance_pt: float = 3.0
```
Say it in one sentence: Allows a horizontal or vertical gap up to 65 points to bridge, as long as the segments align within 3 points.

### `line_processing/__init__.py`
Lines 1-3:
```python
# Exposes LineProcessor and Config
from .line_processor import LineProcessor
from .line_processor_config import LineProcessorConfig
```
Say it in one sentence: Exposes the main class and config so other files can import them cleanly.

### `line_processing/models.py`
Lines 50-58:
```python
@dataclass
class LineFragment:
    id: int # Group_ID
    line_type: str
    segments: List[Segment] = field(default_factory=list)
    matched_tag: Optional[str] = None
```
Say it in one sentence: Represents a group of line segments and the text tag it matched with.

### `line_processing/geometry.py`
Lines 40-57:
```python
def fragment_to_point_distance(
    fragment: LineFragment, px: float, py: float, clamping_penalty: float = 0.0
) -> Tuple[float, float, float]:
```
Say it in one sentence: Finds the shortest distance between a specific point and any segment in a line fragment.

Lines 87-91:
```python
def is_component_on_path(comp_center: Point, fragment: LineFragment, threshold: float) -> bool:
    """Checks if component center is within threshold distance of any segment in fragment (Rule 7b)."""
    min_dist, _, _ = fragment_to_point_distance(fragment, comp_center.x, comp_center.y)
    return min_dist <= threshold
```
Say it in one sentence: Checks if a component's center point is physically close to the line.

Lines 5-39:
```python
def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float, clamping_penalty: float = 0.0
) -> Tuple[float, float, float]:
```
Say it in one sentence: Calculates the exact perpendicular distance from a point to a line segment. (Example: Point (5, 5) to segment (0,0)-(10,0) has a distance of 5).

### `line_processing/adapters.py`
Lines 7-16:
```python
class CoordinateTransformer:
    def __init__(self, page_w_pts: float = 1684.0, page_h_pts: float = 2384.0, rotation: int = 90, dpi: float = 300.0):
```
Say it in one sentence: Initializes the transformer using the PDF page dimensions and rotation.

Lines 18-27:
```python
    @classmethod
    def from_pdf(cls, pdf_path: str, dpi: float = 300.0) -> "CoordinateTransformer":
```
Say it in one sentence: Opens the PDF with PyMuPDF (`fitz`) to read the actual page width, height, and rotation.

Lines 29-39:
```python
    def le_to_pt(self, x_px: float, y_px: float) -> Tuple[float, float]:
```
Say it in one sentence: Transforms a raster pixel coordinate into a PDF vector coordinate. (Example: At 300 DPI, `dpi_scale` is `300/72 = 4.16`; a pixel at `X=416` becomes `100` points).

Lines 41-46:
```python
def parse_bbox_string(s: Any) -> Tuple[float, float, float, float]:
```
Say it in one sentence: Safely converts string representations of lists or tuples into four floats.

Lines 48-94:
```python
def load_line_fragments(
    csv_path: str,
    needed_rows: Optional[Set[int]] = None,
    transformer: Optional[CoordinateTransformer] = None
) -> List[LineFragment]:
```
Say it in one sentence: Reads line segments from CSV, filters by `needed_rows` (if provided), converts pixels to points, and groups them by `Group_ID` with their original orientations.

Lines 96-126:
```python
def load_components(
    csv_path: str
) -> List[InlineComponent]:
```
Say it in one sentence: Reads CE components from CSV, parsing bounding boxes and extracting the `final_type` and `matched_tag`.

### `line_processing/matcher.py`
Lines 16-46:
```python
    def find_inline_components(
        self,
        pipeline: SuperPipeline,
        fragments_map: Dict[int, LineFragment],
        components: List[InlineComponent],
        threshold_px: float
    ) -> List[Tuple[str, str]]:
```
Say it in one sentence: Looks at every component and determines if it sits directly on the super pipeline's path (`on_path`) while actively ignoring components that serve as the pipeline's own anchor tag (Rule 8 deduplication).

### `line_processing/line_processor.py`
Lines 201-218:
```python
    def _sort_candidate_edges(self, edges: List[Dict[str, Any]], fragments: List[LineFragment]) -> None:
        ...
            tag_prio = 0 if t1 == t2 else 4
        ...
            type_prio = 0 if edge['reason'] in ('touch_or_tjunc', 'inline_component') else 1
```
Say it in one sentence: Sorts candidate edges so that matching tags are processed first (priority 0) and collinear gaps are processed after touching components (priority 1).

Lines 219-267:
```python
    def _assemble_super_pipelines(
        self, fragments: List[LineFragment], components: List[InlineComponent], dsu: DisjointSetUnion
    ) -> List[SuperPipeline]:
```
Say it in one sentence: Groups fragments that were successfully merged by the DSU and computes their confidence scores. 
Confidence examples:
- "Single Group": SP has only Fragment 1.
- "Confirmed": SP has Fragments 1 & 2. Fragment 1 tagged "LineA", Fragment 2 tagged "LineA".
- "Unverified": SP has Fragments 1 & 2. Fragment 1 tagged "LineA", Fragment 2 has no tag.

Lines 269-294:
```python
    def _build_branch_connections(
        self, rejected_edges: List[Dict[str, Any]], super_pipelines: List[SuperPipeline], fragments: List[LineFragment]
    ) -> List[Dict[str, Any]]:
```
Say it in one sentence: Takes edges that the DSU refused to merge and records them as branch connections, using `seen_branches` to deduplicate by storing `tuple(sorted([sp_u, sp_v]))`.

Lines 333-340:
```python
    def export_super_pipelines(
        self, super_pipelines: List[SuperPipeline], branches: List[Dict[str, Any]], 
        fragments: List[LineFragment], out_dir: str, stem: str, components: List[InlineComponent]
    ):
        """Exports the generated super pipelines."""
        _export_super_pipelines(
            super_pipelines, branches, fragments, out_dir, stem, components=components
        )
```
Say it in one sentence: Calls the local exporter function to write the results to CSV.

### `line_processing/exporter.py`
Lines 7-72:
```python
def export_super_pipelines(
```
Say it in one sentence: Creates `SUPER_pipelines.csv` (recording ID, group, orientation, tags, and inline components) and `branch_connections.csv`.

### `run_line_processor.py`
Lines 38-38:
```python
    fragments = processor.load_and_filter_fragments(lines_csv, needed_rows=None, transformer=transformer)
```
Say it in one sentence: Calls the loader with `needed_rows=None`, feeding all fragments into the pipeline.

---

## 4. Tag Association (Outside Line Processing)
The runner script performs a dedicated Tag Association stage before merging:
1. `LineTagAssociator.associate` (imported from `line_association`) matches `LineTag`s to `LineFragment`s.
2. It uses a 2-pass approach: Pass 1 finds confident nearest-neighbor matches; Pass 2 expands search distance for unmatched tags.
3. It sets the `matched_tag` property on the raw `LineFragment` object. 
**Note:** This logic stays fully outside `line_processing`.

---

## 5. How It All Connects

**Call Graph:**
```text
run_line_processor.py (main)
 ├── LineProcessorConfig()
 ├── CoordinateTransformer.from_pdf()
 ├── processor.load_and_filter_fragments(needed_rows=None)
 ├── load_line_tags()
 ├── processor.load_components()
 ├── LineAssociationConfig()
 ├── LineTagAssociator.associate()
 ├── processor.build_super_pipelines()
 │    ├── _find_inline_component_edges
 │    ├── _find_touching_edges
 │    ├── _find_collinear_edges
 │    ├── _sort_candidate_edges
 │    ├── DisjointSetUnion
 │    ├── _assemble_super_pipelines
 │    └── _build_branch_connections
 ├── LineAssociationExporter.export_association_results()
 └── processor.export_super_pipelines()
```

**Data Shape Table:**
| Object | Shape / Columns |
| :--- | :--- |
| `LineFragment` | ID, List of Segments, matched_tag |
| `SuperPipeline` | ID, List of Fragment IDs, tag, confidence |
| `SUPER_pipelines.csv` | Super_Pipeline_ID, Group_ID, Segment_In_Group, Line_Type, Orientation, x_start, y_start, x_end, y_end, Pipeline_Tag, Merge_Confidence, Inline_Components |
| `branch_connections.csv` | SP_1, SP_2, Tag_1, Tag_2, Refusal_X, Refusal_Y |

---

## 6. Questions from a Lead

**Q1. What is tested and what is not?**
Only `02.Unmarked` was run end-to-end successfully. `01.Marked` fails because `results/CE_csv/01.Marked_component_tag_associated.csv` is missing from the environment. We have no ground-truth accuracy measure, we only verified byte-for-byte equality with the old legacy script.

**Q2. Why make independent copies instead of a shared package?**
Trade-off: Having duplicated code avoids breaking or modifying the legacy `line_association` module in any way, making the new `line_processing` completely isolated and safe to refactor.

**Q3. Does the row list still exist?**
No. The manual row list is gone. `load_and_filter_fragments` is called with `needed_rows=None`. 

**Q4. Doesn't that mean junk gets merged?**
Yes. This is a known limitation. Borders, title blocks, and revision tables currently go through the merge. Our next planned step is an automatic deterministic border filter relative to page size.

**Q5. Is this using machine learning?**
No. This pipeline is entirely deterministic classical geometry.

**Q6. Why are thresholds in points?**
Because PDF specifications use points. The `CoordinateTransformer` standardizes all pixel coordinates to points early in the process.

**Q7. How are Super Pipeline IDs generated?**
`SP_000`, `SP_001`, etc., follow the order their cluster root is encountered when iterating over the sorted fragments list. They are stable across runs.

**Q8. What happens if multiple refused edges exist between two pipelines?**
`_build_branch_connections` uses `seen_branches` (sorting the IDs into a tuple) to deduplicate them. One branch connection is exported per pair of pipelines.

**Q9. How are tags prevented from merging incorrectly?**
The `DisjointSetUnion` handles this. When `union(u, v)` is called, it checks if `tag_u` and `tag_v` exist and are different. If so, it returns False and refuses the merge.

**Q10. How do you handle unextracted inline valves?**
The collinear gap edge generator looks for segments aligned on the same axis within `collinear_axis_tolerance_pt` and bridges gaps up to `collinear_max_gap_pt`.

**Q11. Do major equipment items act as bridges?**
No. They are explicitly excluded in `_find_collinear_edges` via `touches_major_equipment`, so lines never merge *through* a vessel or pump.

**Q12. What third-party packages do you rely on?**
Only `pandas` for reading/writing CSVs and `PyMuPDF` (`fitz`) inside the Coordinate Transformer to read page geometry.

---

## 7. Known Quirks
* Unused `import pandas as pd` exists in `line_processor.py`.
* `CoordinateTransformer` applies rotation based on `90`, `0`, `180`, `270`.
* The corner reference point extraction logic specifically uses `x_start` and `y_start` blindly in some cases.
* There are two separate configs (`LineProcessorConfig` and `LineAssociationConfig`), both actively used in `run_line_processor.py`.
