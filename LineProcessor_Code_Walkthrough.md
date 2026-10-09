# Line Processor Code Walkthrough

## 1. Cheat Sheet

### Pipeline in 7 Steps
1. **Input CSVs:** Load raw coordinates of line fragments, text tags, and components.
2. **Load & Filter:** Read the fragments and manually drop garbage lines using a curated `needed_rows` list.
3. **Tag Association:** Map text tags to the nearest line fragments using a 2-pass bipartite matcher.
4. **Build Candidate Edges:** Gather all potential connections between fragments (corners, T-junctions, collinear gaps, and shared inline valves).
5. **Union-Find Merge (DSU):** Group connected fragments together, refusing merges where text tags conflict.
6. **Super Pipelines:** Convert grouped fragments into continuous pipelines and assign inline components.
7. **Export:** Write the final pipelines and refused branch connections to CSV files.

### Flow Diagram
```text
[Raw Line CSV] ----+
                   v
[Raw Tags CSV] -> (Tag Associator) -> [Tagged Fragments]
                   |                       |
[Raw Comps CSV] ---+                       v
                                  (Candidate Edges)
                                           |
                                  (Union-Find Merge)
                                           |
                                           v
[Super Pipeline CSV] <------------- [Super Pipelines]
```

### Worked Example
Imagine three small line fragments:
- **Frag 1:** Tagged "Pipe-A"
- **Frag 2:** No tag, but touches Frag 1.
- **Frag 3:** Tagged "Pipe-B", touches Frag 2.

We have two candidate merges: Edge 1-2 and Edge 2-3. Both have exactly one tagged fragment, so they both get a tag priority of `1`. 
Because Python's `sort()` is stable, equal-priority edges are processed in the order they were proposed. 
If Edge 1-2 is processed first:
*Step 1:* Merge Frag 1 and Frag 2. Frag 2 adopts the tag "Pipe-A".
*Step 2:* Try to merge Frag 2 and Frag 3. Frag 2 is now "Pipe-A", Frag 3 is "Pipe-B". They conflict! The merge is refused and becomes a Branch Connection.
If Edge 2-3 were processed first, Frag 2 would become "Pipe-B", and Edge 1-2 would be refused. The stable sort guarantees this ordering never changes between runs, making the output 100% reproducible.

### Method 'Say it in one sentence' Table
| Method | Say it in one sentence |
|---|---|
| `load_and_filter_fragments` | Reads raw line pieces and throws out garbage rows based on a hardcoded list. |
| `load_components` | Reads the pre-associated inline components like valves. |
| `_find_inline_component_edges` | Proposes merges between any two lines that touch the same valve. |
| `_find_touching_edges` | Proposes merges for lines whose endpoints physically touch or form a T-junction. |
| `_find_collinear_edges` | Proposes merges for straight lines that are broken by gaps, as long as the gap doesn't cross a massive tank. |
| `_sort_candidate_edges` | Sorts merge proposals so fragments that already have the same tag merge before untagged or conflicting ones. |
| `_assemble_super_pipelines` | Packages the final grouped fragments into SuperPipeline objects and attaches valves that lie on their path. |
| `_build_branch_connections` | Turns rejected, conflicting merges into explicit branch connections (T-junctions between different pipes). |
| `build_super_pipelines` | The orchestrator that gathers edges, sorts them, merges them using DSU, and builds pipelines and branches. |
| `export_super_pipelines` | Writes the final merged pipeline objects to a CSV file. |

## 2. Concepts in Plain Words

- **PDF points vs pixels:** A PDF uses 'points' (1 pt = 1/72 inch). Computer vision outputs 'pixels'. We convert pixels back to points so the output perfectly overlays the original vector drawing. *Example: Pixel (300, 300) might map to Point (72.0, 72.0).* 
- **Segment / Endpoint:** A segment is a straight line drawn from a start point to an end point. Endpoints are its two tips.
- **Corner vs T-junction:** A corner is when two endpoints touch. A T-junction is when one endpoint hits the solid middle body of another segment.
- **Collinear gap:** Two segments that sit on the exact same invisible straight line (axis) but have an empty space between them. *Example: A dashed line -- --*.
- **Bounding box + padding:** A rectangle tightly wrapping a drawn symbol. Padding inflates it slightly so endpoints that stop just short of the drawing still register as touching it.
- **Union-find (DSU):** A fast algorithm to group items together by giving them all a shared 'Root' leader.
- **Tag-conflict refusal:** A rule saying 'Water' pipes and 'Gas' pipes cannot merge, even if they touch. This prevents entire drawings from blending into a single pipe.
- **`needed_rows`:** A manual, hand-curated list of row IDs that contain actual process pipes, used to discard borders and tables. This is the weakest point of the pipeline because it requires human intervention.

## 3. Full Line-by-Line Walkthrough

### `line_processing/line_processor_config.py`
**Lines 1-3:**
```python
from dataclasses import dataclass, field
from typing import Set
```
- **What it is:** Imports
- **What it does:** Imports dataclass and typing tools.
- **Why it is needed:** Needed to define a clean configuration class without writing a manual __init__.

**Lines 4-13:**
```python
# Hardcoded rows needed for 02.Unmarked (curated to exclude borders, revision tables, ticks)
NEEDED_ROWS_02_UNMARKED = {
    2, 4, 6, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 33, 34, 37, 38, 39, 40, 41, 42, 54, 55, 56, 57, 59, 60, 61, 63, 65, 66,
    70, 73, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 124, 125, 126, 127, 132, 134, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 176, 177, 181, 183, 184,
    191, 192, 193, 195, 202, 218, 224
}
```
- **What it is:** Hardcoded filter list
- **What it does:** A manual list of row indices representing true process pipes.
- **Why it is needed:** Needed to discard borders, revision tables, and tick marks. This manual step is the weakest point of the pipeline.

**Lines 15-20:**
```python
@dataclass
class LineProcessorConfig:
    """
    Configuration parameters for line processing (loading, filtering, merging).
    All geometric coordinates and radii are normalized to PDF points (pt).
    """
```
- **What it is:** Class definition
- **What it does:** Defines the LineProcessorConfig dataclass.
- **Why it is needed:** Keeps all geometric and algorithmic thresholds in one object.

**Lines 21-23:**
```python
# Inline component threshold (Rule 7b: center within 15 pt of line path)
    on_path_threshold_pt: float = 15.0
```
- **What it is:** Inline threshold config
- **What it does:** Sets `on_path_threshold_pt` to 15.0 points.
- **Why it is needed:** Controls how far a valve can float from a line and still be 'inline'. Larger = falsely catching distant valves. Smaller = missing slightly misaligned valves.

**Lines 24-27:**
```python
# Geometric tolerances for corners and bends (in PDF points)
    touch_tolerance_pt: float = 5.0
    padding_pt: float = 12.0 # Bounding box padding for inline valves/fittings
```
- **What it is:** Touch tolerance & padding config
- **What it does:** Sets `touch_tolerance_pt` (5.0) and `padding_pt` (12.0).
- **Why it is needed:** Controls how large a gap endpoints can have before they stop counting as 'touching'. Larger = false corner merges. Smaller = missing sloppy corners.

**Lines 28-31:**
```python
# Collinear gap bridging (bridges across unextracted inline valves and dashed lines)
    collinear_max_gap_pt: float = 65.0
    collinear_axis_tolerance_pt: float = 3.0
```
- **What it is:** Collinear gap config
- **What it does:** Sets `collinear_max_gap_pt` (65.0) and `collinear_axis_tolerance_pt` (3.0).
- **Why it is needed:** Allows jumping over 65 pt gaps if the lines are on the same axis within a 3 pt wiggle room. Larger max gap = jumping across entire unrelated regions.

**Lines 32-38:**
```python
# Major terminal equipment types that lines should NEVER bridge through
    major_equipment_types: Set[str] = field(default_factory=lambda: {
        'Vessel', 'Tank', 'Pump / Compressor', 'Heat Exchanger', 
        'Drum', 'Compressor', 'Complex Equipment',
        'Speed/Safety Valve', 'Pressure Valve', 'Relief Valve', 'Safety Valve'
    })
```
- **What it is:** Equipment stop list
- **What it does:** A set of component names (like 'Vessel' or 'Tank').
- **Why it is needed:** Defines massive objects that collinear gaps should never bridge across. Removing this would cause a pipe entering the left of a tank to merge with a pipe exiting the right.

### `line_processing/__init__.py`
**Lines 1-3:**
```python
# Exposes LineProcessor and Config
from .line_processor import LineProcessor
from .line_processor_config import LineProcessorConfig, NEEDED_ROWS_02_UNMARKED
```
- **What it is:** Module exports
- **What it does:** Exposes the processor and config to the outside module.
- **Why it is needed:** Allows importing `LineProcessor` directly from `line_processing` instead of the nested filename.

### `line_processing/line_processor.py`
**Lines 1-3:**
```python
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional, Any
```
- **What it is:** Imports
- **What it does:** Imports standard types and pandas.
- **Why it is needed:** Pandas is unused here (a harmless leftover), but typing is needed for type hints.

**Lines 4-12:**
```python
# Reuse existing helpers
from line_association.models import LineFragment, InlineComponent, SuperPipeline
from line_association.geometry import (
    segments_touch, endpoint_touches_body, is_point_near_bbox, segments_collinear_gap
)
from line_association.adapters import CoordinateTransformer
from line_association.rules import InlineComponentMatcher
from .line_processor_config import LineProcessorConfig
```
- **What it is:** Internal project imports
- **What it does:** Imports models, math helpers, and the config.
- **Why it is needed:** Reuses existing logic instead of rewriting distance math or data classes.

**Lines 14-18:**
```python
class DisjointSetUnion:
    """Helper class for managing Disjoint Set Union operations with tag conflict refusal."""
    def __init__(self, fragments: List[LineFragment]):
        self.parent = {f.id: f.id for f in fragments}
        self.dsu_tag = {f.id: f.matched_tag for f in fragments}
```
- **What it is:** DSU Constructor
- **What it does:** Initializes the DisjointSetUnion with fragments as their own parents.
- **Why it is needed:** Sets up the initial state where every fragment is an isolated group and stores their initial tags.

**Lines 19-24:**
```python
def find(self, u: int) -> int:
        if self.parent[u] == u:
            return u
        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
```
- **What it is:** DSU Find
- **What it does:** Recursively finds the root parent of a fragment, flattening the tree as it goes.
- **Why it is needed:** Required to quickly check which super-pipeline group a fragment currently belongs to.

**Lines 26-35:**
```python
def union(self, u: int, v: int) -> bool:
        """
        Unions two fragments if their tags do not conflict.
        Returns True if successful, False if rejected due to conflict.
        """
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u == root_v:
            return True
```
- **What it is:** DSU Union setup
- **What it does:** Finds the roots of fragments u and v, returning True if already connected.
- **Why it is needed:** The first step in attempting to merge two fragments.

**Lines 36-42:**
```python
tag_u = self.dsu_tag[root_u]
        tag_v = self.dsu_tag[root_v]

        # Rule 5: Conflict Refusal
        if tag_u and tag_v and tag_u != tag_v:
            return False
```
- **What it is:** Tag Conflict Refusal
- **What it does:** Checks if both groups have tags and if they are different; if so, returns False.
- **Why it is needed:** Rule 5: Prevents merging clashing pipelines (e.g. 'Water' and 'Gas').

**Lines 43-47:**
```python
self.parent[root_u] = root_v
        if not tag_v and tag_u:
            self.dsu_tag[root_v] = tag_u
        return True
```
- **What it is:** DSU Union Execution
- **What it does:** Points u's root to v's root, transferring the tag if v was untagged.
- **Why it is needed:** Completes the physical merge and propagates tags to untagged lines.

**Lines 49-56:**
```python
class LineProcessor:
    """
    Encapsulates line fragment loading, filtering, and SuperPipeline merging.
    Strictly isolated from tag association logic.
    """
    def __init__(self, config: Optional[LineProcessorConfig] = None):
        self.config = config or LineProcessorConfig()
```
- **What it is:** LineProcessor Constructor
- **What it does:** Initializes the LineProcessor with a config.
- **Why it is needed:** Sets up the engine state.

**Lines 57-63:**
```python
# Instantiate the inline component matcher using compatible config
        from line_association.config import LineAssociationConfig
        compat_config = LineAssociationConfig()
        compat_config.on_path_threshold_pt = self.config.on_path_threshold_pt
        compat_config.major_equipment_types = self.config.major_equipment_types
        self.component_matcher = InlineComponentMatcher(compat_config)
```
- **What it is:** Matcher Adapter
- **What it does:** Creates a legacy LineAssociationConfig to initialize the InlineComponentMatcher.
- **Why it is needed:** Allows reusing the old component matching code without modifying it.

**Lines 64-70:**
```python
def load_and_filter_fragments(
        self, csv_path: str, needed_rows: Optional[Set[int]] = None, transformer: Optional[CoordinateTransformer] = None
    ) -> List[LineFragment]:
        """Loads line fragments and applies the needed_rows filter immediately."""
        from line_association.adapters import load_line_fragments
        return load_line_fragments(csv_path, needed_rows=needed_rows, transformer=transformer)
```
- **What it is:** load_and_filter_fragments method
- **What it does:** Calls the adapter to load lines and applies the needed_rows filter.
- **Why it is needed:** Reads CSV data and instantly drops garbage lines.

**Lines 71-75:**
```python
def load_components(self, csv_path: str) -> List[InlineComponent]:
        """Loads components."""
        from line_association.adapters import load_components
        return load_components(csv_path)
```
- **What it is:** load_components method
- **What it does:** Calls the adapter to load components.
- **Why it is needed:** Reads CSV data for valves/equipment.

**Lines 76-83:**
```python
def _find_inline_component_edges(
        self, fragments: List[LineFragment], inline_comps: List[InlineComponent]
    ) -> List[Dict[str, Any]]:
        """Finds edges between fragments that both touch the same inline component."""
        edges = []
        frag_map = {f.id: f for f in fragments}
        padding_pt = self.config.padding_pt
```
- **What it is:** _find_inline_component_edges setup
- **What it does:** Prepares a mapping to track which lines touch which valves.
- **Why it is needed:** Needed to efficiently find pipes connected by a shared valve.

**Lines 84-91:**
```python
comp_to_fragments: Dict[int, Set[int]] = {i: set() for i in range(len(inline_comps))}
        for fid, frag in frag_map.items():
            for seg in frag.segments:
                for c_idx, comp in enumerate(inline_comps):
                    if is_point_near_bbox(seg.x_start, seg.y_start, comp.bbox, padding_pt) or \
                       is_point_near_bbox(seg.x_end, seg.y_end, comp.bbox, padding_pt):
                        comp_to_fragments[c_idx].add(fid)
```
- **What it is:** Bounding box hit test
- **What it does:** Loops over all endpoints and checks if they land inside any valve's padded bounding box.
- **Why it is needed:** Registers physical connections between pipes and valves.

**Lines 92-103:**
```python
for c_idx, touching_fids in comp_to_fragments.items():
            fids_list = list(touching_fids)
            for j in range(len(fids_list)):
                for k in range(j + 1, len(fids_list)):
                    cx, cy = inline_comps[c_idx].center.x, inline_comps[c_idx].center.y
                    edges.append({
                        'u': fids_list[j], 'v': fids_list[k],
                        'x': cx, 'y': cy, 'reason': 'inline_component',
                        'symbol_id': inline_comps[c_idx].symbol_id
                    })
        return edges
```
- **What it is:** Valve edge generation
- **What it does:** For any valve touched by multiple fragments, creates an edge requesting a merge.
- **Why it is needed:** Connects pipes that are interrupted by an inline valve symbol.

**Lines 104-113:**
```python
def _find_touching_edges(self, fragments: List[LineFragment]) -> List[Dict[str, Any]]:
        """Finds edges where fragments physically touch at corners or T-junctions."""
        edges = []
        touch_tol_pt = self.config.touch_tolerance_pt
        
        for i in range(len(fragments)):
            frag_a = fragments[i]
            for j in range(i + 1, len(fragments)):
                frag_b = fragments[j]
```
- **What it is:** _find_touching_edges setup
- **What it does:** Sets up a double loop comparing every fragment to every other fragment.
- **Why it is needed:** Needed to find corners and T-junctions.

**Lines 114-123:**
```python
is_touching = False
                ref_x, ref_y = 0.0, 0.0

                for seg_a in frag_a.segments:
                    for seg_b in frag_b.segments:
                        if segments_touch(seg_a, seg_b, touch_tol_pt):
                            is_touching = True
                            ref_x = (seg_a.x_start + seg_b.x_start) / 2.0
                            ref_y = (seg_a.y_start + seg_b.y_start) / 2.0
                            break
```
- **What it is:** Corner touch check
- **What it does:** Checks if endpoints are within touch_tolerance_pt and records the average of the START points as the coordinate.
- **Why it is needed:** Detects L-shaped corners. Note: storing the average of start points is a quirk copied from the original code.

**Lines 124-131:**
```python
if endpoint_touches_body((seg_a.x_start, seg_a.y_start), seg_b, touch_tol_pt):
                            is_touching = True
                            ref_x, ref_y = seg_a.x_start, seg_a.y_start
                            break
                        if endpoint_touches_body((seg_a.x_end, seg_a.y_end), seg_b, touch_tol_pt):
                            is_touching = True
                            ref_x, ref_y = seg_a.x_end, seg_a.y_end
                            break
```
- **What it is:** T-junction check (first 2 endpoints)
- **What it does:** Checks if segment A's endpoints hit the middle of segment B.
- **Why it is needed:** Detects T-junctions where lines tee off.

**Lines 132-142:**
```python
if endpoint_touches_body((seg_b.x_start, seg_b.y_start), seg_a, touch_tol_pt):
                            is_touching = True
                            ref_x, ref_y = seg_b.x_start, seg_b.y_start
                            break
                        if endpoint_touches_body((seg_b.x_end, seg_b.y_end), seg_a, touch_tol_pt):
                            is_touching = True
                            ref_x, ref_y = seg_b.x_end, seg_b.y_end
                            break
                    if is_touching:
                        break
```
- **What it is:** T-junction check (last 2 endpoints)
- **What it does:** Checks if segment B's endpoints hit the middle of segment A.
- **Why it is needed:** Completes the T-junction detection logic.

**Lines 143-149:**
```python
if is_touching:
                    edges.append({
                        'u': frag_a.id, 'v': frag_b.id,
                        'x': ref_x, 'y': ref_y, 'reason': 'touch_or_tjunc'
                    })
        return edges
```
- **What it is:** Touch edge generation
- **What it does:** Adds the touching edge to the candidates list.
- **Why it is needed:** Proposes merging the physically connected fragments.

**Lines 150-161:**
```python
def _find_collinear_edges(
        self, fragments: List[LineFragment], components: List[InlineComponent]
    ) -> List[Dict[str, Any]]:
        """Finds edges where fragments align linearly across a gap (e.g. unextracted inline valves)."""
        edges = []
        max_collinear_gap = self.config.collinear_max_gap_pt
        collinear_tol = self.config.collinear_axis_tolerance_pt

        major_equipment = [
            c for c in components 
            if c.type in self.config.major_equipment_types or c.group == "Vessel"
        ]
```
- **What it is:** _find_collinear_edges setup
- **What it does:** Filters components to only 'major equipment' like Tanks and Vessels.
- **Why it is needed:** We need this list to ensure straight lines don't bridge through giant obstacles.

**Lines 163-168:**
```python
def touches_major_equipment(px: float, py: float, tol: float = 8.0) -> bool:
            for eq in major_equipment:
                if is_point_near_bbox(px, py, eq.bbox, padding=tol):
                    return True
            return False
```
- **What it is:** touches_major_equipment helper
- **What it does:** Checks if a specific point falls inside a major equipment padded by 8 pt.
- **Why it is needed:** Used to block bridges that pass through tanks.

**Lines 169-180:**
```python
for i in range(len(fragments)):
            frag_a = fragments[i]
            for j in range(i + 1, len(fragments)):
                frag_b = fragments[j]

                is_collinear = False
                ref_x, ref_y = 0.0, 0.0

                for seg_a in frag_a.segments:
                    for seg_b in frag_b.segments:
                        is_gap, gx, gy = segments_collinear_gap(seg_a, seg_b, max_collinear_gap, collinear_tol)
                        if is_gap:
```
- **What it is:** Collinear gap check
- **What it does:** Loops through fragments and checks if they are collinear with a gap <= max_gap.
- **Why it is needed:** Detects straight-line jumps (like dashed lines).

**Lines 181-186:**
```python
if abs(seg_a.y_start - seg_a.y_end) <= collinear_tol:
                                xa = max(seg_a.x_start, seg_a.x_end) if min(seg_a.x_start, seg_a.x_end) < min(seg_b.x_start, seg_b.x_end) else min(seg_a.x_start, seg_a.x_end)
                                xb = min(seg_b.x_start, seg_b.x_end) if min(seg_a.x_start, seg_a.x_end) < min(seg_b.x_start, seg_b.x_end) else max(seg_b.x_start, seg_b.x_end)
                                ya, yb = (seg_a.y_start + seg_a.y_end) / 2.0, (seg_b.y_start + seg_b.y_end) / 2.0
                                if touches_major_equipment(xa, ya) or touches_major_equipment(xb, yb):
                                    continue
```
- **What it is:** Horizontal gap obstacle test
- **What it does:** Calculates the gap's boundary points (xa, xb) and tests BOTH for major equipment intersection.
- **Why it is needed:** Blocks horizontal bridges if either side of the gap touches a tank.

**Lines 187-193:**
```python
else:
                                ya = max(seg_a.y_start, seg_a.y_end) if min(seg_a.y_start, seg_a.y_end) < min(seg_b.y_start, seg_b.y_end) else min(seg_a.y_start, seg_a.y_end)
                                yb = min(seg_b.y_start, seg_b.y_end) if min(seg_a.y_start, seg_a.y_end) < min(seg_b.y_start, seg_b.y_end) else max(seg_b.y_start, seg_b.y_end)
                                xa, xb = (seg_a.x_start + seg_a.x_end) / 2.0, (seg_b.x_start + seg_b.x_end) / 2.0
                                if touches_major_equipment(xa, ya) or touches_major_equipment(xb, yb):
                                    continue
```
- **What it is:** Vertical gap obstacle test
- **What it does:** Calculates the gap's boundary points (ya, yb) and tests BOTH for major equipment intersection.
- **Why it is needed:** Blocks vertical bridges if either side of the gap touches a tank.

**Lines 194-206:**
```python
is_collinear = True
                            ref_x, ref_y = gx, gy
                            break
                    if is_collinear:
                        break

                if is_collinear:
                    edges.append({
                        'u': frag_a.id, 'v': frag_b.id,
                        'x': ref_x, 'y': ref_y, 'reason': 'collinear_gap'
                    })
        return edges
```
- **What it is:** Collinear edge generation
- **What it does:** Adds the collinear edge to candidates.
- **Why it is needed:** Proposes merging the straight-line jump.

**Lines 207-213:**
```python
def _sort_candidate_edges(self, edges: List[Dict[str, Any]], fragments: List[LineFragment]) -> None:
        """Prioritizes edges to evaluate based on tag matching status (Rule 6)."""
        frag_tags = {f.id: f.matched_tag for f in fragments}

        def edge_priority(edge):
            u, v = edge['u'], edge['v']
            t1, t2 = frag_tags[u], frag_tags[v]
```
- **What it is:** _sort_candidate_edges setup
- **What it does:** Extracts tags for priority scoring.
- **Why it is needed:** We must sort edges to guarantee safe merges happen first.

**Lines 214-221:**
```python
if t1 and t2:
                tag_prio = 0 if t1 == t2 else 4
            elif not t1 and not t2:
                tag_prio = 2
            else:
                tag_prio = 1
            type_prio = 0 if edge['reason'] in ('touch_or_tjunc', 'inline_component') else 1
            return (tag_prio, type_prio)
```
- **What it is:** Priority scoring
- **What it does:** Assigns tag_prio (0=match, 1=one tag, 2=no tag, 4=conflict) and type_prio (0=touch, 1=gap).
- **Why it is needed:** Conflicting edges go last so that safe matching tags merge first.

**Lines 223-224:**
```python
edges.sort(key=edge_priority)
```
- **What it is:** Priority sorting
- **What it does:** Sorts the edges list using Python's stable sort.
- **Why it is needed:** Equal priority edges keep the order they were added (inline, touch, collinear). This order guarantees reproducibility.

**Lines 225-238:**
```python
def _assemble_super_pipelines(
        self, fragments: List[LineFragment], components: List[InlineComponent], dsu: DisjointSetUnion
    ) -> List[SuperPipeline]:
        """Groups fragments by DSU roots and builds SuperPipeline objects."""
        frag_map = {f.id: f for f in fragments}
        frag_tags = {f.id: f.matched_tag for f in fragments}
        clusters: Dict[int, List[int]] = {}

        for f in fragments:
            root = dsu.find(f.id)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(f.id)
```
- **What it is:** _assemble_super_pipelines loop
- **What it does:** Loops over fragments and groups them by their DSU root.
- **Why it is needed:** Collects all fragments that successfully merged into discrete clusters.

**Lines 239-259:**
```python
super_pipelines = []
        for i, (root, fids) in enumerate(clusters.items()):
            sp_id = f"SP_{i:03d}"
            sp_tag = dsu.dsu_tag[root]

            if len(fids) == 1:
                conf = "Single Group"
            else:
                independent_tags = [frag_tags[fid] for fid in fids if frag_tags[fid] is not None]
                if len(set(independent_tags)) == 1 and len(independent_tags) > 1:
                    conf = "Confirmed"
                else:
                    conf = "Unverified"

            sp = SuperPipeline(
                id=sp_id,
                fragments=fids,
                tag=sp_tag,
                confidence=conf
            )
```
- **What it is:** SuperPipeline creation
- **What it does:** Calculates confidence based on tags and creates the SuperPipeline object.
- **Why it is needed:** Instantiates the final data model.

**Lines 260-266:**
```python
inline_matches = self.component_matcher.find_inline_components(
                pipeline=sp,
                fragments_map=frag_map,
                components=components,
                threshold_px=self.config.on_path_threshold_pt
            )
```
- **What it is:** Inline component matching
- **What it does:** Calls find_inline_components using threshold_px.
- **Why it is needed:** Finds valves that sit exactly on the pipeline's path. Quirk: passes points to a parameter named 'px'.

**Lines 267-274:**
```python
for sym_id, src in inline_matches:
                sp.inline_components.append(sym_id)
                sp.inline_sources[sym_id] = src

            super_pipelines.append(sp)
        
        return super_pipelines
```
- **What it is:** Inline component assignment
- **What it does:** Adds found symbols to the pipeline object.
- **Why it is needed:** Attaches the valves to the pipeline.

**Lines 275-283:**
```python
def _build_branch_connections(
        self, rejected_edges: List[Dict[str, Any]], super_pipelines: List[SuperPipeline], fragments: List[LineFragment]
    ) -> List[Dict[str, Any]]:
        """Formats the rejected edges into branch connections between distinct Super Pipelines."""
        branch_connections = []
        seen_branches = set()
        frag_to_sp = {fid: sp.id for sp in super_pipelines for fid in sp.fragments}
        frag_tags = {f.id: f.matched_tag for f in fragments}
```
- **What it is:** _build_branch_connections setup
- **What it does:** Maps fragments to their final SuperPipeline ID.
- **Why it is needed:** Needed to trace rejected edges back to their parent pipelines.

**Lines 284-301:**
```python
for edge in rejected_edges:
            u, v = edge['u'], edge['v']
            sp_u = frag_to_sp.get(u)
            sp_v = frag_to_sp.get(v)
            if sp_u and sp_v and sp_u != sp_v:
                branch_key = tuple(sorted([sp_u, sp_v]))
                if branch_key not in seen_branches:
                    seen_branches.add(branch_key)
                    branch_connections.append({
                        'SP_1': sp_u,
                        'SP_2': sp_v,
                        'Tag_1': frag_tags.get(u),
                        'Tag_2': frag_tags.get(v),
                        'Refusal_X': round(edge['x'], 1),
                        'Refusal_Y': round(edge['y'], 1)
                    })
        return branch_connections
```
- **What it is:** Branch connection formatting
- **What it does:** For each rejected edge, creates a branch connection record if the pipelines differ.
- **Why it is needed:** Turns a refused merge into a physical T-junction specification break.

**Lines 302-319:**
```python
def build_super_pipelines(
        self, fragments: List[LineFragment], components: List[InlineComponent]
    ) -> Tuple[List[SuperPipeline], List[Dict[str, Any]]]:
        """
        Orchestrator for the merge process. Generates candidate edges, applies rules,
        merges via DSU, and formats output.
        """
        # 1. Gather all candidate edges
        inline_comps = [
            c for c in components 
            if c.type not in self.config.major_equipment_types and c.group != "Vessel"
        ]
        
        candidate_edges = []
        candidate_edges.extend(self._find_inline_component_edges(fragments, inline_comps))
        candidate_edges.extend(self._find_touching_edges(fragments))
        candidate_edges.extend(self._find_collinear_edges(fragments, components))
```
- **What it is:** build_super_pipelines edges
- **What it does:** Filters inline comps and calls the three edge finders.
- **Why it is needed:** Step 1: Gather all candidate merges.

**Lines 320-322:**
```python
# 2. Sort candidate edges by priority
        self._sort_candidate_edges(candidate_edges, fragments)
```
- **What it is:** build_super_pipelines sort
- **What it does:** Sorts the candidate edges.
- **Why it is needed:** Step 2: Prioritize safe tag matches.

**Lines 323-330:**
```python
# 3. Apply DSU matching
        dsu = DisjointSetUnion(fragments)
        rejected_edges = []

        for edge in candidate_edges:
            if not dsu.union(edge['u'], edge['v']):
                rejected_edges.append(edge)
```
- **What it is:** build_super_pipelines union
- **What it does:** Applies DSU.union and saves rejected edges.
- **Why it is needed:** Step 3: Execute merges while respecting tag conflicts.

**Lines 331-338:**
```python
# 4. Form super pipelines from DSU roots
        super_pipelines = self._assemble_super_pipelines(fragments, components, dsu)

        # 5. Extract branch connections
        branch_connections = self._build_branch_connections(rejected_edges, super_pipelines, fragments)

        return super_pipelines, branch_connections
```
- **What it is:** build_super_pipelines finish
- **What it does:** Assembles pipelines and branches.
- **Why it is needed:** Steps 4 & 5: Format the final output.

**Lines 339-349:**
```python
def export_super_pipelines(
        self, super_pipelines: List[SuperPipeline], branches: List[Dict[str, Any]], 
        fragments: List[LineFragment], out_dir: str, stem: str, components: List[InlineComponent]
    ):
        """Exports the generated super pipelines."""
        from line_association import LineAssociationExporter
        exporter = LineAssociationExporter()
        exporter.export_super_pipelines(
            super_pipelines, branches, fragments, out_dir, stem, components=components
        )
```
- **What it is:** export_super_pipelines
- **What it does:** Wraps the external LineAssociationExporter.
- **Why it is needed:** Writes the final data to disk.

### `run_line_processor.py`
**Lines 1-9:**
```python
import os
import sys
import argparse

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
```
- **What it is:** Imports and Path setup
- **What it does:** Imports sys and adds the root directory to sys.path.
- **Why it is needed:** Ensures the script can run and import project modules from any working directory.

**Lines 10-15:**
```python
from line_association.adapters import CoordinateTransformer, load_line_tags
from line_association import LineTagAssociator, LineAssociationConfig, LineAssociationExporter
from line_processing.line_processor import LineProcessor
from line_processing.line_processor_config import LineProcessorConfig, NEEDED_ROWS_02_UNMARKED
from line_association import summarize_line_association, format_line_stats
```
- **What it is:** Project imports
- **What it does:** Imports all the adapters, configs, processors, and exporters.
- **Why it is needed:** Pulls in the required classes for the pipeline.

**Lines 16-24:**
```python
def run_association(stem: str = "02.Unmarked", out_dir: str = "results/LE_csv"):
    pdf_path = os.path.join(_ROOT, "data", "raw", f"{stem}.pdf")
    lines_csv = os.path.join(_ROOT, "LE", "Results", "LE_csv", f"{stem}_line_coordinates_grouped.csv")
    tags_csv = os.path.join(_ROOT, "TE", "Results", "TE_csv", "process_tags.csv")
    comp_csv = os.path.join(_ROOT, "results", "CE_csv", f"{stem}_component_tag_associated.csv")

    abs_out_dir = os.path.join(_ROOT, out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)
```
- **What it is:** run_association setup
- **What it does:** Defines the main function and sets up file paths.
- **Why it is needed:** Creates absolute paths for input CSVs and ensures the output directory exists.

**Lines 25-31:**
```python
print(f"\n{'='*60}")
    print(f"  RUNNING LINE PROCESSOR ENGINE: {stem}")
    print(f"{'='*60}")

    print(f"[1/5] Building coordinate transformer from {pdf_path}...")
    transformer = CoordinateTransformer.from_pdf(pdf_path, dpi=300.0)
```
- **What it is:** CoordinateTransformer setup
- **What it does:** Instantiates a CoordinateTransformer directly from the PDF.
- **Why it is needed:** Reads the PDF dimensions to allow accurate pixel-to-point translation.

**Lines 32-42:**
```python
# 1. Initialize LineProcessor
    print("[2/5] Initializing LineProcessor and loading inputs...")
    config = LineProcessorConfig()
    processor = LineProcessor(config)

    needed = NEEDED_ROWS_02_UNMARKED if stem == "02.Unmarked" else None
    fragments = processor.load_and_filter_fragments(lines_csv, needed_rows=needed, transformer=transformer)
    tags = load_line_tags(tags_csv, file_name=stem)
    components = processor.load_components(comp_csv)
    print(f"      Loaded {len(fragments)} line fragments, {len(tags)} line tags, {len(components)} components.")
```
- **What it is:** Initialize and load
- **What it does:** Creates the config, initializes LineProcessor, and loads fragments, tags, and components.
- **Why it is needed:** Prepares all the raw data for processing. Passes `NEEDED_ROWS` if stem is 02.Unmarked.

**Lines 43-49:**
```python
# 2. (EXTERNAL) Line Tag Association
    print("[3/5] Running Part 1: Line-Tag Bipartite Association...")
    tag_config = LineAssociationConfig()
    associator = LineTagAssociator(config=tag_config)
    assoc_result = associator.associate(fragments, tags)
    print(f"      Matches: {len(assoc_result.matches)} (Pass 1: {assoc_result.pass1_count}, Pass 2: {assoc_result.pass2_count})")
```
- **What it is:** Tag Association
- **What it does:** Creates a separate LineAssociationConfig and runs the associator.
- **Why it is needed:** Part 1: Tags MUST be assigned before merging so the DSU can refuse clashing lines. Note: two separate configs are used for separation of concerns.

**Lines 50-54:**
```python
# 3. Merging (using LineProcessor)
    print("[4/5] Running Part 2 & Super-Pipeline Merging via LineProcessor...")
    super_pipelines, branches = processor.build_super_pipelines(fragments, components)
    print(f"      Generated {len(super_pipelines)} continuous Super Pipelines and {len(branches)} branch connections.")
```
- **What it is:** Merging via LineProcessor
- **What it does:** Calls processor.build_super_pipelines.
- **Why it is needed:** Part 2: The actual geometric grouping and pipeline generation happens here.

**Lines 55-64:**
```python
# 4. Exporting
    print(f"[5/5] Exporting result CSVs to {abs_out_dir}...")
    exporter = LineAssociationExporter()
    exporter.export_association_results(assoc_result, fragments, abs_out_dir, stem)
    processor.export_super_pipelines(super_pipelines, branches, fragments, abs_out_dir, stem, components=components)

    stats = summarize_line_association(assoc_result, len(fragments), super_pipelines, branches)
    print("\n" + format_line_stats(stats))
    print(f"\nOutputs written to: {abs_out_dir}")
```
- **What it is:** Export and Stats
- **What it does:** Calls both exporters and prints statistics.
- **Why it is needed:** Saves the 5 final CSVs and logs a summary to the console.

**Lines 65-75:**
```python
def main():
    parser = argparse.ArgumentParser(description="Run Line Processor Engine")
    parser.add_argument("--stem", default="02.Unmarked", help="Drawing stem (e.g. 02.Unmarked, 01.Marked)")
    parser.add_argument("--out-dir", default="results/LE_csv", help="Output directory for CSV files")
    args = parser.parse_args()

    run_association(stem=args.stem, out_dir=args.out_dir)

if __name__ == '__main__':
    main()
```
- **What it is:** Main block
- **What it does:** Sets up argparse and executes run_association.
- **Why it is needed:** Allows running the script from the command line with custom stems and output directories.

## 4. Helper Functions and Models

### `line_association/geometry.py`
- **`point_to_segment_distance`**: Computes the shortest distance from a point to a line segment (using projection). *Example: Dist from (0,5) to segment (0,0)-(10,0) is 5.0.*
- **`is_point_near_bbox`**: Checks if a point lies inside a rectangle that has been expanded by padding. *Example: Point (12, 12) is inside Box (0,0,10,10) if padding is 5.*
- **`segments_touch`**: Checks if any of the 4 combinations of segment endpoints are within tolerance. *Example: (0,0)-(10,0) touches (10.1,0)-(20,0) with a tolerance of 1.0.*
- **`endpoint_touches_body`**: Checks if an endpoint lands squarely in the middle interior of another segment (T-junction).
- **`segments_collinear_gap`**: Checks if two segments lie on the exact same axis (within 3 pt tolerance) but have an empty gap > 0 and <= max_gap between them. *Example: (0,0)-(10,0) and (20,0)-(30,0) have a collinear gap of 10.*

### `line_association/adapters.py`
- **`CoordinateTransformer.from_pdf`**: Reads page width/height and rotation from the physical PDF to convert pixels to points. (Rotation and DPI matter because the extracted images are scaled and flipped).
- **`load_line_fragments`**: Reads `line_coordinates_grouped.csv`, drops rows not in `needed_rows`, converts pixels to points, and packs rows sharing a `Group_ID` into `LineFragment` objects.
- **`load_line_tags`**: Reads `process_tags.csv` and filters for 'Line Tag' types.
- **`load_components`**: Reads `component_tag_associated.csv` and handles missing `matched_tag` strings.

### `line_association/associator.py`
- **`LineTagAssociator.associate`**: Orchestrates tag assignment. It maps valid tags to the closest fragments using a bipartite matcher. Pass 1 looks within 40 points; Pass 2 looks within 100 points for leftovers. It sets `fragment.matched_tag` which is critical for the DSU merging.

### `line_association/rules.py`
- **`InlineComponentMatcher.find_inline_components`**: Scans a super pipeline to find components resting directly on its path, ensuring it doesn't accidentally claim the component that anchors the pipeline's own tag (Rule 8).

### `line_association/exporter.py`
- **`LineAssociationExporter.export_association_results`**: Writes `_line_tag_associated.csv` (fragment to tag mappings), `unmatched_line_fragments.csv`, and `orphan_line_tags.csv`.
- **`export_super_pipelines`**: Writes `_SUPER_pipelines.csv` (the final grouped pipe network) and `_branch_connections.csv` (the T-junction locations).

### `line_association/stats.py`
- **`summarize_line_association`** & **`format_line_stats`**: Generates a dictionary of counts (total pipelines, matches, orphans) and formats it into a neat console printout.

### `line_association/models.py` (Dataclasses)
- **`Point`**: Holds an `x` and `y` float.
- **`BBox`**: Holds `x0, y0, x1, y1` defining a rectangle. Has properties for `center`, `width`, `height`.
- **`Segment`**: Holds start/end coordinates, orientation ('H', 'V', 'U'), and segment index.
- **`LineFragment`**: Represents a raw pipe section. Holds a list of `Segment`s, an ID, and the `matched_tag` string populated by the associator.
- **`LineTag`**: Represents extracted text. Holds the `text` string and its `BBox`.
- **`InlineComponent`**: Represents a valve or fitting. Holds `symbol_id`, `type`, `bbox`, and its own `matched_tag`.
- **`LineMatch`**: Represents a successful assignment between a `fragment_id` and a `tag_text`.
- **`SuperPipeline`**: Represents a final merged pipe. Holds a list of fragment IDs, the master `tag`, and a list of `inline_components`.

## 5. How It All Connects

### Call Graph
```text
run_line_processor.py
  ├── CoordinateTransformer.from_pdf
  ├── LineProcessor.load_and_filter_fragments (calls adapters.load_line_fragments)
  ├── adapters.load_line_tags
  ├── LineProcessor.load_components (calls adapters.load_components)
  ├── LineTagAssociator.associate
  │    ├── DistanceMatrixBuilder.build_matrix
  │    └── BipartiteMatcher.match_pass
  ├── LineProcessor.build_super_pipelines
  │    ├── _find_inline_component_edges
  │    ├── _find_touching_edges
  │    ├── _find_collinear_edges
  │    ├── _sort_candidate_edges
  │    ├── DisjointSetUnion.union
  │    ├── _assemble_super_pipelines
  │    │    └── InlineComponentMatcher.find_inline_components
  │    └── _build_branch_connections
  ├── LineAssociationExporter.export_association_results
  └── LineProcessor.export_super_pipelines
       └── LineAssociationExporter.export_super_pipelines
```

### Data Transformations Step-by-Step
| Step | Input | Action | Output |
|---|---|---|---|
| 1 | `line_coordinates_grouped.csv` | `load_line_fragments` | `List[LineFragment]` (raw, no tags) |
| 2 | `List[LineFragment]`, `List[LineTag]` | `LineTagAssociator.associate` | `List[LineFragment]` (mutated: `matched_tag` populated) |
| 3 | `List[LineFragment]`, `List[InlineComponent]` | Edge Finders | `List[Dict]` (Edges with `u`, `v`, `reason`) |
| 4 | `List[Dict]` (Edges) | `DisjointSetUnion.union` | Grouped fragment IDs via common DSU roots |
| 5 | DSU roots, `List[LineFragment]` | `_assemble_super_pipelines` | `List[SuperPipeline]` Objects |
| 6 | `List[SuperPipeline]` | `LineAssociationExporter.export` | `_SUPER_pipelines.csv` (Final Output) |

## 6. Questions from a Lead

**Q1: Why do we assign tags BEFORE merging pipelines?**
**A:** If we merged first, touching pipes with different fluids (e.g. Water and Gas) would merge into one massive pipe. Tagging first lets us refuse merges between conflicting tags.

**Q2: What is Disjoint Set Union (DSU) and why use it?**
**A:** It’s a fast algorithm for grouping connected items. Without it, we would have to write complex, slow, recursive loops to traverse the pipeline chains.

**Q3: What happens if `touch_tolerance_pt` is increased from 5 to 20?**
**A:** Lines that are drawn far apart but roughly nearby will incorrectly snap together, causing unrelated parallel pipes to merge.

**Q4: Why does `_find_collinear_edges` check for `major_equipment`?**
**A:** To prevent the algorithm from bridging a dashed line straight through the middle of a massive tank, which is physically impossible.

**Q5: Why is `needed_rows` hardcoded and what is a better alternative?**
**A:** It manually filters out drawing borders and tables. A better alternative is writing an algorithm to detect large outer bounding boxes or using semantic segmentation to ignore title blocks entirely.

**Q6: What does the `padding_pt` do for inline components?**
**A:** Component bounding boxes are often drawn tightly. Padding expands the box by 12 points so slightly misdrawn pipe endpoints still register as touching the valve.

**Q7: How are tie-breakers handled when multiple lines touch the same point?**
**A:** `_sort_candidate_edges` prioritizes merges where both fragments already share the identical tag, ensuring safe merges happen before risky ones. Because Python's sort is stable, edges with equal priority are resolved in the order they were proposed (inline edges, then touching edges, then collinear edges).

**Q8: What is a 'Branch Connection'?**
**A:** When the DSU refuses a merge between two touching lines (because their tags clash), it records that exact coordinate as a physical T-junction or spec-break branch between two distinct pipelines.

**Q9: Why are we translating pixels to PDF points?**
**A:** The downstream software expects coordinates mapping directly to the original vector PDF file, not our scaled image pixels.

**Q10: What is NOT tested by this code?**
**A:** Only `02.Unmarked` has been run end to end. `01.Marked` fails entirely because its CE csv is missing. The `needed_rows=None` path has never been tested. Finally, there is no ground-truth accuracy measure; we only check for equality against the original script.

**Q11: How was the output proven identical to the original?**
**A:** We ran both the old and new code, directed their CSV outputs to different folders, and compared them byte-for-byte. The stable sorting of candidate edges ensures identical merge ordering.

**Q12: Why split the code into a class, a config, and a runner?**
**A:** It decouples the hardcoded logic from the algorithmic logic, making it possible to write unit tests for `LineProcessor` by passing a mocked configuration, and keeping the orchestration separate from the logic.

## 7. Known Quirks (Copied from original code)
1. **Unused Import:** `import pandas as pd` in `line_processor.py` is an unused leftover.
2. **Misnamed Parameter:** `find_inline_components(threshold_px=self.config.on_path_threshold_pt)` passes a value in PDF *points* to a parameter named `threshold_px` (*pixels*). This is copied from the original and does not change results, but should be double-checked.
3. **Corner Reference Point:** In `_find_touching_edges`, for a corner touch, the reference point is calculated as the average of the two segments' START points, even if their end points are the ones touching. This only affects the coordinate saved for branch connections.
4. **Two Configs:** `run_line_processor.py` instantiates `LineAssociationConfig()` and `LineProcessorConfig()` separately. This maintains separation of concerns between Part 1 (Tag Association) and Part 2 (Line Processor).
