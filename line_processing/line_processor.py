import pandas as pd
from typing import List, Dict, Tuple, Set, Optional, Any

# Reuse existing helpers
from .models import LineFragment, InlineComponent, SuperPipeline
from .geometry import (
    segments_touch, endpoint_touches_body, is_point_near_bbox, segments_collinear_gap
)
from .adapters import CoordinateTransformer, load_line_fragments, load_components
from .matcher import InlineComponentMatcher
from .exporter import export_super_pipelines as _export_super_pipelines
from .line_processor_config import LineProcessorConfig


class DisjointSetUnion:
    """Helper class for managing Disjoint Set Union operations with tag conflict refusal."""
    def __init__(self, fragments: List[LineFragment]):
        self.parent = {f.id: f.id for f in fragments}
        self.dsu_tag = {f.id: f.matched_tag for f in fragments}

    def find(self, u: int) -> int:
        if self.parent[u] == u:
            return u
        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u: int, v: int) -> bool:
        """
        Unions two fragments if their tags do not conflict.
        Returns True if successful, False if rejected due to conflict.
        """
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u == root_v:
            return True

        tag_u = self.dsu_tag[root_u]
        tag_v = self.dsu_tag[root_v]

        # Rule 5: Conflict Refusal
        if tag_u and tag_v and tag_u != tag_v:
            return False

        self.parent[root_u] = root_v
        if not tag_v and tag_u:
            self.dsu_tag[root_v] = tag_u
        return True


class LineProcessor:
    """
    Encapsulates line fragment loading, filtering, and SuperPipeline merging.
    Strictly isolated from tag association logic.
    """
    def __init__(self, config: Optional[LineProcessorConfig] = None):
        self.config = config or LineProcessorConfig()
        
        self.component_matcher = InlineComponentMatcher(self.config)
    
    def load_and_filter_fragments(
        self, csv_path: str, needed_rows: Optional[Set[int]] = None, transformer: Optional[CoordinateTransformer] = None
    ) -> List[LineFragment]:
        """Loads line fragments and applies the needed_rows filter immediately."""
        return load_line_fragments(csv_path, needed_rows=needed_rows, transformer=transformer)

    def load_components(self, csv_path: str) -> List[InlineComponent]:
        """Loads components."""
        return load_components(csv_path)

    def _find_inline_component_edges(
        self, fragments: List[LineFragment], inline_comps: List[InlineComponent]
    ) -> List[Dict[str, Any]]:
        """Finds edges between fragments that both touch the same inline component."""
        edges = []
        frag_map = {f.id: f for f in fragments}
        padding_pt = self.config.padding_pt

        comp_to_fragments: Dict[int, Set[int]] = {i: set() for i in range(len(inline_comps))}
        for fid, frag in frag_map.items():
            for seg in frag.segments:
                for c_idx, comp in enumerate(inline_comps):
                    if is_point_near_bbox(seg.x_start, seg.y_start, comp.bbox, padding_pt) or \
                       is_point_near_bbox(seg.x_end, seg.y_end, comp.bbox, padding_pt):
                        comp_to_fragments[c_idx].add(fid)

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

    def _find_touching_edges(self, fragments: List[LineFragment]) -> List[Dict[str, Any]]:
        """Finds edges where fragments physically touch at corners or T-junctions."""
        edges = []
        touch_tol_pt = self.config.touch_tolerance_pt
        
        for i in range(len(fragments)):
            frag_a = fragments[i]
            for j in range(i + 1, len(fragments)):
                frag_b = fragments[j]

                is_touching = False
                ref_x, ref_y = 0.0, 0.0

                for seg_a in frag_a.segments:
                    for seg_b in frag_b.segments:
                        if segments_touch(seg_a, seg_b, touch_tol_pt):
                            is_touching = True
                            ref_x = (seg_a.x_start + seg_b.x_start) / 2.0
                            ref_y = (seg_a.y_start + seg_b.y_start) / 2.0
                            break
                        if endpoint_touches_body((seg_a.x_start, seg_a.y_start), seg_b, touch_tol_pt):
                            is_touching = True
                            ref_x, ref_y = seg_a.x_start, seg_a.y_start
                            break
                        if endpoint_touches_body((seg_a.x_end, seg_a.y_end), seg_b, touch_tol_pt):
                            is_touching = True
                            ref_x, ref_y = seg_a.x_end, seg_a.y_end
                            break
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

                if is_touching:
                    edges.append({
                        'u': frag_a.id, 'v': frag_b.id,
                        'x': ref_x, 'y': ref_y, 'reason': 'touch_or_tjunc'
                    })
        return edges

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

        def touches_major_equipment(px: float, py: float, tol: float = 8.0) -> bool:
            for eq in major_equipment:
                if is_point_near_bbox(px, py, eq.bbox, padding=tol):
                    return True
            return False

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
                            if abs(seg_a.y_start - seg_a.y_end) <= collinear_tol:
                                xa = max(seg_a.x_start, seg_a.x_end) if min(seg_a.x_start, seg_a.x_end) < min(seg_b.x_start, seg_b.x_end) else min(seg_a.x_start, seg_a.x_end)
                                xb = min(seg_b.x_start, seg_b.x_end) if min(seg_a.x_start, seg_a.x_end) < min(seg_b.x_start, seg_b.x_end) else max(seg_b.x_start, seg_b.x_end)
                                ya, yb = (seg_a.y_start + seg_a.y_end) / 2.0, (seg_b.y_start + seg_b.y_end) / 2.0
                                if touches_major_equipment(xa, ya) or touches_major_equipment(xb, yb):
                                    continue
                            else:
                                ya = max(seg_a.y_start, seg_a.y_end) if min(seg_a.y_start, seg_a.y_end) < min(seg_b.y_start, seg_b.y_end) else min(seg_a.y_start, seg_a.y_end)
                                yb = min(seg_b.y_start, seg_b.y_end) if min(seg_a.y_start, seg_a.y_end) < min(seg_b.y_start, seg_b.y_end) else max(seg_b.y_start, seg_b.y_end)
                                xa, xb = (seg_a.x_start + seg_a.x_end) / 2.0, (seg_b.x_start + seg_b.x_end) / 2.0
                                if touches_major_equipment(xa, ya) or touches_major_equipment(xb, yb):
                                    continue

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

    def _sort_candidate_edges(self, edges: List[Dict[str, Any]], fragments: List[LineFragment]) -> None:
        """Prioritizes edges to evaluate based on tag matching status (Rule 6)."""
        frag_tags = {f.id: f.matched_tag for f in fragments}

        def edge_priority(edge):
            u, v = edge['u'], edge['v']
            t1, t2 = frag_tags[u], frag_tags[v]
            if t1 and t2:
                tag_prio = 0 if t1 == t2 else 4
            elif not t1 and not t2:
                tag_prio = 2
            else:
                tag_prio = 1
            type_prio = 0 if edge['reason'] in ('touch_or_tjunc', 'inline_component') else 1
            return (tag_prio, type_prio)

        edges.sort(key=edge_priority)

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

            inline_matches = self.component_matcher.find_inline_components(
                pipeline=sp,
                fragments_map=frag_map,
                components=components,
                threshold_px=self.config.on_path_threshold_pt
            )

            for sym_id, src in inline_matches:
                sp.inline_components.append(sym_id)
                sp.inline_sources[sym_id] = src

            super_pipelines.append(sp)
        
        return super_pipelines

    def _build_branch_connections(
        self, rejected_edges: List[Dict[str, Any]], super_pipelines: List[SuperPipeline], fragments: List[LineFragment]
    ) -> List[Dict[str, Any]]:
        """Formats the rejected edges into branch connections between distinct Super Pipelines."""
        branch_connections = []
        seen_branches = set()
        frag_to_sp = {fid: sp.id for sp in super_pipelines for fid in sp.fragments}
        frag_tags = {f.id: f.matched_tag for f in fragments}

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

        # 2. Sort candidate edges by priority
        self._sort_candidate_edges(candidate_edges, fragments)

        # 3. Apply DSU matching
        dsu = DisjointSetUnion(fragments)
        rejected_edges = []

        for edge in candidate_edges:
            if not dsu.union(edge['u'], edge['v']):
                rejected_edges.append(edge)

        # 4. Form super pipelines from DSU roots
        super_pipelines = self._assemble_super_pipelines(fragments, components, dsu)

        # 5. Extract branch connections
        branch_connections = self._build_branch_connections(rejected_edges, super_pipelines, fragments)

        return super_pipelines, branch_connections

    def export_super_pipelines(
        self, super_pipelines: List[SuperPipeline], branches: List[Dict[str, Any]], 
        fragments: List[LineFragment], out_dir: str, stem: str, components: List[InlineComponent]
    ):
        """Exports the generated super pipelines."""
        _export_super_pipelines(
            super_pipelines, branches, fragments, out_dir, stem, components=components
        )
