from typing import List, Dict, Tuple, Set, Optional, Any
import math
from .models import LineFragment, InlineComponent, SuperPipeline, BBox
from .geometry import segments_touch, endpoint_touches_body, is_point_near_bbox, segments_collinear_gap
from .config import LineAssociationConfig
from .rules import InlineComponentMatcher

class SuperPipelineMerger:
    """
    Stage 2: Merges line fragments into continuous Super Pipelines using
    priority-ordered Disjoint Set Union (DSU) with tag-conflict refusal.
    """
    def __init__(self, config: Optional[LineAssociationConfig] = None):
        self.config = config or LineAssociationConfig()
        self.component_matcher = InlineComponentMatcher(self.config)

    def merge(
        self,
        fragments: List[LineFragment],
        components: List[InlineComponent]
    ) -> Tuple[List[SuperPipeline], List[Dict[str, Any]]]:
        """
        Merges fragments into continuous SuperPipelines and returns:
          (super_pipelines, rejected_branches)
        """
        frag_map = {f.id: f for f in fragments}
        frag_tags = {f.id: f.matched_tag for f in fragments}
        all_fids = [f.id for f in fragments]

        padding_pt = self.config.padding_pt
        touch_tol_pt = self.config.touch_tolerance_pt

        # 1. Filter Inline Components (exclude major terminal equipment)
        inline_comps = [
            c for c in components 
            if c.type not in self.config.major_equipment_types and c.group != "Vessel"
        ]

        # 2. Map fragments to inline component boxes
        comp_to_fragments: Dict[int, Set[int]] = {i: set() for i in range(len(inline_comps))}
        for fid, frag in frag_map.items():
            for seg in frag.segments:
                for c_idx, comp in enumerate(inline_comps):
                    if is_point_near_bbox(seg.x_start, seg.y_start, comp.bbox, padding_pt) or \
                       is_point_near_bbox(seg.x_end, seg.y_end, comp.bbox, padding_pt):
                        comp_to_fragments[c_idx].add(fid)

        candidate_edges = []

        # Edge Type 1: Inline Component Bridging
        for c_idx, touching_fids in comp_to_fragments.items():
            fids_list = list(touching_fids)
            for j in range(len(fids_list)):
                for k in range(j + 1, len(fids_list)):
                    cx, cy = inline_comps[c_idx].center.x, inline_comps[c_idx].center.y
                    candidate_edges.append({
                        'u': fids_list[j], 'v': fids_list[k],
                        'x': cx, 'y': cy, 'reason': 'inline_component',
                        'symbol_id': inline_comps[c_idx].symbol_id
                    })

        # Edge Type 2 & 3: Direct Endpoint Touching & T-Junctions
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
                        # Check T-junctions
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
                    candidate_edges.append({
                        'u': frag_a.id, 'v': frag_b.id,
                        'x': ref_x, 'y': ref_y, 'reason': 'touch_or_tjunc'
                    })

        # Collect major equipment bboxes to prevent collinear bridging through vessel/exchanger bodies
        major_equipment = [
            c for c in components 
            if c.type in self.config.major_equipment_types or c.group == "Vessel"
        ]

        def touches_major_equipment(px: float, py: float, tol: float = 8.0) -> bool:
            for eq in major_equipment:
                if is_point_near_bbox(px, py, eq.bbox, padding=tol):
                    return True
            return False

        # Edge Type 4: Collinear Gap Bridging (bridges across unextracted valves and dashed lines)
        max_collinear_gap = self.config.collinear_max_gap_pt
        collinear_tol = self.config.collinear_axis_tolerance_pt
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
                            # Verify that facing endpoints do not touch major equipment (e.g. nozzle terminating at vessel)
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
                    candidate_edges.append({
                        'u': frag_a.id, 'v': frag_b.id,
                        'x': ref_x, 'y': ref_y, 'reason': 'collinear_gap'
                    })

        # --- DSU Data Structure with Tag Conflict Refusal (Rule 5 & 6) ---
        parent = {fid: fid for fid in all_fids}
        dsu_tag = {fid: frag_tags[fid] for fid in all_fids}

        def find(u):
            if parent[u] == u:
                return u
            parent[u] = find(parent[u])
            return parent[u]

        def union(u, v) -> bool:
            root_u = find(u)
            root_v = find(v)
            if root_u == root_v:
                return True

            tag_u = dsu_tag[root_u]
            tag_v = dsu_tag[root_v]

            # Rule 5: Conflict Refusal
            if tag_u and tag_v and tag_u != tag_v:
                return False

            parent[root_u] = root_v
            # Propagate tag if one had it
            if not tag_v and tag_u:
                dsu_tag[root_v] = tag_u
            return True

        # Priority sort candidate edges (Rule 6)
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

        candidate_edges.sort(key=edge_priority)

        accepted_edges = []
        rejected_edges = []

        for edge in candidate_edges:
            if not union(edge['u'], edge['v']):
                rejected_edges.append(edge)
            else:
                accepted_edges.append(edge)

        # Build clusters
        clusters: Dict[int, List[int]] = {}
        for fid in all_fids:
            root = find(fid)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(fid)

        # Build SuperPipeline objects
        super_pipelines = []
        on_path_threshold_pt = self.config.on_path_threshold_pt

        for i, (root, fids) in enumerate(clusters.items()):
            sp_id = f"SP_{i:03d}"
            sp_tag = dsu_tag[root]

            # Determine confidence (Rule 7)
            if len(fids) == 1:
                conf = "Single Group"
            else:
                # Check if multiple fragments independently matched the same tag
                independent_tags = [frag_tags[f] for f in fids if frag_tags[f] is not None]
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

            # Part 2: Inline component association (Rules 7 & 8)
            inline_matches = self.component_matcher.find_inline_components(
                pipeline=sp,
                fragments_map=frag_map,
                components=components,
                threshold_px=on_path_threshold_pt
            )

            for sym_id, src in inline_matches:
                sp.inline_components.append(sym_id)
                sp.inline_sources[sym_id] = src

            super_pipelines.append(sp)

        # Format rejected branch connections
        branch_connections = []
        seen_branches = set()
        frag_to_sp = {fid: sp.id for sp in super_pipelines for fid in sp.fragments}

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

        return super_pipelines, branch_connections
