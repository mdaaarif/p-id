import os
import pandas as pd
from typing import List, Dict, Any, Optional
from .models import LineFragment, LineTag, LineMatch, SuperPipeline
from .associator import LineAssociationResult

class LineAssociationExporter:
    """Exports line association and super pipeline outputs to CSV files."""

    @staticmethod
    def export_association_results(
        result: LineAssociationResult,
        fragments: List[LineFragment],
        out_dir: str,
        stem: str
    ):
        os.makedirs(out_dir, exist_ok=True)

        # 1. Main associated fragments CSV
        records = []
        for frag in fragments:
            records.append({
                "Group_ID": frag.id,
                "Line_Type": frag.line_type,
                "Matched_Line_Tag": frag.matched_tag or "",
                "Match_Distance_pt": frag.match_distance if frag.match_distance is not None else "",
                "Match_Pass": frag.match_pass or "",
                "Confidence": frag.confidence or "Unmatched"
            })
        df_assoc = pd.DataFrame(records)
        path_assoc = os.path.join(out_dir, f"{stem}_line_tag_associated.csv")
        df_assoc.to_csv(path_assoc, index=False)

        # 2. Unmatched fragments CSV (Rule 6)
        unmatched_records = []
        for frag in result.unmatched_fragments:
            unmatched_records.append({
                "Group_ID": frag.id,
                "Line_Type": frag.line_type,
                "Num_Segments": len(frag.segments)
            })
        df_unmatched = pd.DataFrame(unmatched_records)
        path_unmatched = os.path.join(out_dir, "unmatched_line_fragments.csv")
        df_unmatched.to_csv(path_unmatched, index=False)

        # 3. Orphan Line Tags CSV (Rule 6)
        orphan_records = []
        for tag in result.orphan_tags:
            orphan_records.append({
                "Tag_ID": tag.id,
                "Text": tag.text,
                "Center_X": round(tag.center.x, 1),
                "Center_Y": round(tag.center.y, 1)
            })
        df_orphan = pd.DataFrame(orphan_records)
        path_orphan = os.path.join(out_dir, "orphan_line_tags.csv")
        df_orphan.to_csv(path_orphan, index=False)

    @staticmethod
    def export_super_pipelines(
        super_pipelines: List[SuperPipeline],
        branches: List[Dict[str, Any]],
        fragments: List[LineFragment],
        out_dir: str,
        stem: str,
        components: Optional[List[Any]] = None
    ):
        os.makedirs(out_dir, exist_ok=True)

        comp_map = {c.symbol_id: c for c in components} if components else {}

        # 1. Segment-level super pipeline mapping CSV
        frag_to_sp = {}
        for sp in super_pipelines:
            for fid in sp.fragments:
                frag_to_sp[fid] = sp

        sp_records = []
        for frag in fragments:
            sp = frag_to_sp.get(frag.id)
            sp_id = sp.id if sp else f"SP_orphan_{frag.id}"
            sp_tag = sp.tag if sp else frag.matched_tag
            sp_conf = sp.confidence if sp else "Single Group"

            if sp and sp.inline_components:
                formatted_items = []
                for sym_id in sp.inline_components:
                    comp = comp_map.get(sym_id)
                    if comp:
                        tag = comp.matched_tag if comp.matched_tag and str(comp.matched_tag).strip() not in ('', 'None', 'nan') else None
                        ctype = comp.type or "Unknown"
                        if tag:
                            formatted_items.append(f"{tag} ({ctype})")
                        else:
                            formatted_items.append(f"Unmatched ({ctype})")
                    else:
                        formatted_items.append(sym_id)
                inline_str = "; ".join(formatted_items)
            else:
                inline_str = ""

            for seg in frag.segments:
                sp_records.append({
                    "Super_Pipeline_ID": sp_id,
                    "Group_ID": frag.id,
                    "Segment_In_Group": seg.segment_in_group,
                    "Line_Type": frag.line_type,
                    "Orientation": seg.orientation,
                    "x_start": seg.x_start,
                    "y_start": seg.y_start,
                    "x_end": seg.x_end,
                    "y_end": seg.y_end,
                    "Pipeline_Tag": sp_tag or "",
                    "Merge_Confidence": sp_conf,
                    "Inline_Components": inline_str
                })

        df_sp = pd.DataFrame(sp_records)
        path_sp = os.path.join(out_dir, f"{stem}_SUPER_pipelines.csv")
        df_sp.to_csv(path_sp, index=False)

        # 2. Branch connections / spec-breaks CSV
        df_branches = pd.DataFrame(branches)
        path_branches = os.path.join(out_dir, f"{stem}_branch_connections.csv")
        df_branches.to_csv(path_branches, index=False)
