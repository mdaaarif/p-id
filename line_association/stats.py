from typing import List, Dict, Any
from .models import LineFragment, LineTag, LineMatch, SuperPipeline
from .associator import LineAssociationResult

def summarize_line_association(
    result: LineAssociationResult,
    total_fragments: int,
    super_pipelines: List[SuperPipeline],
    branches: List[Dict[str, Any]]
) -> Dict[str, Any]:
    matched_count = len(result.matches)
    unmatched_count = len(result.unmatched_fragments)
    orphan_count = len(result.orphan_tags)

    sp_conf_counts = {}
    for sp in super_pipelines:
        sp_conf_counts[sp.confidence] = sp_conf_counts.get(sp.confidence, 0) + 1

    return {
        "total_fragments": total_fragments,
        "matched_fragments": matched_count,
        "unmatched_fragments": unmatched_count,
        "orphan_tags": orphan_count,
        "pass1_matches": result.pass1_count,
        "pass2_matches": result.pass2_count,
        "total_super_pipelines": len(super_pipelines),
        "super_pipeline_confidences": sp_conf_counts,
        "branch_connections": len(branches)
    }

def format_line_stats(stats: Dict[str, Any]) -> str:
    lines = [
        "--- Line Association & Topology Summary ---",
        f"Total Process Line Fragments: {stats['total_fragments']}",
        f"  - Matched with Line Tag:   {stats['matched_fragments']} (Pass 1: {stats['pass1_matches']}, Pass 2: {stats['pass2_matches']})",
        f"  - Unmatched Fragments:     {stats['unmatched_fragments']}",
        f"  - Orphan Line Tags:        {stats['orphan_tags']}",
        f"Continuous Super Pipelines:  {stats['total_super_pipelines']}",
        f"  - Breakdown:               {stats['super_pipeline_confidences']}",
        f"Physical Branch Connections: {stats['branch_connections']}"
    ]
    return "\n".join(lines)
