from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from .models import LineFragment, LineTag, LineMatch, InlineComponent, SuperPipeline
from .config import LineAssociationConfig
from .rules import LineTagValidityRule, DistanceMatrixBuilder, BipartiteMatcher, InlineComponentMatcher, RuleVerdict

@dataclass
class LineAssociationResult:
    matches: List[LineMatch] = field(default_factory=list)
    unmatched_fragments: List[LineFragment] = field(default_factory=list)
    orphan_tags: List[LineTag] = field(default_factory=list)
    dist_matrix: Optional[np.ndarray] = None
    pass1_count: int = 0
    pass2_count: int = 0

class LineTagAssociator:
    """
    Orchestrator for Part 1 (Rules 1 to 6):
    Associates Line Tags to raw line fragments using 2-pass bipartite assignment.
    """
    def __init__(self, config: Optional[LineAssociationConfig] = None):
        self.config = config or LineAssociationConfig()
        self.tag_filter = LineTagValidityRule()
        self.matrix_builder = DistanceMatrixBuilder(self.config)
        self.matcher = BipartiteMatcher(self.config)

    def associate(self, fragments: List[LineFragment], tags: List[LineTag]) -> LineAssociationResult:
        # Rule 2: Filter candidate pool
        valid_tags = [t for t in tags if self.tag_filter.evaluate(t) == RuleVerdict.PASS]

        if not fragments or not valid_tags:
            return LineAssociationResult(
                matches=[],
                unmatched_fragments=list(fragments),
                orphan_tags=list(valid_tags)
            )

        # Rule 1: Build distance matrix
        dist_matrix = self.matrix_builder.build_matrix(fragments, valid_tags)

        # Rule 5: Pass 1 (Tight radius, 40 pt)
        pass1_radius = self.config.pass1_radius_pt
        p1_matches, p1_frag_idx, p1_tag_idx = self.matcher.match_pass(
            fragments=fragments,
            tags=valid_tags,
            dist_matrix=dist_matrix,
            radius_threshold_px=pass1_radius,
            pass_name="Pass 1"
        )

        matched_frag_set = set(p1_frag_idx)
        matched_tag_set = set(p1_tag_idx)

        # Rule 5: Pass 2 (Wide radius, 100 pt) on remaining leftovers
        rem_frag_indices = [i for i in range(len(fragments)) if i not in matched_frag_set]
        rem_tag_indices = [j for j in range(len(valid_tags)) if j not in matched_tag_set]

        p2_matches = []
        if rem_frag_indices and rem_tag_indices:
            sub_dist_matrix = dist_matrix[np.ix_(rem_frag_indices, rem_tag_indices)]
            sub_frags = [fragments[i] for i in rem_frag_indices]
            sub_tags = [valid_tags[j] for j in rem_tag_indices]

            pass2_radius = self.config.pass2_radius_pt
            p2_sub_matches, p2_sub_frag_idx, p2_sub_tag_idx = self.matcher.match_pass(
                fragments=sub_frags,
                tags=sub_tags,
                dist_matrix=sub_dist_matrix,
                radius_threshold_px=pass2_radius,
                pass_name="Pass 2"
            )

            p2_matches = p2_sub_matches
            for sub_i in p2_sub_frag_idx:
                matched_frag_set.add(rem_frag_indices[sub_i])
            for sub_j in p2_sub_tag_idx:
                matched_tag_set.add(rem_tag_indices[sub_j])

        all_matches = p1_matches + p2_matches

        # Update fragment state with matches
        match_by_frag_id = {m.fragment_id: m for m in all_matches}
        for frag in fragments:
            if frag.id in match_by_frag_id:
                m = match_by_frag_id[frag.id]
                frag.matched_tag = m.tag_text
                frag.match_distance = m.distance
                frag.match_pass = m.pass_num
                frag.confidence = m.confidence

        # Rule 6: Track unmatched fragments and orphan tags
        unmatched_frags = [fragments[i] for i in range(len(fragments)) if i not in matched_frag_set]
        orphan_tags = [valid_tags[j] for j in range(len(valid_tags)) if j not in matched_tag_set]

        return LineAssociationResult(
            matches=all_matches,
            unmatched_fragments=unmatched_frags,
            orphan_tags=orphan_tags,
            dist_matrix=dist_matrix,
            pass1_count=len(p1_matches),
            pass2_count=len(p2_matches)
        )
