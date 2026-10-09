from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import LineFragment, LineTag, InlineComponent, LineMatch, SuperPipeline
from .geometry import fragment_to_point_distance, is_component_on_path
from .config import LineAssociationConfig

class RuleVerdict(Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    NEUTRAL = "NEUTRAL"

class BaseLineRule(ABC):
    """Abstract base class for line association and merging rules."""
    name: str = "base_line_rule"
    priority: int = 10

    @abstractmethod
    def evaluate(self, candidate: Any, context: Dict[str, Any]) -> RuleVerdict:
        pass

class LineTagValidityRule(BaseLineRule):
    """Rule 2: Only tags classified as 'Line Tag' by upstream TE enter the candidate pool."""
    name = "line_tag_validity"
    priority = 10

    def evaluate(self, tag: LineTag, context: Dict[str, Any] = None) -> RuleVerdict:
        if tag.tag_type == "Line Tag":
            return RuleVerdict.PASS
        return RuleVerdict.REJECT

class DistanceMatrixBuilder:
    """Rule 1: Builds point-to-segment distance matrix between fragments and candidate tags."""
    def __init__(self, config: LineAssociationConfig):
        self.config = config

    def build_matrix(self, fragments: List[LineFragment], tags: List[LineTag]) -> np.ndarray:
        N = len(fragments)
        M = len(tags)
        dist_matrix = np.full((N, M), np.inf, dtype=float)

        penalty = getattr(self.config, 'endpoint_clamping_penalty_pt', 0.0)
        for i, frag in enumerate(fragments):
            for j, tag in enumerate(tags):
                d, _, _ = fragment_to_point_distance(
                    frag, tag.center.x, tag.center.y, clamping_penalty=penalty
                )
                dist_matrix[i, j] = d

        return dist_matrix

class BipartiteMatcher:
    """
    Rule 4 & 5: Global exclusivity matching via linear_sum_assignment with 2-pass radius gating.
    """
    def __init__(self, config: LineAssociationConfig):
        self.config = config

    def match_pass(
        self,
        fragments: List[LineFragment],
        tags: List[LineTag],
        dist_matrix: np.ndarray,
        radius_threshold_px: float,
        pass_name: str
    ) -> Tuple[List[LineMatch], List[int], List[int]]:
        """
        Runs linear_sum_assignment gated by radius_threshold_px.
        Returns:
          (matches, matched_frag_indices, matched_tag_indices)
        """
        N, M = dist_matrix.shape
        if N == 0 or M == 0:
            return [], [], []

        # Create masked cost matrix: entries > threshold get infinite penalty
        cost_matrix = dist_matrix.copy()
        mask_too_far = cost_matrix > radius_threshold_px
        cost_matrix[mask_too_far] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        matched_frag_idx = []
        matched_tag_idx = []

        for r, c in zip(row_ind, col_ind):
            actual_dist = dist_matrix[r, c]
            if actual_dist <= radius_threshold_px:
                frag = fragments[r]
                tag = tags[c]
                confidence = "High" if pass_name == "Pass 1" else "Medium"
                
                match = LineMatch(
                    fragment_id=frag.id,
                    tag_text=tag.text,
                    distance=round(actual_dist, 2),
                    pass_num=pass_name,
                    confidence=confidence
                )
                matches.append(match)
                matched_frag_idx.append(r)
                matched_tag_idx.append(c)

        return matches, matched_frag_idx, matched_tag_idx

class InlineComponentMatcher:
    """
    Rules 7 & 8: Associates bridging and on-path components to lines,
    excluding the line's own Line Tag anchor.
    """
    def __init__(self, config: LineAssociationConfig):
        self.config = config

    def find_inline_components(
        self,
        pipeline: SuperPipeline,
        fragments_map: Dict[int, LineFragment],
        components: List[InlineComponent],
        threshold_px: float
    ) -> List[Tuple[str, str]]:
        """
        Returns list of (symbol_id, source) for the pipeline, with Rule 8 deduplication applied.
        """
        inline_results = []
        assigned_tag = pipeline.tag

        for comp in components:
            # Rule 8: If component owns the line's own tag, exclude it from inline components
            if assigned_tag and comp.matched_tag and str(comp.matched_tag).strip() == str(assigned_tag).strip():
                continue

            # Check on-path against all fragments in this pipeline
            on_path = False
            for fid in pipeline.fragments:
                frag = fragments_map.get(fid)
                if frag and is_component_on_path(comp.center, frag, threshold_px):
                    on_path = True
                    break

            if on_path:
                source = "on_path"
                inline_results.append((comp.symbol_id, source))

        return inline_results
