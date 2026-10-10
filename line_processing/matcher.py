# Copied from line_association/rules.py so line_processing is self-contained
import math
from typing import List, Optional
from .models import LineFragment, InlineComponent
from .geometry import is_component_on_path
from .line_processor_config import LineProcessorConfig

class InlineComponentMatcher:
    """
    Rules 7 & 8: Associates bridging and on-path components to lines,
    excluding the line's own Line Tag anchor.
    """
    def __init__(self, config: LineProcessorConfig):
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
