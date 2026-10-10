# Copied from line_association/models.py so line_processing is self-contained
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

@dataclass(frozen=True)
class Point:
    x: float
    y: float

@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center(self) -> Point:
        return Point((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def width(self) -> float:
        return abs(self.x1 - self.x0)

    @property
    def height(self) -> float:
        return abs(self.y1 - self.y0)

@dataclass
class Segment:
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    orientation: str = "U" # "H", "V", "U"
    segment_in_group: int = 1

    @property
    def p1(self) -> Point:
        return Point(self.x_start, self.y_start)

    @property
    def p2(self) -> Point:
        return Point(self.x_end, self.y_end)

    @property
    def length(self) -> float:
        return ((self.x_end - self.x_start) ** 2 + (self.y_end - self.y_start) ** 2) ** 0.5

@dataclass
class LineFragment:
    id: int # Group_ID
    line_type: str
    segments: List[Segment] = field(default_factory=list)
    matched_tag: Optional[str] = None
    match_distance: Optional[float] = None
    match_pass: Optional[str] = None
    confidence: Optional[str] = None

@dataclass
class InlineComponent:
    symbol_id: str
    type: str
    group: str
    bbox: BBox
    mounting: Optional[str] = None
    matched_tag: Optional[str] = None

    @property
    def center(self) -> Point:
        return self.bbox.center

@dataclass
class SuperPipeline:
    id: str
    fragments: List[int] # list of Group_IDs
    tag: Optional[str] = None
    confidence: str = "Single Group" # "Confirmed", "Unverified", "Single Group"
    inline_components: List[str] = field(default_factory=list) # symbol_ids
    inline_sources: Dict[str, str] = field(default_factory=dict) # symbol_id -> source ('bridging', 'on_path')
