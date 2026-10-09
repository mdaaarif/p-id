from dataclasses import dataclass, field
from typing import Set

@dataclass
class LineAssociationConfig:
    """
    Configuration parameters for line-tag association, line-component association,
    and super-pipeline merging. All geometric coordinates and radii are normalized
    to PDF points (pt) matching the official specification rules.
    """
    # Radii for Line <-> Line Tag matching (in PDF points, as specified in rules docx)
    pass1_radius_pt: float = 40.0
    pass2_radius_pt: float = 100.0

    # Inline component threshold (Rule 7b: center within 15 pt of line path)
    on_path_threshold_pt: float = 15.0

    # Geometric tolerances for corners and bends (in PDF points)
    touch_tolerance_pt: float = 5.0
    padding_pt: float = 12.0 # Bounding box padding for inline valves/fittings

    # Collinear gap bridging (bridges across unextracted inline valves and dashed lines)
    collinear_max_gap_pt: float = 65.0
    collinear_axis_tolerance_pt: float = 3.0

    # Orthogonal projection preference: penalty applied if tag projection falls beyond segment endpoints
    endpoint_clamping_penalty_pt: float = 20.0

    # Major terminal equipment types that lines should NEVER bridge through
    major_equipment_types: Set[str] = field(default_factory=lambda: {
        'Vessel', 'Tank', 'Pump / Compressor', 'Heat Exchanger', 
        'Drum', 'Compressor', 'Complex Equipment',
        'Speed/Safety Valve', 'Pressure Valve', 'Relief Valve', 'Safety Valve'
    })
