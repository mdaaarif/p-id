# Copied from line_association/geometry.py so line_processing is self-contained
import math
from typing import Tuple, List, Optional
from ..src.models import Point, BBox, Segment, LineFragment

def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float, clamping_penalty: float = 0.0
) -> Tuple[float, float, float]:
    """
    Computes shortest distance from point (px, py) to line segment (x1, y1) - (x2, y2).
    Returns: (distance, proj_x, proj_y)
    Formula from Rule 1:
      v = P2 - P1
      w = P - P1
      t = clip((w . v) / (v . v), 0, 1)
      proj = P1 + t * v
      d = ||P - proj||
    If clamping_penalty > 0 and projection falls beyond endpoints (t < 0 or t > 1), penalty is added.
    """
    vx = x2 - x1
    vy = y2 - y1
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq == 0.0:
        d = math.hypot(px - x1, py - y1)
        return d + clamping_penalty, x1, y1

    wx = px - x1
    wy = py - y1
    t = (wx * vx + wy * vy) / seg_len_sq
    t_clipped = max(0.0, min(1.0, t))

    proj_x = x1 + t_clipped * vx
    proj_y = y1 + t_clipped * vy
    d = math.hypot(px - proj_x, py - proj_y)

    if (t < 0.0 or t > 1.0) and clamping_penalty > 0.0:
        d += clamping_penalty

    return d, proj_x, proj_y

def fragment_to_point_distance(
    fragment: LineFragment, px: float, py: float, clamping_penalty: float = 0.0
) -> Tuple[float, float, float]:
    """
    Computes minimum distance from a point to any segment in a line fragment.
    Returns: (min_dist, closest_x, closest_y)
    """
    min_dist = float('inf')
    best_px, best_py = 0.0, 0.0
    for seg in fragment.segments:
        d, proj_x, proj_y = point_to_segment_distance(
            px, py, seg.x_start, seg.y_start, seg.x_end, seg.y_end, clamping_penalty=clamping_penalty
        )
        if d < min_dist:
            min_dist = d
            best_px, best_py = proj_x, proj_y
    return min_dist, best_px, best_py

def is_point_near_bbox(px: float, py: float, bbox: BBox, padding: float) -> bool:
    """Checks if point (px, py) is inside bbox expanded by padding."""
    min_x = min(bbox.x0, bbox.x1) - padding
    max_x = max(bbox.x0, bbox.x1) + padding
    min_y = min(bbox.y0, bbox.y1) - padding
    max_y = max(bbox.y0, bbox.y1) + padding
    return (min_x <= px <= max_x) and (min_y <= py <= max_y)

def segments_touch(seg_a: Segment, seg_b: Segment, tolerance: float) -> bool:
    """Checks if any endpoint of seg_a is within tolerance of any endpoint of seg_b."""
    pts_a = [(seg_a.x_start, seg_a.y_start), (seg_a.x_end, seg_a.y_end)]
    pts_b = [(seg_b.x_start, seg_b.y_start), (seg_b.x_end, seg_b.y_end)]
    for ax, ay in pts_a:
        for bx, by in pts_b:
            if math.hypot(ax - bx, ay - by) <= tolerance:
                return True
    return False

def endpoint_touches_body(end_pt: Tuple[float, float], seg: Segment, tolerance: float) -> bool:
    """Checks if an endpoint lands squarely on the interior body of another segment (T-junction)."""
    px, py = end_pt
    d, _, _ = point_to_segment_distance(px, py, seg.x_start, seg.y_start, seg.x_end, seg.y_end)
    if d > tolerance:
        return False
    # Ensure it is not just touching near an endpoint
    d1 = math.hypot(px - seg.x_start, py - seg.y_start)
    d2 = math.hypot(px - seg.x_end, py - seg.y_end)
    return d <= tolerance and (d1 > tolerance and d2 > tolerance)

def is_component_on_path(comp_center: Point, fragment: LineFragment, threshold: float) -> bool:
    """Checks if component center is within threshold distance of any segment in fragment (Rule 7b)."""
    min_dist, _, _ = fragment_to_point_distance(fragment, comp_center.x, comp_center.y)
    return min_dist <= threshold

def segments_collinear_gap(
    seg_a: Segment, 
    seg_b: Segment, 
    max_gap: float, 
    axis_tolerance: float = 3.0
) -> Tuple[bool, float, float]:
    """
    Checks if two line segments lie on the exact same collinear axis (both horizontal
    or both vertical) and have an open gap <= max_gap between their closest endpoints.
    Returns: (is_collinear_gap, mid_gap_x, mid_gap_y)
    """
    dx_a = abs(seg_a.x_end - seg_a.x_start)
    dy_a = abs(seg_a.y_end - seg_a.y_start)
    dx_b = abs(seg_b.x_end - seg_b.x_start)
    dy_b = abs(seg_b.y_end - seg_b.y_start)

    # Check Horizontal collinearity: dy is very small for both, and Y-coordinates match
    is_horiz_a = dy_a <= axis_tolerance and dx_a > dy_a
    is_horiz_b = dy_b <= axis_tolerance and dx_b > dy_b

    if is_horiz_a and is_horiz_b:
        y_mean_a = (seg_a.y_start + seg_a.y_end) / 2.0
        y_mean_b = (seg_b.y_start + seg_b.y_end) / 2.0
        if abs(y_mean_a - y_mean_b) <= axis_tolerance:
            xa_min, xa_max = min(seg_a.x_start, seg_a.x_end), max(seg_a.x_start, seg_a.x_end)
            xb_min, xb_max = min(seg_b.x_start, seg_b.x_end), max(seg_b.x_start, seg_b.x_end)

            if xa_max < xb_min:
                gap = xb_min - xa_max
                if 0.0 < gap <= max_gap:
                    return True, (xa_max + xb_min) / 2.0, (y_mean_a + y_mean_b) / 2.0
            elif xb_max < xa_min:
                gap = xa_min - xb_max
                if 0.0 < gap <= max_gap:
                    return True, (xb_max + xa_min) / 2.0, (y_mean_a + y_mean_b) / 2.0

    # Check Vertical collinearity: dx is very small for both, and X-coordinates match
    is_vert_a = dx_a <= axis_tolerance and dy_a > dx_a
    is_vert_b = dx_b <= axis_tolerance and dy_b > dx_b

    if is_vert_a and is_vert_b:
        x_mean_a = (seg_a.x_start + seg_a.x_end) / 2.0
        x_mean_b = (seg_b.x_start + seg_b.x_end) / 2.0
        if abs(x_mean_a - x_mean_b) <= axis_tolerance:
            ya_min, ya_max = min(seg_a.y_start, seg_a.y_end), max(seg_a.y_start, seg_a.y_end)
            yb_min, yb_max = min(seg_b.y_start, seg_b.y_end), max(seg_b.y_start, seg_b.y_end)

            if ya_max < yb_min:
                gap = yb_min - ya_max
                if 0.0 < gap <= max_gap:
                    return True, (x_mean_a + x_mean_b) / 2.0, (ya_max + yb_min) / 2.0
            elif yb_max < ya_min:
                gap = ya_min - yb_max
                if 0.0 < gap <= max_gap:
                    return True, (x_mean_a + x_mean_b) / 2.0, (yb_max + ya_min) / 2.0

    return False, 0.0, 0.0

