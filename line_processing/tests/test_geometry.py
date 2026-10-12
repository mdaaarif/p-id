import unittest
from line_processing.utils.geometry import (
    point_to_segment_distance,
    segments_touch,
    endpoint_touches_body,
    is_point_near_bbox,
    segments_collinear_gap
)
from line_processing.utils.adapters import CoordinateTransformer
from line_processing.src.models import BBox, Segment

class TestGeometry(unittest.TestCase):
    def test_point_to_segment_distance(self):
        d, px, py = point_to_segment_distance(5.0, 5.0, 0.0, 0.0, 10.0, 0.0)
        self.assertEqual(d, 5.0)
        self.assertEqual(px, 5.0)
        self.assertEqual(py, 0.0)
        
        # Beyond endpoint
        d2, px2, py2 = point_to_segment_distance(15.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        self.assertEqual(d2, 5.0)
        self.assertEqual(px2, 10.0)
        self.assertEqual(py2, 0.0)

    def test_segments_touch(self):
        s1 = Segment(0, 0, 10, 0)
        s2 = Segment(10, 0, 10, 10)
        self.assertTrue(segments_touch(s1, s2, 0.1))
        
        s3 = Segment(11, 0, 11, 10)
        self.assertFalse(segments_touch(s1, s3, 0.1))

    def test_endpoint_touches_body(self):
        s_body = Segment(0, 0, 10, 0)
        # T-junction at (5,0)
        self.assertTrue(endpoint_touches_body((5.0, 0.0), s_body, 0.1))
        # Near endpoint - should be False
        self.assertFalse(endpoint_touches_body((0.0, 0.0), s_body, 0.1))

    def test_is_point_near_bbox(self):
        bbox = BBox(0, 0, 10, 10)
        self.assertTrue(is_point_near_bbox(5, 5, bbox, padding=0))
        self.assertTrue(is_point_near_bbox(12, 5, bbox, padding=2.0))
        self.assertFalse(is_point_near_bbox(13, 5, bbox, padding=2.0))

    def test_segments_collinear_gap(self):
        s1 = Segment(0, 0, 10, 0)
        s2 = Segment(30, 0, 40, 0)
        # Horizontal gap 20
        is_gap, gx, gy = segments_collinear_gap(s1, s2, max_gap=65.0, axis_tolerance=3.0)
        self.assertTrue(is_gap)
        self.assertEqual(gx, 20.0)
        self.assertEqual(gy, 0.0)
        
        # Too large gap
        s3 = Segment(80, 0, 90, 0)
        is_gap_large, _, _ = segments_collinear_gap(s1, s3, max_gap=65.0, axis_tolerance=3.0)
        self.assertFalse(is_gap_large)
        
        # Misaligned
        s4 = Segment(30, 5, 40, 5)
        is_gap_misalign, _, _ = segments_collinear_gap(s1, s4, max_gap=65.0, axis_tolerance=3.0)
        self.assertFalse(is_gap_misalign)

    def test_coordinate_transformer(self):
        # 300 DPI -> scale 4.1666
        # Rotation 0
        ct0 = CoordinateTransformer(page_w_pts=1000, page_h_pts=2000, rotation=0, dpi=300)
        x, y = ct0.le_to_pt(300, 300)
        self.assertAlmostEqual(x, 72.0)
        self.assertAlmostEqual(y, 72.0)
        
        # Rotation 90
        ct90 = CoordinateTransformer(page_w_pts=1000, page_h_pts=2000, rotation=90, dpi=300)
        x, y = ct90.le_to_pt(300, 300)
        self.assertAlmostEqual(x, 72.0)
        self.assertAlmostEqual(y, 2000 - 72.0)

if __name__ == "__main__":
    unittest.main()
