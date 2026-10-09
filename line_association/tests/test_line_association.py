import unittest
import numpy as np

from line_association.models import (
    Point, BBox, Segment, LineFragment, LineTag, InlineComponent, SuperPipeline
)
from line_association.geometry import (
    point_to_segment_distance, fragment_to_point_distance, segments_touch, endpoint_touches_body
)
from line_association.config import LineAssociationConfig
from line_association.rules import (
    LineTagValidityRule, RuleVerdict, DistanceMatrixBuilder, BipartiteMatcher, InlineComponentMatcher
)
from line_association.associator import LineTagAssociator
from line_association.merger import SuperPipelineMerger

class TestLineGeometry(unittest.TestCase):
    def test_point_to_segment_orthogonal(self):
        # Horizontal segment from (0, 0) to (100, 0)
        # Point at (50, 20) -> distance should be exactly 20.0
        d, px, py = point_to_segment_distance(50.0, 20.0, 0.0, 0.0, 100.0, 0.0)
        self.assertAlmostEqual(d, 20.0, places=4)
        self.assertAlmostEqual(px, 50.0, places=4)
        self.assertAlmostEqual(py, 0.0, places=4)

    def test_point_to_segment_clamped_to_endpoint(self):
        # Point past the right endpoint: (130, 40)
        # Closest point should be (100, 0), distance = sqrt(30^2 + 40^2) = 50.0
        d, px, py = point_to_segment_distance(130.0, 40.0, 0.0, 0.0, 100.0, 0.0)
        self.assertAlmostEqual(d, 50.0, places=4)
        self.assertAlmostEqual(px, 100.0, places=4)
        self.assertAlmostEqual(py, 0.0, places=4)

    def test_multi_segment_elbow(self):
        # Elbow: segment 1 horizontal (0, 0) -> (50, 0), segment 2 vertical (50, 0) -> (50, 100)
        frag = LineFragment(
            id=1,
            line_type="Elbow",
            segments=[
                Segment(0.0, 0.0, 50.0, 0.0, orientation="H"),
                Segment(50.0, 0.0, 50.0, 100.0, orientation="V")
            ]
        )
        # Point near vertical leg: (60, 50) -> distance to seg 2 should be 10.0
        d, px, py = fragment_to_point_distance(frag, 60.0, 50.0)
        self.assertAlmostEqual(d, 10.0, places=4)
        self.assertAlmostEqual(px, 50.0, places=4)
        self.assertAlmostEqual(py, 50.0, places=4)

    def test_segments_touch_and_t_junction(self):
        seg_a = Segment(0.0, 0.0, 100.0, 0.0)
        seg_corner = Segment(100.0, 0.0, 100.0, 50.0)
        self.assertTrue(segments_touch(seg_a, seg_corner, tolerance=5.0))

        # T-junction: branch endpoint at (50, 0) touches middle of seg_a
        seg_branch = Segment(50.0, 0.0, 50.0, -80.0)
        self.assertTrue(endpoint_touches_body((50.0, 0.0), seg_a, tolerance=5.0))

class TestLineRules(unittest.TestCase):
    def test_tag_validity_rule(self):
        rule = LineTagValidityRule()
        tag_valid = LineTag(id="1", text="12-P-101", bbox=BBox(0, 0, 10, 10), tag_type="Line Tag")
        tag_invalid = LineTag(id="2", text="V-101", bbox=BBox(0, 0, 10, 10), tag_type="Equipment")

        self.assertEqual(rule.evaluate(tag_valid), RuleVerdict.PASS)
        self.assertEqual(rule.evaluate(tag_invalid), RuleVerdict.REJECT)

    def test_bipartite_mutual_exclusivity(self):
        # Two parallel horizontal lines: Line 1 at y=0, Line 2 at y=40
        frag1 = LineFragment(id=1, line_type="Line", segments=[Segment(0, 0, 100, 0)])
        frag2 = LineFragment(id=2, line_type="Line", segments=[Segment(0, 40, 100, 40)])

        # Single Line Tag at (50, 10) -> distance to Line 1 is 10, distance to Line 2 is 30
        tag = LineTag(id="T1", text="12-P-101", bbox=BBox(45, 5, 55, 15), tag_type="Line Tag")

        config = LineAssociationConfig()
        associator = LineTagAssociator(config=config)
        res = associator.associate([frag1, frag2], [tag])

        # Frag 1 should get the tag; Frag 2 should be unmatched
        self.assertEqual(len(res.matches), 1)
        self.assertEqual(res.matches[0].fragment_id, 1)
        self.assertEqual(len(res.unmatched_fragments), 1)
        self.assertEqual(res.unmatched_fragments[0].id, 2)
        self.assertEqual(len(res.orphan_tags), 0)

    def test_two_pass_radius_enforcement(self):
        config = LineAssociationConfig(pass1_radius_pt=40.0, pass2_radius_pt=100.0)
        frag1 = LineFragment(id=1, line_type="Line", segments=[Segment(0, 0, 100, 0)])
        frag2 = LineFragment(id=2, line_type="Line", segments=[Segment(0, 500, 100, 500)])

        # Tag 1 is at y=20 (distance 20 <= 40 -> Pass 1)
        tag1 = LineTag(id="T1", text="TAG-1", bbox=BBox(45, 15, 55, 25), tag_type="Line Tag")
        # Tag 2 is at y=570 (distance 70: > 40 and <= 100 -> Pass 2)
        tag2 = LineTag(id="T2", text="TAG-2", bbox=BBox(45, 565, 55, 575), tag_type="Line Tag")

        associator = LineTagAssociator(config=config)
        res = associator.associate([frag1, frag2], [tag1, tag2])

        self.assertEqual(len(res.matches), 2)
        m1 = next(m for m in res.matches if m.fragment_id == 1)
        m2 = next(m for m in res.matches if m.fragment_id == 2)
        self.assertEqual(m1.pass_num, "Pass 1")
        self.assertEqual(m2.pass_num, "Pass 2")

class TestSuperPipelineMerger(unittest.TestCase):
    def test_tag_propagation_and_corner_merge(self):
        # Two connecting lines forming an elbow: frag 1 tagged, frag 2 untagged
        frag1 = LineFragment(id=1, line_type="Line", segments=[Segment(0, 0, 100, 0)], matched_tag="12-P-101")
        frag2 = LineFragment(id=2, line_type="Line", segments=[Segment(100, 0, 100, 100)], matched_tag=None)

        merger = SuperPipelineMerger()
        pipelines, branches = merger.merge([frag1, frag2], components=[])

        # Should merge into 1 super pipeline
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0].tag, "12-P-101") # Propagated tag
        self.assertEqual(len(branches), 0)

    def test_spec_break_refusal(self):
        # Two lines meeting at (100, 0) but having conflicting tags (spec-break)
        frag1 = LineFragment(id=1, line_type="Line", segments=[Segment(0, 0, 100, 0)], matched_tag="12-P-101")
        frag2 = LineFragment(id=2, line_type="Line", segments=[Segment(100, 0, 200, 0)], matched_tag="2-P-102")

        merger = SuperPipelineMerger()
        pipelines, branches = merger.merge([frag1, frag2], components=[])

        # Should NOT merge: remain 2 distinct pipelines
        self.assertEqual(len(pipelines), 2)
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]['Tag_1'], "12-P-101")
        self.assertEqual(branches[0]['Tag_2'], "2-P-102")

    def test_rule_8_deduplication(self):
        # Line has tag "12-P-101". Valve is on path. Anchor component also on path with matched_tag "12-P-101".
        frag = LineFragment(id=1, line_type="Line", segments=[Segment(0, 0, 100, 0)], matched_tag="12-P-101")
        sp = SuperPipeline(id="SP_001", fragments=[1], tag="12-P-101")

        valve = InlineComponent(symbol_id="V_1", type="Gate Valve", group="Valve", bbox=BBox(45, -5, 55, 5))
        anchor = InlineComponent(symbol_id="V_ANCHOR", type="Spec Tag", group="Tag", bbox=BBox(10, -5, 20, 5), matched_tag="12-P-101")

        matcher = InlineComponentMatcher(LineAssociationConfig())
        results = matcher.find_inline_components(sp, {1: frag}, [valve, anchor], threshold_px=15.0)

        # Only the valve should be kept; anchor with same tag must be deduplicated (Rule 8)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "V_1")

    def test_collinear_gap_bridging(self):
        # Two horizontal collinear lines with a 40 pt gap: frag 1 (0 to 100, y=50) and frag 2 (140 to 240, y=50)
        frag1 = LineFragment(id=1, line_type="Line", segments=[Segment(0, 50, 100, 50)], matched_tag="12-P-101")
        frag2 = LineFragment(id=2, line_type="Line", segments=[Segment(140, 50, 240, 50)], matched_tag=None)

        merger = SuperPipelineMerger(LineAssociationConfig(collinear_max_gap_pt=65.0))
        pipelines, branches = merger.merge([frag1, frag2], components=[])

        # Should bridge the 40 pt gap and merge into 1 super pipeline
        self.assertEqual(len(pipelines), 1)
        self.assertEqual(pipelines[0].tag, "12-P-101")
        self.assertEqual(len(branches), 0)

    def test_collinear_gap_spec_break_refusal(self):
        # Two collinear lines with a 40 pt gap but DIFFERENT tags
        frag1 = LineFragment(id=1, line_type="Line", segments=[Segment(0, 50, 100, 50)], matched_tag="12-P-101")
        frag2 = LineFragment(id=2, line_type="Line", segments=[Segment(140, 50, 240, 50)], matched_tag="2-P-102")

        merger = SuperPipelineMerger(LineAssociationConfig(collinear_max_gap_pt=65.0))
        pipelines, branches = merger.merge([frag1, frag2], components=[])

        # Must NOT merge: remain 2 distinct pipelines
        self.assertEqual(len(pipelines), 2)
        self.assertEqual(len(branches), 1)

if __name__ == '__main__':
    unittest.main()

