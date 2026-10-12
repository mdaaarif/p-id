import unittest
import subprocess
import sys
from line_processing.src.models import LineFragment, Segment, InlineComponent, BBox
from line_processing.src.line_processor import LineProcessor, DisjointSetUnion
from line_processing.config.line_processor_config import LineProcessorConfig

class TestLineProcessor(unittest.TestCase):
    def setUp(self):
        self.config = LineProcessorConfig()
        self.lp = LineProcessor(self.config)

    def test_corner_touch(self):
        f1 = LineFragment(id=1, line_type="Solid", segments=[Segment(0,0,10,0)])
        f2 = LineFragment(id=2, line_type="Solid", segments=[Segment(10,0,10,10)])
        sp, br = self.lp.build_super_pipelines([f1, f2], [])
        self.assertEqual(len(sp), 1)
        self.assertEqual(len(sp[0].fragments), 2)
        self.assertEqual(len(br), 0)

    def test_t_junction(self):
        f1 = LineFragment(id=1, line_type="Solid", segments=[Segment(0,0,20,0)])
        f2 = LineFragment(id=2, line_type="Solid", segments=[Segment(10,0,10,10)])
        sp, br = self.lp.build_super_pipelines([f1, f2], [])
        self.assertEqual(len(sp), 1)
        self.assertEqual(len(sp[0].fragments), 2)

    def test_collinear_gaps(self):
        f1 = LineFragment(id=1, line_type="Solid", segments=[Segment(0,0,10,0)])
        f2 = LineFragment(id=2, line_type="Solid", segments=[Segment(30,0,40,0)])
        sp, br = self.lp.build_super_pipelines([f1, f2], [])
        self.assertEqual(len(sp), 1) # gap = 20 < 65
        
        f3 = LineFragment(id=3, line_type="Solid", segments=[Segment(80,0,90,0)])
        sp2, br2 = self.lp.build_super_pipelines([f1, f3], [])
        self.assertEqual(len(sp2), 2) # gap = 70 > 65

    def test_collinear_major_equipment_block(self):
        f1 = LineFragment(id=1, line_type="Solid", segments=[Segment(0,0,10,0)])
        f2 = LineFragment(id=2, line_type="Solid", segments=[Segment(30,0,40,0)])
        comp = InlineComponent(symbol_id="c1", type="Tank", group="Vessel", bbox=BBox(10,-10,30,10))
        sp, br = self.lp.build_super_pipelines([f1, f2], [comp])
        self.assertEqual(len(sp), 2) # Tank blocked it

    def test_tag_conflict_branch(self):
        f1 = LineFragment(id=1, line_type="Solid", segments=[Segment(0,0,10,0)], matched_tag="A")
        f2 = LineFragment(id=2, line_type="Solid", segments=[Segment(10,0,10,10)], matched_tag="B")
        sp, br = self.lp.build_super_pipelines([f1, f2], [])
        self.assertEqual(len(sp), 2)
        self.assertEqual(len(br), 1)
        self.assertEqual(br[0]['Tag_1'], "A")
        self.assertEqual(br[0]['Tag_2'], "B")

    def test_tag_merge_confidence(self):
        f1 = LineFragment(id=1, line_type="Solid", segments=[Segment(0,0,10,0)], matched_tag="A")
        f2 = LineFragment(id=2, line_type="Solid", segments=[Segment(10,0,10,10)])
        sp, br = self.lp.build_super_pipelines([f1, f2], [])
        self.assertEqual(len(sp), 1)
        self.assertEqual(sp[0].tag, "A")
        self.assertEqual(sp[0].confidence, "Unverified")
        
        f3 = LineFragment(id=3, line_type="Solid", segments=[Segment(0,0,10,0)], matched_tag="A")
        f4 = LineFragment(id=4, line_type="Solid", segments=[Segment(10,0,10,10)], matched_tag="A")
        sp2, br2 = self.lp.build_super_pipelines([f3, f4], [])
        self.assertEqual(sp2[0].confidence, "Confirmed")
        
        f5 = LineFragment(id=5, line_type="Solid", segments=[Segment(0,0,10,0)])
        sp3, br3 = self.lp.build_super_pipelines([f5], [])
        self.assertEqual(sp3[0].confidence, "Single Group")

    def test_deduplicate_branches(self):
        f1 = LineFragment(id=1, line_type="Solid", segments=[Segment(0,0,10,0), Segment(10,0,20,0)], matched_tag="A")
        f2 = LineFragment(id=2, line_type="Solid", segments=[Segment(10,0,10,10)], matched_tag="B")
        # f1 touches f2 at (10,0). They conflict. Will generate multiple touch edges if they had multiple segments meeting, 
        # let's explicitly make them cross in a way that generates multiple edges
        f3 = LineFragment(id=3, line_type="Solid", segments=[Segment(0,0,10,0), Segment(0,10,10,10)], matched_tag="A")
        f4 = LineFragment(id=4, line_type="Solid", segments=[Segment(10,0,10,10)], matched_tag="B")
        sp, br = self.lp.build_super_pipelines([f3, f4], [])
        self.assertEqual(len(br), 1) # Should only be one branch connection

    def test_dsu(self):
        f1 = LineFragment(id=1, line_type="S", matched_tag="A")
        f2 = LineFragment(id=2, line_type="S", matched_tag=None)
        f3 = LineFragment(id=3, line_type="S", matched_tag="B")
        dsu = DisjointSetUnion([f1, f2, f3])
        self.assertTrue(dsu.union(1, 2))
        self.assertFalse(dsu.union(1, 3))
        self.assertEqual(dsu.find(1), dsu.find(2))
        self.assertEqual(dsu.dsu_tag[dsu.find(2)], "A")

    def test_independence(self):
        script = "import sys; [sys.modules.__setitem__(m, None) for m in ('line_association','association','TE','CE','LE')]; sys.path.insert(0,'.'); from line_processing.src.line_processor import LineProcessor; from line_processing.config.line_processor_config import LineProcessorConfig; print('OK')"
        res = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(res.stdout.strip(), "OK")

if __name__ == "__main__":
    unittest.main()
