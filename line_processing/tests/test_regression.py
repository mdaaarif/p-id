import unittest
import os
import tempfile
import filecmp
from line_processing.run.run_line_processor import run_association

class TestRegression(unittest.TestCase):
    def test_regression_02_unmarked(self):
        ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        fixtures_dir = os.path.join(ROOT, "line_processing", "tests", "fixtures")
        
        expected_files = [
            "02.Unmarked_SUPER_pipelines.csv",
            "02.Unmarked_branch_connections.csv",
            "02.Unmarked_line_tag_associated.csv",
            "orphan_line_tags.csv",
            "unmatched_line_fragments.csv"
        ]
        
        for ef in expected_files:
            if not os.path.exists(os.path.join(fixtures_dir, ef)):
                self.skipTest(f"Missing fixture {ef}")
                
        with tempfile.TemporaryDirectory() as tmpdir:
            run_association(stem="02.Unmarked", out_dir=tmpdir)
            for ef in expected_files:
                f1 = os.path.join(fixtures_dir, ef)
                f2 = os.path.join(tmpdir, ef)
                self.assertTrue(os.path.exists(f2), f"Missing output {ef}")
                self.assertTrue(filecmp.cmp(f1, f2, shallow=False), f"Mismatch in {ef}")

if __name__ == "__main__":
    unittest.main()
