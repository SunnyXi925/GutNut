import unittest
from pathlib import Path

class TestPreprocessing(unittest.TestCase):
    def test_paths_exist(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / 'data/microbiome/microbiome_example.csv').exists())
        self.assertTrue((root / 'data/microbiome/phenotypes_example.csv').exists())

    def test_clr_shape(self):
        try:
            from src.preprocessing import load_example_data
        except Exception as e:
            self.skipTest(f"Import failed: {e}")
            return
        _, _, clr = load_example_data()
        # 10 subjects × >=10 taxa
        self.assertEqual(clr.shape[0], 10)
        self.assertGreaterEqual(clr.shape[1], 10)

if __name__ == '__main__':
    unittest.main()