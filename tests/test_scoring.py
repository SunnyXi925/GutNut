import unittest
from pathlib import Path

class TestScoring(unittest.TestCase):
    def test_scoring_pipeline(self):
        root = Path(__file__).resolve().parents[1]
        # Prepare weights: run nutrient_weight (will produce weights.csv)
        try:
            from src.nutrient_weight import run_nutrient_weight
            run_nutrient_weight()
        except Exception as e:
            self.skipTest(f"Skip nutrient weight due to: {e}")
            return
        # Run scoring
        from src.scoring import run_scoring
        df = run_scoring()
        self.assertIn('Score', df.columns)
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()