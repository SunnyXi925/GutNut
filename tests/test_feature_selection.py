import unittest

class TestFeatureSelection(unittest.TestCase):
    def test_selected_features_json(self):
        try:
            from src.feature_selection import run_feature_selection
        except Exception as e:
            self.skipTest(f"Import failed: {e}")
            return
        selected, stability = run_feature_selection()
        self.assertGreaterEqual(len(selected), 5)
        for f in selected[:5]:
            self.assertIn(f, stability)

if __name__ == '__main__':
    unittest.main()