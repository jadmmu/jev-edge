import unittest

import numpy as np

from q8_analysis import js_divergence, quantiles


class AnalysisTests(unittest.TestCase):
    def test_quantiles_use_linear_interpolation(self):
        result = quantiles([1, 2, 3, 4])
        self.assertEqual(result["median_ms"], 2.5)
        self.assertEqual(result["p90_ms"], 3.7)
        self.assertAlmostEqual(result["p95_ms"], 3.85)

    def test_js_divergence_is_zero_for_identical_distributions(self):
        self.assertEqual(js_divergence([0.25, 0.75], [0.25, 0.75]), 0.0)

    def test_js_divergence_is_symmetric_and_bounded(self):
        first = np.array([1.0, 0.0])
        second = np.array([0.0, 1.0])
        self.assertEqual(js_divergence(first, second), 1.0)
        self.assertEqual(js_divergence(first, second), js_divergence(second, first))


if __name__ == "__main__":
    unittest.main()
