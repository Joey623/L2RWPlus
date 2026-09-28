import unittest

from types import SimpleNamespace
import numpy as np
from l2rwplus.evaluation.bupt_metrics import evaluate


class MetricTests(unittest.TestCase):
    def test_bupt_inp_counts_all_relevant_matches_before_cmc_clipping(self):
        # Matches at ranks 1 and 3: INP=2/3, AP=(1+2/3)/2.
        cmc, ap, inp = evaluate(
            np.array([[0.0, 1.0, 2.0]]),
            np.array([7]),
            np.array([7, 8, 7]),
            SimpleNamespace(max_rank=20),
        )
        np.testing.assert_array_equal(cmc, np.ones(3))
        self.assertAlmostEqual(inp, 2 / 3)
        self.assertAlmostEqual(ap, 5 / 6)



if __name__ == "__main__":
    unittest.main()
