import unittest
from pysrc.experiment import BanditExperiment
from numpy.testing import assert_allclose

__author__ = 'kongaloosh'

class TestBandit(unittest.TestCase):

    def test_init(self):
        bandit = BanditExperiment(2)
        self.assertEqual(len(bandit.bandit_means), 2)
        assert_allclose(bandit.bandit_means, [0, 0])


if __name__ == "__main__":
    unittest.main()