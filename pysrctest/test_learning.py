import unittest
from pysrc.learning import BanditSampleAverage, SimpleBandit
from numpy.testing import assert_allclose

__author__ = 'kongaloosh'


class TestSampleAverage(unittest.TestCase):

    def test_init(self):
        test_average = BanditSampleAverage(2, 0)
        self.assertTrue(len(test_average.bandit_estimates) == 2)
        self.assertTrue(len(test_average.bandit_visits) == 2)

        assert_allclose(test_average.bandit_estimates, [0, 0])
        assert_allclose(test_average.bandit_visits, [0, 0])

    def test_get_action(self):
        test_average = BanditSampleAverage(2, 0)
        test_average.update_average(0, 1)
        self.assertEqual(test_average.get_action(), 0)

    def test_update(self):
        test_average = BanditSampleAverage(2, 0)
        test_average.update_average(0, 1)
        self.assertEqual(test_average.bandit_visits[0], 1)
        self.assertEqual(test_average.bandit_estimates[0], 1)

        test_average.update_average(0, 0)
        self.assertEqual(test_average.bandit_visits[0], 2)
        self.assertEqual(test_average.bandit_estimates[0], 0.5)


class Test_Simple_Bandit(unittest.TestCase):

    def test_init(self):
        test_average = SimpleBandit(2, 0, 0.1)
        self.assertTrue(len(test_average.bandit_estimates) == 2)

        assert_allclose(test_average.bandit_estimates, [0, 0])

    def test_get_action(self):
        test_average = SimpleBandit(2, 0, 0.1)
        test_average.update_average(0, 1)
        self.assertEqual(test_average.get_action(), 0)

    def test_update(self):
        test_average = SimpleBandit(2, 0, 0.1)
        test_average.update_average(0, 1)
        self.assertEqual(test_average.bandit_estimates[0], 0.1)

        test_average.update_average(0, 0)
        self.assertEqual(test_average.bandit_estimates[0], 0.09)

if __name__ == "__main__":
    unittest.main()