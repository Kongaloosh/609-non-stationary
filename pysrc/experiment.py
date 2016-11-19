import numpy as np

__author__ = 'kongaloosh'


class BanditExperiment(object):

    def __init__(self, number_of_arms):
        """Creates an n-armed bandit"""
        self.bandit_means = np.zeros(number_of_arms)

    def action(self, arm_number):
        """Returns a value for a specific ban<"""
        return np.random.normal(loc=self.bandit_means[arm_number], scale=1.0)

    def random_walk(self):
        """Randomly moves all of the arms"""
        walk = np.random.normal(loc=0, scale=0.01**2, size=(len(self.bandit_means)))
        self.bandit_means += walk

    def optimal_action(self):
        """Returns the action with the best return. """
        return np.argmax(self.bandit_means)