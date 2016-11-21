import numpy as np

__author__ = 'kongaloosh'


class BanditSampleAverage(object):

    def __init__(self, number_of_arms, epsilon):
        self.number_of_steps = 0
        self.bandit_estimates = np.zeros(number_of_arms)
        # self.bandit_estimates = np.ones(number_of_arms)*10
        self.bandit_visits = np.zeros(number_of_arms)
        self.epsilon = epsilon

    def get_action(self):
        if self.epsilon >= np.random.random():
            return np.random.randint(len(self.bandit_estimates))
        return np.argmax(self.bandit_estimates)

    def update_average(self, arm, observation):
        self.bandit_visits[arm] += 1.
        self.bandit_estimates[arm] += (1/self.bandit_visits[arm]) * (observation - self.bandit_estimates[arm])


class SimpleBandit(object):

    def __init__(self, number_of_arms, epsilon, step_size):
        self.number_of_steps = 0
        self.bandit_estimates = np.zeros(number_of_arms)
        # self.bandit_estimates = np.ones(number_of_arms)*10
        self.epsilon = epsilon
        self.step_size = step_size

    def get_action(self):
        if self.epsilon >= np.random.random():
            return np.random.randint(len(self.bandit_estimates))
        return np.argmax(self.bandit_estimates)

    def update_average(self, arm, observation):
        self.number_of_steps += 1
        self.bandit_estimates[arm] += self.step_size * (observation - self.bandit_estimates[arm])

