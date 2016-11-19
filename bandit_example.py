from pysrc.experiment import BanditExperiment
from pysrc.learning import BanditSampleAverage, SimpleBandit
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'kongaloosh'

simple_average_reward = np.zeros(10000)
simple_average_optimal = np.zeros(10000)
simple_bandit_reward = np.zeros(10000)
simple_bandit_optimal = np.zeros(10000)

class bandit_example(object):

    def __init__(self):
        self.simple_average = BanditSampleAverage(number_of_arms=10, epsilon=0.1)
        self.simple_bandit = SimpleBandit(number_of_arms=10, epsilon=0.1, step_size=0.1)
        self.problem = BanditExperiment(number_of_arms=10)

        self.simple_average_rewards = list()
        self.simple_average_optimal = list()
        self.simple_average_optimal_count = 0.
        self.simple_average_reward_count = 0.

        self.simple_bandit_rewards = list()
        self.simple_bandit_optimal = list()
        self.simple_bandit_optimal_count = 0.
        self.simple_bandit_reward_count = 0.

    def bandit_update(self, episode_number):
        episode_number += 1
        bandit_action = self.simple_bandit.get_action()                                       # pick an action
        # print(bandit_action)
        reward = self.problem.action(bandit_action)                                            # observe reward
        self.simple_bandit_reward_count += reward                                              # update reward count
        optimal_action = self.problem.optimal_action()
        self.simple_bandit.update_average(bandit_action, reward)

        if optimal_action == bandit_action:
                # or self.problem.bandit_means[optimal_action] == self.problem.bandit_means[bandit_action]:
            self.simple_bandit_optimal_count += 1

        self.simple_bandit_optimal.append(self.simple_bandit_optimal_count/episode_number)
        self.simple_bandit_rewards.append(self.simple_bandit_reward_count/episode_number)

    def average_update(self, episode_number):
        episode_number += 1
        average_action \
            = self.simple_average.get_action()                                       # pick an action
        reward = self.problem.action(average_action)                                            # observe reward
        self.simple_average_reward_count += reward                                              # update reward count
        optimal_action = self.problem.optimal_action()
        self.simple_average.update_average(average_action, reward)

        if optimal_action == average_action:
                # or self.problem.bandit_means[optimal_action] == self.problem.bandit_means[average_action]:
            self.simple_average_optimal_count += 1

        self.simple_average_optimal.append(self.simple_average_optimal_count/episode_number)
        self.simple_average_rewards.append(self.simple_average_reward_count/episode_number)


    def run_experiment(self):
        self.problem.random_walk()
        for i in range(10000):
            self.average_update(i)
            self.bandit_update(i)
            # self.problem.random_walk()


if __name__ == "__main__":
    for i in range(2000):
        experiment = bandit_example()
        experiment.run_experiment()
        simple_average_optimal += experiment.simple_average_optimal
        simple_bandit_optimal += experiment.simple_bandit_optimal
        simple_average_reward += experiment.simple_average_rewards
        simple_bandit_reward += experiment.simple_bandit_rewards

    simple_average_optimal /= 2000
    simple_bandit_optimal /= 2000
    simple_average_reward /= 2000
    simple_bandit_reward /= 2000


    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.title("")
    plt.ylabel("% optimal actions")
    plt.xlabel("Steps")
    plt.plot(simple_average_optimal, label='simple average')
    plt.plot(simple_bandit_optimal, label='bandit')

    plt.subplot(1, 2, 2)

    plt.title("")
    plt.ylabel("Average Reward")
    plt.xlabel("Steps")
    plt.plot(simple_average_reward, label='simple average')
    plt.plot(simple_bandit_reward, label='bandit ')

    plt.legend()
    plt.show()
