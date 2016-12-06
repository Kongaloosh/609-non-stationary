import random
random.seed(1994)
from pysrc.experiment import BanditExperiment
from pysrc.learning import BanditSampleAverage, SimpleBandit
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'kongaloosh'

class bandit_example(object):

    def __init__(self):
        self.simple_average = BanditSampleAverage(number_of_arms=10, epsilon=0.1)           # sample average
        self.simple_bandit = SimpleBandit(number_of_arms=10, epsilon=0.1, step_size=0.01)   # action-value learner
        self.problem = BanditExperiment(number_of_arms=10)                                  # source of data

        self.simple_average_rewards = list()
        self.simple_average_optimal = list()
        self.simple_average_optimal_count = 0.
        self.simple_average_reward_count = 0.

        self.simple_bandit_rewards = list()
        self.simple_bandit_optimal = list()
        self.simple_bandit_optimal_count = 0.
        self.simple_bandit_reward_count = 0.

    def bandit_update(self, episode_number):
        episode_number += 1                                                                 # move to next episode
        bandit_action = self.simple_bandit.get_action()                                     # pick an action
        reward = self.problem.action(bandit_action)                                         # observe reward
        self.simple_bandit_reward_count += reward                                           # update reward count
        optimal_action = self.problem.optimal_action()                                      # record the optimal
        self.simple_bandit.update_average(bandit_action, reward)                            # update estimate

        # if the chosen action was optimal, or has the same value as the optimal action...
        if optimal_action == bandit_action or \
                self.problem.bandit_means[optimal_action] == self.problem.bandit_means[bandit_action]:
            self.simple_bandit_optimal.append(1)                                            # add % optimal
        else:
            self.simple_bandit_optimal.append(0)                                            # add % optimal
        self.simple_bandit_rewards.append(reward)                                           # add avg reward

    def average_update(self, episode_number):
        episode_number += 1                                                                 # move to next episode
        average_action = self.simple_average.get_action()                                   # pick an action
        reward = self.problem.action(average_action)                                        # observe reward
        self.simple_average_reward_count += reward                                          # update reward count
        optimal_action = self.problem.optimal_action()                                      # record optimal action
        self.simple_average.update_average(average_action, reward)                          # add simple average

        # if the chosen action was optimal, or has the same value as the optimal action...
        if optimal_action == average_action or \
                self.problem.bandit_means[optimal_action] == self.problem.bandit_means[average_action]:
            self.simple_average_optimal.append(1)
        else:
            self.simple_average_optimal.append(0)
        self.simple_average_rewards.append(reward)

    def run_experiment(self, walk=False):
        """ initializes a new problem"""
        # self.problem.random_walk()                      #
        for step in range(10000):                      # for X many pulls
            self.average_update(step)                  # take a step with the sample average
            self.bandit_update(step)                   # take a step with the action-value
            if walk:                                   # if this is non-stationary, move the arms by a small amount
                self.problem.random_walk()


if __name__ == "__main__":
    simple_average_reward = np.zeros(10000)
    simple_average_optimal = np.zeros(10000)
    simple_bandit_reward = np.zeros(10000)
    simple_bandit_optimal = np.zeros(10000)

    for i in range(2000):
        if i % 100 == 0:
            print(i)
        experiment = bandit_example()
        experiment.run_experiment()
        simple_average_optimal += experiment.simple_average_optimal
        simple_bandit_optimal += experiment.simple_bandit_optimal
        simple_average_reward += experiment.simple_average_rewards
        simple_bandit_reward += experiment.simple_bandit_rewards

    simple_average_optimal /= 2000                                      # avg the outcomes by the number runs
    simple_bandit_optimal /= 2000
    simple_average_reward /= 2000
    simple_bandit_reward /= 2000


    plt.figure(0, figsize=(15,10))
    plt.suptitle("Action Value and Sample Average Performance With Optimistic Initialization")
    plt.subplot(2, 2, 1)
    plt.xlim([-100, 10100])
    plt.title("Percent Optimal Actions Averaged over 2000 Trials Without Random Walk")
    plt.ylabel("% optimal actions")
    plt.xlabel("Steps")
    plt.plot(simple_average_optimal, label='Simple Average')
    plt.plot(simple_bandit_optimal, label='Action Value')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.xlim([-100, 10100])
    plt.title("Average Reward Averaged over 2000 Trials Without Random Walk")
    plt.ylabel("Average Reward")
    plt.xlabel("Steps")
    plt.plot(simple_average_reward, label='Simple Average')
    plt.plot(simple_bandit_reward, label='Action Value')
    plt.legend()

    simple_average_reward = np.zeros(10000)
    simple_average_optimal = np.zeros(10000)
    simple_bandit_reward = np.zeros(10000)
    simple_bandit_optimal = np.zeros(10000)

    for i in range(2000):
        if i % 100 == 0:
            print(i)
        experiment = bandit_example()
        experiment.run_experiment(walk=True)
        simple_average_optimal += experiment.simple_average_optimal
        simple_bandit_optimal += experiment.simple_bandit_optimal
        simple_average_reward += experiment.simple_average_rewards
        simple_bandit_reward += experiment.simple_bandit_rewards

    simple_average_optimal /= 2000
    simple_bandit_optimal /= 2000
    simple_average_reward /= 2000
    simple_bandit_reward /= 2000

    plt.subplot(2, 2, 3)
    plt.xlim([-100, 10100])
    plt.title("Percent Optimal Actions Averaged over 2000 Trials With Random Walk")
    plt.ylabel("% optimal actions")
    plt.xlabel("Steps")
    plt.plot(simple_average_optimal, label='Simple Average')
    plt.plot(simple_bandit_optimal, label='Action Value')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.xlim([-100, 10100])
    plt.title("Average Reward Averaged over 2000 Trials With Random Walk")
    plt.ylabel("Average Reward")
    plt.xlabel("Steps")
    plt.plot(simple_average_reward, label='Simple Average')
    plt.plot(simple_bandit_reward, label='Action Value ')
    plt.legend()

    plt.show()
