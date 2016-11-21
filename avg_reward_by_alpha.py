from pysrc.experiment import BanditExperiment
from pysrc.learning import BanditSampleAverage, SimpleBandit
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'kongaloosh'

# Average Reward based on True Action Values [ ]

class bandit_example(object):

    def __init__(self, epsilon, bandit_experiment=None):
        self.simple_average = BanditSampleAverage(number_of_arms=10, epsilon=epsilon)
        self.simple_bandit = SimpleBandit(number_of_arms=10, epsilon=epsilon, step_size=0.01)

        if bandit_experiment:                                   # if we provided a specific bandit, use it
            self.problem = bandit_experiment
        else:                                                   # otherwise make one
            self.problem = BanditExperiment(number_of_arms=10)

        self.simple_average_rewards = list()                    # the lists to store rewards
        self.simple_average_optimal = list()
        self.simple_average_optimal_count = 0.
        self.simple_average_reward_count = 0.

        self.simple_bandit_rewards = list()
        self.simple_bandit_optimal = list()
        self.simple_bandit_optimal_count = 0.
        self.simple_bandit_reward_count = 0.

        self.optimal_action = 0.

    def bandit_update(self, episode_number):
        """For a given episode number, update the"""
        episode_number += 1                                             # episode numbers
        bandit_action = self.simple_bandit.get_action()                 # pick an action
        reward = self.problem.action(bandit_action)                     # observe reward
        if episode_number > 100000:                                     # only
            self.simple_bandit_reward_count += reward                   # update reward count
        self.simple_bandit.update_average(bandit_action, reward)

    def average_update(self, episode_number):
        """For a given episode number, update the """
        episode_number += 1                                             # making episode non-zero based
        average_action = self.simple_average.get_action()               # pick an action
        # print(average_action)
        reward = self.problem.action(average_action)                    # get a reward for
        if episode_number > 100000:                                     # only count after 100000 samples
            self.simple_average_reward_count += reward                  # update reward count
        self.simple_average.update_average(average_action, reward)      # update the model

    def true_action(self, episode_number):
        if episode_number > 100000:                                     # only take the last 100000 samples
            self.optimal_action += self.problem.action(self.problem.optimal_action())

    def run_experiment(self, timesteps):
        for i in range(timesteps):
            self.average_update(i)                                      # update the sample average
            self.bandit_update(i)                                       # update the step-size action-value
            self.true_action(i)                                         # update the bandit action
            self.problem.random_walk()                                  # take a random walk


if __name__ == "__main__":
    epsilons = [1/128., 1/64., 1/32., 1/16., 1/8., 1/4., 1/2.]          # the epsilons we sweep over
    timesteps = 200000                                                  # the number of timesteps per trial
    # timesteps = 1000
    bandit = []                                                         # where the avg performan
    sample_average = []
    optimal = []

    bandit_experiment = BanditExperiment(number_of_arms=10)     # initialize bandit problem
    bandit_experiment.random_walk()                             # move each arm randomly
    print(bandit_experiment.bandit_means)
    for epsilon in epsilons:                                    # for all the epsilon values we want to check
        bandit_sample = []                                      # the reward for each trial
        sample_average_sample = []                              # the sample avg for each trial
        optimal_sample = []                                     # the reward for optimal choices
        print(epsilon)
        experiment = bandit_example(epsilon, bandit_experiment)             # create a new experiment to run
        print(experiment.problem.optimal_action())
        for i in range(100):                                                # averaged over 100 trials
            experiment.run_experiment(timesteps)                            # run experiment
            # add the averaged reward for each method to the avg performance list
            bandit_sample.append(experiment.simple_bandit_reward_count/(timesteps/2))
            sample_average_sample.append(experiment.simple_average_reward_count/(timesteps/2))
            optimal_sample.append(experiment.optimal_action/(timesteps/2))

        # find the mean performance for the given epsilon and record it
        bandit.append(np.mean(bandit_sample))
        sample_average.append(np.mean(sample_average_sample))
        optimal.append(np.mean(optimal_sample))

    plt.title("The Average Reward From a 10-armed bandit over 100 trials")
    plt.ylabel("Average Reward")
    plt.xlabel("Steps")
    plt.xticks(range(len(epsilons)), epsilons)
    plt.plot(bandit, label="step size")
    plt.plot(sample_average, label="sample average")
    plt.plot(optimal, label="optimal")
    plt.legend()
    plt.show()
