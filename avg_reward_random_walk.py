import random
random.seed(1994)
from pysrc.experiment import BanditExperiment
from pysrc.learning import BanditSampleAverage, SimpleBandit
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

__author__ = 'kongaloosh'

# Average Reward based on True Action Values [ ]

class bandit_example(object):

    def __init__(self, epsilon, bandit_experiment_means=None):
        self.simple_average = BanditSampleAverage(number_of_arms=10, epsilon=epsilon)
        self.simple_bandit = SimpleBandit(number_of_arms=10, epsilon=epsilon, step_size=0.01)

        self.problem = BanditExperiment(number_of_arms=10)
        if bandit_experiment_means is not None:                 # if we provided a specific bandit, use it
            self.problem.bandit_means = \
                np.copy(bandit_experiment_means)                # make a copy of provided means
        else:                                                   # otherwise move the means randomly
            self.problem.random_walk()

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

    def run_experiment(self, timesteps, walk=False):
        for step in range(timesteps):
            self.average_update(step)                                   # update the sample average
            self.bandit_update(step)                                    # update the step-size action-value
            self.true_action(step)                                      # update the bandit action
            if walk:
                self.problem.random_walk()

if __name__ == "__main__":
    epsilons = [1/128., 1/64., 1/32., 1/16., 1/8., 1/4., 1/2.]          # the epsilons we sweep over
    timesteps = 200000                                                  # the number of timesteps per trial
    bandit = []                                                         # where the avg performan
    sample_average = []
    optimal = []

    initial_bandit_means = np.random.normal(loc=0, scale=1, size=10)    # our sample means

    # ==============================================================================================
    #                                   WITH RANDOM WALK
    # ==============================================================================================

    for epsilon in epsilons:                                    # for all the epsilon values we want to check
        bandit_sample = []                                      # the reward for each trial
        sample_average_sample = []                              # the sample avg for each trial
        optimal_sample = []                                     # the reward for optimal choices
        print(epsilon)
        for i in range(1000):                                                # averaged over 100 trials
            experiment = bandit_example(
                epsilon=epsilon,
                bandit_experiment_means=initial_bandit_means)     # create a new experiment to run
            print(i)
            experiment.run_experiment(timesteps, walk=True)                            # run experiment
            # add the averaged reward for each method to the avg performance list
            bandit_sample.append(experiment.simple_bandit_reward_count/(timesteps/2))
            sample_average_sample.append(experiment.simple_average_reward_count/(timesteps/2))
            optimal_sample.append(experiment.optimal_action/(timesteps/2))
        # find the mean performance for the given epsilon and record it
        bandit.append(np.mean(bandit_sample))
        sample_average.append(np.mean(sample_average_sample))
        optimal.append(np.mean(optimal_sample))


    pkl.dump(bandit, open('value_action_walk',"wb"))
    pkl.dump(sample_average, open('sample_average_walk',"wb"))
    pkl.dump(optimal, open('optimal_walk',"wb"))


    plt.title("The Average Reward From a 10-armed bandit over 100 trials")
    plt.ylabel("Average Reward")
    plt.xlim([-1, 1])
    plt.xlabel("Steps")
    plt.xticks(range(len(epsilons)), ['1/128.', '1/64.', '1/32.', '1/16.', '1/8.', '1/4.', '1/2.'])
    plt.plot(bandit, label="action-value")
    plt.plot(sample_average, label="sample average")
    plt.plot(optimal, label="optimal")
    plt.legend()
    plt.show()
