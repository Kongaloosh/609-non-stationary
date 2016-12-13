import random
random.seed(1994)
from pysrc.experiment import BanditExperiment
from pysrc.learning import BanditSampleAverage, SimpleBandit, UCB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import pickle as pkl

__author__ = 'kongaloosh'

initial_bandit_means = np.random.normal(loc=0, scale=1, size=10)    # our sample means


class bandit_example(object):

    def __init__(self, epsilon, bandit_experiment_means=None):
        self.simple_average = BanditSampleAverage(number_of_arms=10, epsilon=epsilon)
        self.simple_bandit = SimpleBandit(number_of_arms=10, epsilon=epsilon, step_size=0.01)
        self.ucb = UCB(step_size=0.01, number_of_arms=10, c=epsilon)

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

        self.ucb_reward_count = 0

        self.optimal_action = 0.

    def bandit_update(self, episode_number):
        """Records reward from """
        episode_number += 1                                             # episode numbers
        bandit_action = self.simple_bandit.get_action()                 # pick an action
        reward = self.problem.action(bandit_action)                     # observe reward
        if episode_number > 100000:                                     # only
            self.simple_bandit_reward_count += reward                   # update reward count
        self.simple_bandit.update_average(bandit_action, reward)        # update EMA model

    def average_update(self, episode_number):
        """Records reward from actions taken using sample averages """
        episode_number += 1                                             # making episode non-zero based
        average_action = self.simple_average.get_action()               # pick an action
        # print(average_action)
        reward = self.problem.action(average_action)                    # get a reward for
        if episode_number > 100000:                                     # only count after 100000 samples
            self.simple_average_reward_count += reward                  # update reward count
        self.simple_average.update_average(average_action, reward)      # update the sample average model

    def ucb_update(self, episode_number):
        """Records reward taken with UCB action selection"""
        episode_number += 1
        ucb_action = self.ucb.get_action(episode_number)
        reward = self.problem.action(ucb_action)
        if episode_number > 100000:
            self.ucb_reward_count += reward
        self.ucb.update_average(ucb_action, reward)

    def true_action(self, episode_number):
        """Records reward from choosing the best arm"""
        if episode_number > 100000:                                     # only take the last 100000 samples
            self.optimal_action += self.problem.action(self.problem.optimal_action())

    def run_experiment(self, timesteps, walk=False):
        for step in range(timesteps):
            self.average_update(step)                                   # update the sample average
            self.bandit_update(step)                                    # update the step-size action-value
            self.ucb_update(step)
            self.true_action(step)                                      # update the bandit action
            if walk:
                self.problem.random_walk()


def run_trial(epsilon):
    """Runs an experiment as outlined in the non-stationary example"""
    bandit_wrapper = np.zeros(len(epsilons))
    sample_average_wrapper = np.zeros(len(epsilons))
    ucb_wrapper = np.zeros(len(epsilons))
    optimal_wrapper = np.zeros(len(epsilons))

    bandit_sample = []                                      # the reward for each trial
    sample_average_sample = []                              # the sample avg for each trial
    ucb_sample = []                                         # the UCB avg for each trial
    optimal_sample = []                                     # the reward for optimal choices
    print(epsilon)
    for i in range(2000):                                   # averaged over this many trials
        experiment = bandit_example(
            epsilon=epsilon,
            bandit_experiment_means=initial_bandit_means)       # create a new experiment to run
        experiment.run_experiment(timesteps, walk=random_walk)  # run experiment

        # add the averaged reward for each method to the avg performance list
        bandit_sample.append(experiment.simple_bandit_reward_count/(timesteps/2))
        sample_average_sample.append(experiment.simple_average_reward_count/(timesteps/2))
        optimal_sample.append(experiment.optimal_action/(timesteps/2))
        ucb_sample.append(experiment.ucb_reward_count / (timesteps/2))

    # find the mean performance for the given epsilon and record it
    index = epsilons.index(epsilon)                                                 # where we should store the results
    bandit_wrapper[index] = np.mean(bandit_sample)                                  # record the results
    sample_average_wrapper[index] = np.mean(sample_average_sample)
    ucb_wrapper[index] = np.mean(ucb_sample)
    optimal_wrapper[index] = np.mean(optimal_sample)
    return bandit_wrapper, sample_average_wrapper, ucb_wrapper, optimal_wrapper     # return the performance


epsilons = [1/128., 1/64., 1/32., 1/16., 1/8., 1/4., 1/2., 1, 2, 4]     # the epsilons we sweep over
timesteps = 200000                                                      # the number of timesteps per trial
random_walk = True
bandit = np.zeros(len(epsilons))                                        # where the avg performance is stored
sample_average = np.zeros(len(epsilons))
ucb = np.zeros(len(epsilons))
optimal = np.zeros(len(epsilons))


if __name__ == "__main__":
    p = Pool(3)                                 # create 3 threads which run the sweep
    results = p.map(run_trial, epsilons)        # each thread runs function run_trial() with an epsilon from epsilons
    p.close()                                   # ends the sweep

    bandit, sample_average, ucb, optimal = np.sum(results, axis=0)                      # collect the sweep results
    pkl.dump((bandit,sample_average,ucb,optimal), open('statpools_sweep_long', "wb"))   # save results to file

    # Figure Plotting
    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.title("The Average Reward From a 10-armed bandit over 1000 trials")
    plt.ylabel("Average Reward")
    plt.xlim([-1, 5])
    plt.xlabel("Epsilon / C")
    plt.xticks(range(len(epsilons)), ['1/128.', '1/64.', '1/32.', '1/16.', '1/8.', '1/4.', '1/2.', '1', '2', '4'])
    plt.plot(bandit, label="action-value")
    plt.plot(sample_average, label="sample average")
    plt.plot(ucb, label="UCB")
    plt.plot(optimal, label="optimal")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('statfoo_slong.png')
    plt.savefig('statfoo_slong.pdf')
    plt.show()
