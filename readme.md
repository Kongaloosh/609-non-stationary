# Alex Kearney's Non-stationary Project
* this was run using python 2.7
* the experiments are as described for the non-stationary bandit project

## Navigating the project

There are two experiments contained within the project:
1. experiment_1.py
    * creates a 10-armed bandit for both a stationary and non-stationary setting
    * the non-stationary bandit arms all start at means of zero and variance of 0.01, then move by adding a small amount
    of noise to the means of each arm
    * the stationary bandit arms all start at a mean which is randomly drawn with a mean of 0 and a variance of 0.01
    * Averaged over 6000 trials of 10000 time steps, plots performance in terms of % optimal action selection and
    average reward.

2. experiment_2.py
    * creates a 10-armed bandit for either stationary or non-stationary setting, as specified by boolean 'random_walk'
    * runs a parallelized experiment which sweeps over the parameter values specified in figure 2.6
    * only uses rewards accumulated after 100,000 steps to calculate performance

There is one plotter which generates the formatted parameter sweep plot: plot_parameter_sweep.py. It does not run any
experiments and can simply be run without any other formatting.

The learning algorithms and bandit specification area clases stored in the pysrc package. There are limited tests for
both these in pysrctest.

