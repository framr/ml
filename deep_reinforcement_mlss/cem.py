#!/usr/bin/env python

import numpy as np
import gym
from gym.spaces import Discrete, Box

# ================================================================
# Policies
# ================================================================

class DeterministicDiscreteActionLinearPolicy(object):

    def __init__(self, theta, obs_space, act_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_obs = obs_space.shape[0]
        n_actions = act_space.n
        assert len(theta) == (dim_obs + 1) * n_actions
        self.W = theta[0 : dim_obs * n_actions].reshape(dim_obs, n_actions)
        self.b = theta[dim_obs * n_actions : None].reshape(1, n_actions)

    def act(self, obs):
        """
        """
        y = obs.dot(self.W) + self.b
        a = y.argmax()
        return a


class DeterministicContinuousActionLinearPolicy(object):

    def __init__(self, theta, obs_space, act_space):
        """
        dim_obs: dimension of observations
        dim_act: dimension of action vector
        theta: flat vector of parameters
        """
        self.act_space = act_space
        dim_obs = obs_space.shape[0]
        dim_act = act_space.shape[0]
        assert len(theta) == (dim_obs + 1) * dim_act
        self.W = theta[0 : dim_obs * dim_act].reshape(dim_obs, dim_act)
        self.b = theta[dim_obs * dim_act : None]

    def act(self, obs):
        a = np.clip(obs.dot(self.W) + self.b, self.act_space.low, self.act_space.high)
        return a


def do_episode(policy, env, num_steps, render=False):
    total_rew = 0
    observation = env.reset()
    for t in range(num_steps):
        a = policy.act(observation)
        (observation, reward, done, _info) = env.step(a)
        total_rew += reward

        if render and t % 3 == 0:
            env.render()    
        if done:
            break
    return total_rew


env = None
def noisy_evaluation(theta):
    policy = make_policy(theta)
    rew = do_episode(policy, env, num_steps)
    return rew

def make_policy(theta):
    if isinstance(env.action_space, Discrete):
        return DeterministicDiscreteActionLinearPolicy(theta,
            env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return DeterministicContinuousActionLinearPolicy(theta,
            env.observation_space, env.action_space)
    else:
        raise NotImplementedError


    
if __name__ == '__main__':

    """
    Slightly modified starter code from MLSS Cadiz 2016
    Sorry for coding style
    """


    # Task settings:
    #env = gym.make('CartPole-v0') # Change as needed
    #env = gym.make('Pendulum-v0') # Change as needed
    #env = gym.make('Acrobot-v1') # Change as needed
    env = gym.make('MountainCar-v0') # Change as needed


    num_steps = 500 # maximum length of episode
    # Alg settings:
    n_iter = 100 # number of iterations of CEM
    batch_size = 25 # number of samples per batch
    elite_frac = 0.2 # fraction of samples used as elite set

    if isinstance(env.action_space, Discrete):
        dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.n
    elif isinstance(env.action_space, Box):
        dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.shape[0]
    else:
        raise NotImplementedError

    # Initialize mean and standard deviation
    theta_mean = np.zeros(dim_theta)
    theta_std = np.ones(dim_theta)

    # Now, for the algorithm
    for iteration in xrange(n_iter):

        # Sample parameter vectors
        thetas = np.random.multivariate_normal(theta_mean, np.diag(theta_std), batch_size)
        print theta_mean.shape
        print thetas.shape
        rewards = [noisy_evaluation(theta) for theta in thetas]
        print len(rewards)

        # Get elite parameters
        n_elite = int(batch_size * elite_frac)
        elite_inds = np.argsort(rewards)[batch_size - n_elite:batch_size]
        elite_thetas = [thetas[i] for i in elite_inds]

        print len(elite_thetas[0])
        # Update theta_mean, theta_std
        theta_mean = np.mean(elite_thetas, axis=0)
        theta_std = np.std(elite_thetas, axis=0)
        print len(theta_mean)

        print "iteration %i. mean f: %8.3g. max f: %8.3g"%(iteration, np.mean(rewards), np.max(rewards))
        do_episode(make_policy(theta_mean), env, num_steps, render=True)



