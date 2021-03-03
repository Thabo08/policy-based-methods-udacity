""" Common imports and utility functions """

import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot(scores):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def sizes(env):
    def size(context):
        if (type(context)) == gym.spaces.box.Box:
            return context.shape[0]
        else:
            return context.n

    print('observation space:', size(env.observation_space))
    print('action space:', size(env.action_space))
    return size(env.observation_space), size(env.action_space)


def progress_report(episode, target_score, mean_score, print_every=100):
    if episode % print_every == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(episode, mean_score))
    if mean_score >= target_score:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, mean_score))
        return True
    return False


def watch_smart_agent(env, policy, steps=1000):
    try:
        state = env.reset()
        img = plt.imshow(env.render(mode='rgb_array'))
        for t in range(steps):
            action, _ = policy.act(state)
            img.set_data(env.render(mode='rgb_array'))
            plt.axis('off')
            state, reward, done, _ = env.step(action)
            if done:
                break
    finally:
        env.close()
