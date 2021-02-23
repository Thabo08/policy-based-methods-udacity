""" Common imports and utility functions """

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym


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
