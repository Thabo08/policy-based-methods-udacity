import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):
    """ The learning agent """
    def __init__(self, env_name, h_size=16):
        """
        :param env_name: The environment to train
        :param h_size: The hidden layer size (number of neurons in the hidden layer)
        """
        super(Agent, self).__init__()

        self.env = gym.make(env_name)
        self.env.seed(101)
        np.random.seed(101)
        self.state_size = self.env.observation_space.shape[0]
        self.h_size = h_size
        self.action_size = self.env.action_space.shape[0]

        # neural network architecture
        self.fc1 = nn.Linear(self.state_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.action_size)

        self.print_env_info(name=env_name)

    def set_weights(self, weights):
        """ Set weights and update them based on the policy return. There weights are used in collecting an episode
        """
        state_size = self.state_size
        h_size = self.h_size
        action_size = self.action_size

        # separate the weights for each layer
        fc1_end = (state_size*h_size)+h_size
        fc1_W = torch.from_numpy(weights[:state_size*h_size].reshape(state_size, h_size))
        fc1_b = torch.from_numpy(weights[state_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*action_size)].reshape(h_size, action_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*action_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self):
        return (self.state_size + 1) * self.h_size + (self.h_size + 1) * self.action_size

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = torch.tanh(self.fc2(state))
        return state.cpu().data

    def evaluate(self, weights, discount_factor=1.0, max_iterations=5000):
        """ Evaluate a return value from an episode
            :param weights: Weights to set during the evaluation
            :param discount_factor: Value by which returns should be discounted
            :param max_iterations: Maximum iterations to evaluate during an episode
         """
        self.set_weights(weights)
        episode_return = .0
        state = self.env.reset()
        for it in range(max_iterations):
            state = torch.from_numpy(state).float().to(device)
            action = self.forward(state)
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(discount_factor, it)
            if done:
                break
        return episode_return

    def env(self):
        return self.env

    def print_env_info(self, name):
        print('environment name:', name)
        print('observation space:', self.env.observation_space)
        print('action space:', self.env.action_space)
        print('  - low:', self.env.action_space.low)
        print('  - high:', self.env.action_space.high)


def cross_entropy_method(agent_, num_episodes=500, sigma=.5, population_size=50, discount_factor=1.0,
                         max_iterations=1000, elite_frac=.2, filename='checkpoint.pth'):
    """ Implements the cross entropy method to train the agent. The goal with cross entropy is to evaluate the policy
    return by passing the average weight of the top X weights to the neural network
    :param filename: file name to store the model weights
    :param elite_frac: percentage of top policies to use in update
    :param discount_factor:
    :param max_iterations:
    :param agent_: the agent to be trained
    :param num_episodes: the maximum number of episodes to train the agent
    :param sigma: standard deviation of additive noise
    :param population_size: size of the population
    :return: the average scores
    """
    scores = []
    scores_deque = deque(maxlen=100)
    additive_noise = sigma * np.random.randn(agent_.get_weights_dim())
    best_avg_weight = additive_noise
    num_elite = int(elite_frac * population_size)
    for episode in range(1, num_episodes + 1):
        population_weights = [best_avg_weight + additive_noise for _ in range(population_size)]
        rewards = np.array([agent_.evaluate(weights, discount_factor, max_iterations) for weights in population_weights])

        elite_indexes = rewards.argsort()[-num_elite:]
        elite_weights = [population_weights[i] for i in elite_indexes]
        best_avg_weight = np.array(elite_weights).mean(axis=0)

        reward = agent_.evaluate(best_avg_weight, discount_factor=discount_factor)
        scores_deque.append(reward)
        scores.append(reward)

        mean_score = np.mean(scores_deque)
        if episode % 10 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, mean_score))

        if mean_score >= 90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(episode - 100, mean_score))
            torch.save(agent.state_dict(), filename)
            break
    return scores


def plot_scores(scores):
    # plot the scores
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def test(agent, filename):
    # load the weights from file
    agent.load_state_dict(torch.load(filename))
    env = agent.env()

    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    while True:
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = agent(state)
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break

    env.close()


if __name__ == '__main__':
    agent = Agent("MountainCarContinuous-v0").to(device)
    train = True
    doing = lambda mode: print("Training agent" if mode else "Testing agent")
    if train:
        doing(train)
        scores = cross_entropy_method(agent, max_iterations=500)
        plot_scores(scores)
    else:
        doing(train)
        test(agent, filename='checkpoint.pth')
