""" This file holds the implementation of the Actor (policy function) and the Critic (value function) models used
 in the DDPG algorithm """

from policy_based_methods_udacity.common import *


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


def reset_parameters(layers):
    for ith_layer in range(0, len(layers) - 1):
        layer = layers[ith_layer]
        layer.weight.data.uniform_(*hidden_init(layer))
    layers[-1].weight.data.uniform_(-3e-3, 3e-3)
    

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=256):
        """
        Initialise instance of the Actor model with its settings
        :param state_size: Size of the state space
        :param action_size: Size of the action space
        :param seed: Random seed
        :param fc_units: Number of nodes in the hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        reset_parameters([self.fc1, self.fc2])

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        return torch.tanh(state)  # this helps to bound the actions to [-1, 1]


class Critic(nn.Module):
    def __init__(self, state_size, action_size, random_seed, fc1_units=256, fc2_units=256, fc3_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(random_seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        reset_parameters([self.fc1, self.fc2, self.fc3, self.fc4])

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

