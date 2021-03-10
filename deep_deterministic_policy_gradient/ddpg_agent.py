""" Implementation of the agent that interacts with and learns from the environment by implementing the
Deep Deterministic Policy Gradient (DDPG) algorithm """

from models import Actor, Critic
from policy_based_methods_udacity.common import *
import torch.optim as optim
import random
import copy
from collections import namedtuple

ACTOR_LR = 10e-4
CRITIC_LR = 10e-3
WEIGHT_DECAY = 10e-2
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = .99
TAU = 1e-3


def minimize_loss(loss, optimizer: optim.Adam):
    """ Minimize the loss and optimize the model weights """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def soft_update(local_model, target_model):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


class Agent:
    def __init__(self, state_size, action_size, seed):
        """
        Initialise the agent
        :param state_size: Dimension of the state space
        :param action_size: Dimension of the action space
        :param seed: Random seed
        """

        random.seed(seed)

        # Initialise the Actor networks (local and target), including the Optimizer
        self.actor_local = Actor(state_size, action_size, seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)

        # Initialise the Critic networks (local and target)
        self.critic_local = Critic(state_size, action_size, seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)

        # Exploration noise process
        self.noise = OUNoise(action_size, seed)

        # Replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.ready_to_learn = len(self.memory) > BATCH_SIZE

    def reset(self):
        """ Reset the action noise """
        self.noise.reset()

    def act(self, state, add_noise=True):
        """ Return action for state as per policy """
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()  # put the policy in evaluation mode
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()  # forward pass
        self.actor_local.train()  # put the policy back in train mode
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, experience):
        """ Add experiences to the experience buffer and learn from a batch """
        self.memory.add(experience)
        if not self.ready_to_learn:
            self.ready_to_learn = len(self.memory) > BATCH_SIZE

        if self.ready_to_learn:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i) - Check https://arxiv.org/pdf/1509.02971.pdf paper
        q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))
        # compute the critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # minimize the loss
        minimize_loss(critic_loss, self.critic_optimizer)

        # update the actor
        actions_predicted = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_predicted).mean()
        # minimize loss
        minimize_loss(actor_loss, self.actor_optimizer)

        # update target networks
        soft_update(self.critic_local, self.critic_target)
        soft_update(self.actor_local, self.actor_target)


class OUNoise:
    """ Ornstein-Uhlenbeck exploration noise process for temporally correlated noise """

    def __init__(self, size, seed, mu=0., theta=.15, sigma=.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class Experience:
    """ Helper class to encapsulate an experience """

    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int):
        """ Initialize a ReplayBuffer object.
            :param buffer_size (int): maximum size of buffer
            :param batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, experience: Experience):
        """Add a new experience to memory."""
        e = self.experience(experience.state, experience.action, experience.reward, experience.next_state,
                            experience.done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def run_ddpg(env: gym.Env, agent: Agent, num_episodes=2000, max_time_steps=700):
    """ Runs the DDPG algorithm
        :param env: The gym environment to use
        :param agent: The agent to train
        :param num_episodes: The maximum number of episodes to train the agent
        :param max_time_steps: Max time steps to interact with the environment per episode
    """
    scores_deque = deque(maxlen=100)
    scores = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        agent.reset()
        score = 0
        for step in range(max_time_steps):
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            agent.step(Experience(state, action, reward, new_state, done))
            state = new_state
            score += reward
            if done:
                break
        scores.append(score)
        scores_deque.append(scores)

        mean_score = np.mean(scores_deque)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score), end="")
        if episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score))

    return scores


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    env.seed(10)
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], seed=10)
    scores = run_ddpg(env, agent, num_episodes=100)
