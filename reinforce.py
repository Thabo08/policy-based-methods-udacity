""" This is a basic implementation of the 'Reinforce' algorithm of Policy Gradient Methods """

from common import *
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, state_size=4, hidden_size=16, action_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = self.fc2(state)  # ??? No need for Relu activation here?
        # state = F.relu(self.fc2(state))  # ??? No need for Relu activation here?
        return F.softmax(state, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        action_probabilities = self.forward(state).cpu()
        m = Categorical(action_probabilities)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(env, policy, optimizer, num_episodes=5000, max_time_steps=1000, discount_factor=.99):
    scores_deque = deque(maxlen=100)
    scores = []
    for episode in range(1, num_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for time_step in range(max_time_steps):
            # collect an episode - or rather more correctly, a trajectory
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores.append(sum(rewards))
        scores_deque.append(sum(rewards))
        cumulative_reward = trajectory_cumulative_reward(discount_factor, rewards)
        # use log probabilities to compute the loss and update the model weights
        update_weights(cumulative_reward, optimizer, saved_log_probs)

        if progress_report(episode, 195., np.mean(scores_deque), print_every=100):
            break

    return scores


def trajectory_cumulative_reward(discount_factor, rewards):
    """ This computes the discounted cumulative reward from the trajectory
        :param discount_factor: Discount factor to apply
        :param rewards: Collected, un-discounted rewards
     """
    discounts = [discount_factor ** i for i in range(len(rewards) + 1)]
    cumulative_reward = sum([discount * reward for discount, reward in zip(discounts, rewards)])
    return cumulative_reward


def update_weights(cumulative_reward, optimizer, saved_log_probs):
    """ This updates the policy weights
        :param cumulative_reward: The discounted, cumulative reward collected in the episode
        :param optimizer: The gradient descent optimizer
        :param saved_log_probs: a list of log probabilities for each action taken per time step
    """
    policy_loss = []
    for log_prob in saved_log_probs:
        policy_loss.append(-log_prob * cumulative_reward)  # what's actually going on here?
    policy_loss = torch.cat(policy_loss).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    policy = Policy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)  # policy.parameters() is how you tell the optimizer
                                                          # which model's weights to update
    scores = reinforce(env, policy, optimizer, num_episodes=5000, max_time_steps=100)
    # plot(scores)
    watch_smart_agent(env, policy, steps=5000)
