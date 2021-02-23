from common import *


class Policy:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.w = 1e-4 * np.random.rand(state_size, self.action_size)  # weights of a single linear policy

    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x)/sum(np.exp(x))

    def act(self, state, stochastic_policy=False):
        probabilities = self.forward(state)
        if stochastic_policy:
            return np.random.choice(self.action_size, p=probabilities)
        else:
            return np.argmax(probabilities)


def test(env, policy):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for t in range(200):
        action = policy.act(state)
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')
        state, reward, done, _ = env.step(action)
        if done:
            break

    env.close()


def hill_climbing(policy, env, episodes=1000, max_iterations=1000, discount_factor=.99, noise_scale=1e-2, target=100.0):
    """

    :return:
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_reward = -np.inf
    best_weight = policy.w
    for episode in range(1, episodes + 1):
        rewards = []
        state = env.reset()
        for it in range(max_iterations):
            action = policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        discounts = [discount_factor ** i for i in range(len(rewards) + 1)]
        cumulative_reward = sum([a_discount * a_reward for a_discount, a_reward in zip(discounts, rewards)])

        if cumulative_reward >= best_reward:  # found better weights
            best_reward = cumulative_reward
            best_weight = policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            policy.w += noise_scale * np.random.rand(*policy.w.shape)
        else:  # did not find better weights
            noise_scale = min(2, noise_scale * 2)
            policy.w = best_weight + noise_scale * np.random.rand(*policy.w.shape)

        mean_scores = np.mean(scores_deque)
        if episode % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, mean_scores))
        if mean_scores >= target:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, mean_scores))
            policy.w = best_weight
            break
    return scores


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # env = gym.make('MountainCarContinuous-v0')
    env.seed(0)
    np.random.seed(0)

    # policy = Policy(env.observation_space.shape[0], env.action_space.n)
    sizes = sizes(env)
    policy = Policy(sizes[0], sizes[1])
    scores = hill_climbing(policy, env, target=195.)
    plot(scores)
    test(env, policy)
