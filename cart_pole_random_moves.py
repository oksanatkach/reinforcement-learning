import gym
import numpy as np
import matplotlib.pyplot as plt


def greedy_epsilon_policy(Q, actions, epsilon):
    def policy_fn(state):
        if np.random.rand() < epsilon:
            return np.random.choice(actions)
        else:
            best = np.argmax(Q.value(state))
            return actions[best]

    return policy_fn


class LinearApproximateQ(object):
    def __init__(self, n_input_state, n_input_action, alpha):
        self.n_input_state = n_input_state
        self.n_input_action = n_input_action
        self.theta = np.random.rand(n_input_state, n_input_action)
        self.alpha = alpha

    def value(self, state):
        return np.matmul(state, self.theta)

    def update(self, error, state, action):

        gradient = self.alpha*error*state
        self.theta[:, action] += gradient
        print(np.linalg.norm(gradient))
        return np.linalg.norm(gradient)

API_KEY = "sk_x96pgtHcREy2bLhVWWbsSQ"
np.random.seed(42)

env = gym.make('CartPole-v0')
# env = gym.wrappers.Monitor(env,'sarsa-linear-approximation', force=True)

learning_rate = 0.001
Q = LinearApproximateQ(4, env.action_space.n, learning_rate)
actions = range(env.action_space.n)
gamma = 0.2
n_episodes = 10000
epsilon = 0.1
epsilon_decay = 0.999

steps_done = []
gradients = []
for j in range(n_episodes):
    done = False
    state = env.reset()
    policy = greedy_epsilon_policy(Q, actions, epsilon*(epsilon_decay**j))
    ep_steps_done = 0


    while not done:
        action = policy(state)
        new_state, reward, done, _ = env.step(action)
        reward = -1 if reward == 0 else reward
        ep_steps_done += 1

        if not done:
            new_action = policy(new_state)
            q_next = Q.value(new_state)[new_action]
        else:
            q_next = 0.0

        Q_true = reward + gamma * q_next
        Q_predicted = Q.value(state)[action]
        error = Q_true-Q_predicted
        gradient = Q.update(error, state, action)
        gradients.append(gradient)

        action = new_action
        state = new_state

    steps_done.append(ep_steps_done)


env.close()
plt.plot(steps_done)
plt.show()
print(Q.theta)
# gym.upload('sarsa-linear-approximation', api_key=API_KEY)