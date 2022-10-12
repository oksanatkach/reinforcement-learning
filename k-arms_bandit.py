import numpy as np
import random
from math import exp


class Arm(object):
    def __init__(self, p, mu):
        self.p = p
        self.mu = mu

    def step(self):
        if np.random.rand() > self.p:
            return 0

        else:
            return 1

    def step_gauss(self):
        return np.random.normal(self.mu, 1)

class Bandit(object):
    def __init__(self, epsilon, n_actions):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.N = np.zeros(n_actions)
        self.Q = np.zeros(n_actions)

    def reset(self):
       self.N = np.zeros(self.n_actions)
       self.Q = np.zeros(self.n_actions)

    def choose_action(self):
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q)
        else:
            return np.random.randint(0, self.n_actions)

    def choose_action_softmax(self, T):

        def softmax(values, t=1.0):
            e = np.exp(np.array(values) / t)
            dist = e / np.sum(e)
            return dist

        totals = np.cumsum(softmax(self.Q))
        norm = totals[-1]
        throw = np.random.rand() * norm
        chosen_action = np.searchsorted(totals, throw)
        return chosen_action

    def update(self, action, reward):
        self.N[action] += 1
        n = self.N[action]
        self.Q[action] = (n-1)/n * self.Q[action] + (1/n) * reward

def test_bandit(bandit, n_arms, n_episodes, n_steps, *args, **kwargs):
    chosen_arms = np.zeros((n_episodes, n_steps))
    cumulative = {}

    for episode in range(n_episodes):

        mues = [ np.random.normal() for i in range(n_arms) ]
        arms = [ Arm(0, mu) for mu in mues ]
        ep_reward = 0
        bandit.reset()

        for step in range(n_steps):
            # chosen_arm = bandit.choose_action()
            chosen_arm = bandit.choose_action_softmax(T=1)
            chosen_arms[episode, step] = chosen_arm
            reward = arms[chosen_arm].step_gauss()
            ep_reward += reward
            bandit.update(chosen_arm, reward)

        cumulative[episode] = float(ep_reward)

    return cumulative

epsilon = test_bandit(bandit=Bandit(epsilon=0.1, n_actions=10), n_arms=10, n_episodes=2000, n_steps=1000)

import matplotlib.pyplot as plt

fig = plt.figure()
cumulative_graph = fig.add_subplot(111)
cumulative_graph.plot(list(epsilon.keys()), list(epsilon.values()))
plt.show()