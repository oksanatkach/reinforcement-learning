import gym
import numpy as np

def epsilon_greedy_policy(Q, epsilon, actions):
    def policy_fn(state):
        if np.random.rand() > epsilon:
            action = np.argmax(Q[state,:])
        else:
            action = np.random.choice(actions)
        return action
    return policy_fn


def choose_action_softmax(lst, T):
    def softmax(values, t=1.0):
        e = np.exp(np.array(values) / t)
        dist = e / np.sum(e)
        return dist

    totals = np.cumsum(softmax(lst))
    norm = totals[-1]
    throw = np.random.rand() * norm
    chosen_action = np.searchsorted(totals, throw)
    return chosen_action

env = gym.make("FrozenLake-v0")
env = gym.wrappers.Monitor(env, "sarsa", force=True)

Q = np.random.rand(env.observation_space.n, env.action_space.n)
R = np.zeros([env.observation_space.n, env.action_space.n])
N = np.zeros([env.observation_space.n, env.action_space.n])

n_episodes = 1000
actions = range(env.action_space.n)
gamma = 1.0
API_KEY = 'sk_x96pgtHcREy2bLhVWWbsSQ'
rewards = 0.0
learning_rate = 0.1

for j in range(n_episodes):
    done = False
    state = env.reset()
    policy = epsilon_greedy_policy(Q, epsilon=0.1*(0.999**j), actions=actions)
    action = policy(state)
    while not done:
        new_state, reward, done, info = env.step(action)

        if not done:
            new_action = policy(new_state)
            q_next = Q[new_state, new_action]

        else:
            q_next = 0

        # N[state, action] += 1
        Q[state, action] += learning_rate*(reward + gamma * q_next - Q[state, action])#/N[state, action]
        state = new_state
        action = new_action
        rewards += reward

env.close()
gym.upload('sarsa', api_key=API_KEY)
print(rewards/n_episodes)