import gym
import numpy as np
# env = gym.make("CartPole-v0")
#
# state = env.reset()
#
# for _ in range(10):
#     env.render()
#     action = np.random.choice(range(env.action_space.n))
#     new_state, reward, done, info = env.step(action)
#
#     print(new_state, reward, done)

def epsilon_greedy_policy(Q, epsilon, actions):
    def policy_fn(state):
        if np.random.rand() > epsilon:
            action = np.argmax(Q[state,:])
        else:
            action = np.random.choice(actions)
        return action
    return policy_fn

env = gym.make("FrozenLake-v0")
env = gym.wrappers.Monitor(env, "first_visit", force=True)

Q = np.random.rand(env.observation_space.n, env.action_space.n)
R = np.zeros([env.observation_space.n, env.action_space.n])
N = np.zeros([env.observation_space.n, env.action_space.n])

n_episodes = 1000
actions = range(env.action_space.n)
gamma = 1.0
API_KEY = 'sk_x96pgtHcREy2bLhVWWbsSQ'
rewards = 0.0

for j in range(n_episodes):
    done = False
    state = env.reset()
    policy = epsilon_greedy_policy(Q, epsilon=0.1, actions=actions)
    episode = []
    while not done:
        action = policy(state)
        new_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = new_state
        rewards += reward

    for s, a, _ in episode:

        first_visit = next( i for i, x in enumerate(episode) if x[0]==s and x[1]==a )
        G = sum(x[2] * (gamma**i) for i, x in enumerate(episode[first_visit:]))


        R[s, a] += G
        N[s, a] += 1
        Q[s, a] = R[s, a]/N[s, a]


env.close()
gym.upload('first_visit', api_key=API_KEY)
print(rewards/n_episodes)