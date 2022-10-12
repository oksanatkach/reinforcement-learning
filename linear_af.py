import gym
import numpy as np
import matplotlib.pyplot as plt

# define policy
def greedy_epsilon_policy(Q, actions, epsilon):
    def policy_fn(state):
        if np.random.rand() < epsilon:
            return np.random.choice(actions)
        else:
            best = np.argmax(Q.value(state))
            return actions[best]

    return policy_fn

# define approximation function
class AF(object):
    def __init__(self, n_input_state, n_input_action, alpha):
        self.n_input_state = n_input_state
        self.n_input_action = n_input_action
        # parameter matrix
        self.theta = np.random.rand(n_input_state, n_input_action)
        # learning rate
        self.alpha = alpha

    def value(self, state):
        # state is a list of 4 parameters, multiply states and weights
        return np.matmul(state, self.theta)

    def update(self, error, state, action):
        # multiply each state by the delta of the predicted and true q and by learning rate
        gradient = self.alpha*error*state
        # add the gradient to theta parameters (move in the direction of the optimal values)
        self.theta[:, action] += gradient

# start env
env = gym.make('CartPole-v0')
# env = gym.wrappers.Monitor(env,'sarsa-linear-approximation', force=True)

# params
API_KEY = "sk_x96pgtHcREy2bLhVWWbsSQ"
actions = range(env.action_space.n)
epsilon = 0.1
gamma = 0.2
epsilon_decay = 0.999
learning_rate = 0.2
n_episodes = 1000
AF = AF(4, env.action_space.n, learning_rate)
graph = []

# each episode is try till success/death, update thetas

for j in range(n_episodes):
    # reset the done flag for each episode
    done = False
    rewards = 0
    state = env.reset()
    policy = greedy_epsilon_policy(AF, actions, epsilon)# * (epsilon_decay ** j))
    ep_steps_done = 0
    while not done:
        action = policy(state)
        new_state, reward, done, _ = env.step(action)
        reward = -1 if reward == 0 else reward
        rewards += reward

        if not done:
            new_action = policy(new_state)
            q_next = AF.value(new_state)[new_action]
        else:
            q_next = 0.0

        Q_true = reward + gamma * q_next
        Q_predicted = AF.value(state)[action]
        error = Q_true - Q_predicted
        AF.update(error, state, action)

        action = new_action
        state = new_state

    graph.append(rewards)

env.close()

plt.plot(graph)
plt.show()
# gym.upload('sarsa-linear-approximation', api_key=API_KEY)