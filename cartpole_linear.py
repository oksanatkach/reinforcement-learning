import tensorflow as tf
import numpy as np
import gym
API_KEY = 'sk_x96pgtHcREy2bLhVWWbsSQ'
env = gym.make("CartPole-v0")
# env = gym.wrappers.Monitor(env, 'linear', force=True)

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

def policy_gradient():
    params = tf.get_variable("policy_parameters",[4,2])
    state = tf.placeholder("float",[None,4])
    actions = tf.placeholder("float",[None,2])
    linear = tf.matmul(state,params)
    probabilities = tf.nn.softmax(linear)
    good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions),reduction_indices=[1])
    # maximize the log probability
    log_probabilities = tf.log(good_probabilities)
    loss = -tf.reduce_sum(log_probabilities)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

def value_gradient():
    # sess.run(calculated) to calculate value of state
    state = tf.placeholder("float", [None, 4])
    w1 = tf.get_variable("w1", [4, 10])
    b1 = tf.get_variable("b1", [10])
    h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
    w2 = tf.get_variable("w2", [10, 1])
    b2 = tf.get_variable("b2", [1])
    calculated = tf.matmul(h1, w2) + b2

    # sess.run(optimizer) to update the value of a state
    newvals = tf.placeholder("float", [None, 1])
    diffs = calculated - newvals
    loss = tf.nn.l2_loss(diffs)
    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

# tensorflow operations to compute probabilties for each action, given a state
pl_probabilities, pl_state = policy_gradient()
observation = env.reset()
actions = []
transitions = []
for _ in range(200):
    # calculate policy
    obs_vector = np.expand_dims(observation, axis=0)
    probs = sess.run(pl_probabilities,feed_dict={pl_state: obs_vector})
    action = 0 if random.uniform(0,1) < probs[0][0] else 1
    # record the transition
    states.append(observation)
    actionblank = np.zeros(2)
    actionblank[action] = 1
    actions.append(actionblank)
    # take the action in the environment
    old_observation = observation
    observation, reward, done, info = env.step(action)
    transitions.append((old_observation, action, reward))
    totalreward += reward

    if done:
        break