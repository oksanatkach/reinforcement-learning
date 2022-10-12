import gym
import numpy as np
import matplotlib.pyplot as plt


def episode_f(w, env):
    done = False
    state = env.reset()
    ep_steps_done = 0
    while not done:
        action = env.reset()
        new_state, _, done, _ = env.step(action)
        ep_steps_done += 1

    return steps_evaluation(ep_steps_done), ep_steps_done

def sigm_policy(weights, state):
    vector = np.concatenate([state,[0]])

def steps_evaluation(steps_done):
    if steps_done < 200