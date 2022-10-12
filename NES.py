import numpy as np
import gym
np.random.seed(0)

# the function we want to optimize
def f(w):
    center = np.array([0.5, 0.1, -0.3])
    reward = -np.sum(np.square(center - w))
    return reward

def nes(npop, n_iter, sigma, alpha, f, w, dim):
    for i in range(n_iter):
        N = np.random.randn(npop, dim)  # generate a population of weight vectors
        R = np.zeros(npop)  # write the rewards for each weight vector here
        for j in range(npop):
            w_try = w + sigma * N[j]  # apply the weights with noise to our function
            R[j] = f(w_try) # apply new weights to the function and record in N
        A = (R - np.mean(R)) / np.std(R) # weighted rewards for each vector
        w = w + alpha / (npop * sigma) * np.dot(N.T, A) # update the initial weights with the weighted sum of all reward coefficients from the vector population
    return(w)

n_iter = 1000
npop = 50    # population size
sigma = 0.1    # noise standard deviation
alpha = 0.001  # learning rate
dim = 3
n_episodes = 100
w = np.random.randn(dim) # initial guess)
print(nes(npop, n_iter, sigma, alpha, f, w, dim))