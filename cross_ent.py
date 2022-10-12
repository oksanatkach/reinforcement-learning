# row, then column

import numpy as np
import operator
np.random.seed(0)
# the function we want to optimize
def f(w):
    center = np.array([0.5, 0.1, -0.3])
    reward = -np.sum(np.square(center - w))
    return reward

def cross(mu, sigma, elite_fraq, npop, n_iter, f, dim):
    for i in range(n_iter):
        # generate a population of weight vectors, already weighted by mu and sigma
        R = np.zeros(npop)  # write the rewards for each weight vector here later
        N = np.random.multivariate_normal(mu, np.diag(sigma), size=npop)

        for j in range(npop): # iterate over all weight vectors
            w_try = N[j]
            R[j] = f(w_try) # apply new weights to the function and record reward in N

        R = enumerate(R)
        R = sorted(R, key=operator.itemgetter(1), reverse=True) # sort by ascending reward
        elite = R[:int(len(R)*elite_fraq)] # get top rewards

        weights = np.zeros([len(elite), dim]) # empty array for weights
        for j in range(len(elite)):
            weights[j] = N[elite[j][0]] # get the weights that gave top rewards

        mu = np.mean(weights, axis=0)
        assert mu.shape[0] == 3
        sigma = np.std(weights, axis=0)
        assert sigma.shape[0] == 3
    return mu, sigma

dim = 3
mu = np.zeros(dim)
sigma = np.ones(dim)
elite_fraq = 0.4
npop = 100
n_iter = 100000

mu, sigma = cross(mu, sigma, elite_fraq, npop, n_iter, f = f, dim = dim)
print(mu)
print(sigma)