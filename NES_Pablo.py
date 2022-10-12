import numpy as np

np.random.seed(0)


# the function we want to optimize
def f(w):
    center = np.array([0.5, 0.1, -0.3])
    reward = -np.sum(np.square(center - w))
    return reward


def nes(npop, n_iter, sigma, alpha, f, w0):
    w = w0
    for i in range(n_iter):
        if i % 20 == 0:
            print('iter {}. w: reward: {}'.format(i, f(w)))

        N = np.random.randn(npop, w.shape[0])
        R = np.zeros(npop)
        for j in range(npop):
            w_try = w + sigma * N[j]
            R[j] = f(w_try)

            # standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R)) / np.std(R)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w + alpha / (npop * sigma) * np.dot(N.T, A)
    return w


if __name__ == "__main__":
    w0 = np.random.randn(3)
    print(nes(npop=50, n_iter=300, sigma=0.1, alpha=0.001, f=f, w0=w0))