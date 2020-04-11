import numpy as np


# Podlubny [Ch. 7]
def fracderivative(f, t, alpha, n=10):

    if t == 0:
        return 0

    h = t / n

    # consider short memory (i.e. taking less coefficients in the past)
    # chapter 7.5
    coeffs = np.cumprod([1] + [(1 - (alpha + 1) / j) for j in range(1, n)])
    fns = np.array([f(t - j * h) for j in range(n)])

    return np.sum(coeffs * fns) / h ** alpha


def lee_right(ttm, u):
    x = u - 1
    return (2 - 4 * (np.sqrt(x ** 2 + x) - x)) / ttm


def lee_left(ttm, u):
    return (2 - 4 * (np.sqrt(u ** 2 - u) + u)) / ttm
