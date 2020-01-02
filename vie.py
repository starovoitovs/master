from scipy.special import gamma
import numpy as np
from numpy import vectorize
import matplotlib.pyplot as plt

# rough heston params
alpha = 0.75
l = 1
rho = 0.1
nu = 1

# discretization params

T = 10
N = 1000

delta = T / N

t = np.linspace(0, delta * N, N + 1)

# generate matrix A

a = np.zeros((N + 2, N + 2))

for k in range(0, N + 1):
    for j in range(0, k + 2):
        if j == 0:
            a[j, k + 1] = delta ** alpha / gamma(alpha + 2) * (k ** (alpha + 1) - (k - alpha) * ((k + 1) ** alpha))
        elif j == k:
            a[j, k + 1] = delta ** alpha / gamma(alpha + 2)
        else:
            a[j, k + 1] = delta ** alpha / gamma(alpha + 2) * (
                    (k - j + 2) ** (alpha + 1) + (k - j) ** (alpha + 1) - 2 * (k - j + 1) ** (alpha + 1))

# generate matrix B

b = np.zeros((N + 2, N + 2))

for k in range(0, N + 1):
    for j in range(0, k + 1):
        b[j, k + 1] = delta ** alpha / gamma(alpha + 1) * ((k - j + 1) ** alpha - (k - j) ** alpha)

# solver

F = lambda p, x: (p ** 2 - p) / 2 + (p * rho * nu - l) * x + (nu ** 2) / 2 * (x ** 2)
F = vectorize(F)

def critical_value(p):
    h = [0]
    kk = 1
    try:
        while kk < N + 1:
            hp = np.sum(b[0:kk, kk] * F(p, h[0:kk]))
            val = np.sum(a[0:kk, kk] * F(p, h[0:kk])) + a[kk, kk] * F(p, hp)
            if val > 1e4:
                raise OverflowError()
            h.append(val)
            kk += 1
    except OverflowError as err:
        # plt.plot(t[0:kk], h)
        # plt.show()
        return t[kk]


x = np.linspace(2, 10, 9)
y = []

for v in np.linspace(2, 10, 9):
    y.append(critical_value(v))

plt.plot(x, y)
plt.show()
