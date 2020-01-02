import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

# speed of mean reversion
l = 2

# mean reversion level
theta = 0.1

# vvol
nu = 0.2

# roughness, H + 1/2
alpha = 0.6

# correlation
rho = -0.8

c1 = lambda u: u * (u - 1) / 2
c2 = lambda u: rho * nu * u - l
c3 = nu ** 2 / 2

d1 = lambda u: c1(u) * c3
d2 = lambda u: c2(u)


def explosion_time(u, n):
    v = lambda n: gamma(alpha * n + 1) / gamma(alpha * n - alpha + 1)

    a = np.array([d1(u) / v(1)])

    for i in range(1, n):
        an = (d2(u) * a[-1] + np.sum(a[:-1] * np.flip(a[:-1]))) / v(i + 1)
        a = np.append(a, [an])

    # case A, left-hand side
    if u < l / nu / rho:
        return (a[-1] * (n ** (1 - alpha)) * (gamma(alpha) ** 2) / (alpha ** alpha) / gamma(2 * alpha)) ** (
                -1 / alpha / (n + 1))
    # case B, right-hand side, get lower bound
    else:
        return np.abs(a[-1]) ** (-1 / alpha / n)


vec_explosion_time = np.vectorize(lambda x: explosion_time(x, 100))

x = np.linspace(-16, -6, 51)
y = vec_explosion_time(x)

plt.plot(x, y)
plt.show()
