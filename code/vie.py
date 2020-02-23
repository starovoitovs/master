import numpy as np
from scipy.special import gamma as Gamma


# [ER16] Theorem 4.1 and Section 5

class VIESolver:

    def __init__(self, params):

        assert -1 / np.sqrt(2) < params['RHO'] < 1 / np.sqrt(2)

        self.delta = 0
        self.n = 0
        self.a = np.zeros(())
        self.b = np.zeros(())

        self._params = params

    def _update_mat_a(self):

        self.a = np.zeros((self.n + 2, self.n + 2))

        for k in range(0, self.n + 1):
            for j in range(0, k + 2):
                if j == 0:
                    self.a[j, k + 1] = self.delta ** self._params['ALPHA'] / Gamma(self._params['ALPHA'] + 2) * \
                                       (k ** (self._params['ALPHA'] + 1) - (k - self._params['ALPHA']) * ((k + 1) ** self._params['ALPHA']))
                elif j == k:
                    self.a[j, k + 1] = self.delta ** self._params['ALPHA'] / Gamma(self._params['ALPHA'] + 2)
                else:
                    self.a[j, k + 1] = self.delta ** self._params['ALPHA'] / Gamma(self._params['ALPHA'] + 2) * \
                                       ((k - j + 2) ** (self._params['ALPHA'] + 1) + (k - j) ** (self._params['ALPHA'] + 1) - 2 * (k - j + 1) ** (self._params['ALPHA'] + 1))

    def _update_mat_b(self):

        self.b = np.zeros((self.n + 2, self.n + 2))

        for k in range(0, self.n + 1):
            for j in range(0, k + 1):
                self.b[j, k + 1] = self.delta ** self._params['ALPHA'] / Gamma(self._params['ALPHA'] + 1) * \
                                   ((k - j + 1) ** self._params['ALPHA'] - (k - j) ** self._params['ALPHA'])

    def solve(self, p, t0, t1, n):

        def f(z):
            return (- p ** 2 - 1j * p) / 2 + \
                   (p * self._params['RHO'] * self._params['XI'] * 1j - self._params['LAMBDA']) * z + \
                   (self._params['XI'] ** 2) / 2 * (z ** 2)

        _delta = (t1 - t0) / n

        if _delta != self.delta or n != self.n:
            self.delta = _delta
            self.n = n

            # update matrices
            self._update_mat_a()
            self._update_mat_b()

        x = np.linspace(t0, t1, self.n + 1)
        y = np.array([0])

        k = 1

        while k < self.n + 1:
            try:

                hp = np.sum(self.b[0:k, k] * np.vectorize(f)(y[0:k]))
                val = np.sum(self.a[0:k, k] * np.vectorize(f)(y[0:k])) + self.a[k, k] * np.vectorize(f)(hp)

                # use with care
                if val > 1e3:
                    raise OverflowError()

                y = np.append(y, val)
                k += 1

            except OverflowError:
                x = x[slice(len(y))]
                break

        assert len(x) == len(y)

        return x, y
