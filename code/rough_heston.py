import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import newton
from scipy.special import gamma as Gamma

from fourier import fourier_call, fourier_call_dampened_itm
from vie import VIESolver
from xutils import fracderivative


def h_pade_small_time(u, t, params):

    def beta(n):
        betas = [-u * (u + 1j) / 2]
        for k in range(1, n + 1):
            betas += [np.sum([betas[i] * betas[j] * Gamma(1 + i*params['ALPHA']) / Gamma(1 + (i+1)*params['ALPHA']) * Gamma(1 + j*params['ALPHA']) / Gamma(1 + (j+1)*params['ALPHA']) if i+j == k-2 else 0 for i in range(k-1) for j in range(k-1)]) + 1j * params['RHO'] * u * betas[k - 1] * Gamma(1 + (k - 1) * params['ALPHA']) / Gamma(1 + k * params['ALPHA'])]

        return betas[n]

    return np.sum([Gamma(1 + j * params['ALPHA']) / Gamma(1 + (j + 1) * params['ALPHA']) * beta(j) * params['XI'] ** j * t ** ((j+1) * params['ALPHA']) for j in range(30)])


# [Gatheral, Radoicic 2019] eqn (4.5); Pade approximant of the solution of fractional Ricatti equation (h33)

def h_pade(u, t, params):

    assert u != 0

    A = np.sqrt(u * (u + 1j) - params['RHO'] ** 2 * u ** 2)
    r_minus = - 1j * params['RHO'] * u - A

    gamma0 = 1
    gamma1 = -1
    gamma2 = 1 + r_minus/2/A * Gamma(1 - 2*params['ALPHA']) / Gamma(1 - params['ALPHA'])**2

    beta0 = -u * (u + 1j) / 2
    beta1 = 1j * params['RHO'] * u * Gamma(1) / Gamma(1 + params['ALPHA']) * beta0
    beta2 = beta0 ** 2 * Gamma(1) ** 2 / Gamma(1 + params['ALPHA']) ** 2 + \
            1j * params['RHO'] * u * Gamma(1 + params['ALPHA']) / Gamma(1 + 2 * params['ALPHA']) * beta1

    # small-time
    b1 = Gamma(1) / Gamma(1 + params['ALPHA']) * beta0 * params['XI'] ** 1
    b2 = Gamma(1 + params['ALPHA']) / Gamma(1 + 2 * params['ALPHA']) * beta1 * params['XI'] ** 2
    b3 = Gamma(1 + 2 * params['ALPHA']) / Gamma(1 + 3 * params['ALPHA']) * beta2 * params['XI'] ** 3

    # large-time
    g0 = r_minus * gamma0 / A ** 0 / Gamma(1)
    g1 = r_minus * gamma1 / A ** 1 / Gamma(1 - params['ALPHA']) * params['XI'] ** (-1)
    g2 = r_minus * gamma2 / A ** 2 / Gamma(1 - 2 * params['ALPHA']) * params['XI'] ** (-2)

    q1 = (b1**2*g1 - b1*b2*g2 + b1*g0**2 - b2*g0*g1 - b3*g0*g2 + b3*g1**2) / (b1**2*g2 + 2*b1*g0*g1 + b2*g0*g2 - b2*g1**2 + g0**3)
    q2 = (b1**2*g0 - b1*b2*g1 - b1*b3*g2 + b2**2*g2 + b2*g0**2 - b3*g0*g1) / (b1**2*g2 + 2*b1*g0*g1 + b2*g0*g2 - b2*g1**2 + g0**3)
    q3 = (b1**3 + 2*b1*b2*g0 + b1*b3*g1 - b2**2*g1 + b3*g0**2) / (b1**2*g2 + 2*b1*g0*g1 + b2*g0*g2 - b2*g1**2 + g0**3)

    p1 = b1
    p2 = (b1**3*g1 + b1**2*g0**2 + b1*b2*g0*g1 - b1*b3*g0*g2 + b1*b3*g1**2 + b2**2*g0*g2 - b2**2*g1**2 + b2*g0**3) / (b1**2*g2 + 2*b1*g0*g1 + b2*g0*g2 - b2*g1**2 + g0**3)
    p3 = g0*q3

    return (p1*t + p2*t**2 + p3*t**3) / (1 + q1*t + q2*t**2 + q3*t**3)


# characteristic function
def rough_heston_cf_pade(u, t, params, n=100):

    assert 1/2 < params['ALPHA'] < 1

    if u == 0:
        return 1

    xs = np.linspace(0, t, n + 1)
    ys = np.array([h_pade(u, x, params) for x in xs])

    h = (xs[1] - xs[0])

    # rectangle rule
    int1 = np.sum(ys) * h

    # frac int
    # Podlubny 7.2 and 7.5 for coefficients
    pwr = -(1 - params['ALPHA'])
    coeffs = np.cumprod([1.] + [(1 - (pwr + 1) / j) for j in range(1, len(xs))])
    fintalpha = np.sum(coeffs * np.flip(ys)) / h ** pwr

    return np.exp(params['VBAR'] * params['LAMBDA'] * int1 +
                  params['V0'] * fintalpha +
                  1j * u * (np.log(params['SPOT']) + params['RATE'] * t))


def rough_heston_cf_adams(u, t, params, n=100):

    solver = VIESolver(params)
    xs, ys = solver.solve(u, 0, t, n)

    h = (xs[1] - xs[0])

    # rectangle rule
    int1 = np.cumsum(ys) * h

    if 1/2 < params['ALPHA'] < 1:
        # fintalpha = np.vectorize(fracderivative)(lambda s: interp1d(xs, ys)(s), xs, -(1 - params['ALPHA']))
        pwr = -(1 - params['ALPHA'])
        coeffs = np.cumprod([1.] + [(1 - (pwr + 1) / j) for j in range(1, len(xs))])
        # fintalpha = np.sum(ys * np.flip(coeffs)) / h ** pwr
        fintalpha = np.array([np.sum(np.flip(coeffs[0:i]) * ys[0:i]) for i in range(1, len(xs) + 1)])
    elif params['ALPHA'] == 1:
        fintalpha = ys
    else:
        raise ValueError("Alpha must be in (1/2, 1].")

    cf = np.exp(params['VBAR'] * params['LAMBDA'] * int1 +
                params['V0'] * fintalpha +
                1j * u * (np.log(params['SPOT']) + params['RATE'] * xs))

    return cf[-1]


# [GGP18] Algorithms 7.5 and 7.6
def rough_heston_critical_time(u, params, n=100):

    def c1(s): return s * (s - 1) / 2
    def c2(s): return params['RHO'] * params['XI'] * s - params['LAMBDA']
    c3 = params['XI'] ** 2 / 2

    def d1(s): return c1(s) * c3
    def d2(s): return c2(s)

    assert c1(u) > 0
    assert c2(u) / 2 >= 0

    def v(k): return Gamma(params['ALPHA'] * k + 1) / Gamma(params['ALPHA'] * k - params['ALPHA'] + 1)

    a = [d1(u) / v(1)]

    for i in range(1, n):
        a += [(d2(u) * a[-1] + np.sum(a[:-1] * np.flip(a[:-1]))) / v(i + 1)]

    # case A, left-hand side
    if u < params['LAMBDA'] / params['XI'] / params['RHO']:
        return (a[-1] * (n ** (1 - params['ALPHA'])) * (Gamma(params['ALPHA']) ** 2) / (params['ALPHA'] ** params['ALPHA']) / Gamma(2 * params['ALPHA'])) ** (-1 / params['ALPHA'] / (n + 1))

    # case B, right-hand side, get lower bound
    else:
        return np.abs(a[-1]) ** (-1 / params['ALPHA'] / n)


# okay i guess here is actually left moment
def rough_heston_critical_moment(t, params):
    root = newton(lambda u: rough_heston_critical_time(u, params) - t, x0=20)
    return root


def rough_heston_fourier_call(ttm, strike, params, damp):
    return fourier_call_dampened_itm(strike, ttm, rough_heston_cf_adams, params, damp)
