import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad

from fourier import fourier_call, fourier_call_dampened_itm


# formulas from https://pdfs.semanticscholar.org/f03b/afa84e4d7e63d51f17a3cf77993f1e6f9fce.pdf [Warrick Poklewski-Koziell]

def heston_cf(u, t, ps):

    alpha = (- u ** 2 - 1j * u) / 2
    beta = ps['LAMBDA'] - ps['RHO'] * ps['XI'] * 1j * u
    gamma = ps['XI'] ** 2 / 2

    d = np.sqrt(beta ** 2 - 4 * alpha * gamma)

    r_pos = (beta + d) / 2 / gamma
    r_neg = (beta - d) / 2 / gamma
    g = r_neg / r_pos

    def C(t):
        return r_neg * t - 2 / ps['XI'] ** 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))

    def D(t):
        return r_neg * (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))

    return np.exp(ps['VBAR'] * ps['LAMBDA'] * C(t) +
                  ps['V0'] * D(t) +
                  1j * u * (np.log(ps['SPOT']) + ps['RATE'] * t))


# Heston (1993)
# https://quant.stackexchange.com/questions/18684/heston-model-option-price-formula

def heston_explicit_call(ttm, strike, ps, cutoff=1e3):

    u1 = 1/2
    u2 = -1/2

    a = ps['LAMBDA'] * ps['VBAR']

    b1 = ps['LAMBDA'] - ps['RHO'] * ps['XI']
    b2 = ps['LAMBDA']

    def d1(phi): return np.sqrt((b1 - ps['RHO'] * ps['XI'] * 1j * phi) ** 2 - ps['XI'] ** 2 * (2j * u1 * phi - phi ** 2))
    def d2(phi): return np.sqrt((b2 - ps['RHO'] * ps['XI'] * 1j * phi) ** 2 - ps['XI'] ** 2 * (2j * u2 * phi - phi ** 2))

    def g1(phi): return (b1 - ps['RHO'] * ps['XI'] * 1j * phi + d1(phi)) / (b1 - ps['RHO'] * ps['XI'] * 1j * phi - d1(phi))
    def g2(phi): return (b2 - ps['RHO'] * ps['XI'] * 1j * phi + d2(phi)) / (b2 - ps['RHO'] * ps['XI'] * 1j * phi - d2(phi))

    def C1(phi): return ps['RATE'] * 1j * phi * ttm + a / ps['XI'] ** 2 * ((b1 - ps['RHO'] * ps['XI'] * 1j * phi + d1(phi)) * ttm - 2 * np.log((1 - g1(phi) * np.exp(d1(phi) * ttm)) / (1 - g1(phi))))
    def C2(phi): return ps['RATE'] * 1j * phi * ttm + a / ps['XI'] ** 2 * ((b2 - ps['RHO'] * ps['XI'] * 1j * phi + d2(phi)) * ttm - 2 * np.log((1 - g2(phi) * np.exp(d2(phi) * ttm)) / (1 - g2(phi))))

    def D1(phi): return (b1 - ps['RHO'] * ps['XI'] * 1j * phi + d1(phi)) / ps['XI'] ** 2 * (1 - np.exp(d1(phi) * ttm)) / (1 - g1(phi) * np.exp(d1(phi) * ttm))
    def D2(phi): return (b2 - ps['RHO'] * ps['XI'] * 1j * phi + d2(phi)) / ps['XI'] ** 2 * (1 - np.exp(d2(phi) * ttm)) / (1 - g2(phi) * np.exp(d2(phi) * ttm))

    def f1(phi): return np.exp(C1(phi) + D1(phi) * ps['V0'] + 1j * phi * np.log(ps['SPOT']))
    def f2(phi): return np.exp(C2(phi) + D2(phi) * ps['V0'] + 1j * phi * np.log(ps['SPOT']))

    def int1(phi): return (np.exp(-1j * phi * np.log(strike)) * f1(phi) / 1j / phi).real
    def int2(phi): return (np.exp(-1j * phi * np.log(strike)) * f2(phi) / 1j / phi).real

    p1 = 1/2 + quad(int1, 0, cutoff)[0] / np.pi
    p2 = 1/2 + quad(int2, 0, cutoff)[0] / np.pi

    return ps['SPOT'] * p1 - strike * np.exp(-ps['RATE'] * ttm) * p2


def heston_fourier_call(ttm, strike, ps):
    return fourier_call(strike, ttm, heston_cf, ps)


def heston_fourier_dampened_call(ttm, strike, ps, damp=0.3, cutoff=5.):
    return fourier_call_dampened_itm(strike, ttm, heston_cf, ps, damp, cutoff)


# GGP18 (2.12)
def heston_critical_time(u, ps):

    def e0(u): return (ps['RHO'] * ps['XI'] * u - ps['LAMBDA']) / 2
    def e1(u): return e0(u) ** 2 - ps['XI'] ** 2 * u * (u - 1) / 4

    if e1(u) < 0:
        return (np.pi / 2 - np.arctan(e0(u) / np.sqrt(np.abs(e1(u))))) / np.sqrt(np.abs(e1(u)))
    elif e0(u) > 0:
        return np.log((e0(u) + np.sqrt(e1(u)) / (e0(u) - np.sqrt(e1(u))))) / 2 / np.sqrt(e1(u))
    else:
        return np.inf


# On refined volatility smile expansion in the Heston model (2.12)
def heston_critical_slope(u, ps):
    delta = (u * ps['RHO'] * ps['XI'] - ps['LAMBDA']) ** 2 - ps['XI'] ** 2 * (u ** 2 - u)

    x0 = (u * ps['RHO'] * ps['XI'] - ps['LAMBDA'])
    x1 = 2 * ps['RHO'] * ps['XI']
    x2 = ps['XI'] ** 2 * (2 * u - 1)

    return - heston_critical_time(u, ps) * (x1 * x0 - x2) / 2 / delta \
           - ((x2 - x1 * x0) * x0 - x1 * delta) / delta / (x0 ** 2 - delta)


def heston_critical_curvature(u, ps):
    eps = np.sqrt(np.finfo(float).eps) * (1.0 + u)
    sigma = (heston_critical_slope(u + eps, ps) - heston_critical_slope(u - eps, ps)) / (2.0 * eps)

    return - sigma


def heston_critical_moment(ttm, ps, x0=20):
    root = newton(lambda u: heston_critical_time(u, ps) - ttm, x0)
    return root


# On refined volatility smile expansion in the Heston model (3.11)
def heston_density_expansion(x, ttm, ps):
    assert ps['RHO'] < 0

    # 2.6
    s_plus = heston_critical_moment(ttm, ps)
    sigma = - heston_critical_slope(s_plus, ps)
    kappa = heston_critical_curvature(s_plus, ps)

    a = ps['VBAR'] * ps['LAMBDA']
    b = ps['XI'] * ps['RHO'] * s_plus - ps['LAMBDA']
    c = ps['XI']

    x1 = 2 * np.sqrt(b**2 + 2 * b * c * ps['RHO'] * s_plus + c**2 * s_plus * (1 - (1 - ps['RHO']**2) * s_plus))
    x2 = c**2 * s_plus * (s_plus - 1) * np.sinh(np.sqrt(b**2 + 2 * b * c * ps['RHO'] * s_plus + c**2 * s_plus * (1 - (1 - ps['RHO'] ** 2) * s_plus)) / 2)

    # Remark 12
    a1 = + 1 / 2 / np.sqrt(np.pi) * (2 * ps['V0']) ** (1/4 - a / c ** 2) \
         * c ** (2 * a / c ** 2 - 1/2) \
         * sigma ** (-a / c ** 2 - 1/4) \
         * np.exp(- ps['V0'] * (b / c ** 2) + kappa / c ** 2 / sigma ** 2 - a * ttm / c ** 2 * b) \
         * (x1 / x2) ** (2 * a / c**2)

    # Theorem 2
    a2 = 2 * np.sqrt(2 * ps['VO']) / ps['XI'] / np.sqrt(sigma)
    a3 = s_plus + 1

    return a1 * \
           (x ** (-a3)) * \
           np.exp(a2 * np.sqrt(np.log(x))) * \
           (np.log(x)) ** (-3/4 + x / ps['XI'] ** 2)


def heston_call_asymptotic(ttm, log_strike, ps):

    s_plus = heston_critical_moment(ttm, ps)
    sigma = - heston_critical_slope(s_plus, ps)
    kappa = heston_critical_curvature(s_plus, ps)

    a = ps['VBAR'] * ps['LAMBDA']
    b = - ps['LAMBDA']
    c = ps['XI']

    x1 = np.sqrt(b**2 + 2 * b * c * ps['RHO'] * s_plus + c**2 * s_plus * (1 - (1 - ps['RHO']**2) * s_plus))
    x2 = c**2 * s_plus * (s_plus - 1) * np.sinh(x1 / 2)

    # Remark 12
    a1 = + 1 / 2 / np.sqrt(np.pi) * (2 * ps['V0']) ** (1/4 - a / c ** 2) \
         * c ** (2 * a / c ** 2 - 1/2) \
         * sigma ** (-a / c ** 2 - 1/4) \
         * np.exp(- ps['V0'] * ((b + s_plus * ps['RHO'] * c) / c ** 2) + kappa / c**2 / sigma**2 - a * ttm / c**2 * (b + c * ps['RHO'] * s_plus)) \
         * (2 * x1 / x2) ** (2 * a / c**2)

    a2 = 2 * np.sqrt(2 * ps['V0']) / ps['XI'] / np.sqrt(sigma)
    a3 = s_plus + 1

    # a1 = 2311.69
    # a2 = 12.3533
    # a3 = 33.2124

    return a1 / (-a3 + 1) / (-a3 + 2) * \
           np.exp(log_strike) ** (-a3 + 2) * \
           np.exp(a2 * log_strike ** 1/2) * \
           log_strike ** (-3/4 + a / c ** 2)


# FGGS (1.7)
def heston_implied_variance_asymptotic(ttm, log_strike, ps):

    s_plus = heston_critical_moment(ttm, ps)
    sigma = - heston_critical_slope(s_plus, ps)
    kappa = heston_critical_curvature(s_plus, ps)

    a = ps['VBAR'] * ps['LAMBDA']
    b = - ps['LAMBDA']
    c = ps['XI']

    x1 = np.sqrt(b**2 + 2 * b * c * ps['RHO'] * s_plus + c**2 * s_plus * (1 - (1 - ps['RHO']**2) * s_plus))
    x2 = c**2 * s_plus * (s_plus - 1) * np.sinh(x1 / 2)

    # Remark 12
    a1 = + 1 / 2 / np.sqrt(np.pi) * (2 * ps['V0']) ** (1/4 - a / c ** 2) \
         * c ** (2 * a / c ** 2 - 1/2) \
         * sigma ** (-a / c ** 2 - 1/4) \
         * np.exp(- ps['V0'] * ((b + s_plus * ps['RHO'] * c) / c ** 2) + kappa / c**2 / sigma**2 - a * ttm / c**2 * (b + c * ps['RHO'] * s_plus)) \
         * (2 * x1 / x2) ** (2 * a / c**2)

    a2 = 2 * np.sqrt(2 * ps['V0']) / ps['XI'] / np.sqrt(sigma)
    a3 = s_plus + 1

    b1 = np.sqrt(2) * (np.sqrt(a3 - 1) - np.sqrt(a3 - 2))
    b2 = a2 / np.sqrt(2) * (1 / np.sqrt(a3 - 2) - 1 / np.sqrt(a3 - 1))
    b3 = 1 / np.sqrt(2) * (1/4 - a / c ** 2) * (1 / np.sqrt(a3 - 1) - 1 / np.sqrt(a3 - 2))

    print(f"b1: {b1}; b2: {b2}; b3: {b3}")

    return (b1 * log_strike ** 0.5 + b2 + b3 * np.log(log_strike) / log_strike ** 0.5) ** 2 / ttm
