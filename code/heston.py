import numpy as np
from scipy.optimize import newton, minimize
from scipy.integrate import quad

from fourier import fourier_call, fourier_call_dampened_itm, fourier_call_dampened_otm


# formulas from https://pdfs.semanticscholar.org/f03b/afa84e4d7e63d51f17a3cf77993f1e6f9fce.pdf [Warrick Poklewski-Koziell]

def heston_cf(u, t, params):

    alpha = (- u ** 2 - 1j * u) / 2
    beta = params['LAMBDA'] - params['RHO'] * params['XI'] * 1j * u
    gamma = params['XI'] ** 2 / 2

    d = np.sqrt(beta ** 2 - 4 * alpha * gamma)

    r_pos = (beta + d) / 2 / gamma
    r_neg = (beta - d) / 2 / gamma
    g = r_neg / r_pos

    def C(t):
        return r_neg * t - 2 / params['XI'] ** 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))

    def D(t):
        return r_neg * (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))

    return np.exp(params['VBAR'] * params['LAMBDA'] * C(t) +
                  params['V0'] * D(t) +
                  1j * u * (np.log(params['SPOT']) + params['RATE'] * t))


# Heston (1993)
# https://quant.stackexchange.com/questions/18684/heston-model-option-price-formula

def heston_explicit_call(ttm, strike, params):

    u1 = 1/2
    u2 = -1/2

    a = params['LAMBDA'] * params['VBAR']

    b1 = params['LAMBDA'] - params['RHO'] * params['XI']
    b2 = params['LAMBDA']

    def d1(phi): return np.sqrt((b1 - params['RHO'] * params['XI'] * 1j * phi) ** 2 - params['XI'] ** 2 * (2j * u1 * phi - phi ** 2))
    def d2(phi): return np.sqrt((b2 - params['RHO'] * params['XI'] * 1j * phi) ** 2 - params['XI'] ** 2 * (2j * u2 * phi - phi ** 2))

    def g1(phi): return (b1 - params['RHO'] * params['XI'] * 1j * phi + d1(phi)) / (b1 - params['RHO'] * params['XI'] * 1j * phi - d1(phi))
    def g2(phi): return (b2 - params['RHO'] * params['XI'] * 1j * phi + d2(phi)) / (b2 - params['RHO'] * params['XI'] * 1j * phi - d2(phi))

    def C1(phi): return params['RATE'] * 1j * phi * ttm + a / params['XI'] ** 2 * ((b1 - params['RHO'] * params['XI'] * 1j * phi + d1(phi)) * ttm - 2 * np.log((1 - g1(phi) * np.exp(d1(phi) * ttm)) / (1 - g1(phi))))
    def C2(phi): return params['RATE'] * 1j * phi * ttm + a / params['XI'] ** 2 * ((b2 - params['RHO'] * params['XI'] * 1j * phi + d2(phi)) * ttm - 2 * np.log((1 - g2(phi) * np.exp(d2(phi) * ttm)) / (1 - g2(phi))))

    def D1(phi): return (b1 - params['RHO'] * params['XI'] * 1j * phi + d1(phi)) / params['XI'] ** 2 * (1 - np.exp(d1(phi) * ttm)) / (1 - g1(phi) * np.exp(d1(phi) * ttm))
    def D2(phi): return (b2 - params['RHO'] * params['XI'] * 1j * phi + d2(phi)) / params['XI'] ** 2 * (1 - np.exp(d2(phi) * ttm)) / (1 - g2(phi) * np.exp(d2(phi) * ttm))

    def f1(phi): return np.exp(C1(phi) + D1(phi) * params['V0'] + 1j * phi * np.log(params['SPOT']))
    def f2(phi): return np.exp(C2(phi) + D2(phi) * params['V0'] + 1j * phi * np.log(params['SPOT']))

    def int1(phi): return (np.exp(-1j * phi * np.log(strike)) * f1(phi) / 1j / phi).real
    def int2(phi): return (np.exp(-1j * phi * np.log(strike)) * f2(phi) / 1j / phi).real

    p1 = 1/2 + quad(int1, 0, 1e3)[0] / np.pi
    p2 = 1/2 + quad(int2, 0, 1e3)[0] / np.pi

    return params['SPOT'] * p1 - strike * np.exp(-params['RATE'] * ttm) * p2


def heston_fourier_call(ttm, strike, params):
    return fourier_call(strike, ttm, heston_cf, params)


def heston_fourier_dampened_call(ttm, strike, params, damp):
    return fourier_call_dampened_itm(strike, ttm, heston_cf, params, damp)


# GGP18 (2.12)
def heston_critical_time(u, params):

    def e0(u): return (params['RHO'] * params['XI'] * u - params['LAMBDA']) / 2
    def e1(u): return e0(u) ** 2 - params['XI'] ** 2 * u * (u - 1) / 4

    if e1(u) < 0:
        return (np.pi / 2 - np.arctan(e0(u) / np.sqrt(np.abs(e1(u))))) / np.sqrt(np.abs(e1(u)))
    elif e0(u) > 0:
        return np.log((e0(u) + np.sqrt(e1(u)) / (e0(u) - np.sqrt(e1(u))))) / 2 / np.sqrt(e1(u))
    else:
        return np.inf


# On refined volatility smile expansion in the Heston model (2.12)
def heston_critical_slope(u, params):
    delta = (u * params['RHO'] * params['XI'] - params['LAMBDA']) ** 2 - params['XI'] ** 2 * (u ** 2 - u)

    x0 = (u * params['RHO'] * params['XI'] - params['LAMBDA'])
    x1 = 2 * params['RHO'] * params['XI']
    x2 = params['XI'] ** 2 * (2 * u - 1)

    return - heston_critical_time(u, params) * (x1 * x0 - x2) / 2 / delta \
           - ((x2 - x1 * x0) * x0 - x1 * delta) / delta / (x0 ** 2 - delta)


def heston_critical_moment(t, params):
    root = newton(lambda u: heston_critical_time(u, params) - t, x0=10)
    return root


# On refined volatility smile expansion in the Heston model (3.11)
def heston_density_expansion(x, t, params):
    assert params['RHO'] < 0

    s_pos = heston_critical_moment(t, params)
    # 2.6
    sigma = - heston_critical_slope(s_pos, params)

    # @todo kappa is unknown!
    kappa = 1

    a = params['VBAR'] * params['LAMBDA']
    b = params['XI'] * params['RHO'] * s_pos - params['LAMBDA']
    c = params['XI']

    x1 = 2 * np.sqrt(b**2 + 2 * b * c * params['RHO'] * s_pos + c**2 * s_pos * (1 - (1 - params['RHO']**2) * s_pos))
    x2 = c**2 * s_pos * (s_pos - 1) * np.sinh(np.sqrt(b**2 + 2 * b * c * params['RHO'] * s_pos + c**2 * s_pos * (1 - (1 - params['RHO'] ** 2) * s_pos)) / 2)

    # Remark 12
    a1 = + 1 / 2 / np.sqrt(np.pi) * (2 * params['V0']) ** (1/4 - a / c ** 2) \
         * c ** (2 * a / c ** 2 - 1/2) \
         * sigma ** (-a / c ** 2 - 1/4) \
         * np.exp(- params['V0'] * (b / c ** 2) + kappa / c**2 / sigma**2 - a * t / c**2 * b) \
         * (x1 / x2) ** (2 * a / c**2)

    # Theorem 2
    a2 = 2 * np.sqrt(2 * params['VO']) / params['XI'] / np.sqrt(sigma)
    a3 = s_pos + 1

    return a1 * \
           (x ** (-a3)) * \
           np.exp(a2 * np.sqrt(np.log(x))) * (np.log(x)) ** (-3/4 + x / params['XI'] ** 2)
