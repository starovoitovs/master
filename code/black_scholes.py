import math

from scipy.optimize import minimize, newton, root_scalar
from scipy.stats import norm
import numpy as np

# tolerance

EPS = 1e-9


def bs_implied_vol(ttm, strike, spot, rate, price, otype='CALL'):

    def f(sigma):
        if otype.upper() == 'CALL':
            return bs_call(ttm, strike, spot, rate, sigma) - price
        elif otype.upper() == 'PUT':
            return bs_put(ttm, strike, spot, rate, sigma) - price
        else:
            raise ValueError("Option type must be CALL or PUT")

    root = newton(f, x0=1)
    return root

    # res = minimize(f, x0=np.array([1]), tol=1e-24)
    # return res.x[0]


# def bs_implied_vol(ttm, strike, spot, rate, price, otype='CALL'):
#     m = spot / strike / math.exp(-rate * ttm)
#     sigma = math.sqrt(2 * abs(math.log(m)) / ttm)
#     delta = EPS + 1
#
#     while delta > EPS:
#
#         dplus = (math.log(spot / strike) + (rate + math.pow(sigma, 2) / 2) * ttm) / sigma / math.sqrt(ttm)
#         dminus = dplus - sigma * math.sqrt(ttm)
#         cbs = spot * norm.cdf(dplus) - strike * math.exp(-rate * ttm) * norm.cdf(dminus)
#         dcbs = spot * math.sqrt(ttm) * norm.pdf(dplus)
#         sigma2 = sigma + (price - cbs) / dcbs
#         delta = abs(sigma2 - sigma)
#         sigma = sigma2
#
#     return sigma


def bs_call(ttm, strike, spot, rate, vol):
    d1 = (np.log(spot / strike) + (rate + vol ** 2 / 2) * ttm) / vol / np.sqrt(ttm)
    d2 = d1 - vol * np.sqrt(ttm)

    return norm.cdf(d1) * spot - norm.cdf(d2) * strike * np.exp(-rate * ttm)


def bs_put(ttm, strike, spot, rate, vol):
    d1 = (np.log(spot / strike) + (rate + vol ** 2 / 2) * ttm) / vol / np.sqrt(ttm)
    d2 = d1 - vol * np.sqrt(ttm)

    return norm.cdf(-d2) * strike * np.exp(-rate * ttm) - norm.cdf(-d1) * spot


def bs_cf(u, t, params):
    return np.exp(1j * u * (np.log(params['SPOT']) + (params['RATE'] - 1/2 * params['SIGMA'] * t)) - 1/2 * u ** 2 * params['SIGMA'] ** 2 * t)
