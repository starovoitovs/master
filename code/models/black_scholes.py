from scipy.optimize import newton
from scipy.stats import norm
import numpy as np


def bs_implied_vol(ttm, strike, spot, rate, price, otype='CALL', x0=0.5):

    def target(sigma):
        if otype.upper() == 'CALL':
            return bs_call(ttm, strike, spot, rate, sigma) - price
        elif otype.upper() == 'PUT':
            return bs_put(ttm, strike, spot, rate, sigma) - price
        else:
            raise ValueError("Option type must be CALL or PUT")

    def vega(sigma):
        d1 = (np.log(spot / strike) + (rate + sigma ** 2 / 2) * ttm) / sigma / np.sqrt(ttm)
        return spot * norm.pdf(d1) * np.sqrt(ttm)

    try:
        return newton(target, x0=x0, fprime=vega)
    except RuntimeError as e:
        print(e)
        return np.nan


# same for calls and puts
def bs_vega(ttm, strike, spot, rate, sigma):
    d1 = (np.log(spot / strike) + (rate + sigma ** 2 / 2) * ttm) / sigma / np.sqrt(ttm)
    return spot * norm.pdf(d1) * np.sqrt(ttm)


def bs_call(ttm, strike, spot, rate, sigma):
    d1 = (np.log(spot / strike) + (rate + sigma ** 2 / 2) * ttm) / sigma / np.sqrt(ttm)
    d2 = d1 - sigma * np.sqrt(ttm)

    return norm.cdf(d1) * spot - norm.cdf(d2) * strike * np.exp(-rate * ttm)


def bs_put(ttm, strike, spot, rate, sigma):
    d1 = (np.log(spot / strike) + (rate + sigma ** 2 / 2) * ttm) / sigma / np.sqrt(ttm)
    d2 = d1 - sigma * np.sqrt(ttm)

    return norm.cdf(-d2) * strike * np.exp(-rate * ttm) - norm.cdf(-d1) * spot


def bs_cf(u, t, params):
    return np.exp(1j * u * (np.log(params['SPOT']) + (params['RATE'] - 1/2 * params['SIGMA'] ** 2 * t)) - 1/2 * u ** 2 * params['SIGMA'] ** 2 * t)
