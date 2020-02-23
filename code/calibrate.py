from scipy.optimize import least_squares, basinhopping
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from fourier import fft_calls
from heston import heston_fourier_call, heston_cf

"""Calibrate to some options to get good initial estimate, than calibrate to the rest, try
also lower max_nfev"""


def heston_calibrate_leastsquares(market_data):

    SPOT = 10000
    RATE = 0

    def call(x):
        LAMBDA, VBAR, RHO, XI, V0 = x
        params = {'LAMBDA': LAMBDA, 'VBAR': VBAR, 'RHO': RHO, 'XI': XI, 'V0': V0, 'SPOT': SPOT, 'RATE': RATE}
        # return np.sum([heston_fourier_call(ttm, strike, params) - price * SPOT for strike, price, ttm in market_data.values[0:5]])
        n = len(market_data.values)
        return np.sqrt(np.sum([(heston_fourier_call(ttm, strike, params) - price * SPOT) ** 2 for strike, price, ttm in market_data.values[0:6]]) / n)

    x0 = np.array([1, 0.3, -0.4, 0.7, 0.3])
    bnds = (0, 0, -1, 0, 0), (np.inf, np.inf, 1, np.inf, np.inf)

    return least_squares(call, x0, bounds=bnds, verbose=2)


def heston_calibrate_basinhopping(market_data):

    SPOT = 10000
    RATE = 0

    def call(x):
        LAMBDA, VBAR, RHO, XI, V0 = x
        params = {'LAMBDA': LAMBDA, 'VBAR': VBAR, 'RHO': RHO, 'XI': XI, 'V0': V0, 'SPOT': SPOT, 'RATE': RATE}
        n = len(market_data.values)
        return np.sqrt(np.sum([(heston_fourier_call(ttm, strike, params) - price * SPOT) ** 2 for strike, price, ttm in market_data.values[0:2]]) / n)

    x0 = np.array([1, 0.3, -0.4, 0.7, 0.3])

    class MyBounds(object):
        def __init__(self, xmin=(0, 0, -1, 0, 0), xmax=(np.inf, np.inf, 1, np.inf, np.inf)):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    bounds = MyBounds()

    return basinhopping(call, x0, accept_test=bounds, disp=True)


# def heston_calibrate_fft(market_data):
#
#     SPOT = 10000
#     RATE = 0
#
#     def call_fft(x):
#         LAMBDA, VBAR, RHO, XI, V0 = x
#         params = {'LAMBDA': LAMBDA, 'VBAR': VBAR, 'RHO': RHO, 'XI': XI, 'V0': V0, 'SPOT': SPOT, 'RATE': RATE}
#
#         total = np.array([])
#
#         for ttm in set(market_data['ttm']):
#             d = market_data[market_data['ttm'] == ttm][['strike', 'price']]
#             x = d['strike']
#             y = d['price']
#             price = interp1d(x, y)
#             inc = (np.log(max(x)) - np.log(min(x))) / 128
#             strikes = np.exp(np.arange(np.log(min(x)) + inc, np.log(max(x)), inc))
#             total = np.concatenate([total, fft_calls(ttm, strikes, heston_cf, params) - np.vectorize(price)(strikes) * SPOT])
#
#         return total
#
#     x0 = np.array([1, 0.3, -0.4, 0.7, 0.3])
#     bnds = (0, 0, -1, 0, 0), (np.inf, np.inf, 1, np.inf, np.inf)
#
#     return least_squares(call_fft, x0, bounds=bnds, verbose=2)


def obtain_market_data(date):

    fname = os.path.join("_input", f"{date}.csv")

    df = pd.read_csv(fname)
    df = df[df['type'] == 'C'][['maturity', 'strike', 'mark_price']]

    def ttm(maturity):
        today = datetime.datetime.strptime(date, '%Y%m%d')
        maturity = datetime.datetime.strptime(maturity, '%d%b%y')

        assert maturity > today

        # depends on the day count conventions
        return (maturity - today).days / 365.25

    df['ttm'] = df['maturity'].apply(ttm)
    df = df.rename(columns={'mark_price': 'price'})

    return df.drop(columns=['maturity'])


def plot_market_data(market_data):
    fig, ax = plt.subplots()
    ax.scatter(market_data['strike'], market_data['ttm'], marker='x')
    fig.savefig(r"_output/market_data_points.pdf")


market_data = obtain_market_data('20200212')
# plot_market_data(market_data)

# output = heston_calibrate_fft(market_data)
# print(output)

output = heston_calibrate_leastsquares(market_data)
print(output)

with open("_output/calibration.txt", "w") as fp:
    fp.write(str(output))
