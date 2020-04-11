import unittest
from scipy.special import gamma as Gamma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from black_scholes import bs_call, bs_implied_vol, bs_cf, bs_put
from fourier import fft_calls, fourier_call, fourier_call_dampened_itm, fourier_call_dampened_otm
from heston import heston_fourier_call, heston_critical_time, heston_cf, heston_explicit_call, heston_critical_moment
from rough_heston import rough_heston_critical_time, rough_heston_cf_pade, rough_heston_fourier_call, h_pade, \
    rough_heston_cf_adams
from xutils import fracderivative
from params import params


class TestBSMImpliedVol(unittest.TestCase):

    TTM = 1
    SPOT = 100
    STRIKE = 90
    RATE = 0.05
    VOL = 0.1
    PRICE = 10

    def test_call_price(self):
        bs_price = bs_call(self.TTM, self.STRIKE, self.SPOT, self.RATE, self.VOL)
        implied_vol = bs_implied_vol(self.TTM, self.STRIKE, self.SPOT, self.RATE, bs_price)
        self.assertAlmostEqual(self.VOL, implied_vol)

    def test_implied_vol(self):
        implied_vol = bs_implied_vol(self.TTM, self.STRIKE, self.SPOT, self.RATE, self.PRICE)
        bs_price = bs_call(self.TTM, self.STRIKE, self.SPOT, self.RATE, implied_vol)
        self.assertAlmostEqual(self.PRICE, bs_price)


class TestFractionalDerivative(unittest.TestCase):

    def test_alpha_one(self):
        integral = fracderivative(lambda t: t, 1, -1)
        self.assertAlmostEqual(integral, 1 / 2, places=3)

    def test_alpha_one_half(self):
        integral = fracderivative(lambda t: t ** (1 / 2), 1, -1/2)
        self.assertAlmostEqual(integral, 1 * Gamma(3 / 2) / Gamma(2), places=3)


class TestRoughHestonExplosionTime(unittest.TestCase):

    def test_compare_heston_critical_time(self):
        for moment in np.arange(-18, -12, 1):
            self.assertAlmostEqual(rough_heston_critical_time(moment, params['HESTON']),
                                   heston_critical_time(moment, params['HESTON']))


class TestBSMFourier(unittest.TestCase):

    def test_bsm_fourier_pricing(self):

        ttm = 0.1

        ps = {
            'SPOT': 1,
            'RATE': 0.03,
            'SIGMA': 0.1,
        }

        strikes = np.exp(np.linspace(-1, 1, 32)) * ps['SPOT']

        prices = []
        columns = []

        cf = bs_cf

        columns += ['BSM CALL']
        prices += [[bs_call(ttm, strike, ps['SPOT'], ps['RATE'], ps['SIGMA']) for strike in strikes]]

        columns += ['BSM PUT']
        prices += [[bs_put(ttm, strike, ps['SPOT'], ps['RATE'], ps['SIGMA']) for strike in strikes]]

        columns += ['Fourier Non-Damped CALL']
        prices += [[fourier_call(strike, ttm, cf, ps) for strike in strikes]]

        columns += ['Fourier Damped ITM CALL 0.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=0.1) for strike in strikes]]

        columns += ['Fourier Damped ITM CALL 1.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=1.1) for strike in strikes]]

        columns += ['Fourier Damped ITM CALL 4.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=4.1) for strike in strikes]]

        columns += ['Fourier Damped ITM PUT -1.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=-1.1) for strike in strikes]]

        columns += ['Fourier Damped ITM PUT -2.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=-2.1) for strike in strikes]]

        columns += ['Fourier Damped ITM PUT -4.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=-4.1) for strike in strikes]]

        columns += ['Fourier Damped OTM CALL']
        prices += [[fourier_call_dampened_otm(strike, ttm, cf, ps) for strike in strikes]]

        df = pd.DataFrame(np.transpose(prices), columns=columns, index=strikes)

        df.to_csv(r"../_output/bsm_prices.csv")


class TestHestonFourier(unittest.TestCase):

    def test_bsm_fourier_pricing(self):

        ttm = 0.1

        ps = params['CUSTOM']

        strikes = np.exp(np.linspace(-1, 1, 32)) * ps['SPOT']

        prices = []
        columns = []

        cf = heston_cf

        columns += ['Fourier Non-Damped']
        prices += [[fourier_call(strike, ttm, cf, ps) for strike in strikes]]

        columns += ['Fourier Damped ITM CALL 0.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=0.1) for strike in strikes]]

        columns += ['Fourier Damped ITM CALL 1.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=1.1) for strike in strikes]]

        columns += ['Fourier Damped ITM CALL 4.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=4.1) for strike in strikes]]

        columns += ['Fourier Damped ITM PUT -1.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=-1.1) for strike in strikes]]

        columns += ['Fourier Damped ITM PUT -2.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=-2.1) for strike in strikes]]

        columns += ['Fourier Damped ITM PUT -4.1']
        prices += [[fourier_call_dampened_itm(strike, ttm, cf, ps, damp=-4.1) for strike in strikes]]

        columns += ['Fourier Damped OTM']
        prices += [[fourier_call_dampened_otm(strike, ttm, cf, ps) for strike in strikes]]

        df = pd.DataFrame(np.transpose(prices), columns=columns, index=strikes)

        df.to_csv(r"../_output/heston_prices.csv")


class TestHestonCallPrice(unittest.TestCase):

    # https://www.mathworks.com/help/fininst/optbyhestonni.html#d117e211346
    def test_heston_fourier_call_price(self):
        TTM = 0.5
        strikes = np.arange(76, 86, 2)
        prices = [7.0401, 5.8053, 4.7007, 3.7316, 2.8991]
        for strike, price in zip(strikes, prices):
            self.assertAlmostEqual(heston_explicit_call(TTM, strike, params['HESTON']),
                                   heston_fourier_call(TTM, strike, params['HESTON']))

    # https://www.mathworks.com/matlabcentral/fileexchange/25771-heston-option-pricer
    def test_heston_fourier_call_price2(self):
        TTM = 0.25
        strikes = np.arange(85, 120, 5)
        prices = [15.9804, 11.4069, 7.2125, 3.9295, 2.1213, 1.2922, 0.8625]

        for strike, price in zip(strikes, prices):
            self.assertAlmostEqual(heston_explicit_call(TTM, strike, params['HESTON2']),
                                   heston_fourier_call(TTM, strike, params['HESTON2']))


class TestHestonFFTPricing(unittest.TestCase):

    def test_heston_fft_pricing(self):
        params['SPOT'] = 1

        ttm = 1
        strikes = np.exp(np.linspace(-1, 1, 128)) * params['SPOT']

        dcs = [heston_fourier_call(ttm, strike, params['HESTON']) for strike in strikes]
        fcs = fft_calls(ttm, strikes, heston_cf, params['HESTON'])

        self.assertAlmostEqual(dcs, fcs)


class TestRoughHestonCharacteristicFunction(unittest.TestCase):

    def test_compare_heston_characteristic_function(self):

        us = np.linspace(-5, 5, 41)
        ttm = 1
        ys1 = [rough_heston_cf_adams(u, ttm, params['CUSTOM']) for u in us]
        ys2 = [heston_cf(u, ttm, params['CUSTOM']) for u in us]

        import matplotlib.pyplot as plt
        plt.plot(us, ys1)
        plt.plot(us, ys2)
        plt.show()

    def test_compare_pade_and_fractional_adams(self):
        us = np.linspace(-50, 50, 41)
        ttm = 1

        ps = params['HESTON3']
        ps['ALPHA'] = 0.99
        ps['RHO'] = -0.6

        import matplotlib.pyplot as plt

        # Heston
        ys3 = [heston_cf(u, ttm, ps) for u in us]
        plt.plot(us, ys3, label="Heston")

        # Rough Heston Pade
        for alpha in 0.99,:
            ps['ALPHA'] = alpha
            ys2 = np.vectorize(rough_heston_cf_pade)(us, ttm, ps)
            plt.plot(us, ys2, label=f"Rough Heston (Pade {alpha})")

        # Rough Heston Adams
        for alpha in 0.99,:
            ps['ALPHA'] = alpha
            ys2 = [rough_heston_cf_adams(u, ttm, ps) for u in us]
            plt.plot(us, ys2, label=f"Rough Heston (Adams {alpha})")

        plt.legend()
        plt.show()

    def test_pade_limiting_case_characteristic_function(self):

        us = np.linspace(-5, 5, 41)
        ttm = 1

        ps = params['CUSTOM']

        ys3 = np.vectorize(heston_cf)(us, ttm, ps)
        plt.plot(us, ys3, label="Heston")

        for alpha in 0.6, 0.8, 0.9, 0.99, 0.999, 0.99999999:
            ps['ALPHA'] = alpha
            ys2 = np.vectorize(rough_heston_cf_pade)(us, ttm, ps)

            plt.plot(us, ys2, label=f"Rough Heston Pade {alpha}")

        plt.legend()
        plt.show()


class TestRoughHestonFourierCallPrice(unittest.TestCase):

    # https://www.mathworks.com/help/fininst/optbyhestonni.html#d117e211346
    def test_rough_heston_fourier_call_price(self):

        strikes = np.arange(76, 86, 1)
        ttms = np.arange(0.05, 0.35, 0.05)
        ps = params['HESTON']

        output = {
            'HESTON': pd.DataFrame(np.nan, index=strikes, columns=ttms),
            'ROUGHHESTON': pd.DataFrame(np.nan, index=strikes, columns=ttms),
        }

        for ttm in ttms:
            print(f"ALPHA: 1; TTM: {ttm}")
            ivs = [bs_implied_vol(ttm, strike, ps['SPOT'], ps['RATE'], heston_fourier_call(ttm, strike, ps))
                   for strike in strikes]

            output['HESTON'][ttm] = ivs

        print(output['HESTON'])

        output['HESTON'].to_csv(r"_output/heston.csv")

        # change model to rough model
        ps['ALPHA'] = 0.6

        for ttm in ttms:
            print(f"ALPHA: 0.6; TTM: {ttm}")
            ivs = [bs_implied_vol(ttm, strike, ps['SPOT'], ps['RATE'], rough_heston_fourier_call(ttm, strike, ps))
                   for strike in strikes]

            output['ROUGHHESTON'][ttm] = ivs

        output['ROUGHHESTON'].to_csv(r"_output/rough_heston.csv")
