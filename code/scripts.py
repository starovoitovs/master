import numpy as np
import pandas as pd

from models.black_scholes import bs_implied_vol
from models.heston import heston_fourier_call
from models.rough_heston import rough_heston_fourier_call
from plotting import plot_skew, plot_surface


def calculate_atm_skew(ps):

    h = 1/100

    tdelta = 1e-5
    ttms = np.concatenate([
            np.arange(tdelta, 11 * tdelta, 2 * tdelta),
            np.arange(10 * tdelta, 11 * 10 * tdelta, 10 * 2 * tdelta)[1:],
            np.arange(10 * 10 * tdelta, 11 * 10 * 10 * tdelta, 10 * 10 * 2 * tdelta)[1:],
            np.arange(10 * 10 * 10 * tdelta, 11 * 10 * 10 * 10 * tdelta, 10 * 10 * 10 * 2 * tdelta)[1:],
            np.arange(10 * 10 * 10 * 10 * tdelta, 11 * 10 * 10 * 10 * 10 * tdelta, 10 * 10 * 10 * 10 * 2 * tdelta)[1:],
           ])

    df = pd.DataFrame(np.nan, index=ttms, columns=['HESTON', 'RHESTON'])

    for ttm in ttms:
        print(f"ALPHA: 1; TTM: {ttm}")
        cp = bs_implied_vol(ttm, 80 + h, ps['SPOT'], ps['RATE'], heston_fourier_call(ttm, 80 + h, ps))
        cm = bs_implied_vol(ttm, 80 - h, ps['SPOT'], ps['RATE'], heston_fourier_call(ttm, 80 - h, ps))

        df['HESTON'][ttm] = np.abs((cp - cm) / h)

    # change model to rough model
    ps['ALPHA'] = 0.6

    for ttm in ttms:
        print(f"ALPHA: 0.6; TTM: {ttm}")
        cp = bs_implied_vol(ttm, 80 + h, ps['SPOT'], ps['RATE'], rough_heston_fourier_call(ttm, 80 + h, ps))
        cm = bs_implied_vol(ttm, 80 - h, ps['SPOT'], ps['RATE'], rough_heston_fourier_call(ttm, 80 - h, ps))

        df['RHESTON'][ttm] = np.abs((cp - cm) / h)

    df.to_csv(r'_output/skew.csv')
    plot_skew(r'_output/skew.csv')


def calculate_vol_surfaces(ps):

    strikes = np.arange(76, 86, 1)
    ttms = np.arange(0.05, 0.35, 0.05)

    output = {
        'HESTON': pd.DataFrame(np.nan, index=ttms, columns=strikes),
        'RHESTON': pd.DataFrame(np.nan, index=ttms, columns=strikes),
    }

    for strike in strikes:
        print(f"ALPHA: 1; STRIKE: {strike}")
        ivs = [bs_implied_vol(ttm, strike, ps['SPOT'], ps['RATE'], heston_fourier_call(ttm, strike, ps))
               for ttm in ttms]

        output['HESTON'][strike] = ivs

    output['HESTON'].to_csv(r'_output/surfaces/heston.surface')

    # change model to rough model
    ps['ALPHA'] = 0.6

    for ttm in ttms:
        print(f"ALPHA: 0.6; TTM: {ttm}")
        ivs = [bs_implied_vol(ttm, strike, ps['SPOT'], ps['RATE'], rough_heston_fourier_call(ttm, strike, ps))
               for strike in strikes]

        output['RHESTON'][ttm] = ivs

    output['RHESTON'].to_csv(r'_output/surfaces/rough_heston.surface')

    plot_surface(r"_output/surfaces/heston.surface")
    plot_surface(r"_output/surfaces/rough_heston.surface")
