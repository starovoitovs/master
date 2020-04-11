import numpy as np
import pandas as pd
import multiprocessing as mp

from models.black_scholes import bs_implied_vol
from models.heston import heston_explicit_call
from models.rough_heston import rough_heston_fourier_call


def pricer_rough_heston_iv(args):

    ttm, log_strike, ps, damp_call = args

    print(f"Pricing for Log-strike: {log_strike}")

    # calculating OTM prices
    strike = ps['SPOT'] * np.exp(log_strike)
    damp = damp_call if log_strike >= 0 else -1.1
    otype = 'CALL' if log_strike >= 0 else 'PUT'

    price = rough_heston_fourier_call(ttm, strike, ps, damp=damp, cutoff=5.)
    iv = bs_implied_vol(ttm, strike, ps['SPOT'], ps['RATE'], price, otype=otype)
    print(f"Log-strike: {log_strike}; Price: {price}; IV: {iv}")

    return log_strike, iv


def pricer_rough_heston_price(args):

    ttm, log_strike, ps = args

    print(f"Pricing {log_strike}")

    strike = ps['SPOT'] * np.exp(log_strike)
    price = rough_heston_fourier_call(ttm, strike, ps, damp=0.3, cutoff=5.)

    return price


def pricer_heston(args):

    ttm, log_strike, ps = args

    print(f"Pricing {log_strike}")

    strike = ps['SPOT'] * np.exp(log_strike)
    price = heston_explicit_call(ttm, strike, ps)

    return price


def calculate_rough_heston_smile(ttm, log_strikes, ps, cores=14, damp=24.1):

    # rough heston

    args = [(ttm, log_strike, ps, damp) for log_strike in log_strikes]

    p = mp.Pool(cores)
    ivs = p.map(pricer_rough_heston_iv, args)

    df = pd.DataFrame(ivs, columns=['logstrikes', 'iv']).set_index('logstrikes')
    df.to_csv(f"_output/smiles/rough_heston__{ps['ALPHA']}.smile")
