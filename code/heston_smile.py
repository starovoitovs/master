import numpy as np
import pandas as pd

from models.black_scholes import bs_implied_vol
from models.heston import heston_fourier_dampened_call


def calculate_heston_smile(ttm, log_strikes, ps):

    calls = np.array([heston_fourier_dampened_call(ttm, np.exp(log_strike), ps, damp=10.1, cutoff=1e2) for log_strike in log_strikes])
    ivs = np.array([bs_implied_vol(ttm, np.exp(log_strike), 1, 0, call, otype='CALL')
                    for log_strike, call in zip(log_strikes, calls)])

    df = pd.DataFrame(zip(log_strikes, ivs), columns=['logstrikes', 'iv']).set_index('logstrikes')
    df.to_csv(f"_output/smiles/heston.smile")
