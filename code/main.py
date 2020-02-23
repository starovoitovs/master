import numpy as np

from black_scholes import bs_implied_vol
from heston import heston_cf, heston_fourier_call, heston_critical_time, heston_critical_moment, heston_explicit_call, \
    heston_fourier_dampened_call
from rough_heston import rough_heston_cf_pade, rough_heston_critical_time, rough_heston_cf_adams, \
    rough_heston_fourier_call, rough_heston_critical_moment
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from params import params
from xutils import lee_right


def plot_rough_heston_critical_time():

    xs = np.linspace(-16, -12.5, 21)
    ys = [rough_heston_critical_time(x, params['GGP18']) for x in xs]

    plt.plot(xs, ys)
    plt.show()


def plot_heston_critical_time():

    ps = params['CUSTOM']

    xs = np.arange(3, 11, 0.1)
    ys = [heston_critical_time(x, ps) for x in xs]

    plt.plot(xs, ys)
    plt.show()


def plot_rough_heston_cf():

    T = 1
    n = 500
    u = 6.5
    # u = -1.4

    ts = np.linspace(0, T, n + 1)

    ps = params['HESTON3']

    # heston
    ys1 = [heston_cf(-1j * u, t, ps).real for t in ts]
    plt.plot(ts, ys1, label="Heston")

    # # rough heston pade
    # ys2 = [rough_heston_cf_pade(-1j * u, t, ps, n).real for t in ts]
    # plt.plot(ts, ys2)

    # rough heston adams
    ys3 = rough_heston_cf_adams(-1j * u, T, ps, n).real
    plt.plot(ts[0:len(ys3)], ys3, label="Rough Heston Adams (alpha = 1)")

    # rough heston adams
    ps['ALPHA'] = 0.6
    ys4 = rough_heston_cf_adams(-1j * u, T, ps, n).real
    plt.plot(ts[0:len(ys4)], ys4, label="Rough Heston Adans (alpha = 0.6)")

    plt.title('both')
    plt.legend()
    plt.show()


def calculate_atm_skew():

    h = 1/100

    tdelta = 1e-5
    ttms = np.concatenate([
            np.arange(tdelta, 11 * tdelta, 2 * tdelta),
            np.arange(10 * tdelta, 11 * 10 * tdelta, 10 * 2 * tdelta)[1:],
            np.arange(10 * 10 * tdelta, 11 * 10 * 10 * tdelta, 10 * 10 * 2 * tdelta)[1:],
            np.arange(10 * 10 * 10 * tdelta, 11 * 10 * 10 * 10 * tdelta, 10 * 10 * 10 * 2 * tdelta)[1:],
            np.arange(10 * 10 * 10 * 10 * tdelta, 11 * 10 * 10 * 10 * 10 * tdelta, 10 * 10 * 10 * 10 * 2 * tdelta)[1:],
           ])

    ps = params['HESTON']

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


def calculate_vol_surfaces():

    strikes = np.arange(76, 86, 1)
    ttms = np.arange(0.05, 0.35, 0.05)

    ps = params['HESTON']

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


def calculate_heston_smile():

    ttm = 1

    ps = params['HESTON3']

    log_strikes = np.arange(-2, 2.2, 0.2)

    df = pd.DataFrame(index=log_strikes, columns=['iv'])
    ivs = []

    u_plus = heston_critical_moment(ttm, ps)
    print(f"Heston Critical moment: {u_plus}")
    print(f"Lee right slope: {lee_right(u_plus, ttm)}")

    for log_strike in log_strikes:

        print(f"LOGSTRIKE: {log_strike}")

        strike = ps['SPOT'] * np.exp(log_strike)

        if log_strike <= 0:
            call = heston_fourier_dampened_call(ttm, strike, ps, damp=0.3)
            ivs += [bs_implied_vol(ttm, strike, ps['SPOT'], ps['RATE'], call, otype='CALL')]
        else:
            put = heston_fourier_dampened_call(ttm, strike, ps, damp=-1.1)
            ivs += [bs_implied_vol(ttm, strike, ps['SPOT'], ps['RATE'], put, otype='PUT')]

    df['iv'] = ivs

    df.to_csv('_output/adams_heston3/heston.smile')
    df.plot().get_figure().show()


def calculate_rough_heston_smile():

    ttm = 1

    ps = params['HESTON3']

    log_strikes = np.arange(-2, 2.2, 0.2)

    # rough heston

    for alpha in 0.6,:

        df = pd.DataFrame(index=log_strikes, columns=['iv'])
        ivs = []

        print(f"ALPHA: {alpha}")

        ps['ALPHA'] = alpha

        for log_strike in log_strikes:

            print(f"LOGSTRIKE: {log_strike}")

            strike = ps['SPOT'] * np.exp(log_strike)

            if log_strike <= 0:
                call = rough_heston_fourier_call(ttm, strike, ps, damp=0.3)
                ivs += [bs_implied_vol(ttm, strike, ps['SPOT'], ps['RATE'], call, otype='CALL')]
            else:
                put = rough_heston_fourier_call(ttm, strike, ps, damp=-1.1)
                ivs += [bs_implied_vol(ttm, strike, ps['SPOT'], ps['RATE'], put, otype='PUT')]

        df['iv'] = ivs

        df.to_csv(f"_output/adams_heston3/rough_heston__{alpha}.smile")


def plot_surface(filename):

    df = pd.read_csv(filename, index_col=0)

    ks = [float(c) for c in df.index]
    ts = [float(c) for c in df.columns]

    K, T = np.meshgrid(ks, ts)
    cs = df.to_numpy().transpose().flatten()

    C = cs.reshape(K.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(K, T, C, cmap=cm.Oranges_r)

    plt.show()


def plot_skew(filename):

    df = pd.read_csv(filename, index_col=0)

    plt.plot(df.index, df['HESTON'])
    plt.plot(df.index, df['RHESTON'])
    plt.show()


# implied variance, not volatility

def plot_variance_smile(ax, filename, left_slope=None, right_slope=None):

    df = pd.read_csv(filename, index_col=0)
    ax.plot(df.index, df['iv'] ** 2)

    if right_slope is not None:
        C = 0
        ts = df.index
        ys = [right_slope * t + C for t in ts]
        ax.plot(ts, ys)

    if left_slope is not None:
        C = -0.12
        ts = df.index[df.index < -1]
        ys = [-left_slope * t + C for t in ts]
        ax.plot(ts, ys, linestyle='dashed')


# calculate_heston_smile()

# fig, ax = plt.subplots()
# plot_variance_smile(ax, r"_output/heston.smile")
# fig.show()

calculate_rough_heston_smile()

# fig, ax = plt.subplots()
#
# plot_variance_smile(ax, r"_output/pade_heston3/ rough_heston__0.6.smile")
# plot_variance_smile(ax, r"_output/pade_heston3/rough_heston__0.7.smile")
# plot_variance_smile(ax, r"_output/pade_heston3/rough_heston__0.8.smile")
# plot_variance_smile(ax, r"_output/pade_heston3/rough_heston__0.9.smile")
# plot_variance_smile(ax, r"_output/pade_heston3/rough_heston__0.95.smile")
# plot_variance_smile(ax, r"_output/pade_heston3/rough_heston__0.99.smile")
#
# ax.set_title("Adams Smiles")
# fig.show()

# plot_rough_heston_cf()
