import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from models.heston import heston_critical_time, heston_cf
from models.rough_heston import rough_heston_critical_time, rough_heston_critical_moment, rough_heston_cf_adams
from mpl_toolkits.mplot3d import Axes3D


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
def plot_variance_smile(ax, filename,
                        left_slope=None, left_intercept=0.,
                        right_slope=None, right_intercept=0.):

    df = pd.read_csv(filename, index_col=0)
    ax.plot(df.index, df['iv'] ** 2, label="Variance smile")

    if right_slope is not None:
        ts = df.index[df.index > 1]
        ys = [right_slope * t + right_intercept for t in ts]
        ax.plot(ts, ys, linestyle='dashed')

    if left_slope is not None:
        ts = df.index[df.index < -1]
        ys = [-left_slope * t + left_intercept for t in ts]
        ax.plot(ts, ys, linestyle='dashed')


def plot_rough_heston_critical_time(ps):

    xs = np.linspace(-16, -12.5, 21)
    ys = [rough_heston_critical_time(x, ps) for x in xs]

    plt.plot(xs, ys)
    plt.show()


def plot_heston_critical_time(ps):

    xs = np.arange(3, 11, 0.1)
    ys = [heston_critical_time(x, ps) for x in xs]

    plt.plot(xs, ys)
    plt.show()


def plot_rough_heston_cf(ps, u=4.55, ttm=1, n=500):

    ts = np.linspace(0, ttm, n + 1)

    # # heston
    # ys1 = [heston_cf(-1j * u, t, ps).real for t in ts]
    # plt.plot(ts, ys1, label="Heston")

    # # rough heston pade
    # ys2 = [rough_heston_cf_pade(-1j * u, t, ps, n).real for t in ts]
    # plt.plot(ts, ys2, label="Rough Heston Pade")

    # rough heston adams
    ys3 = rough_heston_cf_adams(-1j * u, ttm, ps, n, entire=True).real
    plt.plot(ts[0:len(ys3)], ys3, label="Rough Heston Adams")

    plt.legend()
    plt.show()


def plot_critical_slope(ps):

    us = np.linspace(2, 22, 52)
    ys = [rough_heston_critical_time(u, ps) for u in us]

    u = rough_heston_critical_moment(1, ps)
    eps = np.sqrt(np.finfo(float).eps) * (1.0 + u)
    sigma = (rough_heston_critical_time(u + eps, ps) - rough_heston_critical_time(u - eps, ps)) / (2.0 * eps)
    plt.plot(us, ys)

    ys2 = sigma * (us - u) + 1
    ys2 = ys2[ys2 > 0]
    plt.plot(us[0:len(ys2)], ys2)

    plt.show()