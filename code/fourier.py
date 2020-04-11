import numpy as np

from scipy.integrate import quad


# e.g. Carr Madan 1999
def fourier_call(strike, ttm, cf, params):

    def int1(u): return (np.exp(-1j * u * np.log(strike)) * cf(u - 1j, ttm, params) / 1j / u / cf(-1j, ttm, params)).real
    def int2(u): return (np.exp(-1j * u * np.log(strike)) * cf(u, ttm, params) / 1j / u).real

    p1 = 1/2 + 1/np.pi * quad(int1, 0, np.inf)[0]
    p2 = 1/2 + 1/np.pi * quad(int2, 0, np.inf)[0]

    return params['SPOT'] * p1 - strike * np.exp(- params['RATE'] * ttm) * p2


# e.g. FGGS10 Appendix III
# we get call for damp > 0 and put for damp < -1
def fourier_call_dampened_itm(strike, ttm, cf, params, damp=1.9, cutoff=np.inf):

    def psi(u):
        return cf(u - (damp + 1) * 1j, ttm, params) / \
               (damp + 1j * u) / \
               (damp + 1 + 1j * u)

    def int1(u): return np.exp(-1j * u * np.log(strike)) * psi(u)

    itm = np.exp(-params['RATE'] * ttm) * \
          np.exp(-damp * np.log(strike)) / np.pi * \
          quad(int1, 0, cutoff)[0]

    return itm


def fourier_call_dampened_otm(strike, ttm, cf, params, damp=0.1):

    def zeta(u):
        return np.exp(-params['RATE'] * ttm) * (1 / (1 + 1j * u) + np.exp(params['RATE'] * ttm) / 1j / u + cf(u - 1j, ttm, params) / u / (u - 1j))

    def gamma(u):
        return (zeta(u - 1j * damp) - zeta(u + 1j * damp)) / 2

    def int2(u): return np.exp(-1j * u * np.log(strike)) * gamma(u)

    otm = 1 / np.sinh(damp * np.log(strike)) / np.pi * quad(int2, 0, np.inf)[0]

    return otm


def fft_calls(ttm, strikes, cf, params):

    # all log-strikes need to be cyclic to use FFT
    diffs = np.log(strikes)[1:] - np.log(strikes)[:-1]
    assert np.allclose(diffs, diffs[0])
    assert np.isclose(np.log(strikes[-1]), -np.log(strikes[0]))

    b = np.log(strikes[-1])
    n = len(strikes)

    eta = 2 * b / (n - 1)
    delta = 2 * np.pi / n / eta

    # @todo good choice for dampening parameter?
    damp = 0.9

    def psi(u):
        return cf(u - (damp + 1) * 1j, ttm, params) / \
               (damp + 1j * u) / \
               (damp + 1 + 1j * u)

    def zeta(u):
        return np.exp(-params['RATE'] * ttm) * (1 / (1 + 1j * u) + np.exp(params['RATE'] * ttm) / 1j / u + cf(u - 1j, ttm, params) / u / (u - 1j))

    def gamma(u):
        return (zeta(u - 1j * damp) - zeta(u + 1j * damp)) / 2

    # STOCHASTIC VOLATILITY MODELS: CALIBRATION, PRICING AND HEDGING (3.19)
    x1 = [np.exp(1j * b * delta * (j - 1)) * psi(delta * (j - 1)) * delta / 3 * (3 + (-1) ** j - int(j == 1))
          for j in np.arange(1, n + 1, 1)]
    x2 = [np.exp(1j * b * delta * (j - 1)) * gamma(delta * (j - 1)) * delta / 3 * (3 + (-1) ** j - int(j == 1))
          for j in np.arange(1, n + 1, 1)]

    itm = (np.exp(- params['RATE'] * ttm) / np.pi * np.exp(-damp * np.log(strikes)) * np.fft.fftshift(np.fft.fft(x1))).real
    otm = (np.exp(- params['RATE'] * ttm) / np.pi / np.sinh(damp * np.log(strikes)) * np.fft.fftshift(np.fft.fft(x2))).real

    return itm
