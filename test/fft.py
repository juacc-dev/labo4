import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11


def fft(x, y):
    sampling_freq = x.size / x[-1]
    cut = y.size // 2

    yfft = np.fft.fft(y)
    # yfft = 2.0 / len(y) * np.abs(yfft)

    # xf = np.arange(0, sampling_freq, 1 / (x[-1] - x[0]))
    xf = np.linspace(0, sampling_freq / 2, int(x.size / 2))
    yf = 2.0 / y.size * np.abs(yfft[:cut])

    return xf, yf


def ifft(xf, yf):
    yifft = scipy.fft.ifft(yf)
    yifft = 2.0 / yf.size * np.abs(yifft)

    cut = yf.size // 2

    # x = np.linspace(0, 1 / xf[0], )
    # x = np.arange(0, 1 / xf[1], 1 / xf[-1])
    x = None
    y = np.abs(yifft[:cut])

    return x, y


a = 0.1
x0 = 50

xf = np.linspace(0, 100, 1000)
yf = np.exp(-((xf - x0) / a) ** 2)

x, y = ifft(xf, yf)
plt.plot(y)
plt.show()
