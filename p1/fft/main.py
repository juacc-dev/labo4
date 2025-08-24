import scipy.fft
import numpy as np
import matplotlib.pyplot as plt

TMAX = 10
N = 10000


def fourier_transform(t, y):
    sampling_freq = np.max(t) / t.size

    yfft = scipy.fft.fft(y)

    xf = np.linspace(0, sampling_freq, y.size // 2)
    yf = 2.0 / y.size * np.abs(yfft[:N // 2])

    return xf, yf


f = [
    7, 20
]

t = np.linspace(0, TMAX, N)
y = np.sin(2 * np.pi * f[0] * t) + np.sin(2 * np.pi * f[1] * t)

xf, yf = fourier_transform(t, y)

plt.semilogx(xf, yf)
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.show()
