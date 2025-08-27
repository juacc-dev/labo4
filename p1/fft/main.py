import scipy.fft
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

TMAX = 10
N = 10000


def fourier_transform(y, sampling_freq):
    yfft = scipy.fft.fft(y)

    cut = y.size // 2

    xf = np.linspace(0, sampling_freq, cut)
    yf = 2.0 / y.size * np.abs(yfft[:cut])

    return xf, yf


def peaks(y):
    x_peaks, y_peaks = scipy.signal.find_peaks(y, height=y[0])

    return x_peaks, y_peaks


f = 2
gamma = 0.4

t = np.linspace(0, TMAX, N)
y = np.sin(2 * np.pi * f * t) * np.exp(-gamma * t)

# plt.plot(t, y)
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Amplitud")
# plt.show()

sampling_freq = TMAX / N

xf, yf = fourier_transform(y, sampling_freq)

x_peaks, _ = peaks(yf)

print(f"Pico en {xf[x_peaks]}")

plt.semilogx(xf, yf)
plt.axvline(x=xf[x_peaks[0]])
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.show()
