import scipy.fft
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

TMAX = 10
N = 10000


def fourier_transform(y, sampling_freq):
    yfft = scipy.fft.fft(y)

    cut = y.size // 2

    xf = np.linspace(0, sampling_freq / 2, cut)
    yf = 2.0 / y.size * np.abs(yfft[:cut])

    return xf, yf


f = 2
gamma = 0.4

t = np.linspace(0, TMAX, N)
y = np.sin(2 * np.pi * f * t) * np.exp(-gamma * t)

# plt.plot(t, y)
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Amplitud")
# plt.show()

sampling_freq = max(t) / t.size

xf, yf = fourier_transform(y, sampling_freq)

peaks, _ = scipy.signal.find_peaks(yf, height=yf[0])

print(f"Pico en {xf[peaks]}")

plt.semilogx(xf, yf)
for peak in peaks:
    plt.axvline(x=xf[peak])
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.show()
