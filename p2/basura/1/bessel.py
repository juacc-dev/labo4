from scipy.special import jv
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 1, 1000, dtype=complex)

x = (-1 + 1j) * t

y = -jv(2, x) / jv(0, x)

plt.plot(t, y.real, label="Parte real")
plt.plot(t, y.imag, label="Parte imaginaria")

plt.legend()
plt.grid(True)
plt.show()
