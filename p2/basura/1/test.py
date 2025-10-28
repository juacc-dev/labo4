import pandas as pd
import numpy as np
import pylabo.plot as plot
import pylabo.fit as fit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N = 30
SIGMA = 1.0 / 20


def f(x, a, b, c):

    z1 = a * x ** 2 + b * x + c
    z2 = c * np.exp(- (x / (a + b)) ** 2)

    z = np.column_stack((z1, z2))
    return np.asarray(z, dtype=np.float64)


x = np.linspace(-1, 1, N)

noise = np.random.rand(N, 2) * SIGMA

z = f(x, 1, 1, 0) + noise

print(f"x shape: {x.shape}")
print(f"z shape: {z.shape}")

p_opt, p_cov = curve_fit(
    f,
    x,
    z,
    p0=[0.9, 1.2, 0.001],
    sigma=SIGMA,
    absolute_sigma=True
)

x_fit = np.linspace(-1, 1, 1000)
z_fit = f(x_fit, *p_opt)

plt.plot(x, z[0], '.', label="Data 1")
plt.plot(x, z[1], '.', label="Data 2")
plt.plot(x_fit, z_fit[0], '-', label="Fit 1")
plt.plot(x_fit, z_fit[1], '-', label="Fit 2")

plt.legend()
plt.grid(True)
plt.show()
